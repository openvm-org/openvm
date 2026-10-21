// Lean compiler output
// Module: Aesop.RuleTac.Forward
// Imports: public import Init public meta import Init public import Aesop.Forward.Match public import Aesop.RuleTac.Forward.Basic import Batteries.Lean.Meta.UnusedNames import Lean.Meta.CollectFVars import Lean.Meta.Tactic.Apply
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
size_t lean_usize_add(size_t, size_t);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lean_array_get_size(lean_object*);
uint64_t l_Lean_instHashableFVarId_hash(lean_object*);
uint64_t lean_uint64_shift_right(uint64_t, uint64_t);
uint64_t lean_uint64_xor(uint64_t, uint64_t);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_of_nat(lean_object*);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_land(size_t, size_t);
uint8_t l_Lean_instBEqFVarId_beq(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withNewMCtxDepthImp(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_SavedState_restore___redArg(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_LocalDecl_isImplementationDetail(lean_object*);
lean_object* l_Lean_LocalDecl_fvarId(lean_object*);
lean_object* l_Lean_Meta_saveState___redArg(lean_object*, lean_object*);
lean_object* l_Lean_LocalDecl_type(lean_object*);
lean_object* l_Lean_Meta_isExprDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkFVar(lean_object*);
lean_object* lean_st_ref_take(lean_object*);
uint64_t l_Lean_instHashableMVarId_hash(lean_object*);
lean_object* lean_usize_to_nat(size_t);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_instBEqMVarId_beq(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkCollisionNode___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_usize_shift_right(size_t, size_t);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntries(lean_object*, lean_object*);
size_t lean_usize_mul(size_t, size_t);
uint8_t lean_usize_dec_le(size_t, size_t);
lean_object* l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_Meta_abstractMVars(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* lp_aesop_Aesop_isHypRedundantReducibleRigid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lp_aesop_Aesop_isDefEqReducibleRigid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* lean_nat_div(lean_object*, lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
extern lean_object* lp_aesop_Aesop_TraceOption_forward;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
double lean_float_of_nat(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_PersistentArray_push___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* l_Lean_Expr_mvar___override(lean_object*);
lean_object* l_Lean_Meta_synthAppInstances(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_array_get_borrowed(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_getType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_st_mk_ref(lean_object*);
lean_object* l_Lean_Expr_collectFVars(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_fvar___override(lean_object*);
lean_object* l_Lean_Meta_isProof(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkConstWithFreshMVarLevels(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_getForwardHypData(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_tryClearManyS(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_openRuleType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lp_aesop_Aesop_instBEqPremiseIndex_beq(lean_object*, lean_object*);
lean_object* lean_array_get(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_BinderInfo_isInstImplicit(uint8_t);
lean_object* l_Lean_mkAppN(lean_object*, lean_object*);
extern lean_object* lp_aesop_Aesop_forwardHypPrefix;
lean_object* lp_batteries_Lean_LocalContext_getUnusedUserNames(lean_object*, lean_object*, lean_object*);
lean_object* l_Array_toSubarray___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_Script_TacticBuilder_assertHypothesis(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_forwardImplDetailHypName(lean_object*, lean_object*);
lean_object* l_Lean_Meta_ConfigWithKey_setTransparency(uint8_t, lean_object*);
lean_object* l_Lean_MVarId_assertHypotheses(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_withScriptStep___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* l_instMonadLiftBaseIOEIO___lam__0___boxed(lean_object*, lean_object*, lean_object*);
lean_object* l_IO_instMonadLiftSTRealWorldBaseIO___lam__0___boxed(lean_object*, lean_object*, lean_object*);
lean_object* l_instMonadLiftT___lam__0___boxed(lean_object*, lean_object*);
lean_object* l_instMonadLiftTOfMonadLift___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_elabRuleTermForApplyLikeMetaM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instMonadEIO(lean_object*);
lean_object* l_StateRefT_x27_lift___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instMonadLift___lam__0___boxed(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Core_liftIOCore___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_StateRefT_x27_instMonad___redArg(lean_object*);
lean_object* l_instMonadControlStateRefT_x27(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Core_instMonadCoreM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Core_instMonadCoreM___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instFunctorOfMonad___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instFunctorOfMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_pure___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instMonadControlTOfPure___redArg(lean_object*);
lean_object* l_instMonadControlTOfMonadControl___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instMonadControlTOfMonadControl___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_instMonadMetaM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_instMonadMetaM___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_ScriptT_run___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_bind___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_withContext___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(lean_object*, uint8_t);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* lp_aesop_Aesop_CompleteMatch_toMessageData(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_ForwardRuleMatch_apply___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_Lean_MessageData_joinSep(lean_object*, lean_object*);
lean_object* l_Lean_indentD(lean_object*);
extern lean_object* lp_aesop_Aesop_instInhabitedForwardHypData_default;
static lean_once_cell_t lp_aesop_Aesop_RuleTac_ForwardM_instInhabitedContext_default___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_RuleTac_ForwardM_instInhabitedContext_default___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_ForwardM_instInhabitedContext_default;
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_ForwardM_instInhabitedContext;
static const lean_array_object lp_aesop_Aesop_RuleTac_ForwardM_instInhabitedState_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_RuleTac_ForwardM_instInhabitedState_default___closed__0 = (const lean_object*)&lp_aesop_Aesop_RuleTac_ForwardM_instInhabitedState_default___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_RuleTac_ForwardM_instInhabitedState_default___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_RuleTac_ForwardM_instInhabitedState_default___closed__1;
static lean_once_cell_t lp_aesop_Aesop_RuleTac_ForwardM_instInhabitedState_default___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_RuleTac_ForwardM_instInhabitedState_default___closed__2;
static lean_once_cell_t lp_aesop_Aesop_RuleTac_ForwardM_instInhabitedState_default___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_RuleTac_ForwardM_instInhabitedState_default___closed__3;
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_ForwardM_instInhabitedState_default;
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_ForwardM_instInhabitedState;
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_isRedundant_spec__0___redArg(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_isRedundant_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_isRedundant(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_isRedundant___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_isRedundant_spec__0(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_isRedundant_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop___lam__0___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop___lam__0___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__6_spec__12_spec__16_spec__21_spec__25___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__6_spec__12_spec__16_spec__21___redArg(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__6_spec__12_spec__16___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__6_spec__12_spec__16___redArg___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__6_spec__12_spec__16___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__6_spec__12_spec__16_spec__22___redArg(size_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__6_spec__12_spec__16_spec__22___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__6_spec__12_spec__16___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__6_spec__12___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_assign___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__6___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_assign___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_getD___at___00Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__4_spec__8___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_getD___at___00Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__4_spec__8___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__4___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_addTrace___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__3_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_addTrace___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__3_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Lean_addTrace___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__3___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_aesop_Lean_addTrace___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__3___redArg___closed__0;
static const lean_string_object lp_aesop_Lean_addTrace___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__3___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_aesop_Lean_addTrace___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__3___redArg___closed__1 = (const lean_object*)&lp_aesop_Lean_addTrace___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__3___redArg___closed__1_value;
static const lean_array_object lp_aesop_Lean_addTrace___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__3___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Lean_addTrace___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__3___redArg___closed__2 = (const lean_object*)&lp_aesop_Lean_addTrace___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__3___redArg___closed__2_value;
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__8___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__8___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__0_spec__0_spec__2_spec__13_spec__20___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__0_spec__0_spec__2_spec__13___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__0_spec__0_spec__2___redArg(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__0_spec__0_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__0_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__0_spec__1(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__5_spec__10_spec__13_spec__18___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__5_spec__10_spec__13_spec__18___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__5_spec__10_spec__13___redArg(lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__5_spec__10_spec__13___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__5_spec__10___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__5_spec__10___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__5___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_Option_get___at___00Aesop_TraceOption_isEnabled___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__2_spec__4(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Aesop_TraceOption_isEnabled___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__2_spec__4___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__2___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__1(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "instance synthesis failed"};
static const lean_object* lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop___closed__0_value;
static lean_once_cell_t lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop___closed__1;
static const lean_string_object lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "aesop"};
static const lean_object* lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop___closed__2 = (const lean_object*)&lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop___closed__2_value;
static const lean_ctor_object lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop___closed__2_value),LEAN_SCALAR_PTR_LITERAL(29, 147, 5, 234, 46, 7, 240, 183)}};
static const lean_object* lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop___closed__3 = (const lean_object*)&lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop___closed__3_value;
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__7_spec__15_spec__22(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__7_spec__15(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forIn___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop___closed__6 = (const lean_object*)&lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop___closed__6_value;
static lean_once_cell_t lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop___closed__4;
static lean_once_cell_t lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop___closed__5;
static lean_once_cell_t lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop___closed__7;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__7_spec__14_spec__20_spec__26(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__7_spec__14_spec__20(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__7_spec__14(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__7_spec__14_spec__19(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__7_spec__14_spec__19___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forIn___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__7___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__7_spec__14___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__7_spec__15___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__7_spec__14_spec__20___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__7_spec__15_spec__22___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__7_spec__14_spec__20_spec__26___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__4(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_assign___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_assign___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__8(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_getD___at___00Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__4_spec__8(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_getD___at___00Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__4_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__5_spec__10(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__5_spec__10___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__6_spec__12(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__0_spec__0_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__0_spec__0_spec__2(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__5_spec__10_spec__13(lean_object*, lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__5_spec__10_spec__13___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__6_spec__12_spec__16(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__6_spec__12_spec__16___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__0_spec__0_spec__2_spec__13(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__5_spec__10_spec__13_spec__18(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__5_spec__10_spec__13_spec__18___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__6_spec__12_spec__16_spec__21(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__6_spec__12_spec__16_spec__22(lean_object*, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__6_spec__12_spec__16_spec__22___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__0_spec__0_spec__2_spec__13_spec__20(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__6_spec__12_spec__16_spec__21_spec__25(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_withNewMCtxDepth___at___00Aesop_RuleTac_makeForwardHypProofs_spec__2___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_withNewMCtxDepth___at___00Aesop_RuleTac_makeForwardHypProofs_spec__2___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_withNewMCtxDepth___at___00Aesop_RuleTac_makeForwardHypProofs_spec__2___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_withNewMCtxDepth___at___00Aesop_RuleTac_makeForwardHypProofs_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_withNewMCtxDepth___at___00Aesop_RuleTac_makeForwardHypProofs_spec__2(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_withNewMCtxDepth___at___00Aesop_RuleTac_makeForwardHypProofs_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Aesop_UnorderedArraySet_contains___at___00Aesop_RuleTac_makeForwardHypProofs_spec__0_spec__0_spec__2(lean_object*, lean_object*, size_t, size_t);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Aesop_UnorderedArraySet_contains___at___00Aesop_RuleTac_makeForwardHypProofs_spec__0_spec__0_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Array_contains___at___00Aesop_UnorderedArraySet_contains___at___00Aesop_RuleTac_makeForwardHypProofs_spec__0_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_contains___at___00Aesop_UnorderedArraySet_contains___at___00Aesop_RuleTac_makeForwardHypProofs_spec__0_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_UnorderedArraySet_contains___at___00Aesop_RuleTac_makeForwardHypProofs_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_contains___at___00Aesop_RuleTac_makeForwardHypProofs_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleTac_makeForwardHypProofs_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleTac_makeForwardHypProofs_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_makeForwardHypProofs___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_makeForwardHypProofs___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_makeForwardHypProofs(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_makeForwardHypProofs___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleTac_makeForwardHypProofs_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleTac_makeForwardHypProofs_spec__1___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_makeForwardHypProofs_x27_spec__0(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_makeForwardHypProofs_x27_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_makeForwardHypProofs_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_makeForwardHypProofs_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_assertForwardHyp_tacticBuilder___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_assertForwardHyp_tacticBuilder___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_assertForwardHyp_tacticBuilder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_assertForwardHyp_tacticBuilder___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_assertForwardHyp___lam__0(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_RuleTac_assertForwardHyp___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_assertForwardHyp___lam__1___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_RuleTac_assertForwardHyp_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_RuleTac_assertForwardHyp_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_RuleTac_assertForwardHyp___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 79, .m_capacity = 79, .m_length = 78, .m_data = "aesop: internal error in assertForwardHyp: unexpected number of asserted fvars"};
static const lean_object* lp_aesop_Aesop_RuleTac_assertForwardHyp___lam__2___closed__0 = (const lean_object*)&lp_aesop_Aesop_RuleTac_assertForwardHyp___lam__2___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_RuleTac_assertForwardHyp___lam__2___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_RuleTac_assertForwardHyp___lam__2___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_assertForwardHyp___lam__2(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_assertForwardHyp___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_RuleTac_assertForwardHyp___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_RuleTac_assertForwardHyp___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_RuleTac_assertForwardHyp___closed__0 = (const lean_object*)&lp_aesop_Aesop_RuleTac_assertForwardHyp___closed__0_value;
static const lean_closure_object lp_aesop_Aesop_RuleTac_assertForwardHyp___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_RuleTac_assertForwardHyp___lam__1___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_RuleTac_assertForwardHyp___closed__1 = (const lean_object*)&lp_aesop_Aesop_RuleTac_assertForwardHyp___closed__1_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_assertForwardHyp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_assertForwardHyp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_RuleTac_assertForwardHyp_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_RuleTac_assertForwardHyp_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_getUnusedUserNames___at___00Aesop_RuleTac_applyForwardRule_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_getUnusedUserNames___at___00Aesop_RuleTac_applyForwardRule_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_getUnusedUserNames___at___00Aesop_RuleTac_applyForwardRule_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_getUnusedUserNames___at___00Aesop_RuleTac_applyForwardRule_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_RuleTac_applyForwardRule_spec__3___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_RuleTac_applyForwardRule_spec__3___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_RuleTac_applyForwardRule_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_RuleTac_applyForwardRule_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_RuleTac_applyForwardRule_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_RuleTac_applyForwardRule_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_RuleTac_applyForwardRule_spec__2___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_RuleTac_applyForwardRule_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_applyForwardRule___lam__0(lean_object*, lean_object*, lean_object*, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_applyForwardRule___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_applyForwardRule_spec__1(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_applyForwardRule_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_RuleTac_applyForwardRule_spec__5(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_RuleTac_applyForwardRule_spec__6(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_RuleTac_applyForwardRule_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_RuleTac_applyForwardRule_spec__4(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_RuleTac_applyForwardRule_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_RuleTac_applyForwardRule_spec__7___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_RuleTac_applyForwardRule_spec__7___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_RuleTac_applyForwardRule___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 23, .m_capacity = 23, .m_length = 22, .m_data = "found no instances of "};
static const lean_object* lp_aesop_Aesop_RuleTac_applyForwardRule___lam__1___closed__0 = (const lean_object*)&lp_aesop_Aesop_RuleTac_applyForwardRule___lam__1___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_RuleTac_applyForwardRule___lam__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_RuleTac_applyForwardRule___lam__1___closed__1;
static const lean_string_object lp_aesop_Aesop_RuleTac_applyForwardRule___lam__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 78, .m_capacity = 78, .m_length = 77, .m_data = " (other than possibly those which had been previously added by forward rules)"};
static const lean_object* lp_aesop_Aesop_RuleTac_applyForwardRule___lam__1___closed__2 = (const lean_object*)&lp_aesop_Aesop_RuleTac_applyForwardRule___lam__1___closed__2_value;
static lean_once_cell_t lp_aesop_Aesop_RuleTac_applyForwardRule___lam__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_RuleTac_applyForwardRule___lam__1___closed__3;
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_applyForwardRule___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_applyForwardRule___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_applyForwardRule(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_applyForwardRule___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_RuleTac_applyForwardRule_spec__2(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_RuleTac_applyForwardRule_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_RuleTac_applyForwardRule_spec__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_RuleTac_applyForwardRule_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_forwardExpr___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_forwardExpr___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Aesop_RuleTac_forwardExpr___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_RuleTac_forwardExpr___closed__0;
static lean_once_cell_t lp_aesop_Aesop_RuleTac_forwardExpr___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_RuleTac_forwardExpr___closed__1;
static lean_once_cell_t lp_aesop_Aesop_RuleTac_forwardExpr___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_RuleTac_forwardExpr___closed__2;
static const lean_closure_object lp_aesop_Aesop_RuleTac_forwardExpr___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadCoreM___lam__0___boxed, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_RuleTac_forwardExpr___closed__3 = (const lean_object*)&lp_aesop_Aesop_RuleTac_forwardExpr___closed__3_value;
static const lean_closure_object lp_aesop_Aesop_RuleTac_forwardExpr___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadCoreM___lam__1___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_RuleTac_forwardExpr___closed__4 = (const lean_object*)&lp_aesop_Aesop_RuleTac_forwardExpr___closed__4_value;
static const lean_closure_object lp_aesop_Aesop_RuleTac_forwardExpr___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instMonadMetaM___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_RuleTac_forwardExpr___closed__5 = (const lean_object*)&lp_aesop_Aesop_RuleTac_forwardExpr___closed__5_value;
static const lean_closure_object lp_aesop_Aesop_RuleTac_forwardExpr___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instMonadMetaM___lam__1___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_RuleTac_forwardExpr___closed__6 = (const lean_object*)&lp_aesop_Aesop_RuleTac_forwardExpr___closed__6_value;
static const lean_closure_object lp_aesop_Aesop_RuleTac_forwardExpr___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_RuleTac_forwardExpr___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_RuleTac_forwardExpr___closed__7 = (const lean_object*)&lp_aesop_Aesop_RuleTac_forwardExpr___closed__7_value;
static const lean_closure_object lp_aesop_Aesop_RuleTac_forwardExpr___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_StateRefT_x27_lift___boxed, .m_arity = 6, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_aesop_Aesop_RuleTac_forwardExpr___closed__8 = (const lean_object*)&lp_aesop_Aesop_RuleTac_forwardExpr___closed__8_value;
static const lean_closure_object lp_aesop_Aesop_RuleTac_forwardExpr___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_ReaderT_instMonadLift___lam__0___boxed, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_RuleTac_forwardExpr___closed__9 = (const lean_object*)&lp_aesop_Aesop_RuleTac_forwardExpr___closed__9_value;
static const lean_closure_object lp_aesop_Aesop_RuleTac_forwardExpr___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_liftIOCore___boxed, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_RuleTac_forwardExpr___closed__10 = (const lean_object*)&lp_aesop_Aesop_RuleTac_forwardExpr___closed__10_value;
static const lean_closure_object lp_aesop_Aesop_RuleTac_forwardExpr___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instMonadLiftBaseIOEIO___lam__0___boxed, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_RuleTac_forwardExpr___closed__11 = (const lean_object*)&lp_aesop_Aesop_RuleTac_forwardExpr___closed__11_value;
static const lean_closure_object lp_aesop_Aesop_RuleTac_forwardExpr___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_IO_instMonadLiftSTRealWorldBaseIO___lam__0___boxed, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_RuleTac_forwardExpr___closed__12 = (const lean_object*)&lp_aesop_Aesop_RuleTac_forwardExpr___closed__12_value;
static const lean_closure_object lp_aesop_Aesop_RuleTac_forwardExpr___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instMonadLiftT___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_RuleTac_forwardExpr___closed__13 = (const lean_object*)&lp_aesop_Aesop_RuleTac_forwardExpr___closed__13_value;
static const lean_closure_object lp_aesop_Aesop_RuleTac_forwardExpr___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instMonadLiftTOfMonadLift___redArg___lam__0, .m_arity = 4, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_aesop_Aesop_RuleTac_forwardExpr___closed__13_value),((lean_object*)&lp_aesop_Aesop_RuleTac_forwardExpr___closed__12_value)} };
static const lean_object* lp_aesop_Aesop_RuleTac_forwardExpr___closed__14 = (const lean_object*)&lp_aesop_Aesop_RuleTac_forwardExpr___closed__14_value;
static const lean_closure_object lp_aesop_Aesop_RuleTac_forwardExpr___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instMonadLiftTOfMonadLift___redArg___lam__0, .m_arity = 4, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_aesop_Aesop_RuleTac_forwardExpr___closed__14_value),((lean_object*)&lp_aesop_Aesop_RuleTac_forwardExpr___closed__11_value)} };
static const lean_object* lp_aesop_Aesop_RuleTac_forwardExpr___closed__15 = (const lean_object*)&lp_aesop_Aesop_RuleTac_forwardExpr___closed__15_value;
static const lean_closure_object lp_aesop_Aesop_RuleTac_forwardExpr___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instMonadLiftTOfMonadLift___redArg___lam__0, .m_arity = 4, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_aesop_Aesop_RuleTac_forwardExpr___closed__15_value),((lean_object*)&lp_aesop_Aesop_RuleTac_forwardExpr___closed__10_value)} };
static const lean_object* lp_aesop_Aesop_RuleTac_forwardExpr___closed__16 = (const lean_object*)&lp_aesop_Aesop_RuleTac_forwardExpr___closed__16_value;
static const lean_closure_object lp_aesop_Aesop_RuleTac_forwardExpr___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instMonadLiftTOfMonadLift___redArg___lam__0, .m_arity = 4, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_aesop_Aesop_RuleTac_forwardExpr___closed__16_value),((lean_object*)&lp_aesop_Aesop_RuleTac_forwardExpr___closed__8_value)} };
static const lean_object* lp_aesop_Aesop_RuleTac_forwardExpr___closed__17 = (const lean_object*)&lp_aesop_Aesop_RuleTac_forwardExpr___closed__17_value;
static const lean_closure_object lp_aesop_Aesop_RuleTac_forwardExpr___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instMonadLiftTOfMonadLift___redArg___lam__0, .m_arity = 4, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_aesop_Aesop_RuleTac_forwardExpr___closed__17_value),((lean_object*)&lp_aesop_Aesop_RuleTac_forwardExpr___closed__9_value)} };
static const lean_object* lp_aesop_Aesop_RuleTac_forwardExpr___closed__18 = (const lean_object*)&lp_aesop_Aesop_RuleTac_forwardExpr___closed__18_value;
static const lean_closure_object lp_aesop_Aesop_RuleTac_forwardExpr___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instMonadLiftTOfMonadLift___redArg___lam__0, .m_arity = 4, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_aesop_Aesop_RuleTac_forwardExpr___closed__18_value),((lean_object*)&lp_aesop_Aesop_RuleTac_forwardExpr___closed__8_value)} };
static const lean_object* lp_aesop_Aesop_RuleTac_forwardExpr___closed__19 = (const lean_object*)&lp_aesop_Aesop_RuleTac_forwardExpr___closed__19_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_forwardExpr(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_forwardExpr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_forwardConst_spec__0___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_forwardConst_spec__0___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_forwardConst_spec__0___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_forwardConst_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_forwardConst_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_forwardConst_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_forwardConst_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_RuleTac_forwardConst_spec__1___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_RuleTac_forwardConst_spec__1___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_RuleTac_forwardConst_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_RuleTac_forwardConst_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_RuleTac_forwardConst_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_RuleTac_forwardConst_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_forwardConst___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_forwardConst___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_forwardConst(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_forwardConst___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_forwardTerm___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_forwardTerm___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_forwardTerm___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_forwardTerm___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_forwardTerm(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_forwardTerm___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_forward(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_forward___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_forwardMatches___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_forwardMatches___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_RuleTac_forwardMatches_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_RuleTac_forwardMatches_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_forwardMatches_spec__0(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_forwardMatches_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_forwardMatches_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = " "};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_forwardMatches_spec__1___closed__0 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_forwardMatches_spec__1___closed__0_value;
static lean_once_cell_t lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_forwardMatches_spec__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_forwardMatches_spec__1___closed__1;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_forwardMatches_spec__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "global"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_forwardMatches_spec__1___closed__2 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_forwardMatches_spec__1___closed__2_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_forwardMatches_spec__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "local"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_forwardMatches_spec__1___closed__3 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_forwardMatches_spec__1___closed__3_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_forwardMatches_spec__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "|"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_forwardMatches_spec__1___closed__4 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_forwardMatches_spec__1___closed__4_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_forwardMatches_spec__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "apply"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_forwardMatches_spec__1___closed__5 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_forwardMatches_spec__1___closed__5_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_forwardMatches_spec__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "cases"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_forwardMatches_spec__1___closed__6 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_forwardMatches_spec__1___closed__6_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_forwardMatches_spec__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "constructors"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_forwardMatches_spec__1___closed__7 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_forwardMatches_spec__1___closed__7_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_forwardMatches_spec__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "destruct"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_forwardMatches_spec__1___closed__8 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_forwardMatches_spec__1___closed__8_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_forwardMatches_spec__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "forward"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_forwardMatches_spec__1___closed__9 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_forwardMatches_spec__1___closed__9_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_forwardMatches_spec__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "simp"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_forwardMatches_spec__1___closed__10 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_forwardMatches_spec__1___closed__10_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_forwardMatches_spec__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "tactic"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_forwardMatches_spec__1___closed__11 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_forwardMatches_spec__1___closed__11_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_forwardMatches_spec__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "unfold"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_forwardMatches_spec__1___closed__12 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_forwardMatches_spec__1___closed__12_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_forwardMatches_spec__1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "norm"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_forwardMatches_spec__1___closed__13 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_forwardMatches_spec__1___closed__13_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_forwardMatches_spec__1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "safe"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_forwardMatches_spec__1___closed__14 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_forwardMatches_spec__1___closed__14_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_forwardMatches_spec__1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "unsafe"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_forwardMatches_spec__1___closed__15 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_forwardMatches_spec__1___closed__15_value;
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_forwardMatches_spec__1(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_forwardMatches_spec__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_aesop_Aesop_RuleTac_forwardMatches___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_forwardConst_spec__0___redArg___closed__0_value)}};
static const lean_object* lp_aesop_Aesop_RuleTac_forwardMatches___closed__0 = (const lean_object*)&lp_aesop_Aesop_RuleTac_forwardMatches___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_RuleTac_forwardMatches___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_RuleTac_forwardMatches___closed__1;
static lean_once_cell_t lp_aesop_Aesop_RuleTac_forwardMatches___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_RuleTac_forwardMatches___closed__2;
static const lean_string_object lp_aesop_Aesop_RuleTac_forwardMatches___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 66, .m_capacity = 66, .m_length = 65, .m_data = "failed to add hyps for any of the following forward rule matches:"};
static const lean_object* lp_aesop_Aesop_RuleTac_forwardMatches___closed__3 = (const lean_object*)&lp_aesop_Aesop_RuleTac_forwardMatches___closed__3_value;
static lean_once_cell_t lp_aesop_Aesop_RuleTac_forwardMatches___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_RuleTac_forwardMatches___closed__4;
static const lean_string_object lp_aesop_Aesop_RuleTac_forwardMatches___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "\n"};
static const lean_object* lp_aesop_Aesop_RuleTac_forwardMatches___closed__5 = (const lean_object*)&lp_aesop_Aesop_RuleTac_forwardMatches___closed__5_value;
static const lean_ctor_object lp_aesop_Aesop_RuleTac_forwardMatches___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_RuleTac_forwardMatches___closed__5_value)}};
static const lean_object* lp_aesop_Aesop_RuleTac_forwardMatches___closed__6 = (const lean_object*)&lp_aesop_Aesop_RuleTac_forwardMatches___closed__6_value;
static lean_once_cell_t lp_aesop_Aesop_RuleTac_forwardMatches___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_RuleTac_forwardMatches___closed__7;
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_forwardMatches(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_forwardMatches___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_RuleTac_forwardMatches_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_RuleTac_forwardMatches_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_forwardMatch(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_forwardMatch___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_aesop_Aesop_RuleTac_ForwardM_instInhabitedContext_default___closed__0(void){
_start:
{
lean_object* v___x_1_; lean_object* v___x_2_; lean_object* v___x_3_; 
v___x_1_ = lp_aesop_Aesop_instInhabitedForwardHypData_default;
v___x_2_ = lean_box(0);
v___x_3_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3_, 0, v___x_2_);
lean_ctor_set(v___x_3_, 1, v___x_1_);
return v___x_3_;
}
}
static lean_object* _init_lp_aesop_Aesop_RuleTac_ForwardM_instInhabitedContext_default(void){
_start:
{
lean_object* v___x_4_; 
v___x_4_ = lean_obj_once(&lp_aesop_Aesop_RuleTac_ForwardM_instInhabitedContext_default___closed__0, &lp_aesop_Aesop_RuleTac_ForwardM_instInhabitedContext_default___closed__0_once, _init_lp_aesop_Aesop_RuleTac_ForwardM_instInhabitedContext_default___closed__0);
return v___x_4_;
}
}
static lean_object* _init_lp_aesop_Aesop_RuleTac_ForwardM_instInhabitedContext(void){
_start:
{
lean_object* v___x_5_; 
v___x_5_ = lp_aesop_Aesop_RuleTac_ForwardM_instInhabitedContext_default;
return v___x_5_;
}
}
static lean_object* _init_lp_aesop_Aesop_RuleTac_ForwardM_instInhabitedState_default___closed__1(void){
_start:
{
lean_object* v___x_8_; lean_object* v___x_9_; lean_object* v___x_10_; 
v___x_8_ = lean_box(0);
v___x_9_ = lean_unsigned_to_nat(16u);
v___x_10_ = lean_mk_array(v___x_9_, v___x_8_);
return v___x_10_;
}
}
static lean_object* _init_lp_aesop_Aesop_RuleTac_ForwardM_instInhabitedState_default___closed__2(void){
_start:
{
lean_object* v___x_11_; lean_object* v___x_12_; lean_object* v___x_13_; 
v___x_11_ = lean_obj_once(&lp_aesop_Aesop_RuleTac_ForwardM_instInhabitedState_default___closed__1, &lp_aesop_Aesop_RuleTac_ForwardM_instInhabitedState_default___closed__1_once, _init_lp_aesop_Aesop_RuleTac_ForwardM_instInhabitedState_default___closed__1);
v___x_12_ = lean_unsigned_to_nat(0u);
v___x_13_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_13_, 0, v___x_12_);
lean_ctor_set(v___x_13_, 1, v___x_11_);
return v___x_13_;
}
}
static lean_object* _init_lp_aesop_Aesop_RuleTac_ForwardM_instInhabitedState_default___closed__3(void){
_start:
{
lean_object* v___x_14_; lean_object* v___x_15_; lean_object* v___x_16_; 
v___x_14_ = lean_obj_once(&lp_aesop_Aesop_RuleTac_ForwardM_instInhabitedState_default___closed__2, &lp_aesop_Aesop_RuleTac_ForwardM_instInhabitedState_default___closed__2_once, _init_lp_aesop_Aesop_RuleTac_ForwardM_instInhabitedState_default___closed__2);
v___x_15_ = ((lean_object*)(lp_aesop_Aesop_RuleTac_ForwardM_instInhabitedState_default___closed__0));
v___x_16_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_16_, 0, v___x_15_);
lean_ctor_set(v___x_16_, 1, v___x_14_);
return v___x_16_;
}
}
static lean_object* _init_lp_aesop_Aesop_RuleTac_ForwardM_instInhabitedState_default(void){
_start:
{
lean_object* v___x_17_; 
v___x_17_ = lean_obj_once(&lp_aesop_Aesop_RuleTac_ForwardM_instInhabitedState_default___closed__3, &lp_aesop_Aesop_RuleTac_ForwardM_instInhabitedState_default___closed__3_once, _init_lp_aesop_Aesop_RuleTac_ForwardM_instInhabitedState_default___closed__3);
return v___x_17_;
}
}
static lean_object* _init_lp_aesop_Aesop_RuleTac_ForwardM_instInhabitedState(void){
_start:
{
lean_object* v___x_18_; 
v___x_18_ = lp_aesop_Aesop_RuleTac_ForwardM_instInhabitedState_default;
return v___x_18_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_isRedundant_spec__0___redArg(lean_object* v_type_19_, lean_object* v_as_20_, size_t v_i_21_, size_t v_stop_22_, lean_object* v___y_23_, lean_object* v___y_24_, lean_object* v___y_25_, lean_object* v___y_26_){
_start:
{
uint8_t v___x_28_; 
v___x_28_ = lean_usize_dec_eq(v_i_21_, v_stop_22_);
if (v___x_28_ == 0)
{
lean_object* v___x_29_; lean_object* v_fst_30_; lean_object* v___x_31_; 
v___x_29_ = lean_array_uget_borrowed(v_as_20_, v_i_21_);
v_fst_30_ = lean_ctor_get(v___x_29_, 0);
lean_inc_ref(v_type_19_);
lean_inc(v_fst_30_);
v___x_31_ = lp_aesop_Aesop_isDefEqReducibleRigid(v_fst_30_, v_type_19_, v___y_23_, v___y_24_, v___y_25_, v___y_26_);
if (lean_obj_tag(v___x_31_) == 0)
{
lean_object* v_a_32_; lean_object* v___x_34_; uint8_t v_isShared_35_; uint8_t v_isSharedCheck_43_; 
v_a_32_ = lean_ctor_get(v___x_31_, 0);
v_isSharedCheck_43_ = !lean_is_exclusive(v___x_31_);
if (v_isSharedCheck_43_ == 0)
{
v___x_34_ = v___x_31_;
v_isShared_35_ = v_isSharedCheck_43_;
goto v_resetjp_33_;
}
else
{
lean_inc(v_a_32_);
lean_dec(v___x_31_);
v___x_34_ = lean_box(0);
v_isShared_35_ = v_isSharedCheck_43_;
goto v_resetjp_33_;
}
v_resetjp_33_:
{
uint8_t v___x_36_; 
v___x_36_ = lean_unbox(v_a_32_);
if (v___x_36_ == 0)
{
size_t v___x_37_; size_t v___x_38_; 
lean_del_object(v___x_34_);
lean_dec(v_a_32_);
v___x_37_ = ((size_t)1ULL);
v___x_38_ = lean_usize_add(v_i_21_, v___x_37_);
v_i_21_ = v___x_38_;
goto _start;
}
else
{
lean_object* v___x_41_; 
lean_dec_ref(v_type_19_);
if (v_isShared_35_ == 0)
{
v___x_41_ = v___x_34_;
goto v_reusejp_40_;
}
else
{
lean_object* v_reuseFailAlloc_42_; 
v_reuseFailAlloc_42_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_42_, 0, v_a_32_);
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
else
{
lean_dec_ref(v_type_19_);
return v___x_31_;
}
}
else
{
uint8_t v___x_44_; lean_object* v___x_45_; lean_object* v___x_46_; 
lean_dec_ref(v_type_19_);
v___x_44_ = 0;
v___x_45_ = lean_box(v___x_44_);
v___x_46_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_46_, 0, v___x_45_);
return v___x_46_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_isRedundant_spec__0___redArg___boxed(lean_object* v_type_47_, lean_object* v_as_48_, lean_object* v_i_49_, lean_object* v_stop_50_, lean_object* v___y_51_, lean_object* v___y_52_, lean_object* v___y_53_, lean_object* v___y_54_, lean_object* v___y_55_){
_start:
{
size_t v_i_boxed_56_; size_t v_stop_boxed_57_; lean_object* v_res_58_; 
v_i_boxed_56_ = lean_unbox_usize(v_i_49_);
lean_dec(v_i_49_);
v_stop_boxed_57_ = lean_unbox_usize(v_stop_50_);
lean_dec(v_stop_50_);
v_res_58_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_isRedundant_spec__0___redArg(v_type_47_, v_as_48_, v_i_boxed_56_, v_stop_boxed_57_, v___y_51_, v___y_52_, v___y_53_, v___y_54_);
lean_dec(v___y_54_);
lean_dec_ref(v___y_53_);
lean_dec(v___y_52_);
lean_dec_ref(v___y_51_);
lean_dec_ref(v_as_48_);
return v_res_58_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_isRedundant(lean_object* v_type_59_, lean_object* v_a_60_, lean_object* v_a_61_, lean_object* v_a_62_, lean_object* v_a_63_, lean_object* v_a_64_, lean_object* v_a_65_, lean_object* v_a_66_, lean_object* v_a_67_){
_start:
{
lean_object* v___x_69_; lean_object* v_toAssert_70_; lean_object* v___x_71_; lean_object* v___x_72_; uint8_t v___x_73_; 
v___x_69_ = lean_st_ref_get(v_a_61_);
v_toAssert_70_ = lean_ctor_get(v___x_69_, 0);
lean_inc_ref(v_toAssert_70_);
lean_dec(v___x_69_);
v___x_71_ = lean_unsigned_to_nat(0u);
v___x_72_ = lean_array_get_size(v_toAssert_70_);
v___x_73_ = lean_nat_dec_lt(v___x_71_, v___x_72_);
if (v___x_73_ == 0)
{
lean_object* v___x_74_; 
lean_dec_ref(v_toAssert_70_);
v___x_74_ = lp_aesop_Aesop_isHypRedundantReducibleRigid(v_type_59_, v_a_64_, v_a_65_, v_a_66_, v_a_67_);
return v___x_74_;
}
else
{
if (v___x_73_ == 0)
{
lean_object* v___x_75_; 
lean_dec_ref(v_toAssert_70_);
v___x_75_ = lp_aesop_Aesop_isHypRedundantReducibleRigid(v_type_59_, v_a_64_, v_a_65_, v_a_66_, v_a_67_);
return v___x_75_;
}
else
{
size_t v___x_76_; size_t v___x_77_; lean_object* v___x_78_; 
v___x_76_ = ((size_t)0ULL);
v___x_77_ = lean_usize_of_nat(v___x_72_);
lean_inc_ref(v_type_59_);
v___x_78_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_isRedundant_spec__0___redArg(v_type_59_, v_toAssert_70_, v___x_76_, v___x_77_, v_a_64_, v_a_65_, v_a_66_, v_a_67_);
lean_dec_ref(v_toAssert_70_);
if (lean_obj_tag(v___x_78_) == 0)
{
lean_object* v_a_79_; uint8_t v___x_80_; 
v_a_79_ = lean_ctor_get(v___x_78_, 0);
lean_inc(v_a_79_);
v___x_80_ = lean_unbox(v_a_79_);
lean_dec(v_a_79_);
if (v___x_80_ == 0)
{
lean_object* v___x_81_; 
lean_dec_ref_known(v___x_78_, 1);
v___x_81_ = lp_aesop_Aesop_isHypRedundantReducibleRigid(v_type_59_, v_a_64_, v_a_65_, v_a_66_, v_a_67_);
return v___x_81_;
}
else
{
lean_dec_ref(v_type_59_);
return v___x_78_;
}
}
else
{
lean_dec_ref(v_type_59_);
return v___x_78_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_isRedundant___boxed(lean_object* v_type_82_, lean_object* v_a_83_, lean_object* v_a_84_, lean_object* v_a_85_, lean_object* v_a_86_, lean_object* v_a_87_, lean_object* v_a_88_, lean_object* v_a_89_, lean_object* v_a_90_, lean_object* v_a_91_){
_start:
{
lean_object* v_res_92_; 
v_res_92_ = lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_isRedundant(v_type_82_, v_a_83_, v_a_84_, v_a_85_, v_a_86_, v_a_87_, v_a_88_, v_a_89_, v_a_90_);
lean_dec(v_a_90_);
lean_dec_ref(v_a_89_);
lean_dec(v_a_88_);
lean_dec_ref(v_a_87_);
lean_dec(v_a_86_);
lean_dec(v_a_85_);
lean_dec(v_a_84_);
lean_dec_ref(v_a_83_);
return v_res_92_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_isRedundant_spec__0(lean_object* v_type_93_, lean_object* v_as_94_, size_t v_i_95_, size_t v_stop_96_, lean_object* v___y_97_, lean_object* v___y_98_, lean_object* v___y_99_, lean_object* v___y_100_, lean_object* v___y_101_, lean_object* v___y_102_, lean_object* v___y_103_, lean_object* v___y_104_){
_start:
{
lean_object* v___x_106_; 
v___x_106_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_isRedundant_spec__0___redArg(v_type_93_, v_as_94_, v_i_95_, v_stop_96_, v___y_101_, v___y_102_, v___y_103_, v___y_104_);
return v___x_106_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_isRedundant_spec__0___boxed(lean_object* v_type_107_, lean_object* v_as_108_, lean_object* v_i_109_, lean_object* v_stop_110_, lean_object* v___y_111_, lean_object* v___y_112_, lean_object* v___y_113_, lean_object* v___y_114_, lean_object* v___y_115_, lean_object* v___y_116_, lean_object* v___y_117_, lean_object* v___y_118_, lean_object* v___y_119_){
_start:
{
size_t v_i_boxed_120_; size_t v_stop_boxed_121_; lean_object* v_res_122_; 
v_i_boxed_120_ = lean_unbox_usize(v_i_109_);
lean_dec(v_i_109_);
v_stop_boxed_121_ = lean_unbox_usize(v_stop_110_);
lean_dec(v_stop_110_);
v_res_122_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_isRedundant_spec__0(v_type_107_, v_as_108_, v_i_boxed_120_, v_stop_boxed_121_, v___y_111_, v___y_112_, v___y_113_, v___y_114_, v___y_115_, v___y_116_, v___y_117_, v___y_118_);
lean_dec(v___y_118_);
lean_dec_ref(v___y_117_);
lean_dec(v___y_116_);
lean_dec_ref(v___y_115_);
lean_dec(v___y_114_);
lean_dec(v___y_113_);
lean_dec(v___y_112_);
lean_dec_ref(v___y_111_);
lean_dec_ref(v_as_108_);
return v_res_122_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop___lam__0(lean_object* v_____r_125_, lean_object* v___y_126_, lean_object* v___y_127_, lean_object* v___y_128_, lean_object* v___y_129_, lean_object* v___y_130_, lean_object* v___y_131_, lean_object* v___y_132_, lean_object* v___y_133_){
_start:
{
lean_object* v___x_135_; lean_object* v___x_136_; 
v___x_135_ = ((lean_object*)(lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop___lam__0___closed__0));
v___x_136_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_136_, 0, v___x_135_);
return v___x_136_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop___lam__0___boxed(lean_object* v_____r_137_, lean_object* v___y_138_, lean_object* v___y_139_, lean_object* v___y_140_, lean_object* v___y_141_, lean_object* v___y_142_, lean_object* v___y_143_, lean_object* v___y_144_, lean_object* v___y_145_, lean_object* v___y_146_){
_start:
{
lean_object* v_res_147_; 
v_res_147_ = lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop___lam__0(v_____r_137_, v___y_138_, v___y_139_, v___y_140_, v___y_141_, v___y_142_, v___y_143_, v___y_144_, v___y_145_);
lean_dec(v___y_145_);
lean_dec_ref(v___y_144_);
lean_dec(v___y_143_);
lean_dec_ref(v___y_142_);
lean_dec(v___y_141_);
lean_dec(v___y_140_);
lean_dec(v___y_139_);
lean_dec_ref(v___y_138_);
return v_res_147_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__6_spec__12_spec__16_spec__21_spec__25___redArg(lean_object* v_x_148_, lean_object* v_x_149_, lean_object* v_x_150_, lean_object* v_x_151_){
_start:
{
lean_object* v_ks_152_; lean_object* v_vs_153_; lean_object* v___x_155_; uint8_t v_isShared_156_; uint8_t v_isSharedCheck_177_; 
v_ks_152_ = lean_ctor_get(v_x_148_, 0);
v_vs_153_ = lean_ctor_get(v_x_148_, 1);
v_isSharedCheck_177_ = !lean_is_exclusive(v_x_148_);
if (v_isSharedCheck_177_ == 0)
{
v___x_155_ = v_x_148_;
v_isShared_156_ = v_isSharedCheck_177_;
goto v_resetjp_154_;
}
else
{
lean_inc(v_vs_153_);
lean_inc(v_ks_152_);
lean_dec(v_x_148_);
v___x_155_ = lean_box(0);
v_isShared_156_ = v_isSharedCheck_177_;
goto v_resetjp_154_;
}
v_resetjp_154_:
{
lean_object* v___x_157_; uint8_t v___x_158_; 
v___x_157_ = lean_array_get_size(v_ks_152_);
v___x_158_ = lean_nat_dec_lt(v_x_149_, v___x_157_);
if (v___x_158_ == 0)
{
lean_object* v___x_159_; lean_object* v___x_160_; lean_object* v___x_162_; 
lean_dec(v_x_149_);
v___x_159_ = lean_array_push(v_ks_152_, v_x_150_);
v___x_160_ = lean_array_push(v_vs_153_, v_x_151_);
if (v_isShared_156_ == 0)
{
lean_ctor_set(v___x_155_, 1, v___x_160_);
lean_ctor_set(v___x_155_, 0, v___x_159_);
v___x_162_ = v___x_155_;
goto v_reusejp_161_;
}
else
{
lean_object* v_reuseFailAlloc_163_; 
v_reuseFailAlloc_163_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_163_, 0, v___x_159_);
lean_ctor_set(v_reuseFailAlloc_163_, 1, v___x_160_);
v___x_162_ = v_reuseFailAlloc_163_;
goto v_reusejp_161_;
}
v_reusejp_161_:
{
return v___x_162_;
}
}
else
{
lean_object* v_k_x27_164_; uint8_t v___x_165_; 
v_k_x27_164_ = lean_array_fget_borrowed(v_ks_152_, v_x_149_);
v___x_165_ = l_Lean_instBEqMVarId_beq(v_x_150_, v_k_x27_164_);
if (v___x_165_ == 0)
{
lean_object* v___x_167_; 
if (v_isShared_156_ == 0)
{
v___x_167_ = v___x_155_;
goto v_reusejp_166_;
}
else
{
lean_object* v_reuseFailAlloc_171_; 
v_reuseFailAlloc_171_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_171_, 0, v_ks_152_);
lean_ctor_set(v_reuseFailAlloc_171_, 1, v_vs_153_);
v___x_167_ = v_reuseFailAlloc_171_;
goto v_reusejp_166_;
}
v_reusejp_166_:
{
lean_object* v___x_168_; lean_object* v___x_169_; 
v___x_168_ = lean_unsigned_to_nat(1u);
v___x_169_ = lean_nat_add(v_x_149_, v___x_168_);
lean_dec(v_x_149_);
v_x_148_ = v___x_167_;
v_x_149_ = v___x_169_;
goto _start;
}
}
else
{
lean_object* v___x_172_; lean_object* v___x_173_; lean_object* v___x_175_; 
v___x_172_ = lean_array_fset(v_ks_152_, v_x_149_, v_x_150_);
v___x_173_ = lean_array_fset(v_vs_153_, v_x_149_, v_x_151_);
lean_dec(v_x_149_);
if (v_isShared_156_ == 0)
{
lean_ctor_set(v___x_155_, 1, v___x_173_);
lean_ctor_set(v___x_155_, 0, v___x_172_);
v___x_175_ = v___x_155_;
goto v_reusejp_174_;
}
else
{
lean_object* v_reuseFailAlloc_176_; 
v_reuseFailAlloc_176_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_176_, 0, v___x_172_);
lean_ctor_set(v_reuseFailAlloc_176_, 1, v___x_173_);
v___x_175_ = v_reuseFailAlloc_176_;
goto v_reusejp_174_;
}
v_reusejp_174_:
{
return v___x_175_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__6_spec__12_spec__16_spec__21___redArg(lean_object* v_n_178_, lean_object* v_k_179_, lean_object* v_v_180_){
_start:
{
lean_object* v___x_181_; lean_object* v___x_182_; 
v___x_181_ = lean_unsigned_to_nat(0u);
v___x_182_ = lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__6_spec__12_spec__16_spec__21_spec__25___redArg(v_n_178_, v___x_181_, v_k_179_, v_v_180_);
return v___x_182_;
}
}
static lean_object* _init_lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__6_spec__12_spec__16___redArg___closed__0(void){
_start:
{
lean_object* v___x_183_; 
v___x_183_ = l_Lean_PersistentHashMap_mkEmptyEntries(lean_box(0), lean_box(0));
return v___x_183_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__6_spec__12_spec__16___redArg(lean_object* v_x_184_, size_t v_x_185_, size_t v_x_186_, lean_object* v_x_187_, lean_object* v_x_188_){
_start:
{
if (lean_obj_tag(v_x_184_) == 0)
{
lean_object* v_es_189_; size_t v___x_190_; size_t v___x_191_; lean_object* v_j_192_; lean_object* v___x_193_; uint8_t v___x_194_; 
v_es_189_ = lean_ctor_get(v_x_184_, 0);
v___x_190_ = ((size_t)31ULL);
v___x_191_ = lean_usize_land(v_x_185_, v___x_190_);
v_j_192_ = lean_usize_to_nat(v___x_191_);
v___x_193_ = lean_array_get_size(v_es_189_);
v___x_194_ = lean_nat_dec_lt(v_j_192_, v___x_193_);
if (v___x_194_ == 0)
{
lean_dec(v_j_192_);
lean_dec(v_x_188_);
lean_dec(v_x_187_);
return v_x_184_;
}
else
{
lean_object* v___x_196_; uint8_t v_isShared_197_; uint8_t v_isSharedCheck_233_; 
lean_inc_ref(v_es_189_);
v_isSharedCheck_233_ = !lean_is_exclusive(v_x_184_);
if (v_isSharedCheck_233_ == 0)
{
lean_object* v_unused_234_; 
v_unused_234_ = lean_ctor_get(v_x_184_, 0);
lean_dec(v_unused_234_);
v___x_196_ = v_x_184_;
v_isShared_197_ = v_isSharedCheck_233_;
goto v_resetjp_195_;
}
else
{
lean_dec(v_x_184_);
v___x_196_ = lean_box(0);
v_isShared_197_ = v_isSharedCheck_233_;
goto v_resetjp_195_;
}
v_resetjp_195_:
{
lean_object* v_v_198_; lean_object* v___x_199_; lean_object* v_xs_x27_200_; lean_object* v___y_202_; 
v_v_198_ = lean_array_fget(v_es_189_, v_j_192_);
v___x_199_ = lean_box(0);
v_xs_x27_200_ = lean_array_fset(v_es_189_, v_j_192_, v___x_199_);
switch(lean_obj_tag(v_v_198_))
{
case 0:
{
lean_object* v_key_207_; lean_object* v_val_208_; lean_object* v___x_210_; uint8_t v_isShared_211_; uint8_t v_isSharedCheck_218_; 
v_key_207_ = lean_ctor_get(v_v_198_, 0);
v_val_208_ = lean_ctor_get(v_v_198_, 1);
v_isSharedCheck_218_ = !lean_is_exclusive(v_v_198_);
if (v_isSharedCheck_218_ == 0)
{
v___x_210_ = v_v_198_;
v_isShared_211_ = v_isSharedCheck_218_;
goto v_resetjp_209_;
}
else
{
lean_inc(v_val_208_);
lean_inc(v_key_207_);
lean_dec(v_v_198_);
v___x_210_ = lean_box(0);
v_isShared_211_ = v_isSharedCheck_218_;
goto v_resetjp_209_;
}
v_resetjp_209_:
{
uint8_t v___x_212_; 
v___x_212_ = l_Lean_instBEqMVarId_beq(v_x_187_, v_key_207_);
if (v___x_212_ == 0)
{
lean_object* v___x_213_; lean_object* v___x_214_; 
lean_del_object(v___x_210_);
v___x_213_ = l_Lean_PersistentHashMap_mkCollisionNode___redArg(v_key_207_, v_val_208_, v_x_187_, v_x_188_);
v___x_214_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_214_, 0, v___x_213_);
v___y_202_ = v___x_214_;
goto v___jp_201_;
}
else
{
lean_object* v___x_216_; 
lean_dec(v_val_208_);
lean_dec(v_key_207_);
if (v_isShared_211_ == 0)
{
lean_ctor_set(v___x_210_, 1, v_x_188_);
lean_ctor_set(v___x_210_, 0, v_x_187_);
v___x_216_ = v___x_210_;
goto v_reusejp_215_;
}
else
{
lean_object* v_reuseFailAlloc_217_; 
v_reuseFailAlloc_217_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_217_, 0, v_x_187_);
lean_ctor_set(v_reuseFailAlloc_217_, 1, v_x_188_);
v___x_216_ = v_reuseFailAlloc_217_;
goto v_reusejp_215_;
}
v_reusejp_215_:
{
v___y_202_ = v___x_216_;
goto v___jp_201_;
}
}
}
}
case 1:
{
lean_object* v_node_219_; lean_object* v___x_221_; uint8_t v_isShared_222_; uint8_t v_isSharedCheck_231_; 
v_node_219_ = lean_ctor_get(v_v_198_, 0);
v_isSharedCheck_231_ = !lean_is_exclusive(v_v_198_);
if (v_isSharedCheck_231_ == 0)
{
v___x_221_ = v_v_198_;
v_isShared_222_ = v_isSharedCheck_231_;
goto v_resetjp_220_;
}
else
{
lean_inc(v_node_219_);
lean_dec(v_v_198_);
v___x_221_ = lean_box(0);
v_isShared_222_ = v_isSharedCheck_231_;
goto v_resetjp_220_;
}
v_resetjp_220_:
{
size_t v___x_223_; size_t v___x_224_; size_t v___x_225_; size_t v___x_226_; lean_object* v___x_227_; lean_object* v___x_229_; 
v___x_223_ = ((size_t)5ULL);
v___x_224_ = lean_usize_shift_right(v_x_185_, v___x_223_);
v___x_225_ = ((size_t)1ULL);
v___x_226_ = lean_usize_add(v_x_186_, v___x_225_);
v___x_227_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__6_spec__12_spec__16___redArg(v_node_219_, v___x_224_, v___x_226_, v_x_187_, v_x_188_);
if (v_isShared_222_ == 0)
{
lean_ctor_set(v___x_221_, 0, v___x_227_);
v___x_229_ = v___x_221_;
goto v_reusejp_228_;
}
else
{
lean_object* v_reuseFailAlloc_230_; 
v_reuseFailAlloc_230_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_230_, 0, v___x_227_);
v___x_229_ = v_reuseFailAlloc_230_;
goto v_reusejp_228_;
}
v_reusejp_228_:
{
v___y_202_ = v___x_229_;
goto v___jp_201_;
}
}
}
default: 
{
lean_object* v___x_232_; 
v___x_232_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_232_, 0, v_x_187_);
lean_ctor_set(v___x_232_, 1, v_x_188_);
v___y_202_ = v___x_232_;
goto v___jp_201_;
}
}
v___jp_201_:
{
lean_object* v___x_203_; lean_object* v___x_205_; 
v___x_203_ = lean_array_fset(v_xs_x27_200_, v_j_192_, v___y_202_);
lean_dec(v_j_192_);
if (v_isShared_197_ == 0)
{
lean_ctor_set(v___x_196_, 0, v___x_203_);
v___x_205_ = v___x_196_;
goto v_reusejp_204_;
}
else
{
lean_object* v_reuseFailAlloc_206_; 
v_reuseFailAlloc_206_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_206_, 0, v___x_203_);
v___x_205_ = v_reuseFailAlloc_206_;
goto v_reusejp_204_;
}
v_reusejp_204_:
{
return v___x_205_;
}
}
}
}
}
else
{
lean_object* v_ks_235_; lean_object* v_vs_236_; lean_object* v___x_238_; uint8_t v_isShared_239_; uint8_t v_isSharedCheck_256_; 
v_ks_235_ = lean_ctor_get(v_x_184_, 0);
v_vs_236_ = lean_ctor_get(v_x_184_, 1);
v_isSharedCheck_256_ = !lean_is_exclusive(v_x_184_);
if (v_isSharedCheck_256_ == 0)
{
v___x_238_ = v_x_184_;
v_isShared_239_ = v_isSharedCheck_256_;
goto v_resetjp_237_;
}
else
{
lean_inc(v_vs_236_);
lean_inc(v_ks_235_);
lean_dec(v_x_184_);
v___x_238_ = lean_box(0);
v_isShared_239_ = v_isSharedCheck_256_;
goto v_resetjp_237_;
}
v_resetjp_237_:
{
lean_object* v___x_241_; 
if (v_isShared_239_ == 0)
{
v___x_241_ = v___x_238_;
goto v_reusejp_240_;
}
else
{
lean_object* v_reuseFailAlloc_255_; 
v_reuseFailAlloc_255_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_255_, 0, v_ks_235_);
lean_ctor_set(v_reuseFailAlloc_255_, 1, v_vs_236_);
v___x_241_ = v_reuseFailAlloc_255_;
goto v_reusejp_240_;
}
v_reusejp_240_:
{
lean_object* v_newNode_242_; uint8_t v___y_244_; size_t v___x_250_; uint8_t v___x_251_; 
v_newNode_242_ = lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__6_spec__12_spec__16_spec__21___redArg(v___x_241_, v_x_187_, v_x_188_);
v___x_250_ = ((size_t)7ULL);
v___x_251_ = lean_usize_dec_le(v___x_250_, v_x_186_);
if (v___x_251_ == 0)
{
lean_object* v___x_252_; lean_object* v___x_253_; uint8_t v___x_254_; 
v___x_252_ = l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(v_newNode_242_);
v___x_253_ = lean_unsigned_to_nat(4u);
v___x_254_ = lean_nat_dec_lt(v___x_252_, v___x_253_);
lean_dec(v___x_252_);
v___y_244_ = v___x_254_;
goto v___jp_243_;
}
else
{
v___y_244_ = v___x_251_;
goto v___jp_243_;
}
v___jp_243_:
{
if (v___y_244_ == 0)
{
lean_object* v_ks_245_; lean_object* v_vs_246_; lean_object* v___x_247_; lean_object* v___x_248_; lean_object* v___x_249_; 
v_ks_245_ = lean_ctor_get(v_newNode_242_, 0);
lean_inc_ref(v_ks_245_);
v_vs_246_ = lean_ctor_get(v_newNode_242_, 1);
lean_inc_ref(v_vs_246_);
lean_dec_ref(v_newNode_242_);
v___x_247_ = lean_unsigned_to_nat(0u);
v___x_248_ = lean_obj_once(&lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__6_spec__12_spec__16___redArg___closed__0, &lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__6_spec__12_spec__16___redArg___closed__0_once, _init_lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__6_spec__12_spec__16___redArg___closed__0);
v___x_249_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__6_spec__12_spec__16_spec__22___redArg(v_x_186_, v_ks_245_, v_vs_246_, v___x_247_, v___x_248_);
lean_dec_ref(v_vs_246_);
lean_dec_ref(v_ks_245_);
return v___x_249_;
}
else
{
return v_newNode_242_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__6_spec__12_spec__16_spec__22___redArg(size_t v_depth_257_, lean_object* v_keys_258_, lean_object* v_vals_259_, lean_object* v_i_260_, lean_object* v_entries_261_){
_start:
{
lean_object* v___x_262_; uint8_t v___x_263_; 
v___x_262_ = lean_array_get_size(v_keys_258_);
v___x_263_ = lean_nat_dec_lt(v_i_260_, v___x_262_);
if (v___x_263_ == 0)
{
lean_dec(v_i_260_);
return v_entries_261_;
}
else
{
lean_object* v_k_264_; lean_object* v_v_265_; uint64_t v___x_266_; size_t v_h_267_; size_t v___x_268_; lean_object* v___x_269_; size_t v___x_270_; size_t v___x_271_; size_t v___x_272_; size_t v_h_273_; lean_object* v___x_274_; lean_object* v___x_275_; 
v_k_264_ = lean_array_fget_borrowed(v_keys_258_, v_i_260_);
v_v_265_ = lean_array_fget_borrowed(v_vals_259_, v_i_260_);
v___x_266_ = l_Lean_instHashableMVarId_hash(v_k_264_);
v_h_267_ = lean_uint64_to_usize(v___x_266_);
v___x_268_ = ((size_t)5ULL);
v___x_269_ = lean_unsigned_to_nat(1u);
v___x_270_ = ((size_t)1ULL);
v___x_271_ = lean_usize_sub(v_depth_257_, v___x_270_);
v___x_272_ = lean_usize_mul(v___x_268_, v___x_271_);
v_h_273_ = lean_usize_shift_right(v_h_267_, v___x_272_);
v___x_274_ = lean_nat_add(v_i_260_, v___x_269_);
lean_dec(v_i_260_);
lean_inc(v_v_265_);
lean_inc(v_k_264_);
v___x_275_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__6_spec__12_spec__16___redArg(v_entries_261_, v_h_273_, v_depth_257_, v_k_264_, v_v_265_);
v_i_260_ = v___x_274_;
v_entries_261_ = v___x_275_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__6_spec__12_spec__16_spec__22___redArg___boxed(lean_object* v_depth_277_, lean_object* v_keys_278_, lean_object* v_vals_279_, lean_object* v_i_280_, lean_object* v_entries_281_){
_start:
{
size_t v_depth_boxed_282_; lean_object* v_res_283_; 
v_depth_boxed_282_ = lean_unbox_usize(v_depth_277_);
lean_dec(v_depth_277_);
v_res_283_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__6_spec__12_spec__16_spec__22___redArg(v_depth_boxed_282_, v_keys_278_, v_vals_279_, v_i_280_, v_entries_281_);
lean_dec_ref(v_vals_279_);
lean_dec_ref(v_keys_278_);
return v_res_283_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__6_spec__12_spec__16___redArg___boxed(lean_object* v_x_284_, lean_object* v_x_285_, lean_object* v_x_286_, lean_object* v_x_287_, lean_object* v_x_288_){
_start:
{
size_t v_x_63597__boxed_289_; size_t v_x_63598__boxed_290_; lean_object* v_res_291_; 
v_x_63597__boxed_289_ = lean_unbox_usize(v_x_285_);
lean_dec(v_x_285_);
v_x_63598__boxed_290_ = lean_unbox_usize(v_x_286_);
lean_dec(v_x_286_);
v_res_291_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__6_spec__12_spec__16___redArg(v_x_284_, v_x_63597__boxed_289_, v_x_63598__boxed_290_, v_x_287_, v_x_288_);
return v_res_291_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__6_spec__12___redArg(lean_object* v_x_292_, lean_object* v_x_293_, lean_object* v_x_294_){
_start:
{
uint64_t v___x_295_; size_t v___x_296_; size_t v___x_297_; lean_object* v___x_298_; 
v___x_295_ = l_Lean_instHashableMVarId_hash(v_x_293_);
v___x_296_ = lean_uint64_to_usize(v___x_295_);
v___x_297_ = ((size_t)1ULL);
v___x_298_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__6_spec__12_spec__16___redArg(v_x_292_, v___x_296_, v___x_297_, v_x_293_, v_x_294_);
return v___x_298_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_assign___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__6___redArg(lean_object* v_mvarId_299_, lean_object* v_val_300_, lean_object* v___y_301_){
_start:
{
lean_object* v___x_303_; lean_object* v_mctx_304_; lean_object* v_cache_305_; lean_object* v_zetaDeltaFVarIds_306_; lean_object* v_postponed_307_; lean_object* v_diag_308_; lean_object* v___x_310_; uint8_t v_isShared_311_; uint8_t v_isSharedCheck_336_; 
v___x_303_ = lean_st_ref_take(v___y_301_);
v_mctx_304_ = lean_ctor_get(v___x_303_, 0);
v_cache_305_ = lean_ctor_get(v___x_303_, 1);
v_zetaDeltaFVarIds_306_ = lean_ctor_get(v___x_303_, 2);
v_postponed_307_ = lean_ctor_get(v___x_303_, 3);
v_diag_308_ = lean_ctor_get(v___x_303_, 4);
v_isSharedCheck_336_ = !lean_is_exclusive(v___x_303_);
if (v_isSharedCheck_336_ == 0)
{
v___x_310_ = v___x_303_;
v_isShared_311_ = v_isSharedCheck_336_;
goto v_resetjp_309_;
}
else
{
lean_inc(v_diag_308_);
lean_inc(v_postponed_307_);
lean_inc(v_zetaDeltaFVarIds_306_);
lean_inc(v_cache_305_);
lean_inc(v_mctx_304_);
lean_dec(v___x_303_);
v___x_310_ = lean_box(0);
v_isShared_311_ = v_isSharedCheck_336_;
goto v_resetjp_309_;
}
v_resetjp_309_:
{
lean_object* v_depth_312_; lean_object* v_levelAssignDepth_313_; lean_object* v_lmvarCounter_314_; lean_object* v_mvarCounter_315_; lean_object* v_lDecls_316_; lean_object* v_decls_317_; lean_object* v_userNames_318_; lean_object* v_lAssignment_319_; lean_object* v_eAssignment_320_; lean_object* v_dAssignment_321_; lean_object* v___x_323_; uint8_t v_isShared_324_; uint8_t v_isSharedCheck_335_; 
v_depth_312_ = lean_ctor_get(v_mctx_304_, 0);
v_levelAssignDepth_313_ = lean_ctor_get(v_mctx_304_, 1);
v_lmvarCounter_314_ = lean_ctor_get(v_mctx_304_, 2);
v_mvarCounter_315_ = lean_ctor_get(v_mctx_304_, 3);
v_lDecls_316_ = lean_ctor_get(v_mctx_304_, 4);
v_decls_317_ = lean_ctor_get(v_mctx_304_, 5);
v_userNames_318_ = lean_ctor_get(v_mctx_304_, 6);
v_lAssignment_319_ = lean_ctor_get(v_mctx_304_, 7);
v_eAssignment_320_ = lean_ctor_get(v_mctx_304_, 8);
v_dAssignment_321_ = lean_ctor_get(v_mctx_304_, 9);
v_isSharedCheck_335_ = !lean_is_exclusive(v_mctx_304_);
if (v_isSharedCheck_335_ == 0)
{
v___x_323_ = v_mctx_304_;
v_isShared_324_ = v_isSharedCheck_335_;
goto v_resetjp_322_;
}
else
{
lean_inc(v_dAssignment_321_);
lean_inc(v_eAssignment_320_);
lean_inc(v_lAssignment_319_);
lean_inc(v_userNames_318_);
lean_inc(v_decls_317_);
lean_inc(v_lDecls_316_);
lean_inc(v_mvarCounter_315_);
lean_inc(v_lmvarCounter_314_);
lean_inc(v_levelAssignDepth_313_);
lean_inc(v_depth_312_);
lean_dec(v_mctx_304_);
v___x_323_ = lean_box(0);
v_isShared_324_ = v_isSharedCheck_335_;
goto v_resetjp_322_;
}
v_resetjp_322_:
{
lean_object* v___x_325_; lean_object* v___x_327_; 
v___x_325_ = lp_aesop_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__6_spec__12___redArg(v_eAssignment_320_, v_mvarId_299_, v_val_300_);
if (v_isShared_324_ == 0)
{
lean_ctor_set(v___x_323_, 8, v___x_325_);
v___x_327_ = v___x_323_;
goto v_reusejp_326_;
}
else
{
lean_object* v_reuseFailAlloc_334_; 
v_reuseFailAlloc_334_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_334_, 0, v_depth_312_);
lean_ctor_set(v_reuseFailAlloc_334_, 1, v_levelAssignDepth_313_);
lean_ctor_set(v_reuseFailAlloc_334_, 2, v_lmvarCounter_314_);
lean_ctor_set(v_reuseFailAlloc_334_, 3, v_mvarCounter_315_);
lean_ctor_set(v_reuseFailAlloc_334_, 4, v_lDecls_316_);
lean_ctor_set(v_reuseFailAlloc_334_, 5, v_decls_317_);
lean_ctor_set(v_reuseFailAlloc_334_, 6, v_userNames_318_);
lean_ctor_set(v_reuseFailAlloc_334_, 7, v_lAssignment_319_);
lean_ctor_set(v_reuseFailAlloc_334_, 8, v___x_325_);
lean_ctor_set(v_reuseFailAlloc_334_, 9, v_dAssignment_321_);
v___x_327_ = v_reuseFailAlloc_334_;
goto v_reusejp_326_;
}
v_reusejp_326_:
{
lean_object* v___x_329_; 
if (v_isShared_311_ == 0)
{
lean_ctor_set(v___x_310_, 0, v___x_327_);
v___x_329_ = v___x_310_;
goto v_reusejp_328_;
}
else
{
lean_object* v_reuseFailAlloc_333_; 
v_reuseFailAlloc_333_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_333_, 0, v___x_327_);
lean_ctor_set(v_reuseFailAlloc_333_, 1, v_cache_305_);
lean_ctor_set(v_reuseFailAlloc_333_, 2, v_zetaDeltaFVarIds_306_);
lean_ctor_set(v_reuseFailAlloc_333_, 3, v_postponed_307_);
lean_ctor_set(v_reuseFailAlloc_333_, 4, v_diag_308_);
v___x_329_ = v_reuseFailAlloc_333_;
goto v_reusejp_328_;
}
v_reusejp_328_:
{
lean_object* v___x_330_; lean_object* v___x_331_; lean_object* v___x_332_; 
v___x_330_ = lean_st_ref_set(v___y_301_, v___x_329_);
v___x_331_ = lean_box(0);
v___x_332_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_332_, 0, v___x_331_);
return v___x_332_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_assign___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__6___redArg___boxed(lean_object* v_mvarId_337_, lean_object* v_val_338_, lean_object* v___y_339_, lean_object* v___y_340_){
_start:
{
lean_object* v_res_341_; 
v_res_341_ = lp_aesop_Lean_MVarId_assign___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__6___redArg(v_mvarId_337_, v_val_338_, v___y_339_);
lean_dec(v___y_339_);
return v_res_341_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_getD___at___00Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__4_spec__8___redArg(lean_object* v_a_342_, lean_object* v_fallback_343_, lean_object* v_x_344_){
_start:
{
if (lean_obj_tag(v_x_344_) == 0)
{
lean_inc(v_fallback_343_);
return v_fallback_343_;
}
else
{
lean_object* v_key_345_; lean_object* v_value_346_; lean_object* v_tail_347_; uint8_t v___x_348_; 
v_key_345_ = lean_ctor_get(v_x_344_, 0);
v_value_346_ = lean_ctor_get(v_x_344_, 1);
v_tail_347_ = lean_ctor_get(v_x_344_, 2);
v___x_348_ = l_Lean_instBEqFVarId_beq(v_key_345_, v_a_342_);
if (v___x_348_ == 0)
{
v_x_344_ = v_tail_347_;
goto _start;
}
else
{
lean_inc(v_value_346_);
return v_value_346_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_getD___at___00Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__4_spec__8___redArg___boxed(lean_object* v_a_350_, lean_object* v_fallback_351_, lean_object* v_x_352_){
_start:
{
lean_object* v_res_353_; 
v_res_353_ = lp_aesop_Std_DHashMap_Internal_AssocList_getD___at___00Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__4_spec__8___redArg(v_a_350_, v_fallback_351_, v_x_352_);
lean_dec(v_x_352_);
lean_dec(v_fallback_351_);
lean_dec(v_a_350_);
return v_res_353_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__4___redArg(lean_object* v_m_354_, lean_object* v_a_355_, lean_object* v_fallback_356_){
_start:
{
lean_object* v_buckets_357_; lean_object* v___x_358_; uint64_t v___x_359_; uint64_t v___x_360_; uint64_t v___x_361_; uint64_t v_fold_362_; uint64_t v___x_363_; uint64_t v___x_364_; uint64_t v___x_365_; size_t v___x_366_; size_t v___x_367_; size_t v___x_368_; size_t v___x_369_; size_t v___x_370_; lean_object* v___x_371_; lean_object* v___x_372_; 
v_buckets_357_ = lean_ctor_get(v_m_354_, 1);
v___x_358_ = lean_array_get_size(v_buckets_357_);
v___x_359_ = l_Lean_instHashableFVarId_hash(v_a_355_);
v___x_360_ = 32ULL;
v___x_361_ = lean_uint64_shift_right(v___x_359_, v___x_360_);
v_fold_362_ = lean_uint64_xor(v___x_359_, v___x_361_);
v___x_363_ = 16ULL;
v___x_364_ = lean_uint64_shift_right(v_fold_362_, v___x_363_);
v___x_365_ = lean_uint64_xor(v_fold_362_, v___x_364_);
v___x_366_ = lean_uint64_to_usize(v___x_365_);
v___x_367_ = lean_usize_of_nat(v___x_358_);
v___x_368_ = ((size_t)1ULL);
v___x_369_ = lean_usize_sub(v___x_367_, v___x_368_);
v___x_370_ = lean_usize_land(v___x_366_, v___x_369_);
v___x_371_ = lean_array_uget_borrowed(v_buckets_357_, v___x_370_);
v___x_372_ = lp_aesop_Std_DHashMap_Internal_AssocList_getD___at___00Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__4_spec__8___redArg(v_a_355_, v_fallback_356_, v___x_371_);
return v___x_372_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__4___redArg___boxed(lean_object* v_m_373_, lean_object* v_a_374_, lean_object* v_fallback_375_){
_start:
{
lean_object* v_res_376_; 
v_res_376_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__4___redArg(v_m_373_, v_a_374_, v_fallback_375_);
lean_dec(v_fallback_375_);
lean_dec(v_a_374_);
lean_dec_ref(v_m_373_);
return v_res_376_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_addTrace___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__3_spec__6(lean_object* v_msgData_377_, lean_object* v___y_378_, lean_object* v___y_379_, lean_object* v___y_380_, lean_object* v___y_381_){
_start:
{
lean_object* v___x_383_; lean_object* v_env_384_; lean_object* v___x_385_; lean_object* v_mctx_386_; lean_object* v_lctx_387_; lean_object* v_options_388_; lean_object* v___x_389_; lean_object* v___x_390_; lean_object* v___x_391_; 
v___x_383_ = lean_st_ref_get(v___y_381_);
v_env_384_ = lean_ctor_get(v___x_383_, 0);
lean_inc_ref(v_env_384_);
lean_dec(v___x_383_);
v___x_385_ = lean_st_ref_get(v___y_379_);
v_mctx_386_ = lean_ctor_get(v___x_385_, 0);
lean_inc_ref(v_mctx_386_);
lean_dec(v___x_385_);
v_lctx_387_ = lean_ctor_get(v___y_378_, 2);
v_options_388_ = lean_ctor_get(v___y_380_, 2);
lean_inc_ref(v_options_388_);
lean_inc_ref(v_lctx_387_);
v___x_389_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_389_, 0, v_env_384_);
lean_ctor_set(v___x_389_, 1, v_mctx_386_);
lean_ctor_set(v___x_389_, 2, v_lctx_387_);
lean_ctor_set(v___x_389_, 3, v_options_388_);
v___x_390_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_390_, 0, v___x_389_);
lean_ctor_set(v___x_390_, 1, v_msgData_377_);
v___x_391_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_391_, 0, v___x_390_);
return v___x_391_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_addTrace___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__3_spec__6___boxed(lean_object* v_msgData_392_, lean_object* v___y_393_, lean_object* v___y_394_, lean_object* v___y_395_, lean_object* v___y_396_, lean_object* v___y_397_){
_start:
{
lean_object* v_res_398_; 
v_res_398_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_addTrace___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__3_spec__6(v_msgData_392_, v___y_393_, v___y_394_, v___y_395_, v___y_396_);
lean_dec(v___y_396_);
lean_dec_ref(v___y_395_);
lean_dec(v___y_394_);
lean_dec_ref(v___y_393_);
return v_res_398_;
}
}
static double _init_lp_aesop_Lean_addTrace___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__3___redArg___closed__0(void){
_start:
{
lean_object* v___x_399_; double v___x_400_; 
v___x_399_ = lean_unsigned_to_nat(0u);
v___x_400_ = lean_float_of_nat(v___x_399_);
return v___x_400_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__3___redArg(lean_object* v_cls_404_, lean_object* v_msg_405_, lean_object* v___y_406_, lean_object* v___y_407_, lean_object* v___y_408_, lean_object* v___y_409_){
_start:
{
lean_object* v_ref_411_; lean_object* v___x_412_; lean_object* v_a_413_; lean_object* v___x_415_; uint8_t v_isShared_416_; uint8_t v_isSharedCheck_457_; 
v_ref_411_ = lean_ctor_get(v___y_408_, 5);
v___x_412_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_addTrace___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__3_spec__6(v_msg_405_, v___y_406_, v___y_407_, v___y_408_, v___y_409_);
v_a_413_ = lean_ctor_get(v___x_412_, 0);
v_isSharedCheck_457_ = !lean_is_exclusive(v___x_412_);
if (v_isSharedCheck_457_ == 0)
{
v___x_415_ = v___x_412_;
v_isShared_416_ = v_isSharedCheck_457_;
goto v_resetjp_414_;
}
else
{
lean_inc(v_a_413_);
lean_dec(v___x_412_);
v___x_415_ = lean_box(0);
v_isShared_416_ = v_isSharedCheck_457_;
goto v_resetjp_414_;
}
v_resetjp_414_:
{
lean_object* v___x_417_; lean_object* v_traceState_418_; lean_object* v_env_419_; lean_object* v_nextMacroScope_420_; lean_object* v_ngen_421_; lean_object* v_auxDeclNGen_422_; lean_object* v_cache_423_; lean_object* v_messages_424_; lean_object* v_infoState_425_; lean_object* v_snapshotTasks_426_; lean_object* v___x_428_; uint8_t v_isShared_429_; uint8_t v_isSharedCheck_456_; 
v___x_417_ = lean_st_ref_take(v___y_409_);
v_traceState_418_ = lean_ctor_get(v___x_417_, 4);
v_env_419_ = lean_ctor_get(v___x_417_, 0);
v_nextMacroScope_420_ = lean_ctor_get(v___x_417_, 1);
v_ngen_421_ = lean_ctor_get(v___x_417_, 2);
v_auxDeclNGen_422_ = lean_ctor_get(v___x_417_, 3);
v_cache_423_ = lean_ctor_get(v___x_417_, 5);
v_messages_424_ = lean_ctor_get(v___x_417_, 6);
v_infoState_425_ = lean_ctor_get(v___x_417_, 7);
v_snapshotTasks_426_ = lean_ctor_get(v___x_417_, 8);
v_isSharedCheck_456_ = !lean_is_exclusive(v___x_417_);
if (v_isSharedCheck_456_ == 0)
{
v___x_428_ = v___x_417_;
v_isShared_429_ = v_isSharedCheck_456_;
goto v_resetjp_427_;
}
else
{
lean_inc(v_snapshotTasks_426_);
lean_inc(v_infoState_425_);
lean_inc(v_messages_424_);
lean_inc(v_cache_423_);
lean_inc(v_traceState_418_);
lean_inc(v_auxDeclNGen_422_);
lean_inc(v_ngen_421_);
lean_inc(v_nextMacroScope_420_);
lean_inc(v_env_419_);
lean_dec(v___x_417_);
v___x_428_ = lean_box(0);
v_isShared_429_ = v_isSharedCheck_456_;
goto v_resetjp_427_;
}
v_resetjp_427_:
{
uint64_t v_tid_430_; lean_object* v_traces_431_; lean_object* v___x_433_; uint8_t v_isShared_434_; uint8_t v_isSharedCheck_455_; 
v_tid_430_ = lean_ctor_get_uint64(v_traceState_418_, sizeof(void*)*1);
v_traces_431_ = lean_ctor_get(v_traceState_418_, 0);
v_isSharedCheck_455_ = !lean_is_exclusive(v_traceState_418_);
if (v_isSharedCheck_455_ == 0)
{
v___x_433_ = v_traceState_418_;
v_isShared_434_ = v_isSharedCheck_455_;
goto v_resetjp_432_;
}
else
{
lean_inc(v_traces_431_);
lean_dec(v_traceState_418_);
v___x_433_ = lean_box(0);
v_isShared_434_ = v_isSharedCheck_455_;
goto v_resetjp_432_;
}
v_resetjp_432_:
{
lean_object* v___x_435_; double v___x_436_; uint8_t v___x_437_; lean_object* v___x_438_; lean_object* v___x_439_; lean_object* v___x_440_; lean_object* v___x_441_; lean_object* v___x_442_; lean_object* v___x_443_; lean_object* v___x_445_; 
v___x_435_ = lean_box(0);
v___x_436_ = lean_float_once(&lp_aesop_Lean_addTrace___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__3___redArg___closed__0, &lp_aesop_Lean_addTrace___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__3___redArg___closed__0_once, _init_lp_aesop_Lean_addTrace___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__3___redArg___closed__0);
v___x_437_ = 0;
v___x_438_ = ((lean_object*)(lp_aesop_Lean_addTrace___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__3___redArg___closed__1));
v___x_439_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v___x_439_, 0, v_cls_404_);
lean_ctor_set(v___x_439_, 1, v___x_435_);
lean_ctor_set(v___x_439_, 2, v___x_438_);
lean_ctor_set_float(v___x_439_, sizeof(void*)*3, v___x_436_);
lean_ctor_set_float(v___x_439_, sizeof(void*)*3 + 8, v___x_436_);
lean_ctor_set_uint8(v___x_439_, sizeof(void*)*3 + 16, v___x_437_);
v___x_440_ = ((lean_object*)(lp_aesop_Lean_addTrace___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__3___redArg___closed__2));
v___x_441_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v___x_441_, 0, v___x_439_);
lean_ctor_set(v___x_441_, 1, v_a_413_);
lean_ctor_set(v___x_441_, 2, v___x_440_);
lean_inc(v_ref_411_);
v___x_442_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_442_, 0, v_ref_411_);
lean_ctor_set(v___x_442_, 1, v___x_441_);
v___x_443_ = l_Lean_PersistentArray_push___redArg(v_traces_431_, v___x_442_);
if (v_isShared_434_ == 0)
{
lean_ctor_set(v___x_433_, 0, v___x_443_);
v___x_445_ = v___x_433_;
goto v_reusejp_444_;
}
else
{
lean_object* v_reuseFailAlloc_454_; 
v_reuseFailAlloc_454_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_454_, 0, v___x_443_);
lean_ctor_set_uint64(v_reuseFailAlloc_454_, sizeof(void*)*1, v_tid_430_);
v___x_445_ = v_reuseFailAlloc_454_;
goto v_reusejp_444_;
}
v_reusejp_444_:
{
lean_object* v___x_447_; 
if (v_isShared_429_ == 0)
{
lean_ctor_set(v___x_428_, 4, v___x_445_);
v___x_447_ = v___x_428_;
goto v_reusejp_446_;
}
else
{
lean_object* v_reuseFailAlloc_453_; 
v_reuseFailAlloc_453_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_453_, 0, v_env_419_);
lean_ctor_set(v_reuseFailAlloc_453_, 1, v_nextMacroScope_420_);
lean_ctor_set(v_reuseFailAlloc_453_, 2, v_ngen_421_);
lean_ctor_set(v_reuseFailAlloc_453_, 3, v_auxDeclNGen_422_);
lean_ctor_set(v_reuseFailAlloc_453_, 4, v___x_445_);
lean_ctor_set(v_reuseFailAlloc_453_, 5, v_cache_423_);
lean_ctor_set(v_reuseFailAlloc_453_, 6, v_messages_424_);
lean_ctor_set(v_reuseFailAlloc_453_, 7, v_infoState_425_);
lean_ctor_set(v_reuseFailAlloc_453_, 8, v_snapshotTasks_426_);
v___x_447_ = v_reuseFailAlloc_453_;
goto v_reusejp_446_;
}
v_reusejp_446_:
{
lean_object* v___x_448_; lean_object* v___x_449_; lean_object* v___x_451_; 
v___x_448_ = lean_st_ref_set(v___y_409_, v___x_447_);
v___x_449_ = lean_box(0);
if (v_isShared_416_ == 0)
{
lean_ctor_set(v___x_415_, 0, v___x_449_);
v___x_451_ = v___x_415_;
goto v_reusejp_450_;
}
else
{
lean_object* v_reuseFailAlloc_452_; 
v_reuseFailAlloc_452_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_452_, 0, v___x_449_);
v___x_451_ = v_reuseFailAlloc_452_;
goto v_reusejp_450_;
}
v_reusejp_450_:
{
return v___x_451_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__3___redArg___boxed(lean_object* v_cls_458_, lean_object* v_msg_459_, lean_object* v___y_460_, lean_object* v___y_461_, lean_object* v___y_462_, lean_object* v___y_463_, lean_object* v___y_464_){
_start:
{
lean_object* v_res_465_; 
v_res_465_ = lp_aesop_Lean_addTrace___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__3___redArg(v_cls_458_, v_msg_459_, v___y_460_, v___y_461_, v___y_462_, v___y_463_);
lean_dec(v___y_463_);
lean_dec_ref(v___y_462_);
lean_dec(v___y_461_);
lean_dec_ref(v___y_460_);
return v_res_465_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__8___redArg(lean_object* v_as_466_, size_t v_sz_467_, size_t v_i_468_, lean_object* v_b_469_, lean_object* v___y_470_){
_start:
{
lean_object* v_a_473_; uint8_t v___x_477_; 
v___x_477_ = lean_usize_dec_lt(v_i_468_, v_sz_467_);
if (v___x_477_ == 0)
{
lean_object* v___x_478_; 
v___x_478_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_478_, 0, v_b_469_);
return v___x_478_;
}
else
{
lean_object* v_forwardHypData_479_; lean_object* v___x_480_; lean_object* v_a_481_; lean_object* v___x_482_; uint8_t v___x_483_; 
v_forwardHypData_479_ = lean_ctor_get(v___y_470_, 1);
v___x_480_ = lean_unsigned_to_nat(0u);
v_a_481_ = lean_array_uget_borrowed(v_as_466_, v_i_468_);
v___x_482_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__4___redArg(v_forwardHypData_479_, v_a_481_, v___x_480_);
v___x_483_ = lean_nat_dec_le(v_b_469_, v___x_482_);
if (v___x_483_ == 0)
{
lean_dec(v___x_482_);
v_a_473_ = v_b_469_;
goto v___jp_472_;
}
else
{
lean_dec(v_b_469_);
v_a_473_ = v___x_482_;
goto v___jp_472_;
}
}
v___jp_472_:
{
size_t v___x_474_; size_t v___x_475_; 
v___x_474_ = ((size_t)1ULL);
v___x_475_ = lean_usize_add(v_i_468_, v___x_474_);
v_i_468_ = v___x_475_;
v_b_469_ = v_a_473_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__8___redArg___boxed(lean_object* v_as_484_, lean_object* v_sz_485_, lean_object* v_i_486_, lean_object* v_b_487_, lean_object* v___y_488_, lean_object* v___y_489_){
_start:
{
size_t v_sz_boxed_490_; size_t v_i_boxed_491_; lean_object* v_res_492_; 
v_sz_boxed_490_ = lean_unbox_usize(v_sz_485_);
lean_dec(v_sz_485_);
v_i_boxed_491_ = lean_unbox_usize(v_i_486_);
lean_dec(v_i_486_);
v_res_492_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__8___redArg(v_as_484_, v_sz_boxed_490_, v_i_boxed_491_, v_b_487_, v___y_488_);
lean_dec_ref(v___y_488_);
lean_dec_ref(v_as_484_);
return v_res_492_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__0_spec__0_spec__2_spec__13_spec__20___redArg(lean_object* v_x_493_, lean_object* v_x_494_){
_start:
{
if (lean_obj_tag(v_x_494_) == 0)
{
return v_x_493_;
}
else
{
lean_object* v_key_495_; lean_object* v_value_496_; lean_object* v_tail_497_; lean_object* v___x_499_; uint8_t v_isShared_500_; uint8_t v_isSharedCheck_520_; 
v_key_495_ = lean_ctor_get(v_x_494_, 0);
v_value_496_ = lean_ctor_get(v_x_494_, 1);
v_tail_497_ = lean_ctor_get(v_x_494_, 2);
v_isSharedCheck_520_ = !lean_is_exclusive(v_x_494_);
if (v_isSharedCheck_520_ == 0)
{
v___x_499_ = v_x_494_;
v_isShared_500_ = v_isSharedCheck_520_;
goto v_resetjp_498_;
}
else
{
lean_inc(v_tail_497_);
lean_inc(v_value_496_);
lean_inc(v_key_495_);
lean_dec(v_x_494_);
v___x_499_ = lean_box(0);
v_isShared_500_ = v_isSharedCheck_520_;
goto v_resetjp_498_;
}
v_resetjp_498_:
{
lean_object* v___x_501_; uint64_t v___x_502_; uint64_t v___x_503_; uint64_t v___x_504_; uint64_t v_fold_505_; uint64_t v___x_506_; uint64_t v___x_507_; uint64_t v___x_508_; size_t v___x_509_; size_t v___x_510_; size_t v___x_511_; size_t v___x_512_; size_t v___x_513_; lean_object* v___x_514_; lean_object* v___x_516_; 
v___x_501_ = lean_array_get_size(v_x_493_);
v___x_502_ = l_Lean_instHashableFVarId_hash(v_key_495_);
v___x_503_ = 32ULL;
v___x_504_ = lean_uint64_shift_right(v___x_502_, v___x_503_);
v_fold_505_ = lean_uint64_xor(v___x_502_, v___x_504_);
v___x_506_ = 16ULL;
v___x_507_ = lean_uint64_shift_right(v_fold_505_, v___x_506_);
v___x_508_ = lean_uint64_xor(v_fold_505_, v___x_507_);
v___x_509_ = lean_uint64_to_usize(v___x_508_);
v___x_510_ = lean_usize_of_nat(v___x_501_);
v___x_511_ = ((size_t)1ULL);
v___x_512_ = lean_usize_sub(v___x_510_, v___x_511_);
v___x_513_ = lean_usize_land(v___x_509_, v___x_512_);
v___x_514_ = lean_array_uget_borrowed(v_x_493_, v___x_513_);
lean_inc(v___x_514_);
if (v_isShared_500_ == 0)
{
lean_ctor_set(v___x_499_, 2, v___x_514_);
v___x_516_ = v___x_499_;
goto v_reusejp_515_;
}
else
{
lean_object* v_reuseFailAlloc_519_; 
v_reuseFailAlloc_519_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_519_, 0, v_key_495_);
lean_ctor_set(v_reuseFailAlloc_519_, 1, v_value_496_);
lean_ctor_set(v_reuseFailAlloc_519_, 2, v___x_514_);
v___x_516_ = v_reuseFailAlloc_519_;
goto v_reusejp_515_;
}
v_reusejp_515_:
{
lean_object* v___x_517_; 
v___x_517_ = lean_array_uset(v_x_493_, v___x_513_, v___x_516_);
v_x_493_ = v___x_517_;
v_x_494_ = v_tail_497_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__0_spec__0_spec__2_spec__13___redArg(lean_object* v_i_521_, lean_object* v_source_522_, lean_object* v_target_523_){
_start:
{
lean_object* v___x_524_; uint8_t v___x_525_; 
v___x_524_ = lean_array_get_size(v_source_522_);
v___x_525_ = lean_nat_dec_lt(v_i_521_, v___x_524_);
if (v___x_525_ == 0)
{
lean_dec_ref(v_source_522_);
lean_dec(v_i_521_);
return v_target_523_;
}
else
{
lean_object* v_es_526_; lean_object* v___x_527_; lean_object* v_source_528_; lean_object* v_target_529_; lean_object* v___x_530_; lean_object* v___x_531_; 
v_es_526_ = lean_array_fget(v_source_522_, v_i_521_);
v___x_527_ = lean_box(0);
v_source_528_ = lean_array_fset(v_source_522_, v_i_521_, v___x_527_);
v_target_529_ = lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__0_spec__0_spec__2_spec__13_spec__20___redArg(v_target_523_, v_es_526_);
v___x_530_ = lean_unsigned_to_nat(1u);
v___x_531_ = lean_nat_add(v_i_521_, v___x_530_);
lean_dec(v_i_521_);
v_i_521_ = v___x_531_;
v_source_522_ = v_source_528_;
v_target_523_ = v_target_529_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__0_spec__0_spec__2___redArg(lean_object* v_data_533_){
_start:
{
lean_object* v___x_534_; lean_object* v___x_535_; lean_object* v_nbuckets_536_; lean_object* v___x_537_; lean_object* v___x_538_; lean_object* v___x_539_; lean_object* v___x_540_; 
v___x_534_ = lean_array_get_size(v_data_533_);
v___x_535_ = lean_unsigned_to_nat(2u);
v_nbuckets_536_ = lean_nat_mul(v___x_534_, v___x_535_);
v___x_537_ = lean_unsigned_to_nat(0u);
v___x_538_ = lean_box(0);
v___x_539_ = lean_mk_array(v_nbuckets_536_, v___x_538_);
v___x_540_ = lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__0_spec__0_spec__2_spec__13___redArg(v___x_537_, v_data_533_, v___x_539_);
return v___x_540_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__0_spec__0_spec__1___redArg(lean_object* v_a_541_, lean_object* v_x_542_){
_start:
{
if (lean_obj_tag(v_x_542_) == 0)
{
uint8_t v___x_543_; 
v___x_543_ = 0;
return v___x_543_;
}
else
{
lean_object* v_key_544_; lean_object* v_tail_545_; uint8_t v___x_546_; 
v_key_544_ = lean_ctor_get(v_x_542_, 0);
v_tail_545_ = lean_ctor_get(v_x_542_, 2);
v___x_546_ = l_Lean_instBEqFVarId_beq(v_key_544_, v_a_541_);
if (v___x_546_ == 0)
{
v_x_542_ = v_tail_545_;
goto _start;
}
else
{
return v___x_546_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__0_spec__0_spec__1___redArg___boxed(lean_object* v_a_548_, lean_object* v_x_549_){
_start:
{
uint8_t v_res_550_; lean_object* v_r_551_; 
v_res_550_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__0_spec__0_spec__1___redArg(v_a_548_, v_x_549_);
lean_dec(v_x_549_);
lean_dec(v_a_548_);
v_r_551_ = lean_box(v_res_550_);
return v_r_551_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__0_spec__0___redArg(lean_object* v_m_552_, lean_object* v_a_553_, lean_object* v_b_554_){
_start:
{
lean_object* v_size_555_; lean_object* v_buckets_556_; lean_object* v___x_557_; uint64_t v___x_558_; uint64_t v___x_559_; uint64_t v___x_560_; uint64_t v_fold_561_; uint64_t v___x_562_; uint64_t v___x_563_; uint64_t v___x_564_; size_t v___x_565_; size_t v___x_566_; size_t v___x_567_; size_t v___x_568_; size_t v___x_569_; lean_object* v_bkt_570_; uint8_t v___x_571_; 
v_size_555_ = lean_ctor_get(v_m_552_, 0);
v_buckets_556_ = lean_ctor_get(v_m_552_, 1);
v___x_557_ = lean_array_get_size(v_buckets_556_);
v___x_558_ = l_Lean_instHashableFVarId_hash(v_a_553_);
v___x_559_ = 32ULL;
v___x_560_ = lean_uint64_shift_right(v___x_558_, v___x_559_);
v_fold_561_ = lean_uint64_xor(v___x_558_, v___x_560_);
v___x_562_ = 16ULL;
v___x_563_ = lean_uint64_shift_right(v_fold_561_, v___x_562_);
v___x_564_ = lean_uint64_xor(v_fold_561_, v___x_563_);
v___x_565_ = lean_uint64_to_usize(v___x_564_);
v___x_566_ = lean_usize_of_nat(v___x_557_);
v___x_567_ = ((size_t)1ULL);
v___x_568_ = lean_usize_sub(v___x_566_, v___x_567_);
v___x_569_ = lean_usize_land(v___x_565_, v___x_568_);
v_bkt_570_ = lean_array_uget_borrowed(v_buckets_556_, v___x_569_);
v___x_571_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__0_spec__0_spec__1___redArg(v_a_553_, v_bkt_570_);
if (v___x_571_ == 0)
{
lean_object* v___x_573_; uint8_t v_isShared_574_; uint8_t v_isSharedCheck_592_; 
lean_inc_ref(v_buckets_556_);
lean_inc(v_size_555_);
v_isSharedCheck_592_ = !lean_is_exclusive(v_m_552_);
if (v_isSharedCheck_592_ == 0)
{
lean_object* v_unused_593_; lean_object* v_unused_594_; 
v_unused_593_ = lean_ctor_get(v_m_552_, 1);
lean_dec(v_unused_593_);
v_unused_594_ = lean_ctor_get(v_m_552_, 0);
lean_dec(v_unused_594_);
v___x_573_ = v_m_552_;
v_isShared_574_ = v_isSharedCheck_592_;
goto v_resetjp_572_;
}
else
{
lean_dec(v_m_552_);
v___x_573_ = lean_box(0);
v_isShared_574_ = v_isSharedCheck_592_;
goto v_resetjp_572_;
}
v_resetjp_572_:
{
lean_object* v___x_575_; lean_object* v_size_x27_576_; lean_object* v___x_577_; lean_object* v_buckets_x27_578_; lean_object* v___x_579_; lean_object* v___x_580_; lean_object* v___x_581_; lean_object* v___x_582_; lean_object* v___x_583_; uint8_t v___x_584_; 
v___x_575_ = lean_unsigned_to_nat(1u);
v_size_x27_576_ = lean_nat_add(v_size_555_, v___x_575_);
lean_dec(v_size_555_);
lean_inc(v_bkt_570_);
v___x_577_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_577_, 0, v_a_553_);
lean_ctor_set(v___x_577_, 1, v_b_554_);
lean_ctor_set(v___x_577_, 2, v_bkt_570_);
v_buckets_x27_578_ = lean_array_uset(v_buckets_556_, v___x_569_, v___x_577_);
v___x_579_ = lean_unsigned_to_nat(4u);
v___x_580_ = lean_nat_mul(v_size_x27_576_, v___x_579_);
v___x_581_ = lean_unsigned_to_nat(3u);
v___x_582_ = lean_nat_div(v___x_580_, v___x_581_);
lean_dec(v___x_580_);
v___x_583_ = lean_array_get_size(v_buckets_x27_578_);
v___x_584_ = lean_nat_dec_le(v___x_582_, v___x_583_);
lean_dec(v___x_582_);
if (v___x_584_ == 0)
{
lean_object* v_val_585_; lean_object* v___x_587_; 
v_val_585_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__0_spec__0_spec__2___redArg(v_buckets_x27_578_);
if (v_isShared_574_ == 0)
{
lean_ctor_set(v___x_573_, 1, v_val_585_);
lean_ctor_set(v___x_573_, 0, v_size_x27_576_);
v___x_587_ = v___x_573_;
goto v_reusejp_586_;
}
else
{
lean_object* v_reuseFailAlloc_588_; 
v_reuseFailAlloc_588_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_588_, 0, v_size_x27_576_);
lean_ctor_set(v_reuseFailAlloc_588_, 1, v_val_585_);
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
lean_object* v___x_590_; 
if (v_isShared_574_ == 0)
{
lean_ctor_set(v___x_573_, 1, v_buckets_x27_578_);
lean_ctor_set(v___x_573_, 0, v_size_x27_576_);
v___x_590_ = v___x_573_;
goto v_reusejp_589_;
}
else
{
lean_object* v_reuseFailAlloc_591_; 
v_reuseFailAlloc_591_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_591_, 0, v_size_x27_576_);
lean_ctor_set(v_reuseFailAlloc_591_, 1, v_buckets_x27_578_);
v___x_590_ = v_reuseFailAlloc_591_;
goto v_reusejp_589_;
}
v_reusejp_589_:
{
return v___x_590_;
}
}
}
}
else
{
lean_dec(v_b_554_);
lean_dec(v_a_553_);
return v_m_552_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__0_spec__1(lean_object* v_as_595_, size_t v_sz_596_, size_t v_i_597_, lean_object* v_b_598_){
_start:
{
uint8_t v___x_599_; 
v___x_599_ = lean_usize_dec_lt(v_i_597_, v_sz_596_);
if (v___x_599_ == 0)
{
return v_b_598_;
}
else
{
lean_object* v_a_600_; lean_object* v___x_601_; lean_object* v_r_602_; size_t v___x_603_; size_t v___x_604_; 
v_a_600_ = lean_array_uget_borrowed(v_as_595_, v_i_597_);
v___x_601_ = lean_box(0);
lean_inc(v_a_600_);
v_r_602_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__0_spec__0___redArg(v_b_598_, v_a_600_, v___x_601_);
v___x_603_ = ((size_t)1ULL);
v___x_604_ = lean_usize_add(v_i_597_, v___x_603_);
v_i_597_ = v___x_604_;
v_b_598_ = v_r_602_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__0_spec__1___boxed(lean_object* v_as_606_, lean_object* v_sz_607_, lean_object* v_i_608_, lean_object* v_b_609_){
_start:
{
size_t v_sz_boxed_610_; size_t v_i_boxed_611_; lean_object* v_res_612_; 
v_sz_boxed_610_ = lean_unbox_usize(v_sz_607_);
lean_dec(v_sz_607_);
v_i_boxed_611_ = lean_unbox_usize(v_i_608_);
lean_dec(v_i_608_);
v_res_612_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__0_spec__1(v_as_606_, v_sz_boxed_610_, v_i_boxed_611_, v_b_609_);
lean_dec_ref(v_as_606_);
return v_res_612_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__0(lean_object* v_m_613_, lean_object* v_l_614_){
_start:
{
size_t v_sz_615_; size_t v___x_616_; lean_object* v___x_617_; 
v_sz_615_ = lean_array_size(v_l_614_);
v___x_616_ = ((size_t)0ULL);
v___x_617_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__0_spec__1(v_l_614_, v_sz_615_, v___x_616_, v_m_613_);
return v___x_617_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__0___boxed(lean_object* v_m_618_, lean_object* v_l_619_){
_start:
{
lean_object* v_res_620_; 
v_res_620_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__0(v_m_618_, v_l_619_);
lean_dec_ref(v_l_619_);
return v_res_620_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__5_spec__10_spec__13_spec__18___redArg(lean_object* v_keys_621_, lean_object* v_i_622_, lean_object* v_k_623_){
_start:
{
lean_object* v___x_624_; uint8_t v___x_625_; 
v___x_624_ = lean_array_get_size(v_keys_621_);
v___x_625_ = lean_nat_dec_lt(v_i_622_, v___x_624_);
if (v___x_625_ == 0)
{
lean_dec(v_i_622_);
return v___x_625_;
}
else
{
lean_object* v_k_x27_626_; uint8_t v___x_627_; 
v_k_x27_626_ = lean_array_fget_borrowed(v_keys_621_, v_i_622_);
v___x_627_ = l_Lean_instBEqMVarId_beq(v_k_623_, v_k_x27_626_);
if (v___x_627_ == 0)
{
lean_object* v___x_628_; lean_object* v___x_629_; 
v___x_628_ = lean_unsigned_to_nat(1u);
v___x_629_ = lean_nat_add(v_i_622_, v___x_628_);
lean_dec(v_i_622_);
v_i_622_ = v___x_629_;
goto _start;
}
else
{
lean_dec(v_i_622_);
return v___x_627_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__5_spec__10_spec__13_spec__18___redArg___boxed(lean_object* v_keys_631_, lean_object* v_i_632_, lean_object* v_k_633_){
_start:
{
uint8_t v_res_634_; lean_object* v_r_635_; 
v_res_634_ = lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__5_spec__10_spec__13_spec__18___redArg(v_keys_631_, v_i_632_, v_k_633_);
lean_dec(v_k_633_);
lean_dec_ref(v_keys_631_);
v_r_635_ = lean_box(v_res_634_);
return v_r_635_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__5_spec__10_spec__13___redArg(lean_object* v_x_636_, size_t v_x_637_, lean_object* v_x_638_){
_start:
{
if (lean_obj_tag(v_x_636_) == 0)
{
lean_object* v_es_639_; lean_object* v___x_640_; size_t v___x_641_; size_t v___x_642_; lean_object* v_j_643_; lean_object* v___x_644_; 
v_es_639_ = lean_ctor_get(v_x_636_, 0);
v___x_640_ = lean_box(2);
v___x_641_ = ((size_t)31ULL);
v___x_642_ = lean_usize_land(v_x_637_, v___x_641_);
v_j_643_ = lean_usize_to_nat(v___x_642_);
v___x_644_ = lean_array_get_borrowed(v___x_640_, v_es_639_, v_j_643_);
lean_dec(v_j_643_);
switch(lean_obj_tag(v___x_644_))
{
case 0:
{
lean_object* v_key_645_; uint8_t v___x_646_; 
v_key_645_ = lean_ctor_get(v___x_644_, 0);
v___x_646_ = l_Lean_instBEqMVarId_beq(v_x_638_, v_key_645_);
return v___x_646_;
}
case 1:
{
lean_object* v_node_647_; size_t v___x_648_; size_t v___x_649_; 
v_node_647_ = lean_ctor_get(v___x_644_, 0);
v___x_648_ = ((size_t)5ULL);
v___x_649_ = lean_usize_shift_right(v_x_637_, v___x_648_);
v_x_636_ = v_node_647_;
v_x_637_ = v___x_649_;
goto _start;
}
default: 
{
uint8_t v___x_651_; 
v___x_651_ = 0;
return v___x_651_;
}
}
}
else
{
lean_object* v_ks_652_; lean_object* v___x_653_; uint8_t v___x_654_; 
v_ks_652_ = lean_ctor_get(v_x_636_, 0);
v___x_653_ = lean_unsigned_to_nat(0u);
v___x_654_ = lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__5_spec__10_spec__13_spec__18___redArg(v_ks_652_, v___x_653_, v_x_638_);
return v___x_654_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__5_spec__10_spec__13___redArg___boxed(lean_object* v_x_655_, lean_object* v_x_656_, lean_object* v_x_657_){
_start:
{
size_t v_x_64197__boxed_658_; uint8_t v_res_659_; lean_object* v_r_660_; 
v_x_64197__boxed_658_ = lean_unbox_usize(v_x_656_);
lean_dec(v_x_656_);
v_res_659_ = lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__5_spec__10_spec__13___redArg(v_x_655_, v_x_64197__boxed_658_, v_x_657_);
lean_dec(v_x_657_);
lean_dec_ref(v_x_655_);
v_r_660_ = lean_box(v_res_659_);
return v_r_660_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__5_spec__10___redArg(lean_object* v_x_661_, lean_object* v_x_662_){
_start:
{
uint64_t v___x_663_; size_t v___x_664_; uint8_t v___x_665_; 
v___x_663_ = l_Lean_instHashableMVarId_hash(v_x_662_);
v___x_664_ = lean_uint64_to_usize(v___x_663_);
v___x_665_ = lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__5_spec__10_spec__13___redArg(v_x_661_, v___x_664_, v_x_662_);
return v___x_665_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__5_spec__10___redArg___boxed(lean_object* v_x_666_, lean_object* v_x_667_){
_start:
{
uint8_t v_res_668_; lean_object* v_r_669_; 
v_res_668_ = lp_aesop_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__5_spec__10___redArg(v_x_666_, v_x_667_);
lean_dec(v_x_667_);
lean_dec_ref(v_x_666_);
v_r_669_ = lean_box(v_res_668_);
return v_r_669_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__5___redArg(lean_object* v_mvarId_670_, lean_object* v___y_671_){
_start:
{
lean_object* v___x_673_; lean_object* v_mctx_674_; lean_object* v_eAssignment_675_; lean_object* v_dAssignment_676_; uint8_t v___x_677_; 
v___x_673_ = lean_st_ref_get(v___y_671_);
v_mctx_674_ = lean_ctor_get(v___x_673_, 0);
lean_inc_ref(v_mctx_674_);
lean_dec(v___x_673_);
v_eAssignment_675_ = lean_ctor_get(v_mctx_674_, 8);
lean_inc_ref(v_eAssignment_675_);
v_dAssignment_676_ = lean_ctor_get(v_mctx_674_, 9);
lean_inc_ref(v_dAssignment_676_);
lean_dec_ref(v_mctx_674_);
v___x_677_ = lp_aesop_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__5_spec__10___redArg(v_eAssignment_675_, v_mvarId_670_);
lean_dec_ref(v_eAssignment_675_);
if (v___x_677_ == 0)
{
uint8_t v___x_678_; lean_object* v___x_679_; lean_object* v___x_680_; 
v___x_678_ = lp_aesop_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__5_spec__10___redArg(v_dAssignment_676_, v_mvarId_670_);
lean_dec_ref(v_dAssignment_676_);
v___x_679_ = lean_box(v___x_678_);
v___x_680_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_680_, 0, v___x_679_);
return v___x_680_;
}
else
{
lean_object* v___x_681_; lean_object* v___x_682_; 
lean_dec_ref(v_dAssignment_676_);
v___x_681_ = lean_box(v___x_677_);
v___x_682_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_682_, 0, v___x_681_);
return v___x_682_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__5___redArg___boxed(lean_object* v_mvarId_683_, lean_object* v___y_684_, lean_object* v___y_685_){
_start:
{
lean_object* v_res_686_; 
v_res_686_ = lp_aesop_Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__5___redArg(v_mvarId_683_, v___y_684_);
lean_dec(v___y_684_);
lean_dec(v_mvarId_683_);
return v_res_686_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_Option_get___at___00Aesop_TraceOption_isEnabled___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__2_spec__4(lean_object* v_opts_687_, lean_object* v_opt_688_){
_start:
{
lean_object* v_name_689_; lean_object* v_defValue_690_; lean_object* v_map_691_; lean_object* v___x_692_; 
v_name_689_ = lean_ctor_get(v_opt_688_, 0);
v_defValue_690_ = lean_ctor_get(v_opt_688_, 1);
v_map_691_ = lean_ctor_get(v_opts_687_, 0);
v___x_692_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_691_, v_name_689_);
if (lean_obj_tag(v___x_692_) == 0)
{
uint8_t v___x_693_; 
v___x_693_ = lean_unbox(v_defValue_690_);
return v___x_693_;
}
else
{
lean_object* v_val_694_; 
v_val_694_ = lean_ctor_get(v___x_692_, 0);
lean_inc(v_val_694_);
lean_dec_ref_known(v___x_692_, 1);
if (lean_obj_tag(v_val_694_) == 1)
{
uint8_t v_v_695_; 
v_v_695_ = lean_ctor_get_uint8(v_val_694_, 0);
lean_dec_ref_known(v_val_694_, 0);
return v_v_695_;
}
else
{
uint8_t v___x_696_; 
lean_dec(v_val_694_);
v___x_696_ = lean_unbox(v_defValue_690_);
return v___x_696_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Aesop_TraceOption_isEnabled___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__2_spec__4___boxed(lean_object* v_opts_697_, lean_object* v_opt_698_){
_start:
{
uint8_t v_res_699_; lean_object* v_r_700_; 
v_res_699_ = lp_aesop_Lean_Option_get___at___00Aesop_TraceOption_isEnabled___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__2_spec__4(v_opts_697_, v_opt_698_);
lean_dec_ref(v_opt_698_);
lean_dec_ref(v_opts_697_);
v_r_700_ = lean_box(v_res_699_);
return v_r_700_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__2___redArg(lean_object* v_opt_701_, lean_object* v___y_702_){
_start:
{
lean_object* v_options_704_; lean_object* v_option_705_; uint8_t v___x_706_; lean_object* v___x_707_; lean_object* v___x_708_; 
v_options_704_ = lean_ctor_get(v___y_702_, 2);
v_option_705_ = lean_ctor_get(v_opt_701_, 1);
v___x_706_ = lp_aesop_Lean_Option_get___at___00Aesop_TraceOption_isEnabled___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__2_spec__4(v_options_704_, v_option_705_);
v___x_707_ = lean_box(v___x_706_);
v___x_708_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_708_, 0, v___x_707_);
return v___x_708_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__2___redArg___boxed(lean_object* v_opt_709_, lean_object* v___y_710_, lean_object* v___y_711_){
_start:
{
lean_object* v_res_712_; 
v_res_712_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__2___redArg(v_opt_709_, v___y_710_);
lean_dec_ref(v___y_710_);
lean_dec_ref(v_opt_709_);
return v_res_712_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__1(size_t v_sz_713_, size_t v_i_714_, lean_object* v_bs_715_){
_start:
{
uint8_t v___x_716_; 
v___x_716_ = lean_usize_dec_lt(v_i_714_, v_sz_713_);
if (v___x_716_ == 0)
{
return v_bs_715_;
}
else
{
lean_object* v_v_717_; lean_object* v___x_718_; lean_object* v_bs_x27_719_; lean_object* v___x_720_; size_t v___x_721_; size_t v___x_722_; lean_object* v___x_723_; 
v_v_717_ = lean_array_uget(v_bs_715_, v_i_714_);
v___x_718_ = lean_unsigned_to_nat(0u);
v_bs_x27_719_ = lean_array_uset(v_bs_715_, v_i_714_, v___x_718_);
v___x_720_ = l_Lean_Expr_mvar___override(v_v_717_);
v___x_721_ = ((size_t)1ULL);
v___x_722_ = lean_usize_add(v_i_714_, v___x_721_);
v___x_723_ = lean_array_uset(v_bs_x27_719_, v_i_714_, v___x_720_);
v_i_714_ = v___x_722_;
v_bs_715_ = v___x_723_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__1___boxed(lean_object* v_sz_725_, lean_object* v_i_726_, lean_object* v_bs_727_){
_start:
{
size_t v_sz_boxed_728_; size_t v_i_boxed_729_; lean_object* v_res_730_; 
v_sz_boxed_728_ = lean_unbox_usize(v_sz_725_);
lean_dec(v_sz_725_);
v_i_boxed_729_ = lean_unbox_usize(v_i_726_);
lean_dec(v_i_726_);
v_res_730_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__1(v_sz_boxed_728_, v_i_boxed_729_, v_bs_727_);
return v_res_730_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop___closed__1(void){
_start:
{
lean_object* v___x_732_; lean_object* v___x_733_; 
v___x_732_ = ((lean_object*)(lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop___closed__0));
v___x_733_ = l_Lean_stringToMessageData(v___x_732_);
return v___x_733_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__7_spec__15_spec__22(lean_object* v_a_737_, lean_object* v_mvarId_738_, lean_object* v_currentUsedHyps_739_, lean_object* v_i_740_, lean_object* v_app_741_, lean_object* v_instMVars_742_, lean_object* v_immediateMVars_743_, lean_object* v_currentMaxHypDepth_744_, lean_object* v_as_745_, size_t v_sz_746_, size_t v_i_747_, lean_object* v_b_748_, lean_object* v___y_749_, lean_object* v___y_750_, lean_object* v___y_751_, lean_object* v___y_752_, lean_object* v___y_753_, lean_object* v___y_754_, lean_object* v___y_755_, lean_object* v___y_756_){
_start:
{
lean_object* v___y_759_; lean_object* v___y_760_; lean_object* v___y_761_; lean_object* v_a_762_; uint8_t v___x_780_; 
v___x_780_ = lean_usize_dec_lt(v_i_747_, v_sz_746_);
if (v___x_780_ == 0)
{
lean_object* v___x_781_; 
lean_dec(v_currentMaxHypDepth_744_);
lean_dec_ref(v_instMVars_742_);
lean_dec_ref(v_app_741_);
lean_dec_ref(v_currentUsedHyps_739_);
lean_dec(v_mvarId_738_);
lean_dec_ref(v_a_737_);
v___x_781_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_781_, 0, v_b_748_);
return v___x_781_;
}
else
{
lean_object* v_snd_782_; lean_object* v___x_784_; uint8_t v_isShared_785_; uint8_t v_isSharedCheck_860_; 
v_snd_782_ = lean_ctor_get(v_b_748_, 1);
v_isSharedCheck_860_ = !lean_is_exclusive(v_b_748_);
if (v_isSharedCheck_860_ == 0)
{
lean_object* v_unused_861_; 
v_unused_861_ = lean_ctor_get(v_b_748_, 0);
lean_dec(v_unused_861_);
v___x_784_ = v_b_748_;
v_isShared_785_ = v_isSharedCheck_860_;
goto v_resetjp_783_;
}
else
{
lean_inc(v_snd_782_);
lean_dec(v_b_748_);
v___x_784_ = lean_box(0);
v_isShared_785_ = v_isSharedCheck_860_;
goto v_resetjp_783_;
}
v_resetjp_783_:
{
lean_object* v___x_786_; lean_object* v_a_788_; lean_object* v_a_795_; 
v___x_786_ = lean_box(0);
v_a_795_ = lean_array_uget_borrowed(v_as_745_, v_i_747_);
if (lean_obj_tag(v_a_795_) == 0)
{
v_a_788_ = v_snd_782_;
goto v___jp_787_;
}
else
{
lean_object* v_val_796_; lean_object* v___x_797_; lean_object* v___y_799_; lean_object* v___y_800_; lean_object* v___y_801_; lean_object* v___y_812_; lean_object* v___y_813_; lean_object* v___y_814_; lean_object* v___y_815_; uint8_t v___x_817_; 
lean_dec(v_snd_782_);
v_val_796_ = lean_ctor_get(v_a_795_, 0);
v___x_797_ = lean_box(0);
v___x_817_ = l_Lean_LocalDecl_isImplementationDetail(v_val_796_);
if (v___x_817_ == 0)
{
lean_object* v_maxDepth_x3f_818_; lean_object* v_forwardHypData_819_; lean_object* v___x_820_; lean_object* v___x_821_; lean_object* v___x_822_; lean_object* v___y_824_; lean_object* v___y_825_; lean_object* v___y_826_; lean_object* v___y_827_; lean_object* v___y_828_; lean_object* v___y_829_; lean_object* v___y_830_; lean_object* v___y_831_; lean_object* v___y_832_; lean_object* v___y_854_; lean_object* v___x_858_; uint8_t v___x_859_; 
v_maxDepth_x3f_818_ = lean_ctor_get(v___y_749_, 0);
v_forwardHypData_819_ = lean_ctor_get(v___y_749_, 1);
v___x_820_ = lean_unsigned_to_nat(1u);
v___x_821_ = lean_unsigned_to_nat(0u);
v___x_822_ = l_Lean_LocalDecl_fvarId(v_val_796_);
v___x_858_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__4___redArg(v_forwardHypData_819_, v___x_822_, v___x_821_);
v___x_859_ = lean_nat_dec_le(v_currentMaxHypDepth_744_, v___x_858_);
if (v___x_859_ == 0)
{
lean_dec(v___x_858_);
lean_inc(v_currentMaxHypDepth_744_);
v___y_854_ = v_currentMaxHypDepth_744_;
goto v___jp_853_;
}
else
{
v___y_854_ = v___x_858_;
goto v___jp_853_;
}
v___jp_823_:
{
lean_object* v___x_833_; 
v___x_833_ = l_Lean_Meta_saveState___redArg(v___y_830_, v___y_832_);
if (lean_obj_tag(v___x_833_) == 0)
{
lean_object* v_a_834_; lean_object* v___x_835_; lean_object* v___x_836_; 
v_a_834_ = lean_ctor_get(v___x_833_, 0);
lean_inc(v_a_834_);
lean_dec_ref_known(v___x_833_, 1);
v___x_835_ = l_Lean_LocalDecl_type(v_val_796_);
lean_inc_ref(v_a_737_);
v___x_836_ = l_Lean_Meta_isExprDefEq(v___x_835_, v_a_737_, v___y_829_, v___y_830_, v___y_831_, v___y_832_);
if (lean_obj_tag(v___x_836_) == 0)
{
lean_object* v_a_837_; uint8_t v___x_838_; 
v_a_837_ = lean_ctor_get(v___x_836_, 0);
lean_inc(v_a_837_);
lean_dec_ref_known(v___x_836_, 1);
v___x_838_ = lean_unbox(v_a_837_);
lean_dec(v_a_837_);
if (v___x_838_ == 0)
{
lean_dec(v___y_824_);
lean_dec(v___x_822_);
v___y_799_ = v___y_832_;
v___y_800_ = v_a_834_;
v___y_801_ = v___y_830_;
goto v___jp_798_;
}
else
{
lean_object* v___x_839_; lean_object* v___x_840_; 
lean_inc(v___x_822_);
v___x_839_ = l_Lean_mkFVar(v___x_822_);
lean_inc(v_mvarId_738_);
v___x_840_ = lp_aesop_Lean_MVarId_assign___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__6___redArg(v_mvarId_738_, v___x_839_, v___y_830_);
if (lean_obj_tag(v___x_840_) == 0)
{
lean_object* v___x_841_; lean_object* v___x_842_; lean_object* v___x_843_; 
lean_dec_ref_known(v___x_840_, 1);
lean_inc_ref(v_currentUsedHyps_739_);
v___x_841_ = lean_array_push(v_currentUsedHyps_739_, v___x_822_);
v___x_842_ = lean_nat_add(v_i_740_, v___x_820_);
lean_inc_ref(v_instMVars_742_);
lean_inc_ref(v_app_741_);
v___x_843_ = lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop(v_app_741_, v_instMVars_742_, v_immediateMVars_743_, v___x_842_, v___y_824_, v___x_841_, v___y_825_, v___y_826_, v___y_827_, v___y_828_, v___y_829_, v___y_830_, v___y_831_, v___y_832_);
v___y_812_ = v___y_832_;
v___y_813_ = v_a_834_;
v___y_814_ = v___y_830_;
v___y_815_ = v___x_843_;
goto v___jp_811_;
}
else
{
lean_dec(v___y_824_);
lean_dec(v___x_822_);
v___y_812_ = v___y_832_;
v___y_813_ = v_a_834_;
v___y_814_ = v___y_830_;
v___y_815_ = v___x_840_;
goto v___jp_811_;
}
}
}
else
{
lean_object* v_a_844_; 
lean_dec(v___y_824_);
lean_dec(v___x_822_);
lean_del_object(v___x_784_);
lean_dec(v_currentMaxHypDepth_744_);
lean_dec_ref(v_instMVars_742_);
lean_dec_ref(v_app_741_);
lean_dec_ref(v_currentUsedHyps_739_);
lean_dec(v_mvarId_738_);
lean_dec_ref(v_a_737_);
v_a_844_ = lean_ctor_get(v___x_836_, 0);
lean_inc(v_a_844_);
lean_dec_ref_known(v___x_836_, 1);
v___y_759_ = v___y_832_;
v___y_760_ = v_a_834_;
v___y_761_ = v___y_830_;
v_a_762_ = v_a_844_;
goto v___jp_758_;
}
}
else
{
lean_object* v_a_845_; lean_object* v___x_847_; uint8_t v_isShared_848_; uint8_t v_isSharedCheck_852_; 
lean_dec(v___y_824_);
lean_dec(v___x_822_);
lean_del_object(v___x_784_);
lean_dec(v_currentMaxHypDepth_744_);
lean_dec_ref(v_instMVars_742_);
lean_dec_ref(v_app_741_);
lean_dec_ref(v_currentUsedHyps_739_);
lean_dec(v_mvarId_738_);
lean_dec_ref(v_a_737_);
v_a_845_ = lean_ctor_get(v___x_833_, 0);
v_isSharedCheck_852_ = !lean_is_exclusive(v___x_833_);
if (v_isSharedCheck_852_ == 0)
{
v___x_847_ = v___x_833_;
v_isShared_848_ = v_isSharedCheck_852_;
goto v_resetjp_846_;
}
else
{
lean_inc(v_a_845_);
lean_dec(v___x_833_);
v___x_847_ = lean_box(0);
v_isShared_848_ = v_isSharedCheck_852_;
goto v_resetjp_846_;
}
v_resetjp_846_:
{
lean_object* v___x_850_; 
if (v_isShared_848_ == 0)
{
v___x_850_ = v___x_847_;
goto v_reusejp_849_;
}
else
{
lean_object* v_reuseFailAlloc_851_; 
v_reuseFailAlloc_851_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_851_, 0, v_a_845_);
v___x_850_ = v_reuseFailAlloc_851_;
goto v_reusejp_849_;
}
v_reusejp_849_:
{
return v___x_850_;
}
}
}
}
v___jp_853_:
{
if (lean_obj_tag(v_maxDepth_x3f_818_) == 1)
{
lean_object* v_val_855_; lean_object* v___x_856_; uint8_t v___x_857_; 
v_val_855_ = lean_ctor_get(v_maxDepth_x3f_818_, 0);
v___x_856_ = lean_nat_add(v___y_854_, v___x_820_);
v___x_857_ = lean_nat_dec_lt(v_val_855_, v___x_856_);
lean_dec(v___x_856_);
if (v___x_857_ == 0)
{
v___y_824_ = v___y_854_;
v___y_825_ = v___y_749_;
v___y_826_ = v___y_750_;
v___y_827_ = v___y_751_;
v___y_828_ = v___y_752_;
v___y_829_ = v___y_753_;
v___y_830_ = v___y_754_;
v___y_831_ = v___y_755_;
v___y_832_ = v___y_756_;
goto v___jp_823_;
}
else
{
lean_dec(v___y_854_);
lean_dec(v___x_822_);
v_a_788_ = v___x_797_;
goto v___jp_787_;
}
}
else
{
v___y_824_ = v___y_854_;
v___y_825_ = v___y_749_;
v___y_826_ = v___y_750_;
v___y_827_ = v___y_751_;
v___y_828_ = v___y_752_;
v___y_829_ = v___y_753_;
v___y_830_ = v___y_754_;
v___y_831_ = v___y_755_;
v___y_832_ = v___y_756_;
goto v___jp_823_;
}
}
}
else
{
v_a_788_ = v___x_797_;
goto v___jp_787_;
}
v___jp_798_:
{
lean_object* v___x_802_; 
v___x_802_ = l_Lean_Meta_SavedState_restore___redArg(v___y_800_, v___y_801_, v___y_799_);
lean_dec_ref(v___y_800_);
if (lean_obj_tag(v___x_802_) == 0)
{
lean_dec_ref_known(v___x_802_, 1);
v_a_788_ = v___x_797_;
goto v___jp_787_;
}
else
{
lean_object* v_a_803_; lean_object* v___x_805_; uint8_t v_isShared_806_; uint8_t v_isSharedCheck_810_; 
lean_del_object(v___x_784_);
lean_dec(v_currentMaxHypDepth_744_);
lean_dec_ref(v_instMVars_742_);
lean_dec_ref(v_app_741_);
lean_dec_ref(v_currentUsedHyps_739_);
lean_dec(v_mvarId_738_);
lean_dec_ref(v_a_737_);
v_a_803_ = lean_ctor_get(v___x_802_, 0);
v_isSharedCheck_810_ = !lean_is_exclusive(v___x_802_);
if (v_isSharedCheck_810_ == 0)
{
v___x_805_ = v___x_802_;
v_isShared_806_ = v_isSharedCheck_810_;
goto v_resetjp_804_;
}
else
{
lean_inc(v_a_803_);
lean_dec(v___x_802_);
v___x_805_ = lean_box(0);
v_isShared_806_ = v_isSharedCheck_810_;
goto v_resetjp_804_;
}
v_resetjp_804_:
{
lean_object* v___x_808_; 
if (v_isShared_806_ == 0)
{
v___x_808_ = v___x_805_;
goto v_reusejp_807_;
}
else
{
lean_object* v_reuseFailAlloc_809_; 
v_reuseFailAlloc_809_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_809_, 0, v_a_803_);
v___x_808_ = v_reuseFailAlloc_809_;
goto v_reusejp_807_;
}
v_reusejp_807_:
{
return v___x_808_;
}
}
}
}
v___jp_811_:
{
if (lean_obj_tag(v___y_815_) == 0)
{
lean_dec_ref_known(v___y_815_, 1);
v___y_799_ = v___y_812_;
v___y_800_ = v___y_813_;
v___y_801_ = v___y_814_;
goto v___jp_798_;
}
else
{
lean_object* v_a_816_; 
lean_del_object(v___x_784_);
lean_dec(v_currentMaxHypDepth_744_);
lean_dec_ref(v_instMVars_742_);
lean_dec_ref(v_app_741_);
lean_dec_ref(v_currentUsedHyps_739_);
lean_dec(v_mvarId_738_);
lean_dec_ref(v_a_737_);
v_a_816_ = lean_ctor_get(v___y_815_, 0);
lean_inc(v_a_816_);
lean_dec_ref_known(v___y_815_, 1);
v___y_759_ = v___y_812_;
v___y_760_ = v___y_813_;
v___y_761_ = v___y_814_;
v_a_762_ = v_a_816_;
goto v___jp_758_;
}
}
}
v___jp_787_:
{
lean_object* v___x_790_; 
if (v_isShared_785_ == 0)
{
lean_ctor_set(v___x_784_, 1, v_a_788_);
lean_ctor_set(v___x_784_, 0, v___x_786_);
v___x_790_ = v___x_784_;
goto v_reusejp_789_;
}
else
{
lean_object* v_reuseFailAlloc_794_; 
v_reuseFailAlloc_794_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_794_, 0, v___x_786_);
lean_ctor_set(v_reuseFailAlloc_794_, 1, v_a_788_);
v___x_790_ = v_reuseFailAlloc_794_;
goto v_reusejp_789_;
}
v_reusejp_789_:
{
size_t v___x_791_; size_t v___x_792_; 
v___x_791_ = ((size_t)1ULL);
v___x_792_ = lean_usize_add(v_i_747_, v___x_791_);
v_i_747_ = v___x_792_;
v_b_748_ = v___x_790_;
goto _start;
}
}
}
}
v___jp_758_:
{
lean_object* v___x_763_; 
v___x_763_ = l_Lean_Meta_SavedState_restore___redArg(v___y_760_, v___y_761_, v___y_759_);
lean_dec_ref(v___y_760_);
if (lean_obj_tag(v___x_763_) == 0)
{
lean_object* v___x_765_; uint8_t v_isShared_766_; uint8_t v_isSharedCheck_770_; 
v_isSharedCheck_770_ = !lean_is_exclusive(v___x_763_);
if (v_isSharedCheck_770_ == 0)
{
lean_object* v_unused_771_; 
v_unused_771_ = lean_ctor_get(v___x_763_, 0);
lean_dec(v_unused_771_);
v___x_765_ = v___x_763_;
v_isShared_766_ = v_isSharedCheck_770_;
goto v_resetjp_764_;
}
else
{
lean_dec(v___x_763_);
v___x_765_ = lean_box(0);
v_isShared_766_ = v_isSharedCheck_770_;
goto v_resetjp_764_;
}
v_resetjp_764_:
{
lean_object* v___x_768_; 
if (v_isShared_766_ == 0)
{
lean_ctor_set_tag(v___x_765_, 1);
lean_ctor_set(v___x_765_, 0, v_a_762_);
v___x_768_ = v___x_765_;
goto v_reusejp_767_;
}
else
{
lean_object* v_reuseFailAlloc_769_; 
v_reuseFailAlloc_769_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_769_, 0, v_a_762_);
v___x_768_ = v_reuseFailAlloc_769_;
goto v_reusejp_767_;
}
v_reusejp_767_:
{
return v___x_768_;
}
}
}
else
{
lean_object* v_a_772_; lean_object* v___x_774_; uint8_t v_isShared_775_; uint8_t v_isSharedCheck_779_; 
lean_dec_ref(v_a_762_);
v_a_772_ = lean_ctor_get(v___x_763_, 0);
v_isSharedCheck_779_ = !lean_is_exclusive(v___x_763_);
if (v_isSharedCheck_779_ == 0)
{
v___x_774_ = v___x_763_;
v_isShared_775_ = v_isSharedCheck_779_;
goto v_resetjp_773_;
}
else
{
lean_inc(v_a_772_);
lean_dec(v___x_763_);
v___x_774_ = lean_box(0);
v_isShared_775_ = v_isSharedCheck_779_;
goto v_resetjp_773_;
}
v_resetjp_773_:
{
lean_object* v___x_777_; 
if (v_isShared_775_ == 0)
{
v___x_777_ = v___x_774_;
goto v_reusejp_776_;
}
else
{
lean_object* v_reuseFailAlloc_778_; 
v_reuseFailAlloc_778_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_778_, 0, v_a_772_);
v___x_777_ = v_reuseFailAlloc_778_;
goto v_reusejp_776_;
}
v_reusejp_776_:
{
return v___x_777_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__7_spec__15(lean_object* v_a_862_, lean_object* v_mvarId_863_, lean_object* v_currentUsedHyps_864_, lean_object* v_i_865_, lean_object* v_app_866_, lean_object* v_instMVars_867_, lean_object* v_immediateMVars_868_, lean_object* v_currentMaxHypDepth_869_, lean_object* v_as_870_, size_t v_sz_871_, size_t v_i_872_, lean_object* v_b_873_, lean_object* v___y_874_, lean_object* v___y_875_, lean_object* v___y_876_, lean_object* v___y_877_, lean_object* v___y_878_, lean_object* v___y_879_, lean_object* v___y_880_, lean_object* v___y_881_){
_start:
{
lean_object* v___y_884_; lean_object* v___y_885_; lean_object* v___y_886_; lean_object* v_a_887_; uint8_t v___x_905_; 
v___x_905_ = lean_usize_dec_lt(v_i_872_, v_sz_871_);
if (v___x_905_ == 0)
{
lean_object* v___x_906_; 
lean_dec(v_currentMaxHypDepth_869_);
lean_dec_ref(v_instMVars_867_);
lean_dec_ref(v_app_866_);
lean_dec_ref(v_currentUsedHyps_864_);
lean_dec(v_mvarId_863_);
lean_dec_ref(v_a_862_);
v___x_906_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_906_, 0, v_b_873_);
return v___x_906_;
}
else
{
lean_object* v_snd_907_; lean_object* v___x_909_; uint8_t v_isShared_910_; uint8_t v_isSharedCheck_985_; 
v_snd_907_ = lean_ctor_get(v_b_873_, 1);
v_isSharedCheck_985_ = !lean_is_exclusive(v_b_873_);
if (v_isSharedCheck_985_ == 0)
{
lean_object* v_unused_986_; 
v_unused_986_ = lean_ctor_get(v_b_873_, 0);
lean_dec(v_unused_986_);
v___x_909_ = v_b_873_;
v_isShared_910_ = v_isSharedCheck_985_;
goto v_resetjp_908_;
}
else
{
lean_inc(v_snd_907_);
lean_dec(v_b_873_);
v___x_909_ = lean_box(0);
v_isShared_910_ = v_isSharedCheck_985_;
goto v_resetjp_908_;
}
v_resetjp_908_:
{
lean_object* v___x_911_; lean_object* v_a_913_; lean_object* v_a_920_; 
v___x_911_ = lean_box(0);
v_a_920_ = lean_array_uget_borrowed(v_as_870_, v_i_872_);
if (lean_obj_tag(v_a_920_) == 0)
{
v_a_913_ = v_snd_907_;
goto v___jp_912_;
}
else
{
lean_object* v_val_921_; lean_object* v___x_922_; lean_object* v___y_924_; lean_object* v___y_925_; lean_object* v___y_926_; lean_object* v___y_937_; lean_object* v___y_938_; lean_object* v___y_939_; lean_object* v___y_940_; uint8_t v___x_942_; 
lean_dec(v_snd_907_);
v_val_921_ = lean_ctor_get(v_a_920_, 0);
v___x_922_ = lean_box(0);
v___x_942_ = l_Lean_LocalDecl_isImplementationDetail(v_val_921_);
if (v___x_942_ == 0)
{
lean_object* v_maxDepth_x3f_943_; lean_object* v_forwardHypData_944_; lean_object* v___x_945_; lean_object* v___x_946_; lean_object* v___x_947_; lean_object* v___y_949_; lean_object* v___y_950_; lean_object* v___y_951_; lean_object* v___y_952_; lean_object* v___y_953_; lean_object* v___y_954_; lean_object* v___y_955_; lean_object* v___y_956_; lean_object* v___y_957_; lean_object* v___y_979_; lean_object* v___x_983_; uint8_t v___x_984_; 
v_maxDepth_x3f_943_ = lean_ctor_get(v___y_874_, 0);
v_forwardHypData_944_ = lean_ctor_get(v___y_874_, 1);
v___x_945_ = lean_unsigned_to_nat(1u);
v___x_946_ = lean_unsigned_to_nat(0u);
v___x_947_ = l_Lean_LocalDecl_fvarId(v_val_921_);
v___x_983_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__4___redArg(v_forwardHypData_944_, v___x_947_, v___x_946_);
v___x_984_ = lean_nat_dec_le(v_currentMaxHypDepth_869_, v___x_983_);
if (v___x_984_ == 0)
{
lean_dec(v___x_983_);
lean_inc(v_currentMaxHypDepth_869_);
v___y_979_ = v_currentMaxHypDepth_869_;
goto v___jp_978_;
}
else
{
v___y_979_ = v___x_983_;
goto v___jp_978_;
}
v___jp_948_:
{
lean_object* v___x_958_; 
v___x_958_ = l_Lean_Meta_saveState___redArg(v___y_955_, v___y_957_);
if (lean_obj_tag(v___x_958_) == 0)
{
lean_object* v_a_959_; lean_object* v___x_960_; lean_object* v___x_961_; 
v_a_959_ = lean_ctor_get(v___x_958_, 0);
lean_inc(v_a_959_);
lean_dec_ref_known(v___x_958_, 1);
v___x_960_ = l_Lean_LocalDecl_type(v_val_921_);
lean_inc_ref(v_a_862_);
v___x_961_ = l_Lean_Meta_isExprDefEq(v___x_960_, v_a_862_, v___y_954_, v___y_955_, v___y_956_, v___y_957_);
if (lean_obj_tag(v___x_961_) == 0)
{
lean_object* v_a_962_; uint8_t v___x_963_; 
v_a_962_ = lean_ctor_get(v___x_961_, 0);
lean_inc(v_a_962_);
lean_dec_ref_known(v___x_961_, 1);
v___x_963_ = lean_unbox(v_a_962_);
lean_dec(v_a_962_);
if (v___x_963_ == 0)
{
lean_dec(v___y_949_);
lean_dec(v___x_947_);
v___y_924_ = v___y_955_;
v___y_925_ = v_a_959_;
v___y_926_ = v___y_957_;
goto v___jp_923_;
}
else
{
lean_object* v___x_964_; lean_object* v___x_965_; 
lean_inc(v___x_947_);
v___x_964_ = l_Lean_mkFVar(v___x_947_);
lean_inc(v_mvarId_863_);
v___x_965_ = lp_aesop_Lean_MVarId_assign___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__6___redArg(v_mvarId_863_, v___x_964_, v___y_955_);
if (lean_obj_tag(v___x_965_) == 0)
{
lean_object* v___x_966_; lean_object* v___x_967_; lean_object* v___x_968_; 
lean_dec_ref_known(v___x_965_, 1);
lean_inc_ref(v_currentUsedHyps_864_);
v___x_966_ = lean_array_push(v_currentUsedHyps_864_, v___x_947_);
v___x_967_ = lean_nat_add(v_i_865_, v___x_945_);
lean_inc_ref(v_instMVars_867_);
lean_inc_ref(v_app_866_);
v___x_968_ = lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop(v_app_866_, v_instMVars_867_, v_immediateMVars_868_, v___x_967_, v___y_949_, v___x_966_, v___y_950_, v___y_951_, v___y_952_, v___y_953_, v___y_954_, v___y_955_, v___y_956_, v___y_957_);
v___y_937_ = v___y_955_;
v___y_938_ = v_a_959_;
v___y_939_ = v___y_957_;
v___y_940_ = v___x_968_;
goto v___jp_936_;
}
else
{
lean_dec(v___y_949_);
lean_dec(v___x_947_);
v___y_937_ = v___y_955_;
v___y_938_ = v_a_959_;
v___y_939_ = v___y_957_;
v___y_940_ = v___x_965_;
goto v___jp_936_;
}
}
}
else
{
lean_object* v_a_969_; 
lean_dec(v___y_949_);
lean_dec(v___x_947_);
lean_del_object(v___x_909_);
lean_dec(v_currentMaxHypDepth_869_);
lean_dec_ref(v_instMVars_867_);
lean_dec_ref(v_app_866_);
lean_dec_ref(v_currentUsedHyps_864_);
lean_dec(v_mvarId_863_);
lean_dec_ref(v_a_862_);
v_a_969_ = lean_ctor_get(v___x_961_, 0);
lean_inc(v_a_969_);
lean_dec_ref_known(v___x_961_, 1);
v___y_884_ = v___y_955_;
v___y_885_ = v_a_959_;
v___y_886_ = v___y_957_;
v_a_887_ = v_a_969_;
goto v___jp_883_;
}
}
else
{
lean_object* v_a_970_; lean_object* v___x_972_; uint8_t v_isShared_973_; uint8_t v_isSharedCheck_977_; 
lean_dec(v___y_949_);
lean_dec(v___x_947_);
lean_del_object(v___x_909_);
lean_dec(v_currentMaxHypDepth_869_);
lean_dec_ref(v_instMVars_867_);
lean_dec_ref(v_app_866_);
lean_dec_ref(v_currentUsedHyps_864_);
lean_dec(v_mvarId_863_);
lean_dec_ref(v_a_862_);
v_a_970_ = lean_ctor_get(v___x_958_, 0);
v_isSharedCheck_977_ = !lean_is_exclusive(v___x_958_);
if (v_isSharedCheck_977_ == 0)
{
v___x_972_ = v___x_958_;
v_isShared_973_ = v_isSharedCheck_977_;
goto v_resetjp_971_;
}
else
{
lean_inc(v_a_970_);
lean_dec(v___x_958_);
v___x_972_ = lean_box(0);
v_isShared_973_ = v_isSharedCheck_977_;
goto v_resetjp_971_;
}
v_resetjp_971_:
{
lean_object* v___x_975_; 
if (v_isShared_973_ == 0)
{
v___x_975_ = v___x_972_;
goto v_reusejp_974_;
}
else
{
lean_object* v_reuseFailAlloc_976_; 
v_reuseFailAlloc_976_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_976_, 0, v_a_970_);
v___x_975_ = v_reuseFailAlloc_976_;
goto v_reusejp_974_;
}
v_reusejp_974_:
{
return v___x_975_;
}
}
}
}
v___jp_978_:
{
if (lean_obj_tag(v_maxDepth_x3f_943_) == 1)
{
lean_object* v_val_980_; lean_object* v___x_981_; uint8_t v___x_982_; 
v_val_980_ = lean_ctor_get(v_maxDepth_x3f_943_, 0);
v___x_981_ = lean_nat_add(v___y_979_, v___x_945_);
v___x_982_ = lean_nat_dec_lt(v_val_980_, v___x_981_);
lean_dec(v___x_981_);
if (v___x_982_ == 0)
{
v___y_949_ = v___y_979_;
v___y_950_ = v___y_874_;
v___y_951_ = v___y_875_;
v___y_952_ = v___y_876_;
v___y_953_ = v___y_877_;
v___y_954_ = v___y_878_;
v___y_955_ = v___y_879_;
v___y_956_ = v___y_880_;
v___y_957_ = v___y_881_;
goto v___jp_948_;
}
else
{
lean_dec(v___y_979_);
lean_dec(v___x_947_);
v_a_913_ = v___x_922_;
goto v___jp_912_;
}
}
else
{
v___y_949_ = v___y_979_;
v___y_950_ = v___y_874_;
v___y_951_ = v___y_875_;
v___y_952_ = v___y_876_;
v___y_953_ = v___y_877_;
v___y_954_ = v___y_878_;
v___y_955_ = v___y_879_;
v___y_956_ = v___y_880_;
v___y_957_ = v___y_881_;
goto v___jp_948_;
}
}
}
else
{
v_a_913_ = v___x_922_;
goto v___jp_912_;
}
v___jp_923_:
{
lean_object* v___x_927_; 
v___x_927_ = l_Lean_Meta_SavedState_restore___redArg(v___y_925_, v___y_924_, v___y_926_);
lean_dec_ref(v___y_925_);
if (lean_obj_tag(v___x_927_) == 0)
{
lean_dec_ref_known(v___x_927_, 1);
v_a_913_ = v___x_922_;
goto v___jp_912_;
}
else
{
lean_object* v_a_928_; lean_object* v___x_930_; uint8_t v_isShared_931_; uint8_t v_isSharedCheck_935_; 
lean_del_object(v___x_909_);
lean_dec(v_currentMaxHypDepth_869_);
lean_dec_ref(v_instMVars_867_);
lean_dec_ref(v_app_866_);
lean_dec_ref(v_currentUsedHyps_864_);
lean_dec(v_mvarId_863_);
lean_dec_ref(v_a_862_);
v_a_928_ = lean_ctor_get(v___x_927_, 0);
v_isSharedCheck_935_ = !lean_is_exclusive(v___x_927_);
if (v_isSharedCheck_935_ == 0)
{
v___x_930_ = v___x_927_;
v_isShared_931_ = v_isSharedCheck_935_;
goto v_resetjp_929_;
}
else
{
lean_inc(v_a_928_);
lean_dec(v___x_927_);
v___x_930_ = lean_box(0);
v_isShared_931_ = v_isSharedCheck_935_;
goto v_resetjp_929_;
}
v_resetjp_929_:
{
lean_object* v___x_933_; 
if (v_isShared_931_ == 0)
{
v___x_933_ = v___x_930_;
goto v_reusejp_932_;
}
else
{
lean_object* v_reuseFailAlloc_934_; 
v_reuseFailAlloc_934_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_934_, 0, v_a_928_);
v___x_933_ = v_reuseFailAlloc_934_;
goto v_reusejp_932_;
}
v_reusejp_932_:
{
return v___x_933_;
}
}
}
}
v___jp_936_:
{
if (lean_obj_tag(v___y_940_) == 0)
{
lean_dec_ref_known(v___y_940_, 1);
v___y_924_ = v___y_937_;
v___y_925_ = v___y_938_;
v___y_926_ = v___y_939_;
goto v___jp_923_;
}
else
{
lean_object* v_a_941_; 
lean_del_object(v___x_909_);
lean_dec(v_currentMaxHypDepth_869_);
lean_dec_ref(v_instMVars_867_);
lean_dec_ref(v_app_866_);
lean_dec_ref(v_currentUsedHyps_864_);
lean_dec(v_mvarId_863_);
lean_dec_ref(v_a_862_);
v_a_941_ = lean_ctor_get(v___y_940_, 0);
lean_inc(v_a_941_);
lean_dec_ref_known(v___y_940_, 1);
v___y_884_ = v___y_937_;
v___y_885_ = v___y_938_;
v___y_886_ = v___y_939_;
v_a_887_ = v_a_941_;
goto v___jp_883_;
}
}
}
v___jp_912_:
{
lean_object* v___x_915_; 
if (v_isShared_910_ == 0)
{
lean_ctor_set(v___x_909_, 1, v_a_913_);
lean_ctor_set(v___x_909_, 0, v___x_911_);
v___x_915_ = v___x_909_;
goto v_reusejp_914_;
}
else
{
lean_object* v_reuseFailAlloc_919_; 
v_reuseFailAlloc_919_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_919_, 0, v___x_911_);
lean_ctor_set(v_reuseFailAlloc_919_, 1, v_a_913_);
v___x_915_ = v_reuseFailAlloc_919_;
goto v_reusejp_914_;
}
v_reusejp_914_:
{
size_t v___x_916_; size_t v___x_917_; lean_object* v___x_918_; 
v___x_916_ = ((size_t)1ULL);
v___x_917_ = lean_usize_add(v_i_872_, v___x_916_);
v___x_918_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__7_spec__15_spec__22(v_a_862_, v_mvarId_863_, v_currentUsedHyps_864_, v_i_865_, v_app_866_, v_instMVars_867_, v_immediateMVars_868_, v_currentMaxHypDepth_869_, v_as_870_, v_sz_871_, v___x_917_, v___x_915_, v___y_874_, v___y_875_, v___y_876_, v___y_877_, v___y_878_, v___y_879_, v___y_880_, v___y_881_);
return v___x_918_;
}
}
}
}
v___jp_883_:
{
lean_object* v___x_888_; 
v___x_888_ = l_Lean_Meta_SavedState_restore___redArg(v___y_885_, v___y_884_, v___y_886_);
lean_dec_ref(v___y_885_);
if (lean_obj_tag(v___x_888_) == 0)
{
lean_object* v___x_890_; uint8_t v_isShared_891_; uint8_t v_isSharedCheck_895_; 
v_isSharedCheck_895_ = !lean_is_exclusive(v___x_888_);
if (v_isSharedCheck_895_ == 0)
{
lean_object* v_unused_896_; 
v_unused_896_ = lean_ctor_get(v___x_888_, 0);
lean_dec(v_unused_896_);
v___x_890_ = v___x_888_;
v_isShared_891_ = v_isSharedCheck_895_;
goto v_resetjp_889_;
}
else
{
lean_dec(v___x_888_);
v___x_890_ = lean_box(0);
v_isShared_891_ = v_isSharedCheck_895_;
goto v_resetjp_889_;
}
v_resetjp_889_:
{
lean_object* v___x_893_; 
if (v_isShared_891_ == 0)
{
lean_ctor_set_tag(v___x_890_, 1);
lean_ctor_set(v___x_890_, 0, v_a_887_);
v___x_893_ = v___x_890_;
goto v_reusejp_892_;
}
else
{
lean_object* v_reuseFailAlloc_894_; 
v_reuseFailAlloc_894_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_894_, 0, v_a_887_);
v___x_893_ = v_reuseFailAlloc_894_;
goto v_reusejp_892_;
}
v_reusejp_892_:
{
return v___x_893_;
}
}
}
else
{
lean_object* v_a_897_; lean_object* v___x_899_; uint8_t v_isShared_900_; uint8_t v_isSharedCheck_904_; 
lean_dec_ref(v_a_887_);
v_a_897_ = lean_ctor_get(v___x_888_, 0);
v_isSharedCheck_904_ = !lean_is_exclusive(v___x_888_);
if (v_isSharedCheck_904_ == 0)
{
v___x_899_ = v___x_888_;
v_isShared_900_ = v_isSharedCheck_904_;
goto v_resetjp_898_;
}
else
{
lean_inc(v_a_897_);
lean_dec(v___x_888_);
v___x_899_ = lean_box(0);
v_isShared_900_ = v_isSharedCheck_904_;
goto v_resetjp_898_;
}
v_resetjp_898_:
{
lean_object* v___x_902_; 
if (v_isShared_900_ == 0)
{
v___x_902_ = v___x_899_;
goto v_reusejp_901_;
}
else
{
lean_object* v_reuseFailAlloc_903_; 
v_reuseFailAlloc_903_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_903_, 0, v_a_897_);
v___x_902_ = v_reuseFailAlloc_903_;
goto v_reusejp_901_;
}
v_reusejp_901_:
{
return v___x_902_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forIn___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__7(lean_object* v_a_987_, lean_object* v_mvarId_988_, lean_object* v_currentUsedHyps_989_, lean_object* v_i_990_, lean_object* v_app_991_, lean_object* v_instMVars_992_, lean_object* v_immediateMVars_993_, lean_object* v_currentMaxHypDepth_994_, lean_object* v_t_995_, lean_object* v_init_996_, lean_object* v___y_997_, lean_object* v___y_998_, lean_object* v___y_999_, lean_object* v___y_1000_, lean_object* v___y_1001_, lean_object* v___y_1002_, lean_object* v___y_1003_, lean_object* v___y_1004_){
_start:
{
lean_object* v_root_1006_; lean_object* v_tail_1007_; lean_object* v___x_1008_; 
v_root_1006_ = lean_ctor_get(v_t_995_, 0);
v_tail_1007_ = lean_ctor_get(v_t_995_, 1);
lean_inc(v_currentMaxHypDepth_994_);
lean_inc_ref(v_instMVars_992_);
lean_inc_ref(v_app_991_);
lean_inc_ref(v_currentUsedHyps_989_);
lean_inc(v_mvarId_988_);
lean_inc_ref(v_a_987_);
v___x_1008_ = lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__7_spec__14(v_init_996_, v_a_987_, v_mvarId_988_, v_currentUsedHyps_989_, v_i_990_, v_app_991_, v_instMVars_992_, v_immediateMVars_993_, v_currentMaxHypDepth_994_, v_root_1006_, v_init_996_, v___y_997_, v___y_998_, v___y_999_, v___y_1000_, v___y_1001_, v___y_1002_, v___y_1003_, v___y_1004_);
if (lean_obj_tag(v___x_1008_) == 0)
{
lean_object* v_a_1009_; lean_object* v___x_1011_; uint8_t v_isShared_1012_; uint8_t v_isSharedCheck_1045_; 
v_a_1009_ = lean_ctor_get(v___x_1008_, 0);
v_isSharedCheck_1045_ = !lean_is_exclusive(v___x_1008_);
if (v_isSharedCheck_1045_ == 0)
{
v___x_1011_ = v___x_1008_;
v_isShared_1012_ = v_isSharedCheck_1045_;
goto v_resetjp_1010_;
}
else
{
lean_inc(v_a_1009_);
lean_dec(v___x_1008_);
v___x_1011_ = lean_box(0);
v_isShared_1012_ = v_isSharedCheck_1045_;
goto v_resetjp_1010_;
}
v_resetjp_1010_:
{
if (lean_obj_tag(v_a_1009_) == 0)
{
lean_object* v_a_1013_; lean_object* v___x_1015_; 
lean_dec(v_currentMaxHypDepth_994_);
lean_dec_ref(v_instMVars_992_);
lean_dec_ref(v_app_991_);
lean_dec_ref(v_currentUsedHyps_989_);
lean_dec(v_mvarId_988_);
lean_dec_ref(v_a_987_);
v_a_1013_ = lean_ctor_get(v_a_1009_, 0);
lean_inc(v_a_1013_);
lean_dec_ref_known(v_a_1009_, 1);
if (v_isShared_1012_ == 0)
{
lean_ctor_set(v___x_1011_, 0, v_a_1013_);
v___x_1015_ = v___x_1011_;
goto v_reusejp_1014_;
}
else
{
lean_object* v_reuseFailAlloc_1016_; 
v_reuseFailAlloc_1016_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1016_, 0, v_a_1013_);
v___x_1015_ = v_reuseFailAlloc_1016_;
goto v_reusejp_1014_;
}
v_reusejp_1014_:
{
return v___x_1015_;
}
}
else
{
lean_object* v_a_1017_; lean_object* v___x_1018_; lean_object* v___x_1019_; size_t v_sz_1020_; size_t v___x_1021_; lean_object* v___x_1022_; 
lean_del_object(v___x_1011_);
v_a_1017_ = lean_ctor_get(v_a_1009_, 0);
lean_inc(v_a_1017_);
lean_dec_ref_known(v_a_1009_, 1);
v___x_1018_ = lean_box(0);
v___x_1019_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1019_, 0, v___x_1018_);
lean_ctor_set(v___x_1019_, 1, v_a_1017_);
v_sz_1020_ = lean_array_size(v_tail_1007_);
v___x_1021_ = ((size_t)0ULL);
v___x_1022_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__7_spec__15(v_a_987_, v_mvarId_988_, v_currentUsedHyps_989_, v_i_990_, v_app_991_, v_instMVars_992_, v_immediateMVars_993_, v_currentMaxHypDepth_994_, v_tail_1007_, v_sz_1020_, v___x_1021_, v___x_1019_, v___y_997_, v___y_998_, v___y_999_, v___y_1000_, v___y_1001_, v___y_1002_, v___y_1003_, v___y_1004_);
if (lean_obj_tag(v___x_1022_) == 0)
{
lean_object* v_a_1023_; lean_object* v___x_1025_; uint8_t v_isShared_1026_; uint8_t v_isSharedCheck_1036_; 
v_a_1023_ = lean_ctor_get(v___x_1022_, 0);
v_isSharedCheck_1036_ = !lean_is_exclusive(v___x_1022_);
if (v_isSharedCheck_1036_ == 0)
{
v___x_1025_ = v___x_1022_;
v_isShared_1026_ = v_isSharedCheck_1036_;
goto v_resetjp_1024_;
}
else
{
lean_inc(v_a_1023_);
lean_dec(v___x_1022_);
v___x_1025_ = lean_box(0);
v_isShared_1026_ = v_isSharedCheck_1036_;
goto v_resetjp_1024_;
}
v_resetjp_1024_:
{
lean_object* v_fst_1027_; 
v_fst_1027_ = lean_ctor_get(v_a_1023_, 0);
if (lean_obj_tag(v_fst_1027_) == 0)
{
lean_object* v_snd_1028_; lean_object* v___x_1030_; 
v_snd_1028_ = lean_ctor_get(v_a_1023_, 1);
lean_inc(v_snd_1028_);
lean_dec(v_a_1023_);
if (v_isShared_1026_ == 0)
{
lean_ctor_set(v___x_1025_, 0, v_snd_1028_);
v___x_1030_ = v___x_1025_;
goto v_reusejp_1029_;
}
else
{
lean_object* v_reuseFailAlloc_1031_; 
v_reuseFailAlloc_1031_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1031_, 0, v_snd_1028_);
v___x_1030_ = v_reuseFailAlloc_1031_;
goto v_reusejp_1029_;
}
v_reusejp_1029_:
{
return v___x_1030_;
}
}
else
{
lean_object* v_val_1032_; lean_object* v___x_1034_; 
lean_inc_ref(v_fst_1027_);
lean_dec(v_a_1023_);
v_val_1032_ = lean_ctor_get(v_fst_1027_, 0);
lean_inc(v_val_1032_);
lean_dec_ref_known(v_fst_1027_, 1);
if (v_isShared_1026_ == 0)
{
lean_ctor_set(v___x_1025_, 0, v_val_1032_);
v___x_1034_ = v___x_1025_;
goto v_reusejp_1033_;
}
else
{
lean_object* v_reuseFailAlloc_1035_; 
v_reuseFailAlloc_1035_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1035_, 0, v_val_1032_);
v___x_1034_ = v_reuseFailAlloc_1035_;
goto v_reusejp_1033_;
}
v_reusejp_1033_:
{
return v___x_1034_;
}
}
}
}
else
{
lean_object* v_a_1037_; lean_object* v___x_1039_; uint8_t v_isShared_1040_; uint8_t v_isSharedCheck_1044_; 
v_a_1037_ = lean_ctor_get(v___x_1022_, 0);
v_isSharedCheck_1044_ = !lean_is_exclusive(v___x_1022_);
if (v_isSharedCheck_1044_ == 0)
{
v___x_1039_ = v___x_1022_;
v_isShared_1040_ = v_isSharedCheck_1044_;
goto v_resetjp_1038_;
}
else
{
lean_inc(v_a_1037_);
lean_dec(v___x_1022_);
v___x_1039_ = lean_box(0);
v_isShared_1040_ = v_isSharedCheck_1044_;
goto v_resetjp_1038_;
}
v_resetjp_1038_:
{
lean_object* v___x_1042_; 
if (v_isShared_1040_ == 0)
{
v___x_1042_ = v___x_1039_;
goto v_reusejp_1041_;
}
else
{
lean_object* v_reuseFailAlloc_1043_; 
v_reuseFailAlloc_1043_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1043_, 0, v_a_1037_);
v___x_1042_ = v_reuseFailAlloc_1043_;
goto v_reusejp_1041_;
}
v_reusejp_1041_:
{
return v___x_1042_;
}
}
}
}
}
}
else
{
lean_object* v_a_1046_; lean_object* v___x_1048_; uint8_t v_isShared_1049_; uint8_t v_isSharedCheck_1053_; 
lean_dec(v_currentMaxHypDepth_994_);
lean_dec_ref(v_instMVars_992_);
lean_dec_ref(v_app_991_);
lean_dec_ref(v_currentUsedHyps_989_);
lean_dec(v_mvarId_988_);
lean_dec_ref(v_a_987_);
v_a_1046_ = lean_ctor_get(v___x_1008_, 0);
v_isSharedCheck_1053_ = !lean_is_exclusive(v___x_1008_);
if (v_isSharedCheck_1053_ == 0)
{
v___x_1048_ = v___x_1008_;
v_isShared_1049_ = v_isSharedCheck_1053_;
goto v_resetjp_1047_;
}
else
{
lean_inc(v_a_1046_);
lean_dec(v___x_1008_);
v___x_1048_ = lean_box(0);
v_isShared_1049_ = v_isSharedCheck_1053_;
goto v_resetjp_1047_;
}
v_resetjp_1047_:
{
lean_object* v___x_1051_; 
if (v_isShared_1049_ == 0)
{
v___x_1051_ = v___x_1048_;
goto v_reusejp_1050_;
}
else
{
lean_object* v_reuseFailAlloc_1052_; 
v_reuseFailAlloc_1052_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1052_, 0, v_a_1046_);
v___x_1051_ = v_reuseFailAlloc_1052_;
goto v_reusejp_1050_;
}
v_reusejp_1050_:
{
return v___x_1051_;
}
}
}
}
}
static lean_object* _init_lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop___closed__4(void){
_start:
{
lean_object* v___x_1056_; lean_object* v___x_1057_; lean_object* v___x_1058_; 
v___x_1056_ = lean_box(0);
v___x_1057_ = lean_unsigned_to_nat(16u);
v___x_1058_ = lean_mk_array(v___x_1057_, v___x_1056_);
return v___x_1058_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop___closed__5(void){
_start:
{
lean_object* v___x_1059_; lean_object* v___x_1060_; lean_object* v___x_1061_; 
v___x_1059_ = lean_obj_once(&lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop___closed__4, &lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop___closed__4_once, _init_lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop___closed__4);
v___x_1060_ = lean_unsigned_to_nat(0u);
v___x_1061_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1061_, 0, v___x_1060_);
lean_ctor_set(v___x_1061_, 1, v___x_1059_);
return v___x_1061_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop___closed__7(void){
_start:
{
lean_object* v___x_1062_; lean_object* v___x_1063_; lean_object* v___x_1064_; lean_object* v___x_1065_; 
v___x_1062_ = ((lean_object*)(lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop___closed__6));
v___x_1063_ = lean_box(1);
v___x_1064_ = lean_obj_once(&lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop___closed__5, &lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop___closed__5_once, _init_lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop___closed__5);
v___x_1065_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1065_, 0, v___x_1064_);
lean_ctor_set(v___x_1065_, 1, v___x_1063_);
lean_ctor_set(v___x_1065_, 2, v___x_1062_);
return v___x_1065_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop(lean_object* v_app_1066_, lean_object* v_instMVars_1067_, lean_object* v_immediateMVars_1068_, lean_object* v_i_1069_, lean_object* v_currentMaxHypDepth_1070_, lean_object* v_currentUsedHyps_1071_, lean_object* v_a_1072_, lean_object* v_a_1073_, lean_object* v_a_1074_, lean_object* v_a_1075_, lean_object* v_a_1076_, lean_object* v_a_1077_, lean_object* v_a_1078_, lean_object* v_a_1079_){
_start:
{
lean_object* v___y_1145_; lean_object* v___y_1164_; uint8_t v___y_1165_; lean_object* v___x_1200_; lean_object* v___x_1201_; uint8_t v___x_1202_; 
v___x_1200_ = lean_unsigned_to_nat(0u);
v___x_1201_ = lean_array_get_size(v_immediateMVars_1068_);
v___x_1202_ = lean_nat_dec_lt(v___x_1200_, v___x_1201_);
if (v___x_1202_ == 0)
{
lean_dec(v_i_1069_);
goto v___jp_1185_;
}
else
{
uint8_t v___x_1203_; 
v___x_1203_ = lean_nat_dec_lt(v_i_1069_, v___x_1201_);
if (v___x_1203_ == 0)
{
lean_dec(v_i_1069_);
goto v___jp_1185_;
}
else
{
lean_object* v___x_1204_; lean_object* v___x_1205_; lean_object* v___x_1206_; lean_object* v_mvarId_1207_; lean_object* v___x_1208_; 
v___x_1204_ = lean_unsigned_to_nat(1u);
v___x_1205_ = lean_nat_sub(v___x_1201_, v___x_1204_);
v___x_1206_ = lean_nat_sub(v___x_1205_, v_i_1069_);
lean_dec(v___x_1205_);
v_mvarId_1207_ = lean_array_fget_borrowed(v_immediateMVars_1068_, v___x_1206_);
lean_dec(v___x_1206_);
v___x_1208_ = lp_aesop_Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__5___redArg(v_mvarId_1207_, v_a_1077_);
if (lean_obj_tag(v___x_1208_) == 0)
{
lean_object* v_a_1209_; uint8_t v___x_1210_; 
v_a_1209_ = lean_ctor_get(v___x_1208_, 0);
lean_inc(v_a_1209_);
lean_dec_ref_known(v___x_1208_, 1);
v___x_1210_ = lean_unbox(v_a_1209_);
lean_dec(v_a_1209_);
if (v___x_1210_ == 0)
{
lean_object* v___x_1211_; 
lean_inc(v_mvarId_1207_);
v___x_1211_ = l_Lean_MVarId_getType(v_mvarId_1207_, v_a_1076_, v_a_1077_, v_a_1078_, v_a_1079_);
if (lean_obj_tag(v___x_1211_) == 0)
{
lean_object* v_lctx_1212_; lean_object* v_a_1213_; lean_object* v_decls_1214_; lean_object* v___x_1215_; lean_object* v___x_1216_; 
v_lctx_1212_ = lean_ctor_get(v_a_1076_, 2);
v_a_1213_ = lean_ctor_get(v___x_1211_, 0);
lean_inc(v_a_1213_);
lean_dec_ref_known(v___x_1211_, 1);
v_decls_1214_ = lean_ctor_get(v_lctx_1212_, 1);
v___x_1215_ = lean_box(0);
lean_inc(v_mvarId_1207_);
v___x_1216_ = lp_aesop_Lean_PersistentArray_forIn___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__7(v_a_1213_, v_mvarId_1207_, v_currentUsedHyps_1071_, v_i_1069_, v_app_1066_, v_instMVars_1067_, v_immediateMVars_1068_, v_currentMaxHypDepth_1070_, v_decls_1214_, v___x_1215_, v_a_1072_, v_a_1073_, v_a_1074_, v_a_1075_, v_a_1076_, v_a_1077_, v_a_1078_, v_a_1079_);
lean_dec(v_i_1069_);
if (lean_obj_tag(v___x_1216_) == 0)
{
lean_object* v___x_1218_; uint8_t v_isShared_1219_; uint8_t v_isSharedCheck_1223_; 
v_isSharedCheck_1223_ = !lean_is_exclusive(v___x_1216_);
if (v_isSharedCheck_1223_ == 0)
{
lean_object* v_unused_1224_; 
v_unused_1224_ = lean_ctor_get(v___x_1216_, 0);
lean_dec(v_unused_1224_);
v___x_1218_ = v___x_1216_;
v_isShared_1219_ = v_isSharedCheck_1223_;
goto v_resetjp_1217_;
}
else
{
lean_dec(v___x_1216_);
v___x_1218_ = lean_box(0);
v_isShared_1219_ = v_isSharedCheck_1223_;
goto v_resetjp_1217_;
}
v_resetjp_1217_:
{
lean_object* v___x_1221_; 
if (v_isShared_1219_ == 0)
{
lean_ctor_set(v___x_1218_, 0, v___x_1215_);
v___x_1221_ = v___x_1218_;
goto v_reusejp_1220_;
}
else
{
lean_object* v_reuseFailAlloc_1222_; 
v_reuseFailAlloc_1222_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1222_, 0, v___x_1215_);
v___x_1221_ = v_reuseFailAlloc_1222_;
goto v_reusejp_1220_;
}
v_reusejp_1220_:
{
return v___x_1221_;
}
}
}
else
{
return v___x_1216_;
}
}
else
{
lean_object* v_a_1225_; lean_object* v___x_1227_; uint8_t v_isShared_1228_; uint8_t v_isSharedCheck_1232_; 
lean_dec_ref(v_currentUsedHyps_1071_);
lean_dec(v_currentMaxHypDepth_1070_);
lean_dec(v_i_1069_);
lean_dec_ref(v_instMVars_1067_);
lean_dec_ref(v_app_1066_);
v_a_1225_ = lean_ctor_get(v___x_1211_, 0);
v_isSharedCheck_1232_ = !lean_is_exclusive(v___x_1211_);
if (v_isSharedCheck_1232_ == 0)
{
v___x_1227_ = v___x_1211_;
v_isShared_1228_ = v_isSharedCheck_1232_;
goto v_resetjp_1226_;
}
else
{
lean_inc(v_a_1225_);
lean_dec(v___x_1211_);
v___x_1227_ = lean_box(0);
v_isShared_1228_ = v_isSharedCheck_1232_;
goto v_resetjp_1226_;
}
v_resetjp_1226_:
{
lean_object* v___x_1230_; 
if (v_isShared_1228_ == 0)
{
v___x_1230_ = v___x_1227_;
goto v_reusejp_1229_;
}
else
{
lean_object* v_reuseFailAlloc_1231_; 
v_reuseFailAlloc_1231_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1231_, 0, v_a_1225_);
v___x_1230_ = v_reuseFailAlloc_1231_;
goto v_reusejp_1229_;
}
v_reusejp_1229_:
{
return v___x_1230_;
}
}
}
}
else
{
lean_object* v___x_1233_; lean_object* v___x_1234_; lean_object* v___x_1235_; lean_object* v___x_1236_; 
v___x_1233_ = lean_obj_once(&lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop___closed__7, &lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop___closed__7_once, _init_lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop___closed__7);
v___x_1234_ = lean_st_mk_ref(v___x_1233_);
lean_inc(v_mvarId_1207_);
v___x_1235_ = l_Lean_Expr_mvar___override(v_mvarId_1207_);
v___x_1236_ = l_Lean_Expr_collectFVars(v___x_1235_, v___x_1234_, v_a_1076_, v_a_1077_, v_a_1078_, v_a_1079_);
if (lean_obj_tag(v___x_1236_) == 0)
{
lean_object* v___x_1237_; lean_object* v_fvarIds_1238_; size_t v_sz_1239_; size_t v___x_1240_; lean_object* v___x_1241_; 
lean_dec_ref_known(v___x_1236_, 1);
v___x_1237_ = lean_st_ref_get(v___x_1234_);
lean_dec(v___x_1234_);
v_fvarIds_1238_ = lean_ctor_get(v___x_1237_, 2);
lean_inc_ref(v_fvarIds_1238_);
lean_dec(v___x_1237_);
v_sz_1239_ = lean_array_size(v_fvarIds_1238_);
v___x_1240_ = ((size_t)0ULL);
v___x_1241_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__8___redArg(v_fvarIds_1238_, v_sz_1239_, v___x_1240_, v_currentMaxHypDepth_1070_, v_a_1072_);
lean_dec_ref(v_fvarIds_1238_);
if (lean_obj_tag(v___x_1241_) == 0)
{
lean_object* v_a_1242_; lean_object* v___x_1244_; uint8_t v_isShared_1245_; uint8_t v_isSharedCheck_1265_; 
v_a_1242_ = lean_ctor_get(v___x_1241_, 0);
v_isSharedCheck_1265_ = !lean_is_exclusive(v___x_1241_);
if (v_isSharedCheck_1265_ == 0)
{
v___x_1244_ = v___x_1241_;
v_isShared_1245_ = v_isSharedCheck_1265_;
goto v_resetjp_1243_;
}
else
{
lean_inc(v_a_1242_);
lean_dec(v___x_1241_);
v___x_1244_ = lean_box(0);
v_isShared_1245_ = v_isSharedCheck_1265_;
goto v_resetjp_1243_;
}
v_resetjp_1243_:
{
lean_object* v___y_1247_; lean_object* v___y_1248_; lean_object* v___y_1249_; lean_object* v___y_1250_; lean_object* v___y_1251_; lean_object* v___y_1252_; lean_object* v___y_1253_; lean_object* v___y_1254_; lean_object* v_maxDepth_x3f_1257_; 
v_maxDepth_x3f_1257_ = lean_ctor_get(v_a_1072_, 0);
if (lean_obj_tag(v_maxDepth_x3f_1257_) == 1)
{
lean_object* v_val_1258_; lean_object* v___x_1259_; uint8_t v___x_1260_; 
v_val_1258_ = lean_ctor_get(v_maxDepth_x3f_1257_, 0);
v___x_1259_ = lean_nat_add(v_a_1242_, v___x_1204_);
v___x_1260_ = lean_nat_dec_lt(v_val_1258_, v___x_1259_);
lean_dec(v___x_1259_);
if (v___x_1260_ == 0)
{
lean_del_object(v___x_1244_);
v___y_1247_ = v_a_1072_;
v___y_1248_ = v_a_1073_;
v___y_1249_ = v_a_1074_;
v___y_1250_ = v_a_1075_;
v___y_1251_ = v_a_1076_;
v___y_1252_ = v_a_1077_;
v___y_1253_ = v_a_1078_;
v___y_1254_ = v_a_1079_;
goto v___jp_1246_;
}
else
{
lean_object* v___x_1261_; lean_object* v___x_1263_; 
lean_dec(v_a_1242_);
lean_dec_ref(v_currentUsedHyps_1071_);
lean_dec(v_i_1069_);
lean_dec_ref(v_instMVars_1067_);
lean_dec_ref(v_app_1066_);
v___x_1261_ = lean_box(0);
if (v_isShared_1245_ == 0)
{
lean_ctor_set(v___x_1244_, 0, v___x_1261_);
v___x_1263_ = v___x_1244_;
goto v_reusejp_1262_;
}
else
{
lean_object* v_reuseFailAlloc_1264_; 
v_reuseFailAlloc_1264_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1264_, 0, v___x_1261_);
v___x_1263_ = v_reuseFailAlloc_1264_;
goto v_reusejp_1262_;
}
v_reusejp_1262_:
{
return v___x_1263_;
}
}
}
else
{
lean_del_object(v___x_1244_);
v___y_1247_ = v_a_1072_;
v___y_1248_ = v_a_1073_;
v___y_1249_ = v_a_1074_;
v___y_1250_ = v_a_1075_;
v___y_1251_ = v_a_1076_;
v___y_1252_ = v_a_1077_;
v___y_1253_ = v_a_1078_;
v___y_1254_ = v_a_1079_;
goto v___jp_1246_;
}
v___jp_1246_:
{
lean_object* v___x_1255_; 
v___x_1255_ = lean_nat_add(v_i_1069_, v___x_1204_);
lean_dec(v_i_1069_);
v_i_1069_ = v___x_1255_;
v_currentMaxHypDepth_1070_ = v_a_1242_;
v_a_1072_ = v___y_1247_;
v_a_1073_ = v___y_1248_;
v_a_1074_ = v___y_1249_;
v_a_1075_ = v___y_1250_;
v_a_1076_ = v___y_1251_;
v_a_1077_ = v___y_1252_;
v_a_1078_ = v___y_1253_;
v_a_1079_ = v___y_1254_;
goto _start;
}
}
}
else
{
lean_object* v_a_1266_; lean_object* v___x_1268_; uint8_t v_isShared_1269_; uint8_t v_isSharedCheck_1273_; 
lean_dec_ref(v_currentUsedHyps_1071_);
lean_dec(v_i_1069_);
lean_dec_ref(v_instMVars_1067_);
lean_dec_ref(v_app_1066_);
v_a_1266_ = lean_ctor_get(v___x_1241_, 0);
v_isSharedCheck_1273_ = !lean_is_exclusive(v___x_1241_);
if (v_isSharedCheck_1273_ == 0)
{
v___x_1268_ = v___x_1241_;
v_isShared_1269_ = v_isSharedCheck_1273_;
goto v_resetjp_1267_;
}
else
{
lean_inc(v_a_1266_);
lean_dec(v___x_1241_);
v___x_1268_ = lean_box(0);
v_isShared_1269_ = v_isSharedCheck_1273_;
goto v_resetjp_1267_;
}
v_resetjp_1267_:
{
lean_object* v___x_1271_; 
if (v_isShared_1269_ == 0)
{
v___x_1271_ = v___x_1268_;
goto v_reusejp_1270_;
}
else
{
lean_object* v_reuseFailAlloc_1272_; 
v_reuseFailAlloc_1272_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1272_, 0, v_a_1266_);
v___x_1271_ = v_reuseFailAlloc_1272_;
goto v_reusejp_1270_;
}
v_reusejp_1270_:
{
return v___x_1271_;
}
}
}
}
else
{
lean_dec(v___x_1234_);
lean_dec_ref(v_currentUsedHyps_1071_);
lean_dec(v_currentMaxHypDepth_1070_);
lean_dec(v_i_1069_);
lean_dec_ref(v_instMVars_1067_);
lean_dec_ref(v_app_1066_);
return v___x_1236_;
}
}
}
else
{
lean_object* v_a_1274_; lean_object* v___x_1276_; uint8_t v_isShared_1277_; uint8_t v_isSharedCheck_1281_; 
lean_dec_ref(v_currentUsedHyps_1071_);
lean_dec(v_currentMaxHypDepth_1070_);
lean_dec(v_i_1069_);
lean_dec_ref(v_instMVars_1067_);
lean_dec_ref(v_app_1066_);
v_a_1274_ = lean_ctor_get(v___x_1208_, 0);
v_isSharedCheck_1281_ = !lean_is_exclusive(v___x_1208_);
if (v_isSharedCheck_1281_ == 0)
{
v___x_1276_ = v___x_1208_;
v_isShared_1277_ = v_isSharedCheck_1281_;
goto v_resetjp_1275_;
}
else
{
lean_inc(v_a_1274_);
lean_dec(v___x_1208_);
v___x_1276_ = lean_box(0);
v_isShared_1277_ = v_isSharedCheck_1281_;
goto v_resetjp_1275_;
}
v_resetjp_1275_:
{
lean_object* v___x_1279_; 
if (v_isShared_1277_ == 0)
{
v___x_1279_ = v___x_1276_;
goto v_reusejp_1278_;
}
else
{
lean_object* v_reuseFailAlloc_1280_; 
v_reuseFailAlloc_1280_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1280_, 0, v_a_1274_);
v___x_1279_ = v_reuseFailAlloc_1280_;
goto v_reusejp_1278_;
}
v_reusejp_1278_:
{
return v___x_1279_;
}
}
}
}
}
v___jp_1081_:
{
uint8_t v___x_1082_; lean_object* v___x_1083_; 
v___x_1082_ = 1;
v___x_1083_ = l_Lean_Meta_abstractMVars(v_app_1066_, v___x_1082_, v_a_1076_, v_a_1077_, v_a_1078_, v_a_1079_);
if (lean_obj_tag(v___x_1083_) == 0)
{
lean_object* v_a_1084_; lean_object* v_expr_1085_; lean_object* v___x_1086_; 
v_a_1084_ = lean_ctor_get(v___x_1083_, 0);
lean_inc(v_a_1084_);
lean_dec_ref_known(v___x_1083_, 1);
v_expr_1085_ = lean_ctor_get(v_a_1084_, 2);
lean_inc_ref_n(v_expr_1085_, 2);
lean_dec(v_a_1084_);
lean_inc(v_a_1079_);
lean_inc_ref(v_a_1078_);
lean_inc(v_a_1077_);
lean_inc_ref(v_a_1076_);
v___x_1086_ = lean_infer_type(v_expr_1085_, v_a_1076_, v_a_1077_, v_a_1078_, v_a_1079_);
if (lean_obj_tag(v___x_1086_) == 0)
{
lean_object* v_a_1087_; lean_object* v___x_1088_; 
v_a_1087_ = lean_ctor_get(v___x_1086_, 0);
lean_inc_n(v_a_1087_, 2);
lean_dec_ref_known(v___x_1086_, 1);
v___x_1088_ = lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_isRedundant(v_a_1087_, v_a_1072_, v_a_1073_, v_a_1074_, v_a_1075_, v_a_1076_, v_a_1077_, v_a_1078_, v_a_1079_);
if (lean_obj_tag(v___x_1088_) == 0)
{
lean_object* v_a_1089_; lean_object* v___x_1091_; uint8_t v_isShared_1092_; uint8_t v_isSharedCheck_1119_; 
v_a_1089_ = lean_ctor_get(v___x_1088_, 0);
v_isSharedCheck_1119_ = !lean_is_exclusive(v___x_1088_);
if (v_isSharedCheck_1119_ == 0)
{
v___x_1091_ = v___x_1088_;
v_isShared_1092_ = v_isSharedCheck_1119_;
goto v_resetjp_1090_;
}
else
{
lean_inc(v_a_1089_);
lean_dec(v___x_1088_);
v___x_1091_ = lean_box(0);
v_isShared_1092_ = v_isSharedCheck_1119_;
goto v_resetjp_1090_;
}
v_resetjp_1090_:
{
uint8_t v___x_1093_; 
v___x_1093_ = lean_unbox(v_a_1089_);
lean_dec(v_a_1089_);
if (v___x_1093_ == 0)
{
lean_object* v___x_1094_; lean_object* v_toAssert_1095_; lean_object* v_usedHyps_1096_; lean_object* v___x_1098_; uint8_t v_isShared_1099_; uint8_t v_isSharedCheck_1114_; 
v___x_1094_ = lean_st_ref_take(v_a_1073_);
v_toAssert_1095_ = lean_ctor_get(v___x_1094_, 0);
v_usedHyps_1096_ = lean_ctor_get(v___x_1094_, 1);
v_isSharedCheck_1114_ = !lean_is_exclusive(v___x_1094_);
if (v_isSharedCheck_1114_ == 0)
{
v___x_1098_ = v___x_1094_;
v_isShared_1099_ = v_isSharedCheck_1114_;
goto v_resetjp_1097_;
}
else
{
lean_inc(v_usedHyps_1096_);
lean_inc(v_toAssert_1095_);
lean_dec(v___x_1094_);
v___x_1098_ = lean_box(0);
v_isShared_1099_ = v_isSharedCheck_1114_;
goto v_resetjp_1097_;
}
v_resetjp_1097_:
{
lean_object* v___x_1100_; lean_object* v___x_1101_; lean_object* v___x_1102_; lean_object* v___x_1103_; lean_object* v___x_1104_; lean_object* v___x_1105_; lean_object* v___x_1107_; 
v___x_1100_ = lean_unsigned_to_nat(1u);
v___x_1101_ = lean_nat_add(v_currentMaxHypDepth_1070_, v___x_1100_);
lean_dec(v_currentMaxHypDepth_1070_);
v___x_1102_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1102_, 0, v_a_1087_);
lean_ctor_set(v___x_1102_, 1, v___x_1101_);
v___x_1103_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1103_, 0, v_expr_1085_);
lean_ctor_set(v___x_1103_, 1, v___x_1102_);
v___x_1104_ = lean_array_push(v_toAssert_1095_, v___x_1103_);
v___x_1105_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__0(v_usedHyps_1096_, v_currentUsedHyps_1071_);
lean_dec_ref(v_currentUsedHyps_1071_);
if (v_isShared_1099_ == 0)
{
lean_ctor_set(v___x_1098_, 1, v___x_1105_);
lean_ctor_set(v___x_1098_, 0, v___x_1104_);
v___x_1107_ = v___x_1098_;
goto v_reusejp_1106_;
}
else
{
lean_object* v_reuseFailAlloc_1113_; 
v_reuseFailAlloc_1113_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1113_, 0, v___x_1104_);
lean_ctor_set(v_reuseFailAlloc_1113_, 1, v___x_1105_);
v___x_1107_ = v_reuseFailAlloc_1113_;
goto v_reusejp_1106_;
}
v_reusejp_1106_:
{
lean_object* v___x_1108_; lean_object* v___x_1109_; lean_object* v___x_1111_; 
v___x_1108_ = lean_st_ref_set(v_a_1073_, v___x_1107_);
v___x_1109_ = lean_box(0);
if (v_isShared_1092_ == 0)
{
lean_ctor_set(v___x_1091_, 0, v___x_1109_);
v___x_1111_ = v___x_1091_;
goto v_reusejp_1110_;
}
else
{
lean_object* v_reuseFailAlloc_1112_; 
v_reuseFailAlloc_1112_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1112_, 0, v___x_1109_);
v___x_1111_ = v_reuseFailAlloc_1112_;
goto v_reusejp_1110_;
}
v_reusejp_1110_:
{
return v___x_1111_;
}
}
}
}
else
{
lean_object* v___x_1115_; lean_object* v___x_1117_; 
lean_dec(v_a_1087_);
lean_dec_ref(v_expr_1085_);
lean_dec_ref(v_currentUsedHyps_1071_);
lean_dec(v_currentMaxHypDepth_1070_);
v___x_1115_ = lean_box(0);
if (v_isShared_1092_ == 0)
{
lean_ctor_set(v___x_1091_, 0, v___x_1115_);
v___x_1117_ = v___x_1091_;
goto v_reusejp_1116_;
}
else
{
lean_object* v_reuseFailAlloc_1118_; 
v_reuseFailAlloc_1118_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1118_, 0, v___x_1115_);
v___x_1117_ = v_reuseFailAlloc_1118_;
goto v_reusejp_1116_;
}
v_reusejp_1116_:
{
return v___x_1117_;
}
}
}
}
else
{
lean_object* v_a_1120_; lean_object* v___x_1122_; uint8_t v_isShared_1123_; uint8_t v_isSharedCheck_1127_; 
lean_dec(v_a_1087_);
lean_dec_ref(v_expr_1085_);
lean_dec_ref(v_currentUsedHyps_1071_);
lean_dec(v_currentMaxHypDepth_1070_);
v_a_1120_ = lean_ctor_get(v___x_1088_, 0);
v_isSharedCheck_1127_ = !lean_is_exclusive(v___x_1088_);
if (v_isSharedCheck_1127_ == 0)
{
v___x_1122_ = v___x_1088_;
v_isShared_1123_ = v_isSharedCheck_1127_;
goto v_resetjp_1121_;
}
else
{
lean_inc(v_a_1120_);
lean_dec(v___x_1088_);
v___x_1122_ = lean_box(0);
v_isShared_1123_ = v_isSharedCheck_1127_;
goto v_resetjp_1121_;
}
v_resetjp_1121_:
{
lean_object* v___x_1125_; 
if (v_isShared_1123_ == 0)
{
v___x_1125_ = v___x_1122_;
goto v_reusejp_1124_;
}
else
{
lean_object* v_reuseFailAlloc_1126_; 
v_reuseFailAlloc_1126_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1126_, 0, v_a_1120_);
v___x_1125_ = v_reuseFailAlloc_1126_;
goto v_reusejp_1124_;
}
v_reusejp_1124_:
{
return v___x_1125_;
}
}
}
}
else
{
lean_object* v_a_1128_; lean_object* v___x_1130_; uint8_t v_isShared_1131_; uint8_t v_isSharedCheck_1135_; 
lean_dec_ref(v_expr_1085_);
lean_dec_ref(v_currentUsedHyps_1071_);
lean_dec(v_currentMaxHypDepth_1070_);
v_a_1128_ = lean_ctor_get(v___x_1086_, 0);
v_isSharedCheck_1135_ = !lean_is_exclusive(v___x_1086_);
if (v_isSharedCheck_1135_ == 0)
{
v___x_1130_ = v___x_1086_;
v_isShared_1131_ = v_isSharedCheck_1135_;
goto v_resetjp_1129_;
}
else
{
lean_inc(v_a_1128_);
lean_dec(v___x_1086_);
v___x_1130_ = lean_box(0);
v_isShared_1131_ = v_isSharedCheck_1135_;
goto v_resetjp_1129_;
}
v_resetjp_1129_:
{
lean_object* v___x_1133_; 
if (v_isShared_1131_ == 0)
{
v___x_1133_ = v___x_1130_;
goto v_reusejp_1132_;
}
else
{
lean_object* v_reuseFailAlloc_1134_; 
v_reuseFailAlloc_1134_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1134_, 0, v_a_1128_);
v___x_1133_ = v_reuseFailAlloc_1134_;
goto v_reusejp_1132_;
}
v_reusejp_1132_:
{
return v___x_1133_;
}
}
}
}
else
{
lean_object* v_a_1136_; lean_object* v___x_1138_; uint8_t v_isShared_1139_; uint8_t v_isSharedCheck_1143_; 
lean_dec_ref(v_currentUsedHyps_1071_);
lean_dec(v_currentMaxHypDepth_1070_);
v_a_1136_ = lean_ctor_get(v___x_1083_, 0);
v_isSharedCheck_1143_ = !lean_is_exclusive(v___x_1083_);
if (v_isSharedCheck_1143_ == 0)
{
v___x_1138_ = v___x_1083_;
v_isShared_1139_ = v_isSharedCheck_1143_;
goto v_resetjp_1137_;
}
else
{
lean_inc(v_a_1136_);
lean_dec(v___x_1083_);
v___x_1138_ = lean_box(0);
v_isShared_1139_ = v_isSharedCheck_1143_;
goto v_resetjp_1137_;
}
v_resetjp_1137_:
{
lean_object* v___x_1141_; 
if (v_isShared_1139_ == 0)
{
v___x_1141_ = v___x_1138_;
goto v_reusejp_1140_;
}
else
{
lean_object* v_reuseFailAlloc_1142_; 
v_reuseFailAlloc_1142_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1142_, 0, v_a_1136_);
v___x_1141_ = v_reuseFailAlloc_1142_;
goto v_reusejp_1140_;
}
v_reusejp_1140_:
{
return v___x_1141_;
}
}
}
}
v___jp_1144_:
{
if (lean_obj_tag(v___y_1145_) == 0)
{
lean_object* v_a_1146_; lean_object* v___x_1148_; uint8_t v_isShared_1149_; uint8_t v_isSharedCheck_1154_; 
v_a_1146_ = lean_ctor_get(v___y_1145_, 0);
v_isSharedCheck_1154_ = !lean_is_exclusive(v___y_1145_);
if (v_isSharedCheck_1154_ == 0)
{
v___x_1148_ = v___y_1145_;
v_isShared_1149_ = v_isSharedCheck_1154_;
goto v_resetjp_1147_;
}
else
{
lean_inc(v_a_1146_);
lean_dec(v___y_1145_);
v___x_1148_ = lean_box(0);
v_isShared_1149_ = v_isSharedCheck_1154_;
goto v_resetjp_1147_;
}
v_resetjp_1147_:
{
if (lean_obj_tag(v_a_1146_) == 0)
{
lean_object* v_a_1150_; lean_object* v___x_1152_; 
lean_dec_ref(v_currentUsedHyps_1071_);
lean_dec(v_currentMaxHypDepth_1070_);
lean_dec_ref(v_app_1066_);
v_a_1150_ = lean_ctor_get(v_a_1146_, 0);
lean_inc(v_a_1150_);
lean_dec_ref_known(v_a_1146_, 1);
if (v_isShared_1149_ == 0)
{
lean_ctor_set(v___x_1148_, 0, v_a_1150_);
v___x_1152_ = v___x_1148_;
goto v_reusejp_1151_;
}
else
{
lean_object* v_reuseFailAlloc_1153_; 
v_reuseFailAlloc_1153_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1153_, 0, v_a_1150_);
v___x_1152_ = v_reuseFailAlloc_1153_;
goto v_reusejp_1151_;
}
v_reusejp_1151_:
{
return v___x_1152_;
}
}
else
{
lean_dec_ref_known(v_a_1146_, 1);
lean_del_object(v___x_1148_);
goto v___jp_1081_;
}
}
}
else
{
lean_object* v_a_1155_; lean_object* v___x_1157_; uint8_t v_isShared_1158_; uint8_t v_isSharedCheck_1162_; 
lean_dec_ref(v_currentUsedHyps_1071_);
lean_dec(v_currentMaxHypDepth_1070_);
lean_dec_ref(v_app_1066_);
v_a_1155_ = lean_ctor_get(v___y_1145_, 0);
v_isSharedCheck_1162_ = !lean_is_exclusive(v___y_1145_);
if (v_isSharedCheck_1162_ == 0)
{
v___x_1157_ = v___y_1145_;
v_isShared_1158_ = v_isSharedCheck_1162_;
goto v_resetjp_1156_;
}
else
{
lean_inc(v_a_1155_);
lean_dec(v___y_1145_);
v___x_1157_ = lean_box(0);
v_isShared_1158_ = v_isSharedCheck_1162_;
goto v_resetjp_1156_;
}
v_resetjp_1156_:
{
lean_object* v___x_1160_; 
if (v_isShared_1158_ == 0)
{
v___x_1160_ = v___x_1157_;
goto v_reusejp_1159_;
}
else
{
lean_object* v_reuseFailAlloc_1161_; 
v_reuseFailAlloc_1161_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1161_, 0, v_a_1155_);
v___x_1160_ = v_reuseFailAlloc_1161_;
goto v_reusejp_1159_;
}
v_reusejp_1159_:
{
return v___x_1160_;
}
}
}
}
v___jp_1163_:
{
if (v___y_1165_ == 0)
{
lean_object* v___x_1166_; lean_object* v___x_1167_; 
lean_dec_ref(v___y_1164_);
v___x_1166_ = lp_aesop_Aesop_TraceOption_forward;
v___x_1167_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__2___redArg(v___x_1166_, v_a_1078_);
if (lean_obj_tag(v___x_1167_) == 0)
{
lean_object* v_a_1168_; uint8_t v___x_1169_; 
v_a_1168_ = lean_ctor_get(v___x_1167_, 0);
lean_inc(v_a_1168_);
lean_dec_ref_known(v___x_1167_, 1);
v___x_1169_ = lean_unbox(v_a_1168_);
lean_dec(v_a_1168_);
if (v___x_1169_ == 0)
{
lean_object* v___x_1170_; lean_object* v___x_1171_; 
v___x_1170_ = lean_box(0);
v___x_1171_ = lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop___lam__0(v___x_1170_, v_a_1072_, v_a_1073_, v_a_1074_, v_a_1075_, v_a_1076_, v_a_1077_, v_a_1078_, v_a_1079_);
v___y_1145_ = v___x_1171_;
goto v___jp_1144_;
}
else
{
lean_object* v_traceClass_1172_; lean_object* v___x_1173_; lean_object* v___x_1174_; 
v_traceClass_1172_ = lean_ctor_get(v___x_1166_, 0);
v___x_1173_ = lean_obj_once(&lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop___closed__1, &lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop___closed__1_once, _init_lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop___closed__1);
lean_inc(v_traceClass_1172_);
v___x_1174_ = lp_aesop_Lean_addTrace___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__3___redArg(v_traceClass_1172_, v___x_1173_, v_a_1076_, v_a_1077_, v_a_1078_, v_a_1079_);
if (lean_obj_tag(v___x_1174_) == 0)
{
lean_object* v_a_1175_; lean_object* v___x_1176_; 
v_a_1175_ = lean_ctor_get(v___x_1174_, 0);
lean_inc(v_a_1175_);
lean_dec_ref_known(v___x_1174_, 1);
v___x_1176_ = lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop___lam__0(v_a_1175_, v_a_1072_, v_a_1073_, v_a_1074_, v_a_1075_, v_a_1076_, v_a_1077_, v_a_1078_, v_a_1079_);
v___y_1145_ = v___x_1176_;
goto v___jp_1144_;
}
else
{
lean_dec_ref(v_currentUsedHyps_1071_);
lean_dec(v_currentMaxHypDepth_1070_);
lean_dec_ref(v_app_1066_);
return v___x_1174_;
}
}
}
else
{
lean_object* v_a_1177_; lean_object* v___x_1179_; uint8_t v_isShared_1180_; uint8_t v_isSharedCheck_1184_; 
lean_dec_ref(v_currentUsedHyps_1071_);
lean_dec(v_currentMaxHypDepth_1070_);
lean_dec_ref(v_app_1066_);
v_a_1177_ = lean_ctor_get(v___x_1167_, 0);
v_isSharedCheck_1184_ = !lean_is_exclusive(v___x_1167_);
if (v_isSharedCheck_1184_ == 0)
{
v___x_1179_ = v___x_1167_;
v_isShared_1180_ = v_isSharedCheck_1184_;
goto v_resetjp_1178_;
}
else
{
lean_inc(v_a_1177_);
lean_dec(v___x_1167_);
v___x_1179_ = lean_box(0);
v_isShared_1180_ = v_isSharedCheck_1184_;
goto v_resetjp_1178_;
}
v_resetjp_1178_:
{
lean_object* v___x_1182_; 
if (v_isShared_1180_ == 0)
{
v___x_1182_ = v___x_1179_;
goto v_reusejp_1181_;
}
else
{
lean_object* v_reuseFailAlloc_1183_; 
v_reuseFailAlloc_1183_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1183_, 0, v_a_1177_);
v___x_1182_ = v_reuseFailAlloc_1183_;
goto v_reusejp_1181_;
}
v_reusejp_1181_:
{
return v___x_1182_;
}
}
}
}
else
{
lean_dec_ref(v_currentUsedHyps_1071_);
lean_dec(v_currentMaxHypDepth_1070_);
lean_dec_ref(v_app_1066_);
return v___y_1164_;
}
}
v___jp_1185_:
{
lean_object* v___x_1186_; uint8_t v___x_1187_; lean_object* v___x_1188_; lean_object* v_binderInfos_1189_; lean_object* v___x_1190_; lean_object* v___x_1191_; size_t v_sz_1192_; size_t v___x_1193_; lean_object* v___x_1194_; uint8_t v___x_1195_; lean_object* v___x_1196_; 
v___x_1186_ = lean_array_get_size(v_instMVars_1067_);
v___x_1187_ = 3;
v___x_1188_ = lean_box(v___x_1187_);
v_binderInfos_1189_ = lean_mk_array(v___x_1186_, v___x_1188_);
v___x_1190_ = ((lean_object*)(lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop___closed__3));
v___x_1191_ = lean_box(0);
v_sz_1192_ = lean_array_size(v_instMVars_1067_);
v___x_1193_ = ((size_t)0ULL);
v___x_1194_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__1(v_sz_1192_, v___x_1193_, v_instMVars_1067_);
v___x_1195_ = 0;
v___x_1196_ = l_Lean_Meta_synthAppInstances(v___x_1190_, v___x_1191_, v___x_1194_, v_binderInfos_1189_, v___x_1195_, v___x_1195_, v_a_1076_, v_a_1077_, v_a_1078_, v_a_1079_);
lean_dec_ref(v___x_1194_);
if (lean_obj_tag(v___x_1196_) == 0)
{
lean_dec_ref_known(v___x_1196_, 1);
goto v___jp_1081_;
}
else
{
lean_object* v_a_1197_; uint8_t v___x_1198_; 
v_a_1197_ = lean_ctor_get(v___x_1196_, 0);
lean_inc(v_a_1197_);
v___x_1198_ = l_Lean_Exception_isInterrupt(v_a_1197_);
if (v___x_1198_ == 0)
{
uint8_t v___x_1199_; 
v___x_1199_ = l_Lean_Exception_isRuntime(v_a_1197_);
v___y_1164_ = v___x_1196_;
v___y_1165_ = v___x_1199_;
goto v___jp_1163_;
}
else
{
lean_dec(v_a_1197_);
v___y_1164_ = v___x_1196_;
v___y_1165_ = v___x_1198_;
goto v___jp_1163_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__7_spec__14_spec__20_spec__26(lean_object* v_a_1282_, lean_object* v_mvarId_1283_, lean_object* v_currentUsedHyps_1284_, lean_object* v_i_1285_, lean_object* v_app_1286_, lean_object* v_instMVars_1287_, lean_object* v_immediateMVars_1288_, lean_object* v_currentMaxHypDepth_1289_, lean_object* v_as_1290_, size_t v_sz_1291_, size_t v_i_1292_, lean_object* v_b_1293_, lean_object* v___y_1294_, lean_object* v___y_1295_, lean_object* v___y_1296_, lean_object* v___y_1297_, lean_object* v___y_1298_, lean_object* v___y_1299_, lean_object* v___y_1300_, lean_object* v___y_1301_){
_start:
{
lean_object* v___y_1304_; lean_object* v___y_1305_; lean_object* v___y_1306_; lean_object* v_a_1307_; uint8_t v___x_1325_; 
v___x_1325_ = lean_usize_dec_lt(v_i_1292_, v_sz_1291_);
if (v___x_1325_ == 0)
{
lean_object* v___x_1326_; 
lean_dec(v_currentMaxHypDepth_1289_);
lean_dec_ref(v_instMVars_1287_);
lean_dec_ref(v_app_1286_);
lean_dec_ref(v_currentUsedHyps_1284_);
lean_dec(v_mvarId_1283_);
lean_dec_ref(v_a_1282_);
v___x_1326_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1326_, 0, v_b_1293_);
return v___x_1326_;
}
else
{
lean_object* v_snd_1327_; lean_object* v___x_1329_; uint8_t v_isShared_1330_; uint8_t v_isSharedCheck_1405_; 
v_snd_1327_ = lean_ctor_get(v_b_1293_, 1);
v_isSharedCheck_1405_ = !lean_is_exclusive(v_b_1293_);
if (v_isSharedCheck_1405_ == 0)
{
lean_object* v_unused_1406_; 
v_unused_1406_ = lean_ctor_get(v_b_1293_, 0);
lean_dec(v_unused_1406_);
v___x_1329_ = v_b_1293_;
v_isShared_1330_ = v_isSharedCheck_1405_;
goto v_resetjp_1328_;
}
else
{
lean_inc(v_snd_1327_);
lean_dec(v_b_1293_);
v___x_1329_ = lean_box(0);
v_isShared_1330_ = v_isSharedCheck_1405_;
goto v_resetjp_1328_;
}
v_resetjp_1328_:
{
lean_object* v___x_1331_; lean_object* v_a_1333_; lean_object* v_a_1340_; 
v___x_1331_ = lean_box(0);
v_a_1340_ = lean_array_uget_borrowed(v_as_1290_, v_i_1292_);
if (lean_obj_tag(v_a_1340_) == 0)
{
v_a_1333_ = v_snd_1327_;
goto v___jp_1332_;
}
else
{
lean_object* v_val_1341_; lean_object* v___x_1342_; lean_object* v___y_1344_; lean_object* v___y_1345_; lean_object* v___y_1346_; lean_object* v___y_1357_; lean_object* v___y_1358_; lean_object* v___y_1359_; lean_object* v___y_1360_; uint8_t v___x_1362_; 
lean_dec(v_snd_1327_);
v_val_1341_ = lean_ctor_get(v_a_1340_, 0);
v___x_1342_ = lean_box(0);
v___x_1362_ = l_Lean_LocalDecl_isImplementationDetail(v_val_1341_);
if (v___x_1362_ == 0)
{
lean_object* v_maxDepth_x3f_1363_; lean_object* v_forwardHypData_1364_; lean_object* v___x_1365_; lean_object* v___x_1366_; lean_object* v___x_1367_; lean_object* v___y_1369_; lean_object* v___y_1370_; lean_object* v___y_1371_; lean_object* v___y_1372_; lean_object* v___y_1373_; lean_object* v___y_1374_; lean_object* v___y_1375_; lean_object* v___y_1376_; lean_object* v___y_1377_; lean_object* v___y_1399_; lean_object* v___x_1403_; uint8_t v___x_1404_; 
v_maxDepth_x3f_1363_ = lean_ctor_get(v___y_1294_, 0);
v_forwardHypData_1364_ = lean_ctor_get(v___y_1294_, 1);
v___x_1365_ = lean_unsigned_to_nat(1u);
v___x_1366_ = lean_unsigned_to_nat(0u);
v___x_1367_ = l_Lean_LocalDecl_fvarId(v_val_1341_);
v___x_1403_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__4___redArg(v_forwardHypData_1364_, v___x_1367_, v___x_1366_);
v___x_1404_ = lean_nat_dec_le(v_currentMaxHypDepth_1289_, v___x_1403_);
if (v___x_1404_ == 0)
{
lean_dec(v___x_1403_);
lean_inc(v_currentMaxHypDepth_1289_);
v___y_1399_ = v_currentMaxHypDepth_1289_;
goto v___jp_1398_;
}
else
{
v___y_1399_ = v___x_1403_;
goto v___jp_1398_;
}
v___jp_1368_:
{
lean_object* v___x_1378_; 
v___x_1378_ = l_Lean_Meta_saveState___redArg(v___y_1375_, v___y_1377_);
if (lean_obj_tag(v___x_1378_) == 0)
{
lean_object* v_a_1379_; lean_object* v___x_1380_; lean_object* v___x_1381_; 
v_a_1379_ = lean_ctor_get(v___x_1378_, 0);
lean_inc(v_a_1379_);
lean_dec_ref_known(v___x_1378_, 1);
v___x_1380_ = l_Lean_LocalDecl_type(v_val_1341_);
lean_inc_ref(v_a_1282_);
v___x_1381_ = l_Lean_Meta_isExprDefEq(v___x_1380_, v_a_1282_, v___y_1374_, v___y_1375_, v___y_1376_, v___y_1377_);
if (lean_obj_tag(v___x_1381_) == 0)
{
lean_object* v_a_1382_; uint8_t v___x_1383_; 
v_a_1382_ = lean_ctor_get(v___x_1381_, 0);
lean_inc(v_a_1382_);
lean_dec_ref_known(v___x_1381_, 1);
v___x_1383_ = lean_unbox(v_a_1382_);
lean_dec(v_a_1382_);
if (v___x_1383_ == 0)
{
lean_dec(v___y_1369_);
lean_dec(v___x_1367_);
v___y_1344_ = v___y_1375_;
v___y_1345_ = v___y_1377_;
v___y_1346_ = v_a_1379_;
goto v___jp_1343_;
}
else
{
lean_object* v___x_1384_; lean_object* v___x_1385_; 
lean_inc(v___x_1367_);
v___x_1384_ = l_Lean_mkFVar(v___x_1367_);
lean_inc(v_mvarId_1283_);
v___x_1385_ = lp_aesop_Lean_MVarId_assign___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__6___redArg(v_mvarId_1283_, v___x_1384_, v___y_1375_);
if (lean_obj_tag(v___x_1385_) == 0)
{
lean_object* v___x_1386_; lean_object* v___x_1387_; lean_object* v___x_1388_; 
lean_dec_ref_known(v___x_1385_, 1);
lean_inc_ref(v_currentUsedHyps_1284_);
v___x_1386_ = lean_array_push(v_currentUsedHyps_1284_, v___x_1367_);
v___x_1387_ = lean_nat_add(v_i_1285_, v___x_1365_);
lean_inc_ref(v_instMVars_1287_);
lean_inc_ref(v_app_1286_);
v___x_1388_ = lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop(v_app_1286_, v_instMVars_1287_, v_immediateMVars_1288_, v___x_1387_, v___y_1369_, v___x_1386_, v___y_1370_, v___y_1371_, v___y_1372_, v___y_1373_, v___y_1374_, v___y_1375_, v___y_1376_, v___y_1377_);
v___y_1357_ = v___y_1375_;
v___y_1358_ = v___y_1377_;
v___y_1359_ = v_a_1379_;
v___y_1360_ = v___x_1388_;
goto v___jp_1356_;
}
else
{
lean_dec(v___y_1369_);
lean_dec(v___x_1367_);
v___y_1357_ = v___y_1375_;
v___y_1358_ = v___y_1377_;
v___y_1359_ = v_a_1379_;
v___y_1360_ = v___x_1385_;
goto v___jp_1356_;
}
}
}
else
{
lean_object* v_a_1389_; 
lean_dec(v___y_1369_);
lean_dec(v___x_1367_);
lean_del_object(v___x_1329_);
lean_dec(v_currentMaxHypDepth_1289_);
lean_dec_ref(v_instMVars_1287_);
lean_dec_ref(v_app_1286_);
lean_dec_ref(v_currentUsedHyps_1284_);
lean_dec(v_mvarId_1283_);
lean_dec_ref(v_a_1282_);
v_a_1389_ = lean_ctor_get(v___x_1381_, 0);
lean_inc(v_a_1389_);
lean_dec_ref_known(v___x_1381_, 1);
v___y_1304_ = v___y_1375_;
v___y_1305_ = v___y_1377_;
v___y_1306_ = v_a_1379_;
v_a_1307_ = v_a_1389_;
goto v___jp_1303_;
}
}
else
{
lean_object* v_a_1390_; lean_object* v___x_1392_; uint8_t v_isShared_1393_; uint8_t v_isSharedCheck_1397_; 
lean_dec(v___y_1369_);
lean_dec(v___x_1367_);
lean_del_object(v___x_1329_);
lean_dec(v_currentMaxHypDepth_1289_);
lean_dec_ref(v_instMVars_1287_);
lean_dec_ref(v_app_1286_);
lean_dec_ref(v_currentUsedHyps_1284_);
lean_dec(v_mvarId_1283_);
lean_dec_ref(v_a_1282_);
v_a_1390_ = lean_ctor_get(v___x_1378_, 0);
v_isSharedCheck_1397_ = !lean_is_exclusive(v___x_1378_);
if (v_isSharedCheck_1397_ == 0)
{
v___x_1392_ = v___x_1378_;
v_isShared_1393_ = v_isSharedCheck_1397_;
goto v_resetjp_1391_;
}
else
{
lean_inc(v_a_1390_);
lean_dec(v___x_1378_);
v___x_1392_ = lean_box(0);
v_isShared_1393_ = v_isSharedCheck_1397_;
goto v_resetjp_1391_;
}
v_resetjp_1391_:
{
lean_object* v___x_1395_; 
if (v_isShared_1393_ == 0)
{
v___x_1395_ = v___x_1392_;
goto v_reusejp_1394_;
}
else
{
lean_object* v_reuseFailAlloc_1396_; 
v_reuseFailAlloc_1396_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1396_, 0, v_a_1390_);
v___x_1395_ = v_reuseFailAlloc_1396_;
goto v_reusejp_1394_;
}
v_reusejp_1394_:
{
return v___x_1395_;
}
}
}
}
v___jp_1398_:
{
if (lean_obj_tag(v_maxDepth_x3f_1363_) == 1)
{
lean_object* v_val_1400_; lean_object* v___x_1401_; uint8_t v___x_1402_; 
v_val_1400_ = lean_ctor_get(v_maxDepth_x3f_1363_, 0);
v___x_1401_ = lean_nat_add(v___y_1399_, v___x_1365_);
v___x_1402_ = lean_nat_dec_lt(v_val_1400_, v___x_1401_);
lean_dec(v___x_1401_);
if (v___x_1402_ == 0)
{
v___y_1369_ = v___y_1399_;
v___y_1370_ = v___y_1294_;
v___y_1371_ = v___y_1295_;
v___y_1372_ = v___y_1296_;
v___y_1373_ = v___y_1297_;
v___y_1374_ = v___y_1298_;
v___y_1375_ = v___y_1299_;
v___y_1376_ = v___y_1300_;
v___y_1377_ = v___y_1301_;
goto v___jp_1368_;
}
else
{
lean_dec(v___y_1399_);
lean_dec(v___x_1367_);
v_a_1333_ = v___x_1342_;
goto v___jp_1332_;
}
}
else
{
v___y_1369_ = v___y_1399_;
v___y_1370_ = v___y_1294_;
v___y_1371_ = v___y_1295_;
v___y_1372_ = v___y_1296_;
v___y_1373_ = v___y_1297_;
v___y_1374_ = v___y_1298_;
v___y_1375_ = v___y_1299_;
v___y_1376_ = v___y_1300_;
v___y_1377_ = v___y_1301_;
goto v___jp_1368_;
}
}
}
else
{
v_a_1333_ = v___x_1342_;
goto v___jp_1332_;
}
v___jp_1343_:
{
lean_object* v___x_1347_; 
v___x_1347_ = l_Lean_Meta_SavedState_restore___redArg(v___y_1346_, v___y_1344_, v___y_1345_);
lean_dec_ref(v___y_1346_);
if (lean_obj_tag(v___x_1347_) == 0)
{
lean_dec_ref_known(v___x_1347_, 1);
v_a_1333_ = v___x_1342_;
goto v___jp_1332_;
}
else
{
lean_object* v_a_1348_; lean_object* v___x_1350_; uint8_t v_isShared_1351_; uint8_t v_isSharedCheck_1355_; 
lean_del_object(v___x_1329_);
lean_dec(v_currentMaxHypDepth_1289_);
lean_dec_ref(v_instMVars_1287_);
lean_dec_ref(v_app_1286_);
lean_dec_ref(v_currentUsedHyps_1284_);
lean_dec(v_mvarId_1283_);
lean_dec_ref(v_a_1282_);
v_a_1348_ = lean_ctor_get(v___x_1347_, 0);
v_isSharedCheck_1355_ = !lean_is_exclusive(v___x_1347_);
if (v_isSharedCheck_1355_ == 0)
{
v___x_1350_ = v___x_1347_;
v_isShared_1351_ = v_isSharedCheck_1355_;
goto v_resetjp_1349_;
}
else
{
lean_inc(v_a_1348_);
lean_dec(v___x_1347_);
v___x_1350_ = lean_box(0);
v_isShared_1351_ = v_isSharedCheck_1355_;
goto v_resetjp_1349_;
}
v_resetjp_1349_:
{
lean_object* v___x_1353_; 
if (v_isShared_1351_ == 0)
{
v___x_1353_ = v___x_1350_;
goto v_reusejp_1352_;
}
else
{
lean_object* v_reuseFailAlloc_1354_; 
v_reuseFailAlloc_1354_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1354_, 0, v_a_1348_);
v___x_1353_ = v_reuseFailAlloc_1354_;
goto v_reusejp_1352_;
}
v_reusejp_1352_:
{
return v___x_1353_;
}
}
}
}
v___jp_1356_:
{
if (lean_obj_tag(v___y_1360_) == 0)
{
lean_dec_ref_known(v___y_1360_, 1);
v___y_1344_ = v___y_1357_;
v___y_1345_ = v___y_1358_;
v___y_1346_ = v___y_1359_;
goto v___jp_1343_;
}
else
{
lean_object* v_a_1361_; 
lean_del_object(v___x_1329_);
lean_dec(v_currentMaxHypDepth_1289_);
lean_dec_ref(v_instMVars_1287_);
lean_dec_ref(v_app_1286_);
lean_dec_ref(v_currentUsedHyps_1284_);
lean_dec(v_mvarId_1283_);
lean_dec_ref(v_a_1282_);
v_a_1361_ = lean_ctor_get(v___y_1360_, 0);
lean_inc(v_a_1361_);
lean_dec_ref_known(v___y_1360_, 1);
v___y_1304_ = v___y_1357_;
v___y_1305_ = v___y_1358_;
v___y_1306_ = v___y_1359_;
v_a_1307_ = v_a_1361_;
goto v___jp_1303_;
}
}
}
v___jp_1332_:
{
lean_object* v___x_1335_; 
if (v_isShared_1330_ == 0)
{
lean_ctor_set(v___x_1329_, 1, v_a_1333_);
lean_ctor_set(v___x_1329_, 0, v___x_1331_);
v___x_1335_ = v___x_1329_;
goto v_reusejp_1334_;
}
else
{
lean_object* v_reuseFailAlloc_1339_; 
v_reuseFailAlloc_1339_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1339_, 0, v___x_1331_);
lean_ctor_set(v_reuseFailAlloc_1339_, 1, v_a_1333_);
v___x_1335_ = v_reuseFailAlloc_1339_;
goto v_reusejp_1334_;
}
v_reusejp_1334_:
{
size_t v___x_1336_; size_t v___x_1337_; 
v___x_1336_ = ((size_t)1ULL);
v___x_1337_ = lean_usize_add(v_i_1292_, v___x_1336_);
v_i_1292_ = v___x_1337_;
v_b_1293_ = v___x_1335_;
goto _start;
}
}
}
}
v___jp_1303_:
{
lean_object* v___x_1308_; 
v___x_1308_ = l_Lean_Meta_SavedState_restore___redArg(v___y_1306_, v___y_1304_, v___y_1305_);
lean_dec_ref(v___y_1306_);
if (lean_obj_tag(v___x_1308_) == 0)
{
lean_object* v___x_1310_; uint8_t v_isShared_1311_; uint8_t v_isSharedCheck_1315_; 
v_isSharedCheck_1315_ = !lean_is_exclusive(v___x_1308_);
if (v_isSharedCheck_1315_ == 0)
{
lean_object* v_unused_1316_; 
v_unused_1316_ = lean_ctor_get(v___x_1308_, 0);
lean_dec(v_unused_1316_);
v___x_1310_ = v___x_1308_;
v_isShared_1311_ = v_isSharedCheck_1315_;
goto v_resetjp_1309_;
}
else
{
lean_dec(v___x_1308_);
v___x_1310_ = lean_box(0);
v_isShared_1311_ = v_isSharedCheck_1315_;
goto v_resetjp_1309_;
}
v_resetjp_1309_:
{
lean_object* v___x_1313_; 
if (v_isShared_1311_ == 0)
{
lean_ctor_set_tag(v___x_1310_, 1);
lean_ctor_set(v___x_1310_, 0, v_a_1307_);
v___x_1313_ = v___x_1310_;
goto v_reusejp_1312_;
}
else
{
lean_object* v_reuseFailAlloc_1314_; 
v_reuseFailAlloc_1314_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1314_, 0, v_a_1307_);
v___x_1313_ = v_reuseFailAlloc_1314_;
goto v_reusejp_1312_;
}
v_reusejp_1312_:
{
return v___x_1313_;
}
}
}
else
{
lean_object* v_a_1317_; lean_object* v___x_1319_; uint8_t v_isShared_1320_; uint8_t v_isSharedCheck_1324_; 
lean_dec_ref(v_a_1307_);
v_a_1317_ = lean_ctor_get(v___x_1308_, 0);
v_isSharedCheck_1324_ = !lean_is_exclusive(v___x_1308_);
if (v_isSharedCheck_1324_ == 0)
{
v___x_1319_ = v___x_1308_;
v_isShared_1320_ = v_isSharedCheck_1324_;
goto v_resetjp_1318_;
}
else
{
lean_inc(v_a_1317_);
lean_dec(v___x_1308_);
v___x_1319_ = lean_box(0);
v_isShared_1320_ = v_isSharedCheck_1324_;
goto v_resetjp_1318_;
}
v_resetjp_1318_:
{
lean_object* v___x_1322_; 
if (v_isShared_1320_ == 0)
{
v___x_1322_ = v___x_1319_;
goto v_reusejp_1321_;
}
else
{
lean_object* v_reuseFailAlloc_1323_; 
v_reuseFailAlloc_1323_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1323_, 0, v_a_1317_);
v___x_1322_ = v_reuseFailAlloc_1323_;
goto v_reusejp_1321_;
}
v_reusejp_1321_:
{
return v___x_1322_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__7_spec__14_spec__20(lean_object* v_a_1407_, lean_object* v_mvarId_1408_, lean_object* v_currentUsedHyps_1409_, lean_object* v_i_1410_, lean_object* v_app_1411_, lean_object* v_instMVars_1412_, lean_object* v_immediateMVars_1413_, lean_object* v_currentMaxHypDepth_1414_, lean_object* v_as_1415_, size_t v_sz_1416_, size_t v_i_1417_, lean_object* v_b_1418_, lean_object* v___y_1419_, lean_object* v___y_1420_, lean_object* v___y_1421_, lean_object* v___y_1422_, lean_object* v___y_1423_, lean_object* v___y_1424_, lean_object* v___y_1425_, lean_object* v___y_1426_){
_start:
{
lean_object* v___y_1429_; lean_object* v___y_1430_; lean_object* v___y_1431_; lean_object* v_a_1432_; uint8_t v___x_1450_; 
v___x_1450_ = lean_usize_dec_lt(v_i_1417_, v_sz_1416_);
if (v___x_1450_ == 0)
{
lean_object* v___x_1451_; 
lean_dec(v_currentMaxHypDepth_1414_);
lean_dec_ref(v_instMVars_1412_);
lean_dec_ref(v_app_1411_);
lean_dec_ref(v_currentUsedHyps_1409_);
lean_dec(v_mvarId_1408_);
lean_dec_ref(v_a_1407_);
v___x_1451_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1451_, 0, v_b_1418_);
return v___x_1451_;
}
else
{
lean_object* v_snd_1452_; lean_object* v___x_1454_; uint8_t v_isShared_1455_; uint8_t v_isSharedCheck_1530_; 
v_snd_1452_ = lean_ctor_get(v_b_1418_, 1);
v_isSharedCheck_1530_ = !lean_is_exclusive(v_b_1418_);
if (v_isSharedCheck_1530_ == 0)
{
lean_object* v_unused_1531_; 
v_unused_1531_ = lean_ctor_get(v_b_1418_, 0);
lean_dec(v_unused_1531_);
v___x_1454_ = v_b_1418_;
v_isShared_1455_ = v_isSharedCheck_1530_;
goto v_resetjp_1453_;
}
else
{
lean_inc(v_snd_1452_);
lean_dec(v_b_1418_);
v___x_1454_ = lean_box(0);
v_isShared_1455_ = v_isSharedCheck_1530_;
goto v_resetjp_1453_;
}
v_resetjp_1453_:
{
lean_object* v___x_1456_; lean_object* v_a_1458_; lean_object* v_a_1465_; 
v___x_1456_ = lean_box(0);
v_a_1465_ = lean_array_uget_borrowed(v_as_1415_, v_i_1417_);
if (lean_obj_tag(v_a_1465_) == 0)
{
v_a_1458_ = v_snd_1452_;
goto v___jp_1457_;
}
else
{
lean_object* v_val_1466_; lean_object* v___x_1467_; lean_object* v___y_1469_; lean_object* v___y_1470_; lean_object* v___y_1471_; lean_object* v___y_1482_; lean_object* v___y_1483_; lean_object* v___y_1484_; lean_object* v___y_1485_; uint8_t v___x_1487_; 
lean_dec(v_snd_1452_);
v_val_1466_ = lean_ctor_get(v_a_1465_, 0);
v___x_1467_ = lean_box(0);
v___x_1487_ = l_Lean_LocalDecl_isImplementationDetail(v_val_1466_);
if (v___x_1487_ == 0)
{
lean_object* v_maxDepth_x3f_1488_; lean_object* v_forwardHypData_1489_; lean_object* v___x_1490_; lean_object* v___x_1491_; lean_object* v___x_1492_; lean_object* v___y_1494_; lean_object* v___y_1495_; lean_object* v___y_1496_; lean_object* v___y_1497_; lean_object* v___y_1498_; lean_object* v___y_1499_; lean_object* v___y_1500_; lean_object* v___y_1501_; lean_object* v___y_1502_; lean_object* v___y_1524_; lean_object* v___x_1528_; uint8_t v___x_1529_; 
v_maxDepth_x3f_1488_ = lean_ctor_get(v___y_1419_, 0);
v_forwardHypData_1489_ = lean_ctor_get(v___y_1419_, 1);
v___x_1490_ = lean_unsigned_to_nat(1u);
v___x_1491_ = lean_unsigned_to_nat(0u);
v___x_1492_ = l_Lean_LocalDecl_fvarId(v_val_1466_);
v___x_1528_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__4___redArg(v_forwardHypData_1489_, v___x_1492_, v___x_1491_);
v___x_1529_ = lean_nat_dec_le(v_currentMaxHypDepth_1414_, v___x_1528_);
if (v___x_1529_ == 0)
{
lean_dec(v___x_1528_);
lean_inc(v_currentMaxHypDepth_1414_);
v___y_1524_ = v_currentMaxHypDepth_1414_;
goto v___jp_1523_;
}
else
{
v___y_1524_ = v___x_1528_;
goto v___jp_1523_;
}
v___jp_1493_:
{
lean_object* v___x_1503_; 
v___x_1503_ = l_Lean_Meta_saveState___redArg(v___y_1500_, v___y_1502_);
if (lean_obj_tag(v___x_1503_) == 0)
{
lean_object* v_a_1504_; lean_object* v___x_1505_; lean_object* v___x_1506_; 
v_a_1504_ = lean_ctor_get(v___x_1503_, 0);
lean_inc(v_a_1504_);
lean_dec_ref_known(v___x_1503_, 1);
v___x_1505_ = l_Lean_LocalDecl_type(v_val_1466_);
lean_inc_ref(v_a_1407_);
v___x_1506_ = l_Lean_Meta_isExprDefEq(v___x_1505_, v_a_1407_, v___y_1499_, v___y_1500_, v___y_1501_, v___y_1502_);
if (lean_obj_tag(v___x_1506_) == 0)
{
lean_object* v_a_1507_; uint8_t v___x_1508_; 
v_a_1507_ = lean_ctor_get(v___x_1506_, 0);
lean_inc(v_a_1507_);
lean_dec_ref_known(v___x_1506_, 1);
v___x_1508_ = lean_unbox(v_a_1507_);
lean_dec(v_a_1507_);
if (v___x_1508_ == 0)
{
lean_dec(v___y_1494_);
lean_dec(v___x_1492_);
v___y_1469_ = v___y_1502_;
v___y_1470_ = v___y_1500_;
v___y_1471_ = v_a_1504_;
goto v___jp_1468_;
}
else
{
lean_object* v___x_1509_; lean_object* v___x_1510_; 
lean_inc(v___x_1492_);
v___x_1509_ = l_Lean_mkFVar(v___x_1492_);
lean_inc(v_mvarId_1408_);
v___x_1510_ = lp_aesop_Lean_MVarId_assign___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__6___redArg(v_mvarId_1408_, v___x_1509_, v___y_1500_);
if (lean_obj_tag(v___x_1510_) == 0)
{
lean_object* v___x_1511_; lean_object* v___x_1512_; lean_object* v___x_1513_; 
lean_dec_ref_known(v___x_1510_, 1);
lean_inc_ref(v_currentUsedHyps_1409_);
v___x_1511_ = lean_array_push(v_currentUsedHyps_1409_, v___x_1492_);
v___x_1512_ = lean_nat_add(v_i_1410_, v___x_1490_);
lean_inc_ref(v_instMVars_1412_);
lean_inc_ref(v_app_1411_);
v___x_1513_ = lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop(v_app_1411_, v_instMVars_1412_, v_immediateMVars_1413_, v___x_1512_, v___y_1494_, v___x_1511_, v___y_1495_, v___y_1496_, v___y_1497_, v___y_1498_, v___y_1499_, v___y_1500_, v___y_1501_, v___y_1502_);
v___y_1482_ = v___y_1502_;
v___y_1483_ = v___y_1500_;
v___y_1484_ = v_a_1504_;
v___y_1485_ = v___x_1513_;
goto v___jp_1481_;
}
else
{
lean_dec(v___y_1494_);
lean_dec(v___x_1492_);
v___y_1482_ = v___y_1502_;
v___y_1483_ = v___y_1500_;
v___y_1484_ = v_a_1504_;
v___y_1485_ = v___x_1510_;
goto v___jp_1481_;
}
}
}
else
{
lean_object* v_a_1514_; 
lean_dec(v___y_1494_);
lean_dec(v___x_1492_);
lean_del_object(v___x_1454_);
lean_dec(v_currentMaxHypDepth_1414_);
lean_dec_ref(v_instMVars_1412_);
lean_dec_ref(v_app_1411_);
lean_dec_ref(v_currentUsedHyps_1409_);
lean_dec(v_mvarId_1408_);
lean_dec_ref(v_a_1407_);
v_a_1514_ = lean_ctor_get(v___x_1506_, 0);
lean_inc(v_a_1514_);
lean_dec_ref_known(v___x_1506_, 1);
v___y_1429_ = v___y_1502_;
v___y_1430_ = v___y_1500_;
v___y_1431_ = v_a_1504_;
v_a_1432_ = v_a_1514_;
goto v___jp_1428_;
}
}
else
{
lean_object* v_a_1515_; lean_object* v___x_1517_; uint8_t v_isShared_1518_; uint8_t v_isSharedCheck_1522_; 
lean_dec(v___y_1494_);
lean_dec(v___x_1492_);
lean_del_object(v___x_1454_);
lean_dec(v_currentMaxHypDepth_1414_);
lean_dec_ref(v_instMVars_1412_);
lean_dec_ref(v_app_1411_);
lean_dec_ref(v_currentUsedHyps_1409_);
lean_dec(v_mvarId_1408_);
lean_dec_ref(v_a_1407_);
v_a_1515_ = lean_ctor_get(v___x_1503_, 0);
v_isSharedCheck_1522_ = !lean_is_exclusive(v___x_1503_);
if (v_isSharedCheck_1522_ == 0)
{
v___x_1517_ = v___x_1503_;
v_isShared_1518_ = v_isSharedCheck_1522_;
goto v_resetjp_1516_;
}
else
{
lean_inc(v_a_1515_);
lean_dec(v___x_1503_);
v___x_1517_ = lean_box(0);
v_isShared_1518_ = v_isSharedCheck_1522_;
goto v_resetjp_1516_;
}
v_resetjp_1516_:
{
lean_object* v___x_1520_; 
if (v_isShared_1518_ == 0)
{
v___x_1520_ = v___x_1517_;
goto v_reusejp_1519_;
}
else
{
lean_object* v_reuseFailAlloc_1521_; 
v_reuseFailAlloc_1521_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1521_, 0, v_a_1515_);
v___x_1520_ = v_reuseFailAlloc_1521_;
goto v_reusejp_1519_;
}
v_reusejp_1519_:
{
return v___x_1520_;
}
}
}
}
v___jp_1523_:
{
if (lean_obj_tag(v_maxDepth_x3f_1488_) == 1)
{
lean_object* v_val_1525_; lean_object* v___x_1526_; uint8_t v___x_1527_; 
v_val_1525_ = lean_ctor_get(v_maxDepth_x3f_1488_, 0);
v___x_1526_ = lean_nat_add(v___y_1524_, v___x_1490_);
v___x_1527_ = lean_nat_dec_lt(v_val_1525_, v___x_1526_);
lean_dec(v___x_1526_);
if (v___x_1527_ == 0)
{
v___y_1494_ = v___y_1524_;
v___y_1495_ = v___y_1419_;
v___y_1496_ = v___y_1420_;
v___y_1497_ = v___y_1421_;
v___y_1498_ = v___y_1422_;
v___y_1499_ = v___y_1423_;
v___y_1500_ = v___y_1424_;
v___y_1501_ = v___y_1425_;
v___y_1502_ = v___y_1426_;
goto v___jp_1493_;
}
else
{
lean_dec(v___y_1524_);
lean_dec(v___x_1492_);
v_a_1458_ = v___x_1467_;
goto v___jp_1457_;
}
}
else
{
v___y_1494_ = v___y_1524_;
v___y_1495_ = v___y_1419_;
v___y_1496_ = v___y_1420_;
v___y_1497_ = v___y_1421_;
v___y_1498_ = v___y_1422_;
v___y_1499_ = v___y_1423_;
v___y_1500_ = v___y_1424_;
v___y_1501_ = v___y_1425_;
v___y_1502_ = v___y_1426_;
goto v___jp_1493_;
}
}
}
else
{
v_a_1458_ = v___x_1467_;
goto v___jp_1457_;
}
v___jp_1468_:
{
lean_object* v___x_1472_; 
v___x_1472_ = l_Lean_Meta_SavedState_restore___redArg(v___y_1471_, v___y_1470_, v___y_1469_);
lean_dec_ref(v___y_1471_);
if (lean_obj_tag(v___x_1472_) == 0)
{
lean_dec_ref_known(v___x_1472_, 1);
v_a_1458_ = v___x_1467_;
goto v___jp_1457_;
}
else
{
lean_object* v_a_1473_; lean_object* v___x_1475_; uint8_t v_isShared_1476_; uint8_t v_isSharedCheck_1480_; 
lean_del_object(v___x_1454_);
lean_dec(v_currentMaxHypDepth_1414_);
lean_dec_ref(v_instMVars_1412_);
lean_dec_ref(v_app_1411_);
lean_dec_ref(v_currentUsedHyps_1409_);
lean_dec(v_mvarId_1408_);
lean_dec_ref(v_a_1407_);
v_a_1473_ = lean_ctor_get(v___x_1472_, 0);
v_isSharedCheck_1480_ = !lean_is_exclusive(v___x_1472_);
if (v_isSharedCheck_1480_ == 0)
{
v___x_1475_ = v___x_1472_;
v_isShared_1476_ = v_isSharedCheck_1480_;
goto v_resetjp_1474_;
}
else
{
lean_inc(v_a_1473_);
lean_dec(v___x_1472_);
v___x_1475_ = lean_box(0);
v_isShared_1476_ = v_isSharedCheck_1480_;
goto v_resetjp_1474_;
}
v_resetjp_1474_:
{
lean_object* v___x_1478_; 
if (v_isShared_1476_ == 0)
{
v___x_1478_ = v___x_1475_;
goto v_reusejp_1477_;
}
else
{
lean_object* v_reuseFailAlloc_1479_; 
v_reuseFailAlloc_1479_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1479_, 0, v_a_1473_);
v___x_1478_ = v_reuseFailAlloc_1479_;
goto v_reusejp_1477_;
}
v_reusejp_1477_:
{
return v___x_1478_;
}
}
}
}
v___jp_1481_:
{
if (lean_obj_tag(v___y_1485_) == 0)
{
lean_dec_ref_known(v___y_1485_, 1);
v___y_1469_ = v___y_1482_;
v___y_1470_ = v___y_1483_;
v___y_1471_ = v___y_1484_;
goto v___jp_1468_;
}
else
{
lean_object* v_a_1486_; 
lean_del_object(v___x_1454_);
lean_dec(v_currentMaxHypDepth_1414_);
lean_dec_ref(v_instMVars_1412_);
lean_dec_ref(v_app_1411_);
lean_dec_ref(v_currentUsedHyps_1409_);
lean_dec(v_mvarId_1408_);
lean_dec_ref(v_a_1407_);
v_a_1486_ = lean_ctor_get(v___y_1485_, 0);
lean_inc(v_a_1486_);
lean_dec_ref_known(v___y_1485_, 1);
v___y_1429_ = v___y_1482_;
v___y_1430_ = v___y_1483_;
v___y_1431_ = v___y_1484_;
v_a_1432_ = v_a_1486_;
goto v___jp_1428_;
}
}
}
v___jp_1457_:
{
lean_object* v___x_1460_; 
if (v_isShared_1455_ == 0)
{
lean_ctor_set(v___x_1454_, 1, v_a_1458_);
lean_ctor_set(v___x_1454_, 0, v___x_1456_);
v___x_1460_ = v___x_1454_;
goto v_reusejp_1459_;
}
else
{
lean_object* v_reuseFailAlloc_1464_; 
v_reuseFailAlloc_1464_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1464_, 0, v___x_1456_);
lean_ctor_set(v_reuseFailAlloc_1464_, 1, v_a_1458_);
v___x_1460_ = v_reuseFailAlloc_1464_;
goto v_reusejp_1459_;
}
v_reusejp_1459_:
{
size_t v___x_1461_; size_t v___x_1462_; lean_object* v___x_1463_; 
v___x_1461_ = ((size_t)1ULL);
v___x_1462_ = lean_usize_add(v_i_1417_, v___x_1461_);
v___x_1463_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__7_spec__14_spec__20_spec__26(v_a_1407_, v_mvarId_1408_, v_currentUsedHyps_1409_, v_i_1410_, v_app_1411_, v_instMVars_1412_, v_immediateMVars_1413_, v_currentMaxHypDepth_1414_, v_as_1415_, v_sz_1416_, v___x_1462_, v___x_1460_, v___y_1419_, v___y_1420_, v___y_1421_, v___y_1422_, v___y_1423_, v___y_1424_, v___y_1425_, v___y_1426_);
return v___x_1463_;
}
}
}
}
v___jp_1428_:
{
lean_object* v___x_1433_; 
v___x_1433_ = l_Lean_Meta_SavedState_restore___redArg(v___y_1431_, v___y_1430_, v___y_1429_);
lean_dec_ref(v___y_1431_);
if (lean_obj_tag(v___x_1433_) == 0)
{
lean_object* v___x_1435_; uint8_t v_isShared_1436_; uint8_t v_isSharedCheck_1440_; 
v_isSharedCheck_1440_ = !lean_is_exclusive(v___x_1433_);
if (v_isSharedCheck_1440_ == 0)
{
lean_object* v_unused_1441_; 
v_unused_1441_ = lean_ctor_get(v___x_1433_, 0);
lean_dec(v_unused_1441_);
v___x_1435_ = v___x_1433_;
v_isShared_1436_ = v_isSharedCheck_1440_;
goto v_resetjp_1434_;
}
else
{
lean_dec(v___x_1433_);
v___x_1435_ = lean_box(0);
v_isShared_1436_ = v_isSharedCheck_1440_;
goto v_resetjp_1434_;
}
v_resetjp_1434_:
{
lean_object* v___x_1438_; 
if (v_isShared_1436_ == 0)
{
lean_ctor_set_tag(v___x_1435_, 1);
lean_ctor_set(v___x_1435_, 0, v_a_1432_);
v___x_1438_ = v___x_1435_;
goto v_reusejp_1437_;
}
else
{
lean_object* v_reuseFailAlloc_1439_; 
v_reuseFailAlloc_1439_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1439_, 0, v_a_1432_);
v___x_1438_ = v_reuseFailAlloc_1439_;
goto v_reusejp_1437_;
}
v_reusejp_1437_:
{
return v___x_1438_;
}
}
}
else
{
lean_object* v_a_1442_; lean_object* v___x_1444_; uint8_t v_isShared_1445_; uint8_t v_isSharedCheck_1449_; 
lean_dec_ref(v_a_1432_);
v_a_1442_ = lean_ctor_get(v___x_1433_, 0);
v_isSharedCheck_1449_ = !lean_is_exclusive(v___x_1433_);
if (v_isSharedCheck_1449_ == 0)
{
v___x_1444_ = v___x_1433_;
v_isShared_1445_ = v_isSharedCheck_1449_;
goto v_resetjp_1443_;
}
else
{
lean_inc(v_a_1442_);
lean_dec(v___x_1433_);
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
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__7_spec__14(lean_object* v_init_1532_, lean_object* v_a_1533_, lean_object* v_mvarId_1534_, lean_object* v_currentUsedHyps_1535_, lean_object* v_i_1536_, lean_object* v_app_1537_, lean_object* v_instMVars_1538_, lean_object* v_immediateMVars_1539_, lean_object* v_currentMaxHypDepth_1540_, lean_object* v_n_1541_, lean_object* v_b_1542_, lean_object* v___y_1543_, lean_object* v___y_1544_, lean_object* v___y_1545_, lean_object* v___y_1546_, lean_object* v___y_1547_, lean_object* v___y_1548_, lean_object* v___y_1549_, lean_object* v___y_1550_){
_start:
{
if (lean_obj_tag(v_n_1541_) == 0)
{
lean_object* v_cs_1552_; lean_object* v___x_1553_; lean_object* v___x_1554_; size_t v_sz_1555_; size_t v___x_1556_; lean_object* v___x_1557_; 
v_cs_1552_ = lean_ctor_get(v_n_1541_, 0);
v___x_1553_ = lean_box(0);
v___x_1554_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1554_, 0, v___x_1553_);
lean_ctor_set(v___x_1554_, 1, v_b_1542_);
v_sz_1555_ = lean_array_size(v_cs_1552_);
v___x_1556_ = ((size_t)0ULL);
v___x_1557_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__7_spec__14_spec__19(v_init_1532_, v_a_1533_, v_mvarId_1534_, v_currentUsedHyps_1535_, v_i_1536_, v_app_1537_, v_instMVars_1538_, v_immediateMVars_1539_, v_currentMaxHypDepth_1540_, v_cs_1552_, v_sz_1555_, v___x_1556_, v___x_1554_, v___y_1543_, v___y_1544_, v___y_1545_, v___y_1546_, v___y_1547_, v___y_1548_, v___y_1549_, v___y_1550_);
if (lean_obj_tag(v___x_1557_) == 0)
{
lean_object* v_a_1558_; lean_object* v___x_1560_; uint8_t v_isShared_1561_; uint8_t v_isSharedCheck_1572_; 
v_a_1558_ = lean_ctor_get(v___x_1557_, 0);
v_isSharedCheck_1572_ = !lean_is_exclusive(v___x_1557_);
if (v_isSharedCheck_1572_ == 0)
{
v___x_1560_ = v___x_1557_;
v_isShared_1561_ = v_isSharedCheck_1572_;
goto v_resetjp_1559_;
}
else
{
lean_inc(v_a_1558_);
lean_dec(v___x_1557_);
v___x_1560_ = lean_box(0);
v_isShared_1561_ = v_isSharedCheck_1572_;
goto v_resetjp_1559_;
}
v_resetjp_1559_:
{
lean_object* v_fst_1562_; 
v_fst_1562_ = lean_ctor_get(v_a_1558_, 0);
if (lean_obj_tag(v_fst_1562_) == 0)
{
lean_object* v_snd_1563_; lean_object* v___x_1564_; lean_object* v___x_1566_; 
v_snd_1563_ = lean_ctor_get(v_a_1558_, 1);
lean_inc(v_snd_1563_);
lean_dec(v_a_1558_);
v___x_1564_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1564_, 0, v_snd_1563_);
if (v_isShared_1561_ == 0)
{
lean_ctor_set(v___x_1560_, 0, v___x_1564_);
v___x_1566_ = v___x_1560_;
goto v_reusejp_1565_;
}
else
{
lean_object* v_reuseFailAlloc_1567_; 
v_reuseFailAlloc_1567_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1567_, 0, v___x_1564_);
v___x_1566_ = v_reuseFailAlloc_1567_;
goto v_reusejp_1565_;
}
v_reusejp_1565_:
{
return v___x_1566_;
}
}
else
{
lean_object* v_val_1568_; lean_object* v___x_1570_; 
lean_inc_ref(v_fst_1562_);
lean_dec(v_a_1558_);
v_val_1568_ = lean_ctor_get(v_fst_1562_, 0);
lean_inc(v_val_1568_);
lean_dec_ref_known(v_fst_1562_, 1);
if (v_isShared_1561_ == 0)
{
lean_ctor_set(v___x_1560_, 0, v_val_1568_);
v___x_1570_ = v___x_1560_;
goto v_reusejp_1569_;
}
else
{
lean_object* v_reuseFailAlloc_1571_; 
v_reuseFailAlloc_1571_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1571_, 0, v_val_1568_);
v___x_1570_ = v_reuseFailAlloc_1571_;
goto v_reusejp_1569_;
}
v_reusejp_1569_:
{
return v___x_1570_;
}
}
}
}
else
{
lean_object* v_a_1573_; lean_object* v___x_1575_; uint8_t v_isShared_1576_; uint8_t v_isSharedCheck_1580_; 
v_a_1573_ = lean_ctor_get(v___x_1557_, 0);
v_isSharedCheck_1580_ = !lean_is_exclusive(v___x_1557_);
if (v_isSharedCheck_1580_ == 0)
{
v___x_1575_ = v___x_1557_;
v_isShared_1576_ = v_isSharedCheck_1580_;
goto v_resetjp_1574_;
}
else
{
lean_inc(v_a_1573_);
lean_dec(v___x_1557_);
v___x_1575_ = lean_box(0);
v_isShared_1576_ = v_isSharedCheck_1580_;
goto v_resetjp_1574_;
}
v_resetjp_1574_:
{
lean_object* v___x_1578_; 
if (v_isShared_1576_ == 0)
{
v___x_1578_ = v___x_1575_;
goto v_reusejp_1577_;
}
else
{
lean_object* v_reuseFailAlloc_1579_; 
v_reuseFailAlloc_1579_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1579_, 0, v_a_1573_);
v___x_1578_ = v_reuseFailAlloc_1579_;
goto v_reusejp_1577_;
}
v_reusejp_1577_:
{
return v___x_1578_;
}
}
}
}
else
{
lean_object* v_vs_1581_; lean_object* v___x_1582_; lean_object* v___x_1583_; size_t v_sz_1584_; size_t v___x_1585_; lean_object* v___x_1586_; 
v_vs_1581_ = lean_ctor_get(v_n_1541_, 0);
v___x_1582_ = lean_box(0);
v___x_1583_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1583_, 0, v___x_1582_);
lean_ctor_set(v___x_1583_, 1, v_b_1542_);
v_sz_1584_ = lean_array_size(v_vs_1581_);
v___x_1585_ = ((size_t)0ULL);
v___x_1586_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__7_spec__14_spec__20(v_a_1533_, v_mvarId_1534_, v_currentUsedHyps_1535_, v_i_1536_, v_app_1537_, v_instMVars_1538_, v_immediateMVars_1539_, v_currentMaxHypDepth_1540_, v_vs_1581_, v_sz_1584_, v___x_1585_, v___x_1583_, v___y_1543_, v___y_1544_, v___y_1545_, v___y_1546_, v___y_1547_, v___y_1548_, v___y_1549_, v___y_1550_);
if (lean_obj_tag(v___x_1586_) == 0)
{
lean_object* v_a_1587_; lean_object* v___x_1589_; uint8_t v_isShared_1590_; uint8_t v_isSharedCheck_1601_; 
v_a_1587_ = lean_ctor_get(v___x_1586_, 0);
v_isSharedCheck_1601_ = !lean_is_exclusive(v___x_1586_);
if (v_isSharedCheck_1601_ == 0)
{
v___x_1589_ = v___x_1586_;
v_isShared_1590_ = v_isSharedCheck_1601_;
goto v_resetjp_1588_;
}
else
{
lean_inc(v_a_1587_);
lean_dec(v___x_1586_);
v___x_1589_ = lean_box(0);
v_isShared_1590_ = v_isSharedCheck_1601_;
goto v_resetjp_1588_;
}
v_resetjp_1588_:
{
lean_object* v_fst_1591_; 
v_fst_1591_ = lean_ctor_get(v_a_1587_, 0);
if (lean_obj_tag(v_fst_1591_) == 0)
{
lean_object* v_snd_1592_; lean_object* v___x_1593_; lean_object* v___x_1595_; 
v_snd_1592_ = lean_ctor_get(v_a_1587_, 1);
lean_inc(v_snd_1592_);
lean_dec(v_a_1587_);
v___x_1593_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1593_, 0, v_snd_1592_);
if (v_isShared_1590_ == 0)
{
lean_ctor_set(v___x_1589_, 0, v___x_1593_);
v___x_1595_ = v___x_1589_;
goto v_reusejp_1594_;
}
else
{
lean_object* v_reuseFailAlloc_1596_; 
v_reuseFailAlloc_1596_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1596_, 0, v___x_1593_);
v___x_1595_ = v_reuseFailAlloc_1596_;
goto v_reusejp_1594_;
}
v_reusejp_1594_:
{
return v___x_1595_;
}
}
else
{
lean_object* v_val_1597_; lean_object* v___x_1599_; 
lean_inc_ref(v_fst_1591_);
lean_dec(v_a_1587_);
v_val_1597_ = lean_ctor_get(v_fst_1591_, 0);
lean_inc(v_val_1597_);
lean_dec_ref_known(v_fst_1591_, 1);
if (v_isShared_1590_ == 0)
{
lean_ctor_set(v___x_1589_, 0, v_val_1597_);
v___x_1599_ = v___x_1589_;
goto v_reusejp_1598_;
}
else
{
lean_object* v_reuseFailAlloc_1600_; 
v_reuseFailAlloc_1600_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1600_, 0, v_val_1597_);
v___x_1599_ = v_reuseFailAlloc_1600_;
goto v_reusejp_1598_;
}
v_reusejp_1598_:
{
return v___x_1599_;
}
}
}
}
else
{
lean_object* v_a_1602_; lean_object* v___x_1604_; uint8_t v_isShared_1605_; uint8_t v_isSharedCheck_1609_; 
v_a_1602_ = lean_ctor_get(v___x_1586_, 0);
v_isSharedCheck_1609_ = !lean_is_exclusive(v___x_1586_);
if (v_isSharedCheck_1609_ == 0)
{
v___x_1604_ = v___x_1586_;
v_isShared_1605_ = v_isSharedCheck_1609_;
goto v_resetjp_1603_;
}
else
{
lean_inc(v_a_1602_);
lean_dec(v___x_1586_);
v___x_1604_ = lean_box(0);
v_isShared_1605_ = v_isSharedCheck_1609_;
goto v_resetjp_1603_;
}
v_resetjp_1603_:
{
lean_object* v___x_1607_; 
if (v_isShared_1605_ == 0)
{
v___x_1607_ = v___x_1604_;
goto v_reusejp_1606_;
}
else
{
lean_object* v_reuseFailAlloc_1608_; 
v_reuseFailAlloc_1608_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1608_, 0, v_a_1602_);
v___x_1607_ = v_reuseFailAlloc_1608_;
goto v_reusejp_1606_;
}
v_reusejp_1606_:
{
return v___x_1607_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__7_spec__14_spec__19(lean_object* v_init_1610_, lean_object* v_a_1611_, lean_object* v_mvarId_1612_, lean_object* v_currentUsedHyps_1613_, lean_object* v_i_1614_, lean_object* v_app_1615_, lean_object* v_instMVars_1616_, lean_object* v_immediateMVars_1617_, lean_object* v_currentMaxHypDepth_1618_, lean_object* v_as_1619_, size_t v_sz_1620_, size_t v_i_1621_, lean_object* v_b_1622_, lean_object* v___y_1623_, lean_object* v___y_1624_, lean_object* v___y_1625_, lean_object* v___y_1626_, lean_object* v___y_1627_, lean_object* v___y_1628_, lean_object* v___y_1629_, lean_object* v___y_1630_){
_start:
{
uint8_t v___x_1632_; 
v___x_1632_ = lean_usize_dec_lt(v_i_1621_, v_sz_1620_);
if (v___x_1632_ == 0)
{
lean_object* v___x_1633_; 
lean_dec(v_currentMaxHypDepth_1618_);
lean_dec_ref(v_instMVars_1616_);
lean_dec_ref(v_app_1615_);
lean_dec_ref(v_currentUsedHyps_1613_);
lean_dec(v_mvarId_1612_);
lean_dec_ref(v_a_1611_);
v___x_1633_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1633_, 0, v_b_1622_);
return v___x_1633_;
}
else
{
lean_object* v_snd_1634_; lean_object* v___x_1636_; uint8_t v_isShared_1637_; uint8_t v_isSharedCheck_1668_; 
v_snd_1634_ = lean_ctor_get(v_b_1622_, 1);
v_isSharedCheck_1668_ = !lean_is_exclusive(v_b_1622_);
if (v_isSharedCheck_1668_ == 0)
{
lean_object* v_unused_1669_; 
v_unused_1669_ = lean_ctor_get(v_b_1622_, 0);
lean_dec(v_unused_1669_);
v___x_1636_ = v_b_1622_;
v_isShared_1637_ = v_isSharedCheck_1668_;
goto v_resetjp_1635_;
}
else
{
lean_inc(v_snd_1634_);
lean_dec(v_b_1622_);
v___x_1636_ = lean_box(0);
v_isShared_1637_ = v_isSharedCheck_1668_;
goto v_resetjp_1635_;
}
v_resetjp_1635_:
{
lean_object* v_a_1638_; lean_object* v___x_1639_; 
v_a_1638_ = lean_array_uget_borrowed(v_as_1619_, v_i_1621_);
lean_inc(v_snd_1634_);
lean_inc(v_currentMaxHypDepth_1618_);
lean_inc_ref(v_instMVars_1616_);
lean_inc_ref(v_app_1615_);
lean_inc_ref(v_currentUsedHyps_1613_);
lean_inc(v_mvarId_1612_);
lean_inc_ref(v_a_1611_);
v___x_1639_ = lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__7_spec__14(v_init_1610_, v_a_1611_, v_mvarId_1612_, v_currentUsedHyps_1613_, v_i_1614_, v_app_1615_, v_instMVars_1616_, v_immediateMVars_1617_, v_currentMaxHypDepth_1618_, v_a_1638_, v_snd_1634_, v___y_1623_, v___y_1624_, v___y_1625_, v___y_1626_, v___y_1627_, v___y_1628_, v___y_1629_, v___y_1630_);
if (lean_obj_tag(v___x_1639_) == 0)
{
lean_object* v_a_1640_; lean_object* v___x_1642_; uint8_t v_isShared_1643_; uint8_t v_isSharedCheck_1659_; 
v_a_1640_ = lean_ctor_get(v___x_1639_, 0);
v_isSharedCheck_1659_ = !lean_is_exclusive(v___x_1639_);
if (v_isSharedCheck_1659_ == 0)
{
v___x_1642_ = v___x_1639_;
v_isShared_1643_ = v_isSharedCheck_1659_;
goto v_resetjp_1641_;
}
else
{
lean_inc(v_a_1640_);
lean_dec(v___x_1639_);
v___x_1642_ = lean_box(0);
v_isShared_1643_ = v_isSharedCheck_1659_;
goto v_resetjp_1641_;
}
v_resetjp_1641_:
{
if (lean_obj_tag(v_a_1640_) == 0)
{
lean_object* v___x_1644_; lean_object* v___x_1646_; 
lean_dec(v_currentMaxHypDepth_1618_);
lean_dec_ref(v_instMVars_1616_);
lean_dec_ref(v_app_1615_);
lean_dec_ref(v_currentUsedHyps_1613_);
lean_dec(v_mvarId_1612_);
lean_dec_ref(v_a_1611_);
v___x_1644_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1644_, 0, v_a_1640_);
if (v_isShared_1637_ == 0)
{
lean_ctor_set(v___x_1636_, 0, v___x_1644_);
v___x_1646_ = v___x_1636_;
goto v_reusejp_1645_;
}
else
{
lean_object* v_reuseFailAlloc_1650_; 
v_reuseFailAlloc_1650_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1650_, 0, v___x_1644_);
lean_ctor_set(v_reuseFailAlloc_1650_, 1, v_snd_1634_);
v___x_1646_ = v_reuseFailAlloc_1650_;
goto v_reusejp_1645_;
}
v_reusejp_1645_:
{
lean_object* v___x_1648_; 
if (v_isShared_1643_ == 0)
{
lean_ctor_set(v___x_1642_, 0, v___x_1646_);
v___x_1648_ = v___x_1642_;
goto v_reusejp_1647_;
}
else
{
lean_object* v_reuseFailAlloc_1649_; 
v_reuseFailAlloc_1649_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1649_, 0, v___x_1646_);
v___x_1648_ = v_reuseFailAlloc_1649_;
goto v_reusejp_1647_;
}
v_reusejp_1647_:
{
return v___x_1648_;
}
}
}
else
{
lean_object* v_a_1651_; lean_object* v___x_1652_; lean_object* v___x_1654_; 
lean_del_object(v___x_1642_);
lean_dec(v_snd_1634_);
v_a_1651_ = lean_ctor_get(v_a_1640_, 0);
lean_inc(v_a_1651_);
lean_dec_ref_known(v_a_1640_, 1);
v___x_1652_ = lean_box(0);
if (v_isShared_1637_ == 0)
{
lean_ctor_set(v___x_1636_, 1, v_a_1651_);
lean_ctor_set(v___x_1636_, 0, v___x_1652_);
v___x_1654_ = v___x_1636_;
goto v_reusejp_1653_;
}
else
{
lean_object* v_reuseFailAlloc_1658_; 
v_reuseFailAlloc_1658_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1658_, 0, v___x_1652_);
lean_ctor_set(v_reuseFailAlloc_1658_, 1, v_a_1651_);
v___x_1654_ = v_reuseFailAlloc_1658_;
goto v_reusejp_1653_;
}
v_reusejp_1653_:
{
size_t v___x_1655_; size_t v___x_1656_; 
v___x_1655_ = ((size_t)1ULL);
v___x_1656_ = lean_usize_add(v_i_1621_, v___x_1655_);
v_i_1621_ = v___x_1656_;
v_b_1622_ = v___x_1654_;
goto _start;
}
}
}
}
else
{
lean_object* v_a_1660_; lean_object* v___x_1662_; uint8_t v_isShared_1663_; uint8_t v_isSharedCheck_1667_; 
lean_del_object(v___x_1636_);
lean_dec(v_snd_1634_);
lean_dec(v_currentMaxHypDepth_1618_);
lean_dec_ref(v_instMVars_1616_);
lean_dec_ref(v_app_1615_);
lean_dec_ref(v_currentUsedHyps_1613_);
lean_dec(v_mvarId_1612_);
lean_dec_ref(v_a_1611_);
v_a_1660_ = lean_ctor_get(v___x_1639_, 0);
v_isSharedCheck_1667_ = !lean_is_exclusive(v___x_1639_);
if (v_isSharedCheck_1667_ == 0)
{
v___x_1662_ = v___x_1639_;
v_isShared_1663_ = v_isSharedCheck_1667_;
goto v_resetjp_1661_;
}
else
{
lean_inc(v_a_1660_);
lean_dec(v___x_1639_);
v___x_1662_ = lean_box(0);
v_isShared_1663_ = v_isSharedCheck_1667_;
goto v_resetjp_1661_;
}
v_resetjp_1661_:
{
lean_object* v___x_1665_; 
if (v_isShared_1663_ == 0)
{
v___x_1665_ = v___x_1662_;
goto v_reusejp_1664_;
}
else
{
lean_object* v_reuseFailAlloc_1666_; 
v_reuseFailAlloc_1666_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1666_, 0, v_a_1660_);
v___x_1665_ = v_reuseFailAlloc_1666_;
goto v_reusejp_1664_;
}
v_reusejp_1664_:
{
return v___x_1665_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__7_spec__14_spec__19___boxed(lean_object** _args){
lean_object* v_init_1670_ = _args[0];
lean_object* v_a_1671_ = _args[1];
lean_object* v_mvarId_1672_ = _args[2];
lean_object* v_currentUsedHyps_1673_ = _args[3];
lean_object* v_i_1674_ = _args[4];
lean_object* v_app_1675_ = _args[5];
lean_object* v_instMVars_1676_ = _args[6];
lean_object* v_immediateMVars_1677_ = _args[7];
lean_object* v_currentMaxHypDepth_1678_ = _args[8];
lean_object* v_as_1679_ = _args[9];
lean_object* v_sz_1680_ = _args[10];
lean_object* v_i_1681_ = _args[11];
lean_object* v_b_1682_ = _args[12];
lean_object* v___y_1683_ = _args[13];
lean_object* v___y_1684_ = _args[14];
lean_object* v___y_1685_ = _args[15];
lean_object* v___y_1686_ = _args[16];
lean_object* v___y_1687_ = _args[17];
lean_object* v___y_1688_ = _args[18];
lean_object* v___y_1689_ = _args[19];
lean_object* v___y_1690_ = _args[20];
lean_object* v___y_1691_ = _args[21];
_start:
{
size_t v_sz_boxed_1692_; size_t v_i_boxed_1693_; lean_object* v_res_1694_; 
v_sz_boxed_1692_ = lean_unbox_usize(v_sz_1680_);
lean_dec(v_sz_1680_);
v_i_boxed_1693_ = lean_unbox_usize(v_i_1681_);
lean_dec(v_i_1681_);
v_res_1694_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__7_spec__14_spec__19(v_init_1670_, v_a_1671_, v_mvarId_1672_, v_currentUsedHyps_1673_, v_i_1674_, v_app_1675_, v_instMVars_1676_, v_immediateMVars_1677_, v_currentMaxHypDepth_1678_, v_as_1679_, v_sz_boxed_1692_, v_i_boxed_1693_, v_b_1682_, v___y_1683_, v___y_1684_, v___y_1685_, v___y_1686_, v___y_1687_, v___y_1688_, v___y_1689_, v___y_1690_);
lean_dec(v___y_1690_);
lean_dec_ref(v___y_1689_);
lean_dec(v___y_1688_);
lean_dec_ref(v___y_1687_);
lean_dec(v___y_1686_);
lean_dec(v___y_1685_);
lean_dec(v___y_1684_);
lean_dec_ref(v___y_1683_);
lean_dec_ref(v_as_1679_);
lean_dec_ref(v_immediateMVars_1677_);
lean_dec(v_i_1674_);
return v_res_1694_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forIn___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__7___boxed(lean_object** _args){
lean_object* v_a_1695_ = _args[0];
lean_object* v_mvarId_1696_ = _args[1];
lean_object* v_currentUsedHyps_1697_ = _args[2];
lean_object* v_i_1698_ = _args[3];
lean_object* v_app_1699_ = _args[4];
lean_object* v_instMVars_1700_ = _args[5];
lean_object* v_immediateMVars_1701_ = _args[6];
lean_object* v_currentMaxHypDepth_1702_ = _args[7];
lean_object* v_t_1703_ = _args[8];
lean_object* v_init_1704_ = _args[9];
lean_object* v___y_1705_ = _args[10];
lean_object* v___y_1706_ = _args[11];
lean_object* v___y_1707_ = _args[12];
lean_object* v___y_1708_ = _args[13];
lean_object* v___y_1709_ = _args[14];
lean_object* v___y_1710_ = _args[15];
lean_object* v___y_1711_ = _args[16];
lean_object* v___y_1712_ = _args[17];
lean_object* v___y_1713_ = _args[18];
_start:
{
lean_object* v_res_1714_; 
v_res_1714_ = lp_aesop_Lean_PersistentArray_forIn___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__7(v_a_1695_, v_mvarId_1696_, v_currentUsedHyps_1697_, v_i_1698_, v_app_1699_, v_instMVars_1700_, v_immediateMVars_1701_, v_currentMaxHypDepth_1702_, v_t_1703_, v_init_1704_, v___y_1705_, v___y_1706_, v___y_1707_, v___y_1708_, v___y_1709_, v___y_1710_, v___y_1711_, v___y_1712_);
lean_dec(v___y_1712_);
lean_dec_ref(v___y_1711_);
lean_dec(v___y_1710_);
lean_dec_ref(v___y_1709_);
lean_dec(v___y_1708_);
lean_dec(v___y_1707_);
lean_dec(v___y_1706_);
lean_dec_ref(v___y_1705_);
lean_dec_ref(v_t_1703_);
lean_dec_ref(v_immediateMVars_1701_);
lean_dec(v_i_1698_);
return v_res_1714_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__7_spec__14___boxed(lean_object** _args){
lean_object* v_init_1715_ = _args[0];
lean_object* v_a_1716_ = _args[1];
lean_object* v_mvarId_1717_ = _args[2];
lean_object* v_currentUsedHyps_1718_ = _args[3];
lean_object* v_i_1719_ = _args[4];
lean_object* v_app_1720_ = _args[5];
lean_object* v_instMVars_1721_ = _args[6];
lean_object* v_immediateMVars_1722_ = _args[7];
lean_object* v_currentMaxHypDepth_1723_ = _args[8];
lean_object* v_n_1724_ = _args[9];
lean_object* v_b_1725_ = _args[10];
lean_object* v___y_1726_ = _args[11];
lean_object* v___y_1727_ = _args[12];
lean_object* v___y_1728_ = _args[13];
lean_object* v___y_1729_ = _args[14];
lean_object* v___y_1730_ = _args[15];
lean_object* v___y_1731_ = _args[16];
lean_object* v___y_1732_ = _args[17];
lean_object* v___y_1733_ = _args[18];
lean_object* v___y_1734_ = _args[19];
_start:
{
lean_object* v_res_1735_; 
v_res_1735_ = lp_aesop_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__7_spec__14(v_init_1715_, v_a_1716_, v_mvarId_1717_, v_currentUsedHyps_1718_, v_i_1719_, v_app_1720_, v_instMVars_1721_, v_immediateMVars_1722_, v_currentMaxHypDepth_1723_, v_n_1724_, v_b_1725_, v___y_1726_, v___y_1727_, v___y_1728_, v___y_1729_, v___y_1730_, v___y_1731_, v___y_1732_, v___y_1733_);
lean_dec(v___y_1733_);
lean_dec_ref(v___y_1732_);
lean_dec(v___y_1731_);
lean_dec_ref(v___y_1730_);
lean_dec(v___y_1729_);
lean_dec(v___y_1728_);
lean_dec(v___y_1727_);
lean_dec_ref(v___y_1726_);
lean_dec_ref(v_n_1724_);
lean_dec_ref(v_immediateMVars_1722_);
lean_dec(v_i_1719_);
return v_res_1735_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__7_spec__15___boxed(lean_object** _args){
lean_object* v_a_1736_ = _args[0];
lean_object* v_mvarId_1737_ = _args[1];
lean_object* v_currentUsedHyps_1738_ = _args[2];
lean_object* v_i_1739_ = _args[3];
lean_object* v_app_1740_ = _args[4];
lean_object* v_instMVars_1741_ = _args[5];
lean_object* v_immediateMVars_1742_ = _args[6];
lean_object* v_currentMaxHypDepth_1743_ = _args[7];
lean_object* v_as_1744_ = _args[8];
lean_object* v_sz_1745_ = _args[9];
lean_object* v_i_1746_ = _args[10];
lean_object* v_b_1747_ = _args[11];
lean_object* v___y_1748_ = _args[12];
lean_object* v___y_1749_ = _args[13];
lean_object* v___y_1750_ = _args[14];
lean_object* v___y_1751_ = _args[15];
lean_object* v___y_1752_ = _args[16];
lean_object* v___y_1753_ = _args[17];
lean_object* v___y_1754_ = _args[18];
lean_object* v___y_1755_ = _args[19];
lean_object* v___y_1756_ = _args[20];
_start:
{
size_t v_sz_boxed_1757_; size_t v_i_boxed_1758_; lean_object* v_res_1759_; 
v_sz_boxed_1757_ = lean_unbox_usize(v_sz_1745_);
lean_dec(v_sz_1745_);
v_i_boxed_1758_ = lean_unbox_usize(v_i_1746_);
lean_dec(v_i_1746_);
v_res_1759_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__7_spec__15(v_a_1736_, v_mvarId_1737_, v_currentUsedHyps_1738_, v_i_1739_, v_app_1740_, v_instMVars_1741_, v_immediateMVars_1742_, v_currentMaxHypDepth_1743_, v_as_1744_, v_sz_boxed_1757_, v_i_boxed_1758_, v_b_1747_, v___y_1748_, v___y_1749_, v___y_1750_, v___y_1751_, v___y_1752_, v___y_1753_, v___y_1754_, v___y_1755_);
lean_dec(v___y_1755_);
lean_dec_ref(v___y_1754_);
lean_dec(v___y_1753_);
lean_dec_ref(v___y_1752_);
lean_dec(v___y_1751_);
lean_dec(v___y_1750_);
lean_dec(v___y_1749_);
lean_dec_ref(v___y_1748_);
lean_dec_ref(v_as_1744_);
lean_dec_ref(v_immediateMVars_1742_);
lean_dec(v_i_1739_);
return v_res_1759_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__7_spec__14_spec__20___boxed(lean_object** _args){
lean_object* v_a_1760_ = _args[0];
lean_object* v_mvarId_1761_ = _args[1];
lean_object* v_currentUsedHyps_1762_ = _args[2];
lean_object* v_i_1763_ = _args[3];
lean_object* v_app_1764_ = _args[4];
lean_object* v_instMVars_1765_ = _args[5];
lean_object* v_immediateMVars_1766_ = _args[6];
lean_object* v_currentMaxHypDepth_1767_ = _args[7];
lean_object* v_as_1768_ = _args[8];
lean_object* v_sz_1769_ = _args[9];
lean_object* v_i_1770_ = _args[10];
lean_object* v_b_1771_ = _args[11];
lean_object* v___y_1772_ = _args[12];
lean_object* v___y_1773_ = _args[13];
lean_object* v___y_1774_ = _args[14];
lean_object* v___y_1775_ = _args[15];
lean_object* v___y_1776_ = _args[16];
lean_object* v___y_1777_ = _args[17];
lean_object* v___y_1778_ = _args[18];
lean_object* v___y_1779_ = _args[19];
lean_object* v___y_1780_ = _args[20];
_start:
{
size_t v_sz_boxed_1781_; size_t v_i_boxed_1782_; lean_object* v_res_1783_; 
v_sz_boxed_1781_ = lean_unbox_usize(v_sz_1769_);
lean_dec(v_sz_1769_);
v_i_boxed_1782_ = lean_unbox_usize(v_i_1770_);
lean_dec(v_i_1770_);
v_res_1783_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__7_spec__14_spec__20(v_a_1760_, v_mvarId_1761_, v_currentUsedHyps_1762_, v_i_1763_, v_app_1764_, v_instMVars_1765_, v_immediateMVars_1766_, v_currentMaxHypDepth_1767_, v_as_1768_, v_sz_boxed_1781_, v_i_boxed_1782_, v_b_1771_, v___y_1772_, v___y_1773_, v___y_1774_, v___y_1775_, v___y_1776_, v___y_1777_, v___y_1778_, v___y_1779_);
lean_dec(v___y_1779_);
lean_dec_ref(v___y_1778_);
lean_dec(v___y_1777_);
lean_dec_ref(v___y_1776_);
lean_dec(v___y_1775_);
lean_dec(v___y_1774_);
lean_dec(v___y_1773_);
lean_dec_ref(v___y_1772_);
lean_dec_ref(v_as_1768_);
lean_dec_ref(v_immediateMVars_1766_);
lean_dec(v_i_1763_);
return v_res_1783_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__7_spec__15_spec__22___boxed(lean_object** _args){
lean_object* v_a_1784_ = _args[0];
lean_object* v_mvarId_1785_ = _args[1];
lean_object* v_currentUsedHyps_1786_ = _args[2];
lean_object* v_i_1787_ = _args[3];
lean_object* v_app_1788_ = _args[4];
lean_object* v_instMVars_1789_ = _args[5];
lean_object* v_immediateMVars_1790_ = _args[6];
lean_object* v_currentMaxHypDepth_1791_ = _args[7];
lean_object* v_as_1792_ = _args[8];
lean_object* v_sz_1793_ = _args[9];
lean_object* v_i_1794_ = _args[10];
lean_object* v_b_1795_ = _args[11];
lean_object* v___y_1796_ = _args[12];
lean_object* v___y_1797_ = _args[13];
lean_object* v___y_1798_ = _args[14];
lean_object* v___y_1799_ = _args[15];
lean_object* v___y_1800_ = _args[16];
lean_object* v___y_1801_ = _args[17];
lean_object* v___y_1802_ = _args[18];
lean_object* v___y_1803_ = _args[19];
lean_object* v___y_1804_ = _args[20];
_start:
{
size_t v_sz_boxed_1805_; size_t v_i_boxed_1806_; lean_object* v_res_1807_; 
v_sz_boxed_1805_ = lean_unbox_usize(v_sz_1793_);
lean_dec(v_sz_1793_);
v_i_boxed_1806_ = lean_unbox_usize(v_i_1794_);
lean_dec(v_i_1794_);
v_res_1807_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__7_spec__15_spec__22(v_a_1784_, v_mvarId_1785_, v_currentUsedHyps_1786_, v_i_1787_, v_app_1788_, v_instMVars_1789_, v_immediateMVars_1790_, v_currentMaxHypDepth_1791_, v_as_1792_, v_sz_boxed_1805_, v_i_boxed_1806_, v_b_1795_, v___y_1796_, v___y_1797_, v___y_1798_, v___y_1799_, v___y_1800_, v___y_1801_, v___y_1802_, v___y_1803_);
lean_dec(v___y_1803_);
lean_dec_ref(v___y_1802_);
lean_dec(v___y_1801_);
lean_dec_ref(v___y_1800_);
lean_dec(v___y_1799_);
lean_dec(v___y_1798_);
lean_dec(v___y_1797_);
lean_dec_ref(v___y_1796_);
lean_dec_ref(v_as_1792_);
lean_dec_ref(v_immediateMVars_1790_);
lean_dec(v_i_1787_);
return v_res_1807_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__7_spec__14_spec__20_spec__26___boxed(lean_object** _args){
lean_object* v_a_1808_ = _args[0];
lean_object* v_mvarId_1809_ = _args[1];
lean_object* v_currentUsedHyps_1810_ = _args[2];
lean_object* v_i_1811_ = _args[3];
lean_object* v_app_1812_ = _args[4];
lean_object* v_instMVars_1813_ = _args[5];
lean_object* v_immediateMVars_1814_ = _args[6];
lean_object* v_currentMaxHypDepth_1815_ = _args[7];
lean_object* v_as_1816_ = _args[8];
lean_object* v_sz_1817_ = _args[9];
lean_object* v_i_1818_ = _args[10];
lean_object* v_b_1819_ = _args[11];
lean_object* v___y_1820_ = _args[12];
lean_object* v___y_1821_ = _args[13];
lean_object* v___y_1822_ = _args[14];
lean_object* v___y_1823_ = _args[15];
lean_object* v___y_1824_ = _args[16];
lean_object* v___y_1825_ = _args[17];
lean_object* v___y_1826_ = _args[18];
lean_object* v___y_1827_ = _args[19];
lean_object* v___y_1828_ = _args[20];
_start:
{
size_t v_sz_boxed_1829_; size_t v_i_boxed_1830_; lean_object* v_res_1831_; 
v_sz_boxed_1829_ = lean_unbox_usize(v_sz_1817_);
lean_dec(v_sz_1817_);
v_i_boxed_1830_ = lean_unbox_usize(v_i_1818_);
lean_dec(v_i_1818_);
v_res_1831_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__7_spec__14_spec__20_spec__26(v_a_1808_, v_mvarId_1809_, v_currentUsedHyps_1810_, v_i_1811_, v_app_1812_, v_instMVars_1813_, v_immediateMVars_1814_, v_currentMaxHypDepth_1815_, v_as_1816_, v_sz_boxed_1829_, v_i_boxed_1830_, v_b_1819_, v___y_1820_, v___y_1821_, v___y_1822_, v___y_1823_, v___y_1824_, v___y_1825_, v___y_1826_, v___y_1827_);
lean_dec(v___y_1827_);
lean_dec_ref(v___y_1826_);
lean_dec(v___y_1825_);
lean_dec_ref(v___y_1824_);
lean_dec(v___y_1823_);
lean_dec(v___y_1822_);
lean_dec(v___y_1821_);
lean_dec_ref(v___y_1820_);
lean_dec_ref(v_as_1816_);
lean_dec_ref(v_immediateMVars_1814_);
lean_dec(v_i_1811_);
return v_res_1831_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop___boxed(lean_object* v_app_1832_, lean_object* v_instMVars_1833_, lean_object* v_immediateMVars_1834_, lean_object* v_i_1835_, lean_object* v_currentMaxHypDepth_1836_, lean_object* v_currentUsedHyps_1837_, lean_object* v_a_1838_, lean_object* v_a_1839_, lean_object* v_a_1840_, lean_object* v_a_1841_, lean_object* v_a_1842_, lean_object* v_a_1843_, lean_object* v_a_1844_, lean_object* v_a_1845_, lean_object* v_a_1846_){
_start:
{
lean_object* v_res_1847_; 
v_res_1847_ = lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop(v_app_1832_, v_instMVars_1833_, v_immediateMVars_1834_, v_i_1835_, v_currentMaxHypDepth_1836_, v_currentUsedHyps_1837_, v_a_1838_, v_a_1839_, v_a_1840_, v_a_1841_, v_a_1842_, v_a_1843_, v_a_1844_, v_a_1845_);
lean_dec(v_a_1845_);
lean_dec_ref(v_a_1844_);
lean_dec(v_a_1843_);
lean_dec_ref(v_a_1842_);
lean_dec(v_a_1841_);
lean_dec(v_a_1840_);
lean_dec(v_a_1839_);
lean_dec_ref(v_a_1838_);
lean_dec_ref(v_immediateMVars_1834_);
return v_res_1847_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__2(lean_object* v_opt_1848_, lean_object* v___y_1849_, lean_object* v___y_1850_, lean_object* v___y_1851_, lean_object* v___y_1852_, lean_object* v___y_1853_, lean_object* v___y_1854_, lean_object* v___y_1855_, lean_object* v___y_1856_){
_start:
{
lean_object* v___x_1858_; 
v___x_1858_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__2___redArg(v_opt_1848_, v___y_1855_);
return v___x_1858_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__2___boxed(lean_object* v_opt_1859_, lean_object* v___y_1860_, lean_object* v___y_1861_, lean_object* v___y_1862_, lean_object* v___y_1863_, lean_object* v___y_1864_, lean_object* v___y_1865_, lean_object* v___y_1866_, lean_object* v___y_1867_, lean_object* v___y_1868_){
_start:
{
lean_object* v_res_1869_; 
v_res_1869_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__2(v_opt_1859_, v___y_1860_, v___y_1861_, v___y_1862_, v___y_1863_, v___y_1864_, v___y_1865_, v___y_1866_, v___y_1867_);
lean_dec(v___y_1867_);
lean_dec_ref(v___y_1866_);
lean_dec(v___y_1865_);
lean_dec_ref(v___y_1864_);
lean_dec(v___y_1863_);
lean_dec(v___y_1862_);
lean_dec(v___y_1861_);
lean_dec_ref(v___y_1860_);
lean_dec_ref(v_opt_1859_);
return v_res_1869_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__3(lean_object* v_cls_1870_, lean_object* v_msg_1871_, lean_object* v___y_1872_, lean_object* v___y_1873_, lean_object* v___y_1874_, lean_object* v___y_1875_, lean_object* v___y_1876_, lean_object* v___y_1877_, lean_object* v___y_1878_, lean_object* v___y_1879_){
_start:
{
lean_object* v___x_1881_; 
v___x_1881_ = lp_aesop_Lean_addTrace___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__3___redArg(v_cls_1870_, v_msg_1871_, v___y_1876_, v___y_1877_, v___y_1878_, v___y_1879_);
return v___x_1881_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__3___boxed(lean_object* v_cls_1882_, lean_object* v_msg_1883_, lean_object* v___y_1884_, lean_object* v___y_1885_, lean_object* v___y_1886_, lean_object* v___y_1887_, lean_object* v___y_1888_, lean_object* v___y_1889_, lean_object* v___y_1890_, lean_object* v___y_1891_, lean_object* v___y_1892_){
_start:
{
lean_object* v_res_1893_; 
v_res_1893_ = lp_aesop_Lean_addTrace___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__3(v_cls_1882_, v_msg_1883_, v___y_1884_, v___y_1885_, v___y_1886_, v___y_1887_, v___y_1888_, v___y_1889_, v___y_1890_, v___y_1891_);
lean_dec(v___y_1891_);
lean_dec_ref(v___y_1890_);
lean_dec(v___y_1889_);
lean_dec_ref(v___y_1888_);
lean_dec(v___y_1887_);
lean_dec(v___y_1886_);
lean_dec(v___y_1885_);
lean_dec_ref(v___y_1884_);
return v_res_1893_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__4(lean_object* v_00_u03b2_1894_, lean_object* v_m_1895_, lean_object* v_a_1896_, lean_object* v_fallback_1897_){
_start:
{
lean_object* v___x_1898_; 
v___x_1898_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__4___redArg(v_m_1895_, v_a_1896_, v_fallback_1897_);
return v___x_1898_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__4___boxed(lean_object* v_00_u03b2_1899_, lean_object* v_m_1900_, lean_object* v_a_1901_, lean_object* v_fallback_1902_){
_start:
{
lean_object* v_res_1903_; 
v_res_1903_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__4(v_00_u03b2_1899_, v_m_1900_, v_a_1901_, v_fallback_1902_);
lean_dec(v_fallback_1902_);
lean_dec(v_a_1901_);
lean_dec_ref(v_m_1900_);
return v_res_1903_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__5(lean_object* v_mvarId_1904_, lean_object* v___y_1905_, lean_object* v___y_1906_, lean_object* v___y_1907_, lean_object* v___y_1908_, lean_object* v___y_1909_, lean_object* v___y_1910_, lean_object* v___y_1911_, lean_object* v___y_1912_){
_start:
{
lean_object* v___x_1914_; 
v___x_1914_ = lp_aesop_Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__5___redArg(v_mvarId_1904_, v___y_1910_);
return v___x_1914_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__5___boxed(lean_object* v_mvarId_1915_, lean_object* v___y_1916_, lean_object* v___y_1917_, lean_object* v___y_1918_, lean_object* v___y_1919_, lean_object* v___y_1920_, lean_object* v___y_1921_, lean_object* v___y_1922_, lean_object* v___y_1923_, lean_object* v___y_1924_){
_start:
{
lean_object* v_res_1925_; 
v_res_1925_ = lp_aesop_Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__5(v_mvarId_1915_, v___y_1916_, v___y_1917_, v___y_1918_, v___y_1919_, v___y_1920_, v___y_1921_, v___y_1922_, v___y_1923_);
lean_dec(v___y_1923_);
lean_dec_ref(v___y_1922_);
lean_dec(v___y_1921_);
lean_dec_ref(v___y_1920_);
lean_dec(v___y_1919_);
lean_dec(v___y_1918_);
lean_dec(v___y_1917_);
lean_dec_ref(v___y_1916_);
lean_dec(v_mvarId_1915_);
return v_res_1925_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_assign___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__6(lean_object* v_mvarId_1926_, lean_object* v_val_1927_, lean_object* v___y_1928_, lean_object* v___y_1929_, lean_object* v___y_1930_, lean_object* v___y_1931_, lean_object* v___y_1932_, lean_object* v___y_1933_, lean_object* v___y_1934_, lean_object* v___y_1935_){
_start:
{
lean_object* v___x_1937_; 
v___x_1937_ = lp_aesop_Lean_MVarId_assign___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__6___redArg(v_mvarId_1926_, v_val_1927_, v___y_1933_);
return v___x_1937_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_assign___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__6___boxed(lean_object* v_mvarId_1938_, lean_object* v_val_1939_, lean_object* v___y_1940_, lean_object* v___y_1941_, lean_object* v___y_1942_, lean_object* v___y_1943_, lean_object* v___y_1944_, lean_object* v___y_1945_, lean_object* v___y_1946_, lean_object* v___y_1947_, lean_object* v___y_1948_){
_start:
{
lean_object* v_res_1949_; 
v_res_1949_ = lp_aesop_Lean_MVarId_assign___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__6(v_mvarId_1938_, v_val_1939_, v___y_1940_, v___y_1941_, v___y_1942_, v___y_1943_, v___y_1944_, v___y_1945_, v___y_1946_, v___y_1947_);
lean_dec(v___y_1947_);
lean_dec_ref(v___y_1946_);
lean_dec(v___y_1945_);
lean_dec_ref(v___y_1944_);
lean_dec(v___y_1943_);
lean_dec(v___y_1942_);
lean_dec(v___y_1941_);
lean_dec_ref(v___y_1940_);
return v_res_1949_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__8(lean_object* v_as_1950_, size_t v_sz_1951_, size_t v_i_1952_, lean_object* v_b_1953_, lean_object* v___y_1954_, lean_object* v___y_1955_, lean_object* v___y_1956_, lean_object* v___y_1957_, lean_object* v___y_1958_, lean_object* v___y_1959_, lean_object* v___y_1960_, lean_object* v___y_1961_){
_start:
{
lean_object* v___x_1963_; 
v___x_1963_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__8___redArg(v_as_1950_, v_sz_1951_, v_i_1952_, v_b_1953_, v___y_1954_);
return v___x_1963_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__8___boxed(lean_object* v_as_1964_, lean_object* v_sz_1965_, lean_object* v_i_1966_, lean_object* v_b_1967_, lean_object* v___y_1968_, lean_object* v___y_1969_, lean_object* v___y_1970_, lean_object* v___y_1971_, lean_object* v___y_1972_, lean_object* v___y_1973_, lean_object* v___y_1974_, lean_object* v___y_1975_, lean_object* v___y_1976_){
_start:
{
size_t v_sz_boxed_1977_; size_t v_i_boxed_1978_; lean_object* v_res_1979_; 
v_sz_boxed_1977_ = lean_unbox_usize(v_sz_1965_);
lean_dec(v_sz_1965_);
v_i_boxed_1978_ = lean_unbox_usize(v_i_1966_);
lean_dec(v_i_1966_);
v_res_1979_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__8(v_as_1964_, v_sz_boxed_1977_, v_i_boxed_1978_, v_b_1967_, v___y_1968_, v___y_1969_, v___y_1970_, v___y_1971_, v___y_1972_, v___y_1973_, v___y_1974_, v___y_1975_);
lean_dec(v___y_1975_);
lean_dec_ref(v___y_1974_);
lean_dec(v___y_1973_);
lean_dec_ref(v___y_1972_);
lean_dec(v___y_1971_);
lean_dec(v___y_1970_);
lean_dec(v___y_1969_);
lean_dec_ref(v___y_1968_);
lean_dec_ref(v_as_1964_);
return v_res_1979_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__0_spec__0(lean_object* v_00_u03b2_1980_, lean_object* v_m_1981_, lean_object* v_a_1982_, lean_object* v_b_1983_){
_start:
{
lean_object* v___x_1984_; 
v___x_1984_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__0_spec__0___redArg(v_m_1981_, v_a_1982_, v_b_1983_);
return v___x_1984_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_getD___at___00Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__4_spec__8(lean_object* v_00_u03b2_1985_, lean_object* v_a_1986_, lean_object* v_fallback_1987_, lean_object* v_x_1988_){
_start:
{
lean_object* v___x_1989_; 
v___x_1989_ = lp_aesop_Std_DHashMap_Internal_AssocList_getD___at___00Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__4_spec__8___redArg(v_a_1986_, v_fallback_1987_, v_x_1988_);
return v___x_1989_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_getD___at___00Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__4_spec__8___boxed(lean_object* v_00_u03b2_1990_, lean_object* v_a_1991_, lean_object* v_fallback_1992_, lean_object* v_x_1993_){
_start:
{
lean_object* v_res_1994_; 
v_res_1994_ = lp_aesop_Std_DHashMap_Internal_AssocList_getD___at___00Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__4_spec__8(v_00_u03b2_1990_, v_a_1991_, v_fallback_1992_, v_x_1993_);
lean_dec(v_x_1993_);
lean_dec(v_fallback_1992_);
lean_dec(v_a_1991_);
return v_res_1994_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__5_spec__10(lean_object* v_00_u03b2_1995_, lean_object* v_x_1996_, lean_object* v_x_1997_){
_start:
{
uint8_t v___x_1998_; 
v___x_1998_ = lp_aesop_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__5_spec__10___redArg(v_x_1996_, v_x_1997_);
return v___x_1998_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__5_spec__10___boxed(lean_object* v_00_u03b2_1999_, lean_object* v_x_2000_, lean_object* v_x_2001_){
_start:
{
uint8_t v_res_2002_; lean_object* v_r_2003_; 
v_res_2002_ = lp_aesop_Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__5_spec__10(v_00_u03b2_1999_, v_x_2000_, v_x_2001_);
lean_dec(v_x_2001_);
lean_dec_ref(v_x_2000_);
v_r_2003_ = lean_box(v_res_2002_);
return v_r_2003_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__6_spec__12(lean_object* v_00_u03b2_2004_, lean_object* v_x_2005_, lean_object* v_x_2006_, lean_object* v_x_2007_){
_start:
{
lean_object* v___x_2008_; 
v___x_2008_ = lp_aesop_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__6_spec__12___redArg(v_x_2005_, v_x_2006_, v_x_2007_);
return v___x_2008_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__0_spec__0_spec__1(lean_object* v_00_u03b2_2009_, lean_object* v_a_2010_, lean_object* v_x_2011_){
_start:
{
uint8_t v___x_2012_; 
v___x_2012_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__0_spec__0_spec__1___redArg(v_a_2010_, v_x_2011_);
return v___x_2012_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__0_spec__0_spec__1___boxed(lean_object* v_00_u03b2_2013_, lean_object* v_a_2014_, lean_object* v_x_2015_){
_start:
{
uint8_t v_res_2016_; lean_object* v_r_2017_; 
v_res_2016_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__0_spec__0_spec__1(v_00_u03b2_2013_, v_a_2014_, v_x_2015_);
lean_dec(v_x_2015_);
lean_dec(v_a_2014_);
v_r_2017_ = lean_box(v_res_2016_);
return v_r_2017_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__0_spec__0_spec__2(lean_object* v_00_u03b2_2018_, lean_object* v_data_2019_){
_start:
{
lean_object* v___x_2020_; 
v___x_2020_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__0_spec__0_spec__2___redArg(v_data_2019_);
return v___x_2020_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__5_spec__10_spec__13(lean_object* v_00_u03b2_2021_, lean_object* v_x_2022_, size_t v_x_2023_, lean_object* v_x_2024_){
_start:
{
uint8_t v___x_2025_; 
v___x_2025_ = lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__5_spec__10_spec__13___redArg(v_x_2022_, v_x_2023_, v_x_2024_);
return v___x_2025_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__5_spec__10_spec__13___boxed(lean_object* v_00_u03b2_2026_, lean_object* v_x_2027_, lean_object* v_x_2028_, lean_object* v_x_2029_){
_start:
{
size_t v_x_66218__boxed_2030_; uint8_t v_res_2031_; lean_object* v_r_2032_; 
v_x_66218__boxed_2030_ = lean_unbox_usize(v_x_2028_);
lean_dec(v_x_2028_);
v_res_2031_ = lp_aesop_Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__5_spec__10_spec__13(v_00_u03b2_2026_, v_x_2027_, v_x_66218__boxed_2030_, v_x_2029_);
lean_dec(v_x_2029_);
lean_dec_ref(v_x_2027_);
v_r_2032_ = lean_box(v_res_2031_);
return v_r_2032_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__6_spec__12_spec__16(lean_object* v_00_u03b2_2033_, lean_object* v_x_2034_, size_t v_x_2035_, size_t v_x_2036_, lean_object* v_x_2037_, lean_object* v_x_2038_){
_start:
{
lean_object* v___x_2039_; 
v___x_2039_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__6_spec__12_spec__16___redArg(v_x_2034_, v_x_2035_, v_x_2036_, v_x_2037_, v_x_2038_);
return v___x_2039_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__6_spec__12_spec__16___boxed(lean_object* v_00_u03b2_2040_, lean_object* v_x_2041_, lean_object* v_x_2042_, lean_object* v_x_2043_, lean_object* v_x_2044_, lean_object* v_x_2045_){
_start:
{
size_t v_x_66229__boxed_2046_; size_t v_x_66230__boxed_2047_; lean_object* v_res_2048_; 
v_x_66229__boxed_2046_ = lean_unbox_usize(v_x_2042_);
lean_dec(v_x_2042_);
v_x_66230__boxed_2047_ = lean_unbox_usize(v_x_2043_);
lean_dec(v_x_2043_);
v_res_2048_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__6_spec__12_spec__16(v_00_u03b2_2040_, v_x_2041_, v_x_66229__boxed_2046_, v_x_66230__boxed_2047_, v_x_2044_, v_x_2045_);
return v_res_2048_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__0_spec__0_spec__2_spec__13(lean_object* v_00_u03b2_2049_, lean_object* v_i_2050_, lean_object* v_source_2051_, lean_object* v_target_2052_){
_start:
{
lean_object* v___x_2053_; 
v___x_2053_ = lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__0_spec__0_spec__2_spec__13___redArg(v_i_2050_, v_source_2051_, v_target_2052_);
return v___x_2053_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__5_spec__10_spec__13_spec__18(lean_object* v_00_u03b2_2054_, lean_object* v_keys_2055_, lean_object* v_vals_2056_, lean_object* v_heq_2057_, lean_object* v_i_2058_, lean_object* v_k_2059_){
_start:
{
uint8_t v___x_2060_; 
v___x_2060_ = lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__5_spec__10_spec__13_spec__18___redArg(v_keys_2055_, v_i_2058_, v_k_2059_);
return v___x_2060_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__5_spec__10_spec__13_spec__18___boxed(lean_object* v_00_u03b2_2061_, lean_object* v_keys_2062_, lean_object* v_vals_2063_, lean_object* v_heq_2064_, lean_object* v_i_2065_, lean_object* v_k_2066_){
_start:
{
uint8_t v_res_2067_; lean_object* v_r_2068_; 
v_res_2067_ = lp_aesop_Lean_PersistentHashMap_containsAtAux___at___00Lean_PersistentHashMap_containsAux___at___00Lean_PersistentHashMap_contains___at___00Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__5_spec__10_spec__13_spec__18(v_00_u03b2_2061_, v_keys_2062_, v_vals_2063_, v_heq_2064_, v_i_2065_, v_k_2066_);
lean_dec(v_k_2066_);
lean_dec_ref(v_vals_2063_);
lean_dec_ref(v_keys_2062_);
v_r_2068_ = lean_box(v_res_2067_);
return v_r_2068_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__6_spec__12_spec__16_spec__21(lean_object* v_00_u03b2_2069_, lean_object* v_n_2070_, lean_object* v_k_2071_, lean_object* v_v_2072_){
_start:
{
lean_object* v___x_2073_; 
v___x_2073_ = lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__6_spec__12_spec__16_spec__21___redArg(v_n_2070_, v_k_2071_, v_v_2072_);
return v___x_2073_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__6_spec__12_spec__16_spec__22(lean_object* v_00_u03b2_2074_, size_t v_depth_2075_, lean_object* v_keys_2076_, lean_object* v_vals_2077_, lean_object* v_heq_2078_, lean_object* v_i_2079_, lean_object* v_entries_2080_){
_start:
{
lean_object* v___x_2081_; 
v___x_2081_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__6_spec__12_spec__16_spec__22___redArg(v_depth_2075_, v_keys_2076_, v_vals_2077_, v_i_2079_, v_entries_2080_);
return v___x_2081_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__6_spec__12_spec__16_spec__22___boxed(lean_object* v_00_u03b2_2082_, lean_object* v_depth_2083_, lean_object* v_keys_2084_, lean_object* v_vals_2085_, lean_object* v_heq_2086_, lean_object* v_i_2087_, lean_object* v_entries_2088_){
_start:
{
size_t v_depth_boxed_2089_; lean_object* v_res_2090_; 
v_depth_boxed_2089_ = lean_unbox_usize(v_depth_2083_);
lean_dec(v_depth_2083_);
v_res_2090_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__6_spec__12_spec__16_spec__22(v_00_u03b2_2082_, v_depth_boxed_2089_, v_keys_2084_, v_vals_2085_, v_heq_2086_, v_i_2087_, v_entries_2088_);
lean_dec_ref(v_vals_2085_);
lean_dec_ref(v_keys_2084_);
return v_res_2090_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__0_spec__0_spec__2_spec__13_spec__20(lean_object* v_00_u03b2_2091_, lean_object* v_x_2092_, lean_object* v_x_2093_){
_start:
{
lean_object* v___x_2094_; 
v___x_2094_ = lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__0_spec__0_spec__2_spec__13_spec__20___redArg(v_x_2092_, v_x_2093_);
return v___x_2094_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__6_spec__12_spec__16_spec__21_spec__25(lean_object* v_00_u03b2_2095_, lean_object* v_x_2096_, lean_object* v_x_2097_, lean_object* v_x_2098_, lean_object* v_x_2099_){
_start:
{
lean_object* v___x_2100_; 
v___x_2100_ = lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__6_spec__12_spec__16_spec__21_spec__25___redArg(v_x_2096_, v_x_2097_, v_x_2098_, v_x_2099_);
return v___x_2100_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_withNewMCtxDepth___at___00Aesop_RuleTac_makeForwardHypProofs_spec__2___redArg___lam__0(lean_object* v_k_2101_, lean_object* v___y_2102_, lean_object* v___y_2103_, lean_object* v___y_2104_, lean_object* v___y_2105_, lean_object* v___y_2106_, lean_object* v___y_2107_, lean_object* v___y_2108_, lean_object* v___y_2109_){
_start:
{
lean_object* v___x_2111_; 
lean_inc(v___y_2105_);
lean_inc(v___y_2104_);
lean_inc(v___y_2103_);
lean_inc_ref(v___y_2102_);
v___x_2111_ = lean_apply_9(v_k_2101_, v___y_2102_, v___y_2103_, v___y_2104_, v___y_2105_, v___y_2106_, v___y_2107_, v___y_2108_, v___y_2109_, lean_box(0));
return v___x_2111_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_withNewMCtxDepth___at___00Aesop_RuleTac_makeForwardHypProofs_spec__2___redArg___lam__0___boxed(lean_object* v_k_2112_, lean_object* v___y_2113_, lean_object* v___y_2114_, lean_object* v___y_2115_, lean_object* v___y_2116_, lean_object* v___y_2117_, lean_object* v___y_2118_, lean_object* v___y_2119_, lean_object* v___y_2120_, lean_object* v___y_2121_){
_start:
{
lean_object* v_res_2122_; 
v_res_2122_ = lp_aesop_Lean_Meta_withNewMCtxDepth___at___00Aesop_RuleTac_makeForwardHypProofs_spec__2___redArg___lam__0(v_k_2112_, v___y_2113_, v___y_2114_, v___y_2115_, v___y_2116_, v___y_2117_, v___y_2118_, v___y_2119_, v___y_2120_);
lean_dec(v___y_2116_);
lean_dec(v___y_2115_);
lean_dec(v___y_2114_);
lean_dec_ref(v___y_2113_);
return v_res_2122_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_withNewMCtxDepth___at___00Aesop_RuleTac_makeForwardHypProofs_spec__2___redArg(lean_object* v_k_2123_, uint8_t v_allowLevelAssignments_2124_, lean_object* v___y_2125_, lean_object* v___y_2126_, lean_object* v___y_2127_, lean_object* v___y_2128_, lean_object* v___y_2129_, lean_object* v___y_2130_, lean_object* v___y_2131_, lean_object* v___y_2132_){
_start:
{
lean_object* v___f_2134_; lean_object* v___x_2135_; 
lean_inc(v___y_2128_);
lean_inc(v___y_2127_);
lean_inc(v___y_2126_);
lean_inc_ref(v___y_2125_);
v___f_2134_ = lean_alloc_closure((void*)(lp_aesop_Lean_Meta_withNewMCtxDepth___at___00Aesop_RuleTac_makeForwardHypProofs_spec__2___redArg___lam__0___boxed), 10, 5);
lean_closure_set(v___f_2134_, 0, v_k_2123_);
lean_closure_set(v___f_2134_, 1, v___y_2125_);
lean_closure_set(v___f_2134_, 2, v___y_2126_);
lean_closure_set(v___f_2134_, 3, v___y_2127_);
lean_closure_set(v___f_2134_, 4, v___y_2128_);
v___x_2135_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withNewMCtxDepthImp(lean_box(0), v_allowLevelAssignments_2124_, v___f_2134_, v___y_2129_, v___y_2130_, v___y_2131_, v___y_2132_);
if (lean_obj_tag(v___x_2135_) == 0)
{
return v___x_2135_;
}
else
{
lean_object* v_a_2136_; lean_object* v___x_2138_; uint8_t v_isShared_2139_; uint8_t v_isSharedCheck_2143_; 
v_a_2136_ = lean_ctor_get(v___x_2135_, 0);
v_isSharedCheck_2143_ = !lean_is_exclusive(v___x_2135_);
if (v_isSharedCheck_2143_ == 0)
{
v___x_2138_ = v___x_2135_;
v_isShared_2139_ = v_isSharedCheck_2143_;
goto v_resetjp_2137_;
}
else
{
lean_inc(v_a_2136_);
lean_dec(v___x_2135_);
v___x_2138_ = lean_box(0);
v_isShared_2139_ = v_isSharedCheck_2143_;
goto v_resetjp_2137_;
}
v_resetjp_2137_:
{
lean_object* v___x_2141_; 
if (v_isShared_2139_ == 0)
{
v___x_2141_ = v___x_2138_;
goto v_reusejp_2140_;
}
else
{
lean_object* v_reuseFailAlloc_2142_; 
v_reuseFailAlloc_2142_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2142_, 0, v_a_2136_);
v___x_2141_ = v_reuseFailAlloc_2142_;
goto v_reusejp_2140_;
}
v_reusejp_2140_:
{
return v___x_2141_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_withNewMCtxDepth___at___00Aesop_RuleTac_makeForwardHypProofs_spec__2___redArg___boxed(lean_object* v_k_2144_, lean_object* v_allowLevelAssignments_2145_, lean_object* v___y_2146_, lean_object* v___y_2147_, lean_object* v___y_2148_, lean_object* v___y_2149_, lean_object* v___y_2150_, lean_object* v___y_2151_, lean_object* v___y_2152_, lean_object* v___y_2153_, lean_object* v___y_2154_){
_start:
{
uint8_t v_allowLevelAssignments_boxed_2155_; lean_object* v_res_2156_; 
v_allowLevelAssignments_boxed_2155_ = lean_unbox(v_allowLevelAssignments_2145_);
v_res_2156_ = lp_aesop_Lean_Meta_withNewMCtxDepth___at___00Aesop_RuleTac_makeForwardHypProofs_spec__2___redArg(v_k_2144_, v_allowLevelAssignments_boxed_2155_, v___y_2146_, v___y_2147_, v___y_2148_, v___y_2149_, v___y_2150_, v___y_2151_, v___y_2152_, v___y_2153_);
lean_dec(v___y_2153_);
lean_dec_ref(v___y_2152_);
lean_dec(v___y_2151_);
lean_dec_ref(v___y_2150_);
lean_dec(v___y_2149_);
lean_dec(v___y_2148_);
lean_dec(v___y_2147_);
lean_dec_ref(v___y_2146_);
return v_res_2156_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_withNewMCtxDepth___at___00Aesop_RuleTac_makeForwardHypProofs_spec__2(lean_object* v_00_u03b1_2157_, lean_object* v_k_2158_, uint8_t v_allowLevelAssignments_2159_, lean_object* v___y_2160_, lean_object* v___y_2161_, lean_object* v___y_2162_, lean_object* v___y_2163_, lean_object* v___y_2164_, lean_object* v___y_2165_, lean_object* v___y_2166_, lean_object* v___y_2167_){
_start:
{
lean_object* v___x_2169_; 
v___x_2169_ = lp_aesop_Lean_Meta_withNewMCtxDepth___at___00Aesop_RuleTac_makeForwardHypProofs_spec__2___redArg(v_k_2158_, v_allowLevelAssignments_2159_, v___y_2160_, v___y_2161_, v___y_2162_, v___y_2163_, v___y_2164_, v___y_2165_, v___y_2166_, v___y_2167_);
return v___x_2169_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_withNewMCtxDepth___at___00Aesop_RuleTac_makeForwardHypProofs_spec__2___boxed(lean_object* v_00_u03b1_2170_, lean_object* v_k_2171_, lean_object* v_allowLevelAssignments_2172_, lean_object* v___y_2173_, lean_object* v___y_2174_, lean_object* v___y_2175_, lean_object* v___y_2176_, lean_object* v___y_2177_, lean_object* v___y_2178_, lean_object* v___y_2179_, lean_object* v___y_2180_, lean_object* v___y_2181_){
_start:
{
uint8_t v_allowLevelAssignments_boxed_2182_; lean_object* v_res_2183_; 
v_allowLevelAssignments_boxed_2182_ = lean_unbox(v_allowLevelAssignments_2172_);
v_res_2183_ = lp_aesop_Lean_Meta_withNewMCtxDepth___at___00Aesop_RuleTac_makeForwardHypProofs_spec__2(v_00_u03b1_2170_, v_k_2171_, v_allowLevelAssignments_boxed_2182_, v___y_2173_, v___y_2174_, v___y_2175_, v___y_2176_, v___y_2177_, v___y_2178_, v___y_2179_, v___y_2180_);
lean_dec(v___y_2180_);
lean_dec_ref(v___y_2179_);
lean_dec(v___y_2178_);
lean_dec_ref(v___y_2177_);
lean_dec(v___y_2176_);
lean_dec(v___y_2175_);
lean_dec(v___y_2174_);
lean_dec_ref(v___y_2173_);
return v_res_2183_;
}
}
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Aesop_UnorderedArraySet_contains___at___00Aesop_RuleTac_makeForwardHypProofs_spec__0_spec__0_spec__2(lean_object* v_a_2184_, lean_object* v_as_2185_, size_t v_i_2186_, size_t v_stop_2187_){
_start:
{
uint8_t v___x_2188_; 
v___x_2188_ = lean_usize_dec_eq(v_i_2186_, v_stop_2187_);
if (v___x_2188_ == 0)
{
lean_object* v___x_2189_; uint8_t v___x_2190_; 
v___x_2189_ = lean_array_uget_borrowed(v_as_2185_, v_i_2186_);
v___x_2190_ = lp_aesop_Aesop_instBEqPremiseIndex_beq(v_a_2184_, v___x_2189_);
if (v___x_2190_ == 0)
{
size_t v___x_2191_; size_t v___x_2192_; 
v___x_2191_ = ((size_t)1ULL);
v___x_2192_ = lean_usize_add(v_i_2186_, v___x_2191_);
v_i_2186_ = v___x_2192_;
goto _start;
}
else
{
return v___x_2190_;
}
}
else
{
uint8_t v___x_2194_; 
v___x_2194_ = 0;
return v___x_2194_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Aesop_UnorderedArraySet_contains___at___00Aesop_RuleTac_makeForwardHypProofs_spec__0_spec__0_spec__2___boxed(lean_object* v_a_2195_, lean_object* v_as_2196_, lean_object* v_i_2197_, lean_object* v_stop_2198_){
_start:
{
size_t v_i_boxed_2199_; size_t v_stop_boxed_2200_; uint8_t v_res_2201_; lean_object* v_r_2202_; 
v_i_boxed_2199_ = lean_unbox_usize(v_i_2197_);
lean_dec(v_i_2197_);
v_stop_boxed_2200_ = lean_unbox_usize(v_stop_2198_);
lean_dec(v_stop_2198_);
v_res_2201_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Aesop_UnorderedArraySet_contains___at___00Aesop_RuleTac_makeForwardHypProofs_spec__0_spec__0_spec__2(v_a_2195_, v_as_2196_, v_i_boxed_2199_, v_stop_boxed_2200_);
lean_dec_ref(v_as_2196_);
lean_dec(v_a_2195_);
v_r_2202_ = lean_box(v_res_2201_);
return v_r_2202_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Array_contains___at___00Aesop_UnorderedArraySet_contains___at___00Aesop_RuleTac_makeForwardHypProofs_spec__0_spec__0(lean_object* v_as_2203_, lean_object* v_a_2204_){
_start:
{
lean_object* v___x_2205_; lean_object* v___x_2206_; uint8_t v___x_2207_; 
v___x_2205_ = lean_unsigned_to_nat(0u);
v___x_2206_ = lean_array_get_size(v_as_2203_);
v___x_2207_ = lean_nat_dec_lt(v___x_2205_, v___x_2206_);
if (v___x_2207_ == 0)
{
return v___x_2207_;
}
else
{
if (v___x_2207_ == 0)
{
return v___x_2207_;
}
else
{
size_t v___x_2208_; size_t v___x_2209_; uint8_t v___x_2210_; 
v___x_2208_ = ((size_t)0ULL);
v___x_2209_ = lean_usize_of_nat(v___x_2206_);
v___x_2210_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Aesop_UnorderedArraySet_contains___at___00Aesop_RuleTac_makeForwardHypProofs_spec__0_spec__0_spec__2(v_a_2204_, v_as_2203_, v___x_2208_, v___x_2209_);
return v___x_2210_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_contains___at___00Aesop_UnorderedArraySet_contains___at___00Aesop_RuleTac_makeForwardHypProofs_spec__0_spec__0___boxed(lean_object* v_as_2211_, lean_object* v_a_2212_){
_start:
{
uint8_t v_res_2213_; lean_object* v_r_2214_; 
v_res_2213_ = lp_aesop_Array_contains___at___00Aesop_UnorderedArraySet_contains___at___00Aesop_RuleTac_makeForwardHypProofs_spec__0_spec__0(v_as_2211_, v_a_2212_);
lean_dec(v_a_2212_);
lean_dec_ref(v_as_2211_);
v_r_2214_ = lean_box(v_res_2213_);
return v_r_2214_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_UnorderedArraySet_contains___at___00Aesop_RuleTac_makeForwardHypProofs_spec__0(lean_object* v_x_2215_, lean_object* v_s_2216_){
_start:
{
uint8_t v___x_2217_; 
v___x_2217_ = lp_aesop_Array_contains___at___00Aesop_UnorderedArraySet_contains___at___00Aesop_RuleTac_makeForwardHypProofs_spec__0_spec__0(v_s_2216_, v_x_2215_);
return v___x_2217_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_contains___at___00Aesop_RuleTac_makeForwardHypProofs_spec__0___boxed(lean_object* v_x_2218_, lean_object* v_s_2219_){
_start:
{
uint8_t v_res_2220_; lean_object* v_r_2221_; 
v_res_2220_ = lp_aesop_Aesop_UnorderedArraySet_contains___at___00Aesop_RuleTac_makeForwardHypProofs_spec__0(v_x_2218_, v_s_2219_);
lean_dec_ref(v_s_2219_);
lean_dec(v_x_2218_);
v_r_2221_ = lean_box(v_res_2220_);
return v_r_2221_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleTac_makeForwardHypProofs_spec__1___redArg(lean_object* v_fst_2222_, lean_object* v_immediate_2223_, lean_object* v_fst_2224_, lean_object* v_range_2225_, lean_object* v_b_2226_, lean_object* v_i_2227_, lean_object* v___y_2228_){
_start:
{
lean_object* v_stop_2230_; lean_object* v_step_2231_; lean_object* v_a_2233_; uint8_t v___x_2236_; 
v_stop_2230_ = lean_ctor_get(v_range_2225_, 1);
v_step_2231_ = lean_ctor_get(v_range_2225_, 2);
v___x_2236_ = lean_nat_dec_lt(v_i_2227_, v_stop_2230_);
if (v___x_2236_ == 0)
{
lean_object* v___x_2237_; 
lean_dec(v_i_2227_);
v___x_2237_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2237_, 0, v_b_2226_);
return v___x_2237_;
}
else
{
lean_object* v___x_2238_; lean_object* v___x_2239_; lean_object* v_a_2240_; uint8_t v___x_2241_; 
v___x_2238_ = lean_array_fget_borrowed(v_fst_2222_, v_i_2227_);
v___x_2239_ = lp_aesop_Lean_MVarId_isAssignedOrDelayedAssigned___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__5___redArg(v___x_2238_, v___y_2228_);
v_a_2240_ = lean_ctor_get(v___x_2239_, 0);
lean_inc(v_a_2240_);
lean_dec_ref(v___x_2239_);
v___x_2241_ = lean_unbox(v_a_2240_);
lean_dec(v_a_2240_);
if (v___x_2241_ == 0)
{
lean_object* v_fst_2242_; lean_object* v_snd_2243_; lean_object* v___x_2245_; uint8_t v_isShared_2246_; uint8_t v_isSharedCheck_2264_; 
v_fst_2242_ = lean_ctor_get(v_b_2226_, 0);
v_snd_2243_ = lean_ctor_get(v_b_2226_, 1);
v_isSharedCheck_2264_ = !lean_is_exclusive(v_b_2226_);
if (v_isSharedCheck_2264_ == 0)
{
v___x_2245_ = v_b_2226_;
v_isShared_2246_ = v_isSharedCheck_2264_;
goto v_resetjp_2244_;
}
else
{
lean_inc(v_snd_2243_);
lean_inc(v_fst_2242_);
lean_dec(v_b_2226_);
v___x_2245_ = lean_box(0);
v_isShared_2246_ = v_isSharedCheck_2264_;
goto v_resetjp_2244_;
}
v_resetjp_2244_:
{
uint8_t v___x_2247_; 
v___x_2247_ = lp_aesop_Array_contains___at___00Aesop_UnorderedArraySet_contains___at___00Aesop_RuleTac_makeForwardHypProofs_spec__0_spec__0(v_immediate_2223_, v_i_2227_);
if (v___x_2247_ == 0)
{
uint8_t v___x_2248_; lean_object* v___x_2249_; lean_object* v___x_2250_; uint8_t v___x_2251_; uint8_t v___x_2252_; 
v___x_2248_ = 0;
v___x_2249_ = lean_box(v___x_2248_);
v___x_2250_ = lean_array_get(v___x_2249_, v_fst_2224_, v_i_2227_);
lean_dec(v___x_2249_);
v___x_2251_ = lean_unbox(v___x_2250_);
lean_dec(v___x_2250_);
v___x_2252_ = l_Lean_BinderInfo_isInstImplicit(v___x_2251_);
if (v___x_2252_ == 0)
{
lean_object* v___x_2254_; 
if (v_isShared_2246_ == 0)
{
v___x_2254_ = v___x_2245_;
goto v_reusejp_2253_;
}
else
{
lean_object* v_reuseFailAlloc_2255_; 
v_reuseFailAlloc_2255_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2255_, 0, v_fst_2242_);
lean_ctor_set(v_reuseFailAlloc_2255_, 1, v_snd_2243_);
v___x_2254_ = v_reuseFailAlloc_2255_;
goto v_reusejp_2253_;
}
v_reusejp_2253_:
{
v_a_2233_ = v___x_2254_;
goto v___jp_2232_;
}
}
else
{
lean_object* v___x_2256_; lean_object* v___x_2258_; 
lean_inc(v___x_2238_);
v___x_2256_ = lean_array_push(v_fst_2242_, v___x_2238_);
if (v_isShared_2246_ == 0)
{
lean_ctor_set(v___x_2245_, 0, v___x_2256_);
v___x_2258_ = v___x_2245_;
goto v_reusejp_2257_;
}
else
{
lean_object* v_reuseFailAlloc_2259_; 
v_reuseFailAlloc_2259_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2259_, 0, v___x_2256_);
lean_ctor_set(v_reuseFailAlloc_2259_, 1, v_snd_2243_);
v___x_2258_ = v_reuseFailAlloc_2259_;
goto v_reusejp_2257_;
}
v_reusejp_2257_:
{
v_a_2233_ = v___x_2258_;
goto v___jp_2232_;
}
}
}
else
{
lean_object* v___x_2260_; lean_object* v___x_2262_; 
lean_inc(v___x_2238_);
v___x_2260_ = lean_array_push(v_snd_2243_, v___x_2238_);
if (v_isShared_2246_ == 0)
{
lean_ctor_set(v___x_2245_, 1, v___x_2260_);
v___x_2262_ = v___x_2245_;
goto v_reusejp_2261_;
}
else
{
lean_object* v_reuseFailAlloc_2263_; 
v_reuseFailAlloc_2263_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2263_, 0, v_fst_2242_);
lean_ctor_set(v_reuseFailAlloc_2263_, 1, v___x_2260_);
v___x_2262_ = v_reuseFailAlloc_2263_;
goto v_reusejp_2261_;
}
v_reusejp_2261_:
{
v_a_2233_ = v___x_2262_;
goto v___jp_2232_;
}
}
}
}
else
{
lean_object* v_fst_2265_; lean_object* v_snd_2266_; lean_object* v___x_2268_; uint8_t v_isShared_2269_; uint8_t v_isSharedCheck_2273_; 
v_fst_2265_ = lean_ctor_get(v_b_2226_, 0);
v_snd_2266_ = lean_ctor_get(v_b_2226_, 1);
v_isSharedCheck_2273_ = !lean_is_exclusive(v_b_2226_);
if (v_isSharedCheck_2273_ == 0)
{
v___x_2268_ = v_b_2226_;
v_isShared_2269_ = v_isSharedCheck_2273_;
goto v_resetjp_2267_;
}
else
{
lean_inc(v_snd_2266_);
lean_inc(v_fst_2265_);
lean_dec(v_b_2226_);
v___x_2268_ = lean_box(0);
v_isShared_2269_ = v_isSharedCheck_2273_;
goto v_resetjp_2267_;
}
v_resetjp_2267_:
{
lean_object* v___x_2271_; 
if (v_isShared_2269_ == 0)
{
v___x_2271_ = v___x_2268_;
goto v_reusejp_2270_;
}
else
{
lean_object* v_reuseFailAlloc_2272_; 
v_reuseFailAlloc_2272_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2272_, 0, v_fst_2265_);
lean_ctor_set(v_reuseFailAlloc_2272_, 1, v_snd_2266_);
v___x_2271_ = v_reuseFailAlloc_2272_;
goto v_reusejp_2270_;
}
v_reusejp_2270_:
{
v_a_2233_ = v___x_2271_;
goto v___jp_2232_;
}
}
}
}
v___jp_2232_:
{
lean_object* v___x_2234_; 
v___x_2234_ = lean_nat_add(v_i_2227_, v_step_2231_);
lean_dec(v_i_2227_);
v_b_2226_ = v_a_2233_;
v_i_2227_ = v___x_2234_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleTac_makeForwardHypProofs_spec__1___redArg___boxed(lean_object* v_fst_2274_, lean_object* v_immediate_2275_, lean_object* v_fst_2276_, lean_object* v_range_2277_, lean_object* v_b_2278_, lean_object* v_i_2279_, lean_object* v___y_2280_, lean_object* v___y_2281_){
_start:
{
lean_object* v_res_2282_; 
v_res_2282_ = lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleTac_makeForwardHypProofs_spec__1___redArg(v_fst_2274_, v_immediate_2275_, v_fst_2276_, v_range_2277_, v_b_2278_, v_i_2279_, v___y_2280_);
lean_dec(v___y_2280_);
lean_dec_ref(v_range_2277_);
lean_dec_ref(v_fst_2276_);
lean_dec_ref(v_immediate_2275_);
lean_dec_ref(v_fst_2274_);
return v_res_2282_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_makeForwardHypProofs___lam__0(lean_object* v_e_2283_, lean_object* v_patSubst_x3f_2284_, lean_object* v_immediate_2285_, lean_object* v___y_2286_, lean_object* v___y_2287_, lean_object* v___y_2288_, lean_object* v___y_2289_, lean_object* v___y_2290_, lean_object* v___y_2291_, lean_object* v___y_2292_, lean_object* v___y_2293_){
_start:
{
lean_object* v___x_2295_; 
lean_inc(v___y_2293_);
lean_inc_ref(v___y_2292_);
lean_inc(v___y_2291_);
lean_inc_ref(v___y_2290_);
lean_inc_ref(v_e_2283_);
v___x_2295_ = lean_infer_type(v_e_2283_, v___y_2290_, v___y_2291_, v___y_2292_, v___y_2293_);
if (lean_obj_tag(v___x_2295_) == 0)
{
lean_object* v___x_2296_; 
lean_dec_ref_known(v___x_2295_, 1);
lean_inc_ref(v_e_2283_);
v___x_2296_ = lp_aesop_Aesop_openRuleType(v_patSubst_x3f_2284_, v_e_2283_, v___y_2290_, v___y_2291_, v___y_2292_, v___y_2293_);
if (lean_obj_tag(v___x_2296_) == 0)
{
lean_object* v_a_2297_; lean_object* v_snd_2298_; lean_object* v_fst_2299_; lean_object* v_fst_2300_; lean_object* v___x_2302_; uint8_t v_isShared_2303_; uint8_t v_isSharedCheck_2330_; 
v_a_2297_ = lean_ctor_get(v___x_2296_, 0);
lean_inc(v_a_2297_);
lean_dec_ref_known(v___x_2296_, 1);
v_snd_2298_ = lean_ctor_get(v_a_2297_, 1);
lean_inc(v_snd_2298_);
v_fst_2299_ = lean_ctor_get(v_a_2297_, 0);
lean_inc(v_fst_2299_);
lean_dec(v_a_2297_);
v_fst_2300_ = lean_ctor_get(v_snd_2298_, 0);
v_isSharedCheck_2330_ = !lean_is_exclusive(v_snd_2298_);
if (v_isSharedCheck_2330_ == 0)
{
lean_object* v_unused_2331_; 
v_unused_2331_ = lean_ctor_get(v_snd_2298_, 1);
lean_dec(v_unused_2331_);
v___x_2302_ = v_snd_2298_;
v_isShared_2303_ = v_isSharedCheck_2330_;
goto v_resetjp_2301_;
}
else
{
lean_inc(v_fst_2300_);
lean_dec(v_snd_2298_);
v___x_2302_ = lean_box(0);
v_isShared_2303_ = v_isSharedCheck_2330_;
goto v_resetjp_2301_;
}
v_resetjp_2301_:
{
size_t v_sz_2304_; lean_object* v___x_2305_; lean_object* v___x_2306_; lean_object* v___x_2307_; lean_object* v___x_2308_; lean_object* v___x_2309_; lean_object* v___x_2311_; 
v_sz_2304_ = lean_array_size(v_fst_2299_);
v___x_2305_ = lean_array_get_size(v_fst_2299_);
v___x_2306_ = lean_mk_empty_array_with_capacity(v___x_2305_);
v___x_2307_ = lean_unsigned_to_nat(0u);
v___x_2308_ = lean_unsigned_to_nat(1u);
v___x_2309_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_2309_, 0, v___x_2307_);
lean_ctor_set(v___x_2309_, 1, v___x_2305_);
lean_ctor_set(v___x_2309_, 2, v___x_2308_);
lean_inc_ref(v___x_2306_);
if (v_isShared_2303_ == 0)
{
lean_ctor_set(v___x_2302_, 1, v___x_2306_);
lean_ctor_set(v___x_2302_, 0, v___x_2306_);
v___x_2311_ = v___x_2302_;
goto v_reusejp_2310_;
}
else
{
lean_object* v_reuseFailAlloc_2329_; 
v_reuseFailAlloc_2329_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2329_, 0, v___x_2306_);
lean_ctor_set(v_reuseFailAlloc_2329_, 1, v___x_2306_);
v___x_2311_ = v_reuseFailAlloc_2329_;
goto v_reusejp_2310_;
}
v_reusejp_2310_:
{
lean_object* v___x_2312_; 
v___x_2312_ = lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleTac_makeForwardHypProofs_spec__1___redArg(v_fst_2299_, v_immediate_2285_, v_fst_2300_, v___x_2309_, v___x_2311_, v___x_2307_, v___y_2291_);
lean_dec_ref_known(v___x_2309_, 3);
lean_dec(v_fst_2300_);
if (lean_obj_tag(v___x_2312_) == 0)
{
lean_object* v_a_2313_; lean_object* v_fst_2314_; lean_object* v_snd_2315_; size_t v___x_2316_; lean_object* v___x_2317_; lean_object* v___x_2318_; lean_object* v___x_2319_; lean_object* v___x_2320_; 
v_a_2313_ = lean_ctor_get(v___x_2312_, 0);
lean_inc(v_a_2313_);
lean_dec_ref_known(v___x_2312_, 1);
v_fst_2314_ = lean_ctor_get(v_a_2313_, 0);
lean_inc(v_fst_2314_);
v_snd_2315_ = lean_ctor_get(v_a_2313_, 1);
lean_inc(v_snd_2315_);
lean_dec(v_a_2313_);
v___x_2316_ = ((size_t)0ULL);
v___x_2317_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__1(v_sz_2304_, v___x_2316_, v_fst_2299_);
v___x_2318_ = l_Lean_mkAppN(v_e_2283_, v___x_2317_);
lean_dec_ref(v___x_2317_);
v___x_2319_ = ((lean_object*)(lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop___closed__6));
v___x_2320_ = lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop(v___x_2318_, v_fst_2314_, v_snd_2315_, v___x_2307_, v___x_2307_, v___x_2319_, v___y_2286_, v___y_2287_, v___y_2288_, v___y_2289_, v___y_2290_, v___y_2291_, v___y_2292_, v___y_2293_);
lean_dec(v_snd_2315_);
return v___x_2320_;
}
else
{
lean_object* v_a_2321_; lean_object* v___x_2323_; uint8_t v_isShared_2324_; uint8_t v_isSharedCheck_2328_; 
lean_dec(v_fst_2299_);
lean_dec_ref(v_e_2283_);
v_a_2321_ = lean_ctor_get(v___x_2312_, 0);
v_isSharedCheck_2328_ = !lean_is_exclusive(v___x_2312_);
if (v_isSharedCheck_2328_ == 0)
{
v___x_2323_ = v___x_2312_;
v_isShared_2324_ = v_isSharedCheck_2328_;
goto v_resetjp_2322_;
}
else
{
lean_inc(v_a_2321_);
lean_dec(v___x_2312_);
v___x_2323_ = lean_box(0);
v_isShared_2324_ = v_isSharedCheck_2328_;
goto v_resetjp_2322_;
}
v_resetjp_2322_:
{
lean_object* v___x_2326_; 
if (v_isShared_2324_ == 0)
{
v___x_2326_ = v___x_2323_;
goto v_reusejp_2325_;
}
else
{
lean_object* v_reuseFailAlloc_2327_; 
v_reuseFailAlloc_2327_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2327_, 0, v_a_2321_);
v___x_2326_ = v_reuseFailAlloc_2327_;
goto v_reusejp_2325_;
}
v_reusejp_2325_:
{
return v___x_2326_;
}
}
}
}
}
}
else
{
lean_object* v_a_2332_; lean_object* v___x_2334_; uint8_t v_isShared_2335_; uint8_t v_isSharedCheck_2339_; 
lean_dec_ref(v_e_2283_);
v_a_2332_ = lean_ctor_get(v___x_2296_, 0);
v_isSharedCheck_2339_ = !lean_is_exclusive(v___x_2296_);
if (v_isSharedCheck_2339_ == 0)
{
v___x_2334_ = v___x_2296_;
v_isShared_2335_ = v_isSharedCheck_2339_;
goto v_resetjp_2333_;
}
else
{
lean_inc(v_a_2332_);
lean_dec(v___x_2296_);
v___x_2334_ = lean_box(0);
v_isShared_2335_ = v_isSharedCheck_2339_;
goto v_resetjp_2333_;
}
v_resetjp_2333_:
{
lean_object* v___x_2337_; 
if (v_isShared_2335_ == 0)
{
v___x_2337_ = v___x_2334_;
goto v_reusejp_2336_;
}
else
{
lean_object* v_reuseFailAlloc_2338_; 
v_reuseFailAlloc_2338_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2338_, 0, v_a_2332_);
v___x_2337_ = v_reuseFailAlloc_2338_;
goto v_reusejp_2336_;
}
v_reusejp_2336_:
{
return v___x_2337_;
}
}
}
}
else
{
lean_object* v_a_2340_; lean_object* v___x_2342_; uint8_t v_isShared_2343_; uint8_t v_isSharedCheck_2347_; 
lean_dec(v_patSubst_x3f_2284_);
lean_dec_ref(v_e_2283_);
v_a_2340_ = lean_ctor_get(v___x_2295_, 0);
v_isSharedCheck_2347_ = !lean_is_exclusive(v___x_2295_);
if (v_isSharedCheck_2347_ == 0)
{
v___x_2342_ = v___x_2295_;
v_isShared_2343_ = v_isSharedCheck_2347_;
goto v_resetjp_2341_;
}
else
{
lean_inc(v_a_2340_);
lean_dec(v___x_2295_);
v___x_2342_ = lean_box(0);
v_isShared_2343_ = v_isSharedCheck_2347_;
goto v_resetjp_2341_;
}
v_resetjp_2341_:
{
lean_object* v___x_2345_; 
if (v_isShared_2343_ == 0)
{
v___x_2345_ = v___x_2342_;
goto v_reusejp_2344_;
}
else
{
lean_object* v_reuseFailAlloc_2346_; 
v_reuseFailAlloc_2346_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2346_, 0, v_a_2340_);
v___x_2345_ = v_reuseFailAlloc_2346_;
goto v_reusejp_2344_;
}
v_reusejp_2344_:
{
return v___x_2345_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_makeForwardHypProofs___lam__0___boxed(lean_object* v_e_2348_, lean_object* v_patSubst_x3f_2349_, lean_object* v_immediate_2350_, lean_object* v___y_2351_, lean_object* v___y_2352_, lean_object* v___y_2353_, lean_object* v___y_2354_, lean_object* v___y_2355_, lean_object* v___y_2356_, lean_object* v___y_2357_, lean_object* v___y_2358_, lean_object* v___y_2359_){
_start:
{
lean_object* v_res_2360_; 
v_res_2360_ = lp_aesop_Aesop_RuleTac_makeForwardHypProofs___lam__0(v_e_2348_, v_patSubst_x3f_2349_, v_immediate_2350_, v___y_2351_, v___y_2352_, v___y_2353_, v___y_2354_, v___y_2355_, v___y_2356_, v___y_2357_, v___y_2358_);
lean_dec(v___y_2358_);
lean_dec_ref(v___y_2357_);
lean_dec(v___y_2356_);
lean_dec_ref(v___y_2355_);
lean_dec(v___y_2354_);
lean_dec(v___y_2353_);
lean_dec(v___y_2352_);
lean_dec_ref(v___y_2351_);
lean_dec_ref(v_immediate_2350_);
return v_res_2360_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_makeForwardHypProofs(lean_object* v_e_2361_, lean_object* v_patSubst_x3f_2362_, lean_object* v_immediate_2363_, lean_object* v_a_2364_, lean_object* v_a_2365_, lean_object* v_a_2366_, lean_object* v_a_2367_, lean_object* v_a_2368_, lean_object* v_a_2369_, lean_object* v_a_2370_, lean_object* v_a_2371_){
_start:
{
lean_object* v___f_2373_; uint8_t v___x_2374_; lean_object* v___x_2375_; 
v___f_2373_ = lean_alloc_closure((void*)(lp_aesop_Aesop_RuleTac_makeForwardHypProofs___lam__0___boxed), 12, 3);
lean_closure_set(v___f_2373_, 0, v_e_2361_);
lean_closure_set(v___f_2373_, 1, v_patSubst_x3f_2362_);
lean_closure_set(v___f_2373_, 2, v_immediate_2363_);
v___x_2374_ = 1;
v___x_2375_ = lp_aesop_Lean_Meta_withNewMCtxDepth___at___00Aesop_RuleTac_makeForwardHypProofs_spec__2___redArg(v___f_2373_, v___x_2374_, v_a_2364_, v_a_2365_, v_a_2366_, v_a_2367_, v_a_2368_, v_a_2369_, v_a_2370_, v_a_2371_);
return v___x_2375_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_makeForwardHypProofs___boxed(lean_object* v_e_2376_, lean_object* v_patSubst_x3f_2377_, lean_object* v_immediate_2378_, lean_object* v_a_2379_, lean_object* v_a_2380_, lean_object* v_a_2381_, lean_object* v_a_2382_, lean_object* v_a_2383_, lean_object* v_a_2384_, lean_object* v_a_2385_, lean_object* v_a_2386_, lean_object* v_a_2387_){
_start:
{
lean_object* v_res_2388_; 
v_res_2388_ = lp_aesop_Aesop_RuleTac_makeForwardHypProofs(v_e_2376_, v_patSubst_x3f_2377_, v_immediate_2378_, v_a_2379_, v_a_2380_, v_a_2381_, v_a_2382_, v_a_2383_, v_a_2384_, v_a_2385_, v_a_2386_);
lean_dec(v_a_2386_);
lean_dec_ref(v_a_2385_);
lean_dec(v_a_2384_);
lean_dec_ref(v_a_2383_);
lean_dec(v_a_2382_);
lean_dec(v_a_2381_);
lean_dec(v_a_2380_);
lean_dec_ref(v_a_2379_);
return v_res_2388_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleTac_makeForwardHypProofs_spec__1(lean_object* v_fst_2389_, lean_object* v_immediate_2390_, lean_object* v_fst_2391_, lean_object* v_range_2392_, lean_object* v_b_2393_, lean_object* v_i_2394_, lean_object* v_hs_2395_, lean_object* v_hl_2396_, lean_object* v___y_2397_, lean_object* v___y_2398_, lean_object* v___y_2399_, lean_object* v___y_2400_, lean_object* v___y_2401_, lean_object* v___y_2402_, lean_object* v___y_2403_, lean_object* v___y_2404_){
_start:
{
lean_object* v___x_2406_; 
v___x_2406_ = lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleTac_makeForwardHypProofs_spec__1___redArg(v_fst_2389_, v_immediate_2390_, v_fst_2391_, v_range_2392_, v_b_2393_, v_i_2394_, v___y_2402_);
return v___x_2406_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleTac_makeForwardHypProofs_spec__1___boxed(lean_object** _args){
lean_object* v_fst_2407_ = _args[0];
lean_object* v_immediate_2408_ = _args[1];
lean_object* v_fst_2409_ = _args[2];
lean_object* v_range_2410_ = _args[3];
lean_object* v_b_2411_ = _args[4];
lean_object* v_i_2412_ = _args[5];
lean_object* v_hs_2413_ = _args[6];
lean_object* v_hl_2414_ = _args[7];
lean_object* v___y_2415_ = _args[8];
lean_object* v___y_2416_ = _args[9];
lean_object* v___y_2417_ = _args[10];
lean_object* v___y_2418_ = _args[11];
lean_object* v___y_2419_ = _args[12];
lean_object* v___y_2420_ = _args[13];
lean_object* v___y_2421_ = _args[14];
lean_object* v___y_2422_ = _args[15];
lean_object* v___y_2423_ = _args[16];
_start:
{
lean_object* v_res_2424_; 
v_res_2424_ = lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_RuleTac_makeForwardHypProofs_spec__1(v_fst_2407_, v_immediate_2408_, v_fst_2409_, v_range_2410_, v_b_2411_, v_i_2412_, v_hs_2413_, v_hl_2414_, v___y_2415_, v___y_2416_, v___y_2417_, v___y_2418_, v___y_2419_, v___y_2420_, v___y_2421_, v___y_2422_);
lean_dec(v___y_2422_);
lean_dec_ref(v___y_2421_);
lean_dec(v___y_2420_);
lean_dec_ref(v___y_2419_);
lean_dec(v___y_2418_);
lean_dec(v___y_2417_);
lean_dec(v___y_2416_);
lean_dec_ref(v___y_2415_);
lean_dec_ref(v_range_2410_);
lean_dec_ref(v_fst_2409_);
lean_dec_ref(v_immediate_2408_);
lean_dec_ref(v_fst_2407_);
return v_res_2424_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_makeForwardHypProofs_x27_spec__0(lean_object* v_e_2425_, lean_object* v_immediate_2426_, lean_object* v_as_2427_, size_t v_sz_2428_, size_t v_i_2429_, lean_object* v_b_2430_, lean_object* v___y_2431_, lean_object* v___y_2432_, lean_object* v___y_2433_, lean_object* v___y_2434_, lean_object* v___y_2435_, lean_object* v___y_2436_, lean_object* v___y_2437_, lean_object* v___y_2438_){
_start:
{
uint8_t v___x_2440_; 
v___x_2440_ = lean_usize_dec_lt(v_i_2429_, v_sz_2428_);
if (v___x_2440_ == 0)
{
lean_object* v___x_2441_; 
lean_dec_ref(v_immediate_2426_);
lean_dec_ref(v_e_2425_);
v___x_2441_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2441_, 0, v_b_2430_);
return v___x_2441_;
}
else
{
lean_object* v_a_2442_; lean_object* v___x_2443_; lean_object* v___x_2444_; 
v_a_2442_ = lean_array_uget_borrowed(v_as_2427_, v_i_2429_);
lean_inc(v_a_2442_);
v___x_2443_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2443_, 0, v_a_2442_);
lean_inc_ref(v_immediate_2426_);
lean_inc_ref(v_e_2425_);
v___x_2444_ = lp_aesop_Aesop_RuleTac_makeForwardHypProofs(v_e_2425_, v___x_2443_, v_immediate_2426_, v___y_2431_, v___y_2432_, v___y_2433_, v___y_2434_, v___y_2435_, v___y_2436_, v___y_2437_, v___y_2438_);
if (lean_obj_tag(v___x_2444_) == 0)
{
lean_object* v___x_2445_; size_t v___x_2446_; size_t v___x_2447_; 
lean_dec_ref_known(v___x_2444_, 1);
v___x_2445_ = lean_box(0);
v___x_2446_ = ((size_t)1ULL);
v___x_2447_ = lean_usize_add(v_i_2429_, v___x_2446_);
v_i_2429_ = v___x_2447_;
v_b_2430_ = v___x_2445_;
goto _start;
}
else
{
lean_dec_ref(v_immediate_2426_);
lean_dec_ref(v_e_2425_);
return v___x_2444_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_makeForwardHypProofs_x27_spec__0___boxed(lean_object* v_e_2449_, lean_object* v_immediate_2450_, lean_object* v_as_2451_, lean_object* v_sz_2452_, lean_object* v_i_2453_, lean_object* v_b_2454_, lean_object* v___y_2455_, lean_object* v___y_2456_, lean_object* v___y_2457_, lean_object* v___y_2458_, lean_object* v___y_2459_, lean_object* v___y_2460_, lean_object* v___y_2461_, lean_object* v___y_2462_, lean_object* v___y_2463_){
_start:
{
size_t v_sz_boxed_2464_; size_t v_i_boxed_2465_; lean_object* v_res_2466_; 
v_sz_boxed_2464_ = lean_unbox_usize(v_sz_2452_);
lean_dec(v_sz_2452_);
v_i_boxed_2465_ = lean_unbox_usize(v_i_2453_);
lean_dec(v_i_2453_);
v_res_2466_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_makeForwardHypProofs_x27_spec__0(v_e_2449_, v_immediate_2450_, v_as_2451_, v_sz_boxed_2464_, v_i_boxed_2465_, v_b_2454_, v___y_2455_, v___y_2456_, v___y_2457_, v___y_2458_, v___y_2459_, v___y_2460_, v___y_2461_, v___y_2462_);
lean_dec(v___y_2462_);
lean_dec_ref(v___y_2461_);
lean_dec(v___y_2460_);
lean_dec_ref(v___y_2459_);
lean_dec(v___y_2458_);
lean_dec(v___y_2457_);
lean_dec(v___y_2456_);
lean_dec_ref(v___y_2455_);
lean_dec_ref(v_as_2451_);
return v_res_2466_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_makeForwardHypProofs_x27(lean_object* v_e_2467_, lean_object* v_patSubsts_x3f_2468_, lean_object* v_immediate_2469_, lean_object* v_a_2470_, lean_object* v_a_2471_, lean_object* v_a_2472_, lean_object* v_a_2473_, lean_object* v_a_2474_, lean_object* v_a_2475_, lean_object* v_a_2476_, lean_object* v_a_2477_){
_start:
{
if (lean_obj_tag(v_patSubsts_x3f_2468_) == 1)
{
lean_object* v_val_2479_; lean_object* v___x_2480_; size_t v_sz_2481_; size_t v___x_2482_; lean_object* v___x_2483_; 
v_val_2479_ = lean_ctor_get(v_patSubsts_x3f_2468_, 0);
v___x_2480_ = lean_box(0);
v_sz_2481_ = lean_array_size(v_val_2479_);
v___x_2482_ = ((size_t)0ULL);
v___x_2483_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_makeForwardHypProofs_x27_spec__0(v_e_2467_, v_immediate_2469_, v_val_2479_, v_sz_2481_, v___x_2482_, v___x_2480_, v_a_2470_, v_a_2471_, v_a_2472_, v_a_2473_, v_a_2474_, v_a_2475_, v_a_2476_, v_a_2477_);
if (lean_obj_tag(v___x_2483_) == 0)
{
lean_object* v___x_2485_; uint8_t v_isShared_2486_; uint8_t v_isSharedCheck_2490_; 
v_isSharedCheck_2490_ = !lean_is_exclusive(v___x_2483_);
if (v_isSharedCheck_2490_ == 0)
{
lean_object* v_unused_2491_; 
v_unused_2491_ = lean_ctor_get(v___x_2483_, 0);
lean_dec(v_unused_2491_);
v___x_2485_ = v___x_2483_;
v_isShared_2486_ = v_isSharedCheck_2490_;
goto v_resetjp_2484_;
}
else
{
lean_dec(v___x_2483_);
v___x_2485_ = lean_box(0);
v_isShared_2486_ = v_isSharedCheck_2490_;
goto v_resetjp_2484_;
}
v_resetjp_2484_:
{
lean_object* v___x_2488_; 
if (v_isShared_2486_ == 0)
{
lean_ctor_set(v___x_2485_, 0, v___x_2480_);
v___x_2488_ = v___x_2485_;
goto v_reusejp_2487_;
}
else
{
lean_object* v_reuseFailAlloc_2489_; 
v_reuseFailAlloc_2489_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2489_, 0, v___x_2480_);
v___x_2488_ = v_reuseFailAlloc_2489_;
goto v_reusejp_2487_;
}
v_reusejp_2487_:
{
return v___x_2488_;
}
}
}
else
{
return v___x_2483_;
}
}
else
{
lean_object* v___x_2492_; lean_object* v___x_2493_; 
v___x_2492_ = lean_box(0);
v___x_2493_ = lp_aesop_Aesop_RuleTac_makeForwardHypProofs(v_e_2467_, v___x_2492_, v_immediate_2469_, v_a_2470_, v_a_2471_, v_a_2472_, v_a_2473_, v_a_2474_, v_a_2475_, v_a_2476_, v_a_2477_);
return v___x_2493_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_makeForwardHypProofs_x27___boxed(lean_object* v_e_2494_, lean_object* v_patSubsts_x3f_2495_, lean_object* v_immediate_2496_, lean_object* v_a_2497_, lean_object* v_a_2498_, lean_object* v_a_2499_, lean_object* v_a_2500_, lean_object* v_a_2501_, lean_object* v_a_2502_, lean_object* v_a_2503_, lean_object* v_a_2504_, lean_object* v_a_2505_){
_start:
{
lean_object* v_res_2506_; 
v_res_2506_ = lp_aesop_Aesop_RuleTac_makeForwardHypProofs_x27(v_e_2494_, v_patSubsts_x3f_2495_, v_immediate_2496_, v_a_2497_, v_a_2498_, v_a_2499_, v_a_2500_, v_a_2501_, v_a_2502_, v_a_2503_, v_a_2504_);
lean_dec(v_a_2504_);
lean_dec_ref(v_a_2503_);
lean_dec(v_a_2502_);
lean_dec_ref(v_a_2501_);
lean_dec(v_a_2500_);
lean_dec(v_a_2499_);
lean_dec(v_a_2498_);
lean_dec_ref(v_a_2497_);
lean_dec(v_patSubsts_x3f_2495_);
return v_res_2506_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_assertForwardHyp_tacticBuilder___redArg(lean_object* v_goal_2507_, lean_object* v_hyp_2508_, lean_object* v_a_2509_, lean_object* v_a_2510_, lean_object* v_a_2511_, lean_object* v_a_2512_){
_start:
{
uint8_t v___x_2514_; lean_object* v___x_2515_; 
v___x_2514_ = 2;
v___x_2515_ = lp_aesop_Aesop_Script_TacticBuilder_assertHypothesis(v_goal_2507_, v_hyp_2508_, v___x_2514_, v_a_2509_, v_a_2510_, v_a_2511_, v_a_2512_);
return v___x_2515_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_assertForwardHyp_tacticBuilder___redArg___boxed(lean_object* v_goal_2516_, lean_object* v_hyp_2517_, lean_object* v_a_2518_, lean_object* v_a_2519_, lean_object* v_a_2520_, lean_object* v_a_2521_, lean_object* v_a_2522_){
_start:
{
lean_object* v_res_2523_; 
v_res_2523_ = lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_assertForwardHyp_tacticBuilder___redArg(v_goal_2516_, v_hyp_2517_, v_a_2518_, v_a_2519_, v_a_2520_, v_a_2521_);
lean_dec(v_a_2521_);
lean_dec_ref(v_a_2520_);
lean_dec(v_a_2519_);
lean_dec_ref(v_a_2518_);
return v_res_2523_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_assertForwardHyp_tacticBuilder(lean_object* v_goal_2524_, lean_object* v_hyp_2525_, lean_object* v_x_2526_, lean_object* v_a_2527_, lean_object* v_a_2528_, lean_object* v_a_2529_, lean_object* v_a_2530_){
_start:
{
lean_object* v___x_2532_; 
v___x_2532_ = lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_assertForwardHyp_tacticBuilder___redArg(v_goal_2524_, v_hyp_2525_, v_a_2527_, v_a_2528_, v_a_2529_, v_a_2530_);
return v___x_2532_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_assertForwardHyp_tacticBuilder___boxed(lean_object* v_goal_2533_, lean_object* v_hyp_2534_, lean_object* v_x_2535_, lean_object* v_a_2536_, lean_object* v_a_2537_, lean_object* v_a_2538_, lean_object* v_a_2539_, lean_object* v_a_2540_){
_start:
{
lean_object* v_res_2541_; 
v_res_2541_ = lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_assertForwardHyp_tacticBuilder(v_goal_2533_, v_hyp_2534_, v_x_2535_, v_a_2536_, v_a_2537_, v_a_2538_, v_a_2539_);
lean_dec(v_a_2539_);
lean_dec_ref(v_a_2538_);
lean_dec(v_a_2537_);
lean_dec_ref(v_a_2536_);
lean_dec_ref(v_x_2535_);
return v_res_2541_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_assertForwardHyp___lam__0(lean_object* v_x_2542_){
_start:
{
lean_object* v_snd_2543_; lean_object* v___x_2544_; lean_object* v___x_2545_; lean_object* v___x_2546_; 
v_snd_2543_ = lean_ctor_get(v_x_2542_, 1);
lean_inc(v_snd_2543_);
lean_dec_ref(v_x_2542_);
v___x_2544_ = lean_unsigned_to_nat(1u);
v___x_2545_ = lean_mk_empty_array_with_capacity(v___x_2544_);
v___x_2546_ = lean_array_push(v___x_2545_, v_snd_2543_);
return v___x_2546_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_RuleTac_assertForwardHyp___lam__1(lean_object* v_x_2547_){
_start:
{
uint8_t v___x_2548_; 
v___x_2548_ = 1;
return v___x_2548_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_assertForwardHyp___lam__1___boxed(lean_object* v_x_2549_){
_start:
{
uint8_t v_res_2550_; lean_object* v_r_2551_; 
v_res_2550_ = lp_aesop_Aesop_RuleTac_assertForwardHyp___lam__1(v_x_2549_);
lean_dec_ref(v_x_2549_);
v_r_2551_ = lean_box(v_res_2550_);
return v_r_2551_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_RuleTac_assertForwardHyp_spec__0___redArg(lean_object* v_msg_2552_, lean_object* v___y_2553_, lean_object* v___y_2554_, lean_object* v___y_2555_, lean_object* v___y_2556_){
_start:
{
lean_object* v_ref_2558_; lean_object* v___x_2559_; lean_object* v_a_2560_; lean_object* v___x_2562_; uint8_t v_isShared_2563_; uint8_t v_isSharedCheck_2568_; 
v_ref_2558_ = lean_ctor_get(v___y_2555_, 5);
v___x_2559_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_addTrace___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__3_spec__6(v_msg_2552_, v___y_2553_, v___y_2554_, v___y_2555_, v___y_2556_);
v_a_2560_ = lean_ctor_get(v___x_2559_, 0);
v_isSharedCheck_2568_ = !lean_is_exclusive(v___x_2559_);
if (v_isSharedCheck_2568_ == 0)
{
v___x_2562_ = v___x_2559_;
v_isShared_2563_ = v_isSharedCheck_2568_;
goto v_resetjp_2561_;
}
else
{
lean_inc(v_a_2560_);
lean_dec(v___x_2559_);
v___x_2562_ = lean_box(0);
v_isShared_2563_ = v_isSharedCheck_2568_;
goto v_resetjp_2561_;
}
v_resetjp_2561_:
{
lean_object* v___x_2564_; lean_object* v___x_2566_; 
lean_inc(v_ref_2558_);
v___x_2564_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2564_, 0, v_ref_2558_);
lean_ctor_set(v___x_2564_, 1, v_a_2560_);
if (v_isShared_2563_ == 0)
{
lean_ctor_set_tag(v___x_2562_, 1);
lean_ctor_set(v___x_2562_, 0, v___x_2564_);
v___x_2566_ = v___x_2562_;
goto v_reusejp_2565_;
}
else
{
lean_object* v_reuseFailAlloc_2567_; 
v_reuseFailAlloc_2567_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2567_, 0, v___x_2564_);
v___x_2566_ = v_reuseFailAlloc_2567_;
goto v_reusejp_2565_;
}
v_reusejp_2565_:
{
return v___x_2566_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_RuleTac_assertForwardHyp_spec__0___redArg___boxed(lean_object* v_msg_2569_, lean_object* v___y_2570_, lean_object* v___y_2571_, lean_object* v___y_2572_, lean_object* v___y_2573_, lean_object* v___y_2574_){
_start:
{
lean_object* v_res_2575_; 
v_res_2575_ = lp_aesop_Lean_throwError___at___00Aesop_RuleTac_assertForwardHyp_spec__0___redArg(v_msg_2569_, v___y_2570_, v___y_2571_, v___y_2572_, v___y_2573_);
lean_dec(v___y_2573_);
lean_dec_ref(v___y_2572_);
lean_dec(v___y_2571_);
lean_dec_ref(v___y_2570_);
return v_res_2575_;
}
}
static lean_object* _init_lp_aesop_Aesop_RuleTac_assertForwardHyp___lam__2___closed__1(void){
_start:
{
lean_object* v___x_2577_; lean_object* v___x_2578_; 
v___x_2577_ = ((lean_object*)(lp_aesop_Aesop_RuleTac_assertForwardHyp___lam__2___closed__0));
v___x_2578_ = l_Lean_stringToMessageData(v___x_2577_);
return v___x_2578_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_assertForwardHyp___lam__2(uint8_t v___x_2579_, lean_object* v_goal_2580_, lean_object* v___x_2581_, lean_object* v___x_2582_, lean_object* v___y_2583_, lean_object* v___y_2584_, lean_object* v___y_2585_, lean_object* v___y_2586_){
_start:
{
lean_object* v_keyedConfig_2588_; uint8_t v_trackZetaDelta_2589_; lean_object* v_zetaDeltaSet_2590_; lean_object* v_lctx_2591_; lean_object* v_localInstances_2592_; lean_object* v_defEqCtx_x3f_2593_; lean_object* v_synthPendingDepth_2594_; lean_object* v_customCanUnfoldPredicate_x3f_2595_; uint8_t v_univApprox_2596_; uint8_t v_inTypeClassResolution_2597_; uint8_t v_cacheInferType_2598_; lean_object* v___x_2600_; uint8_t v_isShared_2601_; uint8_t v_isSharedCheck_2638_; 
v_keyedConfig_2588_ = lean_ctor_get(v___y_2583_, 0);
v_trackZetaDelta_2589_ = lean_ctor_get_uint8(v___y_2583_, sizeof(void*)*7);
v_zetaDeltaSet_2590_ = lean_ctor_get(v___y_2583_, 1);
v_lctx_2591_ = lean_ctor_get(v___y_2583_, 2);
v_localInstances_2592_ = lean_ctor_get(v___y_2583_, 3);
v_defEqCtx_x3f_2593_ = lean_ctor_get(v___y_2583_, 4);
v_synthPendingDepth_2594_ = lean_ctor_get(v___y_2583_, 5);
v_customCanUnfoldPredicate_x3f_2595_ = lean_ctor_get(v___y_2583_, 6);
v_univApprox_2596_ = lean_ctor_get_uint8(v___y_2583_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_2597_ = lean_ctor_get_uint8(v___y_2583_, sizeof(void*)*7 + 2);
v_cacheInferType_2598_ = lean_ctor_get_uint8(v___y_2583_, sizeof(void*)*7 + 3);
v_isSharedCheck_2638_ = !lean_is_exclusive(v___y_2583_);
if (v_isSharedCheck_2638_ == 0)
{
v___x_2600_ = v___y_2583_;
v_isShared_2601_ = v_isSharedCheck_2638_;
goto v_resetjp_2599_;
}
else
{
lean_inc(v_customCanUnfoldPredicate_x3f_2595_);
lean_inc(v_synthPendingDepth_2594_);
lean_inc(v_defEqCtx_x3f_2593_);
lean_inc(v_localInstances_2592_);
lean_inc(v_lctx_2591_);
lean_inc(v_zetaDeltaSet_2590_);
lean_inc(v_keyedConfig_2588_);
lean_dec(v___y_2583_);
v___x_2600_ = lean_box(0);
v_isShared_2601_ = v_isSharedCheck_2638_;
goto v_resetjp_2599_;
}
v_resetjp_2599_:
{
lean_object* v___x_2602_; lean_object* v___x_2604_; 
v___x_2602_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_2579_, v_keyedConfig_2588_);
if (v_isShared_2601_ == 0)
{
lean_ctor_set(v___x_2600_, 0, v___x_2602_);
v___x_2604_ = v___x_2600_;
goto v_reusejp_2603_;
}
else
{
lean_object* v_reuseFailAlloc_2637_; 
v_reuseFailAlloc_2637_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v_reuseFailAlloc_2637_, 0, v___x_2602_);
lean_ctor_set(v_reuseFailAlloc_2637_, 1, v_zetaDeltaSet_2590_);
lean_ctor_set(v_reuseFailAlloc_2637_, 2, v_lctx_2591_);
lean_ctor_set(v_reuseFailAlloc_2637_, 3, v_localInstances_2592_);
lean_ctor_set(v_reuseFailAlloc_2637_, 4, v_defEqCtx_x3f_2593_);
lean_ctor_set(v_reuseFailAlloc_2637_, 5, v_synthPendingDepth_2594_);
lean_ctor_set(v_reuseFailAlloc_2637_, 6, v_customCanUnfoldPredicate_x3f_2595_);
lean_ctor_set_uint8(v_reuseFailAlloc_2637_, sizeof(void*)*7, v_trackZetaDelta_2589_);
lean_ctor_set_uint8(v_reuseFailAlloc_2637_, sizeof(void*)*7 + 1, v_univApprox_2596_);
lean_ctor_set_uint8(v_reuseFailAlloc_2637_, sizeof(void*)*7 + 2, v_inTypeClassResolution_2597_);
lean_ctor_set_uint8(v_reuseFailAlloc_2637_, sizeof(void*)*7 + 3, v_cacheInferType_2598_);
v___x_2604_ = v_reuseFailAlloc_2637_;
goto v_reusejp_2603_;
}
v_reusejp_2603_:
{
lean_object* v___x_2605_; 
v___x_2605_ = l_Lean_MVarId_assertHypotheses(v_goal_2580_, v___x_2581_, v___x_2604_, v___y_2584_, v___y_2585_, v___y_2586_);
if (lean_obj_tag(v___x_2605_) == 0)
{
lean_object* v_a_2606_; lean_object* v___x_2608_; uint8_t v_isShared_2609_; uint8_t v_isSharedCheck_2628_; 
v_a_2606_ = lean_ctor_get(v___x_2605_, 0);
v_isSharedCheck_2628_ = !lean_is_exclusive(v___x_2605_);
if (v_isSharedCheck_2628_ == 0)
{
v___x_2608_ = v___x_2605_;
v_isShared_2609_ = v_isSharedCheck_2628_;
goto v_resetjp_2607_;
}
else
{
lean_inc(v_a_2606_);
lean_dec(v___x_2605_);
v___x_2608_ = lean_box(0);
v_isShared_2609_ = v_isSharedCheck_2628_;
goto v_resetjp_2607_;
}
v_resetjp_2607_:
{
lean_object* v_fst_2610_; lean_object* v_snd_2611_; lean_object* v___x_2613_; uint8_t v_isShared_2614_; uint8_t v_isSharedCheck_2627_; 
v_fst_2610_ = lean_ctor_get(v_a_2606_, 0);
v_snd_2611_ = lean_ctor_get(v_a_2606_, 1);
v_isSharedCheck_2627_ = !lean_is_exclusive(v_a_2606_);
if (v_isSharedCheck_2627_ == 0)
{
v___x_2613_ = v_a_2606_;
v_isShared_2614_ = v_isSharedCheck_2627_;
goto v_resetjp_2612_;
}
else
{
lean_inc(v_snd_2611_);
lean_inc(v_fst_2610_);
lean_dec(v_a_2606_);
v___x_2613_ = lean_box(0);
v_isShared_2614_ = v_isSharedCheck_2627_;
goto v_resetjp_2612_;
}
v_resetjp_2612_:
{
lean_object* v___x_2615_; uint8_t v___x_2616_; 
v___x_2615_ = lean_array_get_size(v_fst_2610_);
v___x_2616_ = lean_nat_dec_eq(v___x_2615_, v___x_2582_);
if (v___x_2616_ == 0)
{
lean_object* v___x_2617_; lean_object* v___x_2618_; 
lean_del_object(v___x_2613_);
lean_dec(v_snd_2611_);
lean_dec(v_fst_2610_);
lean_del_object(v___x_2608_);
v___x_2617_ = lean_obj_once(&lp_aesop_Aesop_RuleTac_assertForwardHyp___lam__2___closed__1, &lp_aesop_Aesop_RuleTac_assertForwardHyp___lam__2___closed__1_once, _init_lp_aesop_Aesop_RuleTac_assertForwardHyp___lam__2___closed__1);
v___x_2618_ = lp_aesop_Lean_throwError___at___00Aesop_RuleTac_assertForwardHyp_spec__0___redArg(v___x_2617_, v___x_2604_, v___y_2584_, v___y_2585_, v___y_2586_);
lean_dec_ref(v___x_2604_);
return v___x_2618_;
}
else
{
lean_object* v___x_2619_; lean_object* v___x_2620_; lean_object* v___x_2622_; 
lean_dec_ref(v___x_2604_);
v___x_2619_ = lean_unsigned_to_nat(0u);
v___x_2620_ = lean_array_fget(v_fst_2610_, v___x_2619_);
lean_dec(v_fst_2610_);
if (v_isShared_2614_ == 0)
{
lean_ctor_set(v___x_2613_, 0, v___x_2620_);
v___x_2622_ = v___x_2613_;
goto v_reusejp_2621_;
}
else
{
lean_object* v_reuseFailAlloc_2626_; 
v_reuseFailAlloc_2626_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2626_, 0, v___x_2620_);
lean_ctor_set(v_reuseFailAlloc_2626_, 1, v_snd_2611_);
v___x_2622_ = v_reuseFailAlloc_2626_;
goto v_reusejp_2621_;
}
v_reusejp_2621_:
{
lean_object* v___x_2624_; 
if (v_isShared_2609_ == 0)
{
lean_ctor_set(v___x_2608_, 0, v___x_2622_);
v___x_2624_ = v___x_2608_;
goto v_reusejp_2623_;
}
else
{
lean_object* v_reuseFailAlloc_2625_; 
v_reuseFailAlloc_2625_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2625_, 0, v___x_2622_);
v___x_2624_ = v_reuseFailAlloc_2625_;
goto v_reusejp_2623_;
}
v_reusejp_2623_:
{
return v___x_2624_;
}
}
}
}
}
}
else
{
lean_object* v_a_2629_; lean_object* v___x_2631_; uint8_t v_isShared_2632_; uint8_t v_isSharedCheck_2636_; 
lean_dec_ref(v___x_2604_);
v_a_2629_ = lean_ctor_get(v___x_2605_, 0);
v_isSharedCheck_2636_ = !lean_is_exclusive(v___x_2605_);
if (v_isSharedCheck_2636_ == 0)
{
v___x_2631_ = v___x_2605_;
v_isShared_2632_ = v_isSharedCheck_2636_;
goto v_resetjp_2630_;
}
else
{
lean_inc(v_a_2629_);
lean_dec(v___x_2605_);
v___x_2631_ = lean_box(0);
v_isShared_2632_ = v_isSharedCheck_2636_;
goto v_resetjp_2630_;
}
v_resetjp_2630_:
{
lean_object* v___x_2634_; 
if (v_isShared_2632_ == 0)
{
v___x_2634_ = v___x_2631_;
goto v_reusejp_2633_;
}
else
{
lean_object* v_reuseFailAlloc_2635_; 
v_reuseFailAlloc_2635_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2635_, 0, v_a_2629_);
v___x_2634_ = v_reuseFailAlloc_2635_;
goto v_reusejp_2633_;
}
v_reusejp_2633_:
{
return v___x_2634_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_assertForwardHyp___lam__2___boxed(lean_object* v___x_2639_, lean_object* v_goal_2640_, lean_object* v___x_2641_, lean_object* v___x_2642_, lean_object* v___y_2643_, lean_object* v___y_2644_, lean_object* v___y_2645_, lean_object* v___y_2646_, lean_object* v___y_2647_){
_start:
{
uint8_t v___x_1305__boxed_2648_; lean_object* v_res_2649_; 
v___x_1305__boxed_2648_ = lean_unbox(v___x_2639_);
v_res_2649_ = lp_aesop_Aesop_RuleTac_assertForwardHyp___lam__2(v___x_1305__boxed_2648_, v_goal_2640_, v___x_2641_, v___x_2642_, v___y_2643_, v___y_2644_, v___y_2645_, v___y_2646_);
lean_dec(v___y_2646_);
lean_dec_ref(v___y_2645_);
lean_dec(v___y_2644_);
lean_dec(v___x_2642_);
return v_res_2649_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_assertForwardHyp(lean_object* v_goal_2652_, lean_object* v_hyp_2653_, lean_object* v_depth_2654_, lean_object* v_a_2655_, lean_object* v_a_2656_, lean_object* v_a_2657_, lean_object* v_a_2658_, lean_object* v_a_2659_, lean_object* v_a_2660_){
_start:
{
lean_object* v_userName_2662_; lean_object* v_type_2663_; lean_object* v_value_2664_; lean_object* v___f_2665_; lean_object* v___f_2666_; lean_object* v___x_2667_; uint8_t v___x_2668_; uint8_t v___x_2669_; lean_object* v_hyp_2670_; lean_object* v___x_2671_; uint8_t v___x_2672_; lean_object* v_implDetailHyp_2673_; lean_object* v___x_2674_; lean_object* v___x_2675_; lean_object* v___x_2676_; lean_object* v___x_2677_; uint8_t v___x_2678_; lean_object* v___x_2679_; lean_object* v___f_2680_; lean_object* v___x_2681_; 
v_userName_2662_ = lean_ctor_get(v_hyp_2653_, 0);
lean_inc_n(v_userName_2662_, 2);
v_type_2663_ = lean_ctor_get(v_hyp_2653_, 1);
lean_inc_ref_n(v_type_2663_, 2);
v_value_2664_ = lean_ctor_get(v_hyp_2653_, 2);
lean_inc_ref_n(v_value_2664_, 2);
v___f_2665_ = ((lean_object*)(lp_aesop_Aesop_RuleTac_assertForwardHyp___closed__0));
v___f_2666_ = ((lean_object*)(lp_aesop_Aesop_RuleTac_assertForwardHyp___closed__1));
lean_inc_n(v_goal_2652_, 2);
v___x_2667_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_assertForwardHyp_tacticBuilder___boxed), 8, 2);
lean_closure_set(v___x_2667_, 0, v_goal_2652_);
lean_closure_set(v___x_2667_, 1, v_hyp_2653_);
v___x_2668_ = 0;
v___x_2669_ = 0;
v_hyp_2670_ = lean_alloc_ctor(0, 3, 2);
lean_ctor_set(v_hyp_2670_, 0, v_userName_2662_);
lean_ctor_set(v_hyp_2670_, 1, v_type_2663_);
lean_ctor_set(v_hyp_2670_, 2, v_value_2664_);
lean_ctor_set_uint8(v_hyp_2670_, sizeof(void*)*3, v___x_2668_);
lean_ctor_set_uint8(v_hyp_2670_, sizeof(void*)*3 + 1, v___x_2669_);
v___x_2671_ = lp_aesop_Aesop_forwardImplDetailHypName(v_userName_2662_, v_depth_2654_);
v___x_2672_ = 1;
v_implDetailHyp_2673_ = lean_alloc_ctor(0, 3, 2);
lean_ctor_set(v_implDetailHyp_2673_, 0, v___x_2671_);
lean_ctor_set(v_implDetailHyp_2673_, 1, v_type_2663_);
lean_ctor_set(v_implDetailHyp_2673_, 2, v_value_2664_);
lean_ctor_set_uint8(v_implDetailHyp_2673_, sizeof(void*)*3, v___x_2668_);
lean_ctor_set_uint8(v_implDetailHyp_2673_, sizeof(void*)*3 + 1, v___x_2672_);
v___x_2674_ = lean_unsigned_to_nat(2u);
v___x_2675_ = lean_mk_empty_array_with_capacity(v___x_2674_);
v___x_2676_ = lean_array_push(v___x_2675_, v_hyp_2670_);
v___x_2677_ = lean_array_push(v___x_2676_, v_implDetailHyp_2673_);
v___x_2678_ = 2;
v___x_2679_ = lean_box(v___x_2678_);
v___f_2680_ = lean_alloc_closure((void*)(lp_aesop_Aesop_RuleTac_assertForwardHyp___lam__2___boxed), 9, 4);
lean_closure_set(v___f_2680_, 0, v___x_2679_);
lean_closure_set(v___f_2680_, 1, v_goal_2652_);
lean_closure_set(v___f_2680_, 2, v___x_2677_);
lean_closure_set(v___f_2680_, 3, v___x_2674_);
v___x_2681_ = lp_aesop_Aesop_withScriptStep___redArg(v_goal_2652_, v___f_2665_, v___f_2666_, v___x_2667_, v___f_2680_, v_a_2655_, v_a_2656_, v_a_2657_, v_a_2658_, v_a_2659_, v_a_2660_);
return v___x_2681_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_assertForwardHyp___boxed(lean_object* v_goal_2682_, lean_object* v_hyp_2683_, lean_object* v_depth_2684_, lean_object* v_a_2685_, lean_object* v_a_2686_, lean_object* v_a_2687_, lean_object* v_a_2688_, lean_object* v_a_2689_, lean_object* v_a_2690_, lean_object* v_a_2691_){
_start:
{
lean_object* v_res_2692_; 
v_res_2692_ = lp_aesop_Aesop_RuleTac_assertForwardHyp(v_goal_2682_, v_hyp_2683_, v_depth_2684_, v_a_2685_, v_a_2686_, v_a_2687_, v_a_2688_, v_a_2689_, v_a_2690_);
lean_dec(v_a_2690_);
lean_dec_ref(v_a_2689_);
lean_dec(v_a_2688_);
lean_dec_ref(v_a_2687_);
lean_dec(v_a_2686_);
lean_dec(v_a_2685_);
return v_res_2692_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_RuleTac_assertForwardHyp_spec__0(lean_object* v_00_u03b1_2693_, lean_object* v_msg_2694_, lean_object* v___y_2695_, lean_object* v___y_2696_, lean_object* v___y_2697_, lean_object* v___y_2698_){
_start:
{
lean_object* v___x_2700_; 
v___x_2700_ = lp_aesop_Lean_throwError___at___00Aesop_RuleTac_assertForwardHyp_spec__0___redArg(v_msg_2694_, v___y_2695_, v___y_2696_, v___y_2697_, v___y_2698_);
return v___x_2700_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_RuleTac_assertForwardHyp_spec__0___boxed(lean_object* v_00_u03b1_2701_, lean_object* v_msg_2702_, lean_object* v___y_2703_, lean_object* v___y_2704_, lean_object* v___y_2705_, lean_object* v___y_2706_, lean_object* v___y_2707_){
_start:
{
lean_object* v_res_2708_; 
v_res_2708_ = lp_aesop_Lean_throwError___at___00Aesop_RuleTac_assertForwardHyp_spec__0(v_00_u03b1_2701_, v_msg_2702_, v___y_2703_, v___y_2704_, v___y_2705_, v___y_2706_);
lean_dec(v___y_2706_);
lean_dec_ref(v___y_2705_);
lean_dec(v___y_2704_);
lean_dec_ref(v___y_2703_);
return v_res_2708_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_getUnusedUserNames___at___00Aesop_RuleTac_applyForwardRule_spec__0___redArg(lean_object* v_n_2709_, lean_object* v_suggestion_2710_, lean_object* v___y_2711_){
_start:
{
lean_object* v_lctx_2713_; lean_object* v___x_2714_; lean_object* v___x_2715_; 
v_lctx_2713_ = lean_ctor_get(v___y_2711_, 2);
v___x_2714_ = lp_batteries_Lean_LocalContext_getUnusedUserNames(v_lctx_2713_, v_n_2709_, v_suggestion_2710_);
v___x_2715_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2715_, 0, v___x_2714_);
return v___x_2715_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_getUnusedUserNames___at___00Aesop_RuleTac_applyForwardRule_spec__0___redArg___boxed(lean_object* v_n_2716_, lean_object* v_suggestion_2717_, lean_object* v___y_2718_, lean_object* v___y_2719_){
_start:
{
lean_object* v_res_2720_; 
v_res_2720_ = lp_aesop_Lean_Meta_getUnusedUserNames___at___00Aesop_RuleTac_applyForwardRule_spec__0___redArg(v_n_2716_, v_suggestion_2717_, v___y_2718_);
lean_dec_ref(v___y_2718_);
return v_res_2720_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_getUnusedUserNames___at___00Aesop_RuleTac_applyForwardRule_spec__0(lean_object* v_n_2721_, lean_object* v_suggestion_2722_, lean_object* v___y_2723_, lean_object* v___y_2724_, lean_object* v___y_2725_, lean_object* v___y_2726_, lean_object* v___y_2727_, lean_object* v___y_2728_){
_start:
{
lean_object* v___x_2730_; 
v___x_2730_ = lp_aesop_Lean_Meta_getUnusedUserNames___at___00Aesop_RuleTac_applyForwardRule_spec__0___redArg(v_n_2721_, v_suggestion_2722_, v___y_2725_);
return v___x_2730_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_getUnusedUserNames___at___00Aesop_RuleTac_applyForwardRule_spec__0___boxed(lean_object* v_n_2731_, lean_object* v_suggestion_2732_, lean_object* v___y_2733_, lean_object* v___y_2734_, lean_object* v___y_2735_, lean_object* v___y_2736_, lean_object* v___y_2737_, lean_object* v___y_2738_, lean_object* v___y_2739_){
_start:
{
lean_object* v_res_2740_; 
v_res_2740_ = lp_aesop_Lean_Meta_getUnusedUserNames___at___00Aesop_RuleTac_applyForwardRule_spec__0(v_n_2731_, v_suggestion_2732_, v___y_2733_, v___y_2734_, v___y_2735_, v___y_2736_, v___y_2737_, v___y_2738_);
lean_dec(v___y_2738_);
lean_dec_ref(v___y_2737_);
lean_dec(v___y_2736_);
lean_dec_ref(v___y_2735_);
lean_dec(v___y_2734_);
lean_dec(v___y_2733_);
return v_res_2740_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_RuleTac_applyForwardRule_spec__3___redArg___lam__0(lean_object* v_x_2741_, lean_object* v___y_2742_, lean_object* v___y_2743_, lean_object* v___y_2744_, lean_object* v___y_2745_, lean_object* v___y_2746_, lean_object* v___y_2747_){
_start:
{
lean_object* v___x_2749_; 
lean_inc(v___y_2743_);
lean_inc(v___y_2742_);
v___x_2749_ = lean_apply_7(v_x_2741_, v___y_2742_, v___y_2743_, v___y_2744_, v___y_2745_, v___y_2746_, v___y_2747_, lean_box(0));
return v___x_2749_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_RuleTac_applyForwardRule_spec__3___redArg___lam__0___boxed(lean_object* v_x_2750_, lean_object* v___y_2751_, lean_object* v___y_2752_, lean_object* v___y_2753_, lean_object* v___y_2754_, lean_object* v___y_2755_, lean_object* v___y_2756_, lean_object* v___y_2757_){
_start:
{
lean_object* v_res_2758_; 
v_res_2758_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_RuleTac_applyForwardRule_spec__3___redArg___lam__0(v_x_2750_, v___y_2751_, v___y_2752_, v___y_2753_, v___y_2754_, v___y_2755_, v___y_2756_);
lean_dec(v___y_2752_);
lean_dec(v___y_2751_);
return v_res_2758_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_RuleTac_applyForwardRule_spec__3___redArg(lean_object* v_mvarId_2759_, lean_object* v_x_2760_, lean_object* v___y_2761_, lean_object* v___y_2762_, lean_object* v___y_2763_, lean_object* v___y_2764_, lean_object* v___y_2765_, lean_object* v___y_2766_){
_start:
{
lean_object* v___f_2768_; lean_object* v___x_2769_; 
lean_inc(v___y_2762_);
lean_inc(v___y_2761_);
v___f_2768_ = lean_alloc_closure((void*)(lp_aesop_Lean_MVarId_withContext___at___00Aesop_RuleTac_applyForwardRule_spec__3___redArg___lam__0___boxed), 8, 3);
lean_closure_set(v___f_2768_, 0, v_x_2760_);
lean_closure_set(v___f_2768_, 1, v___y_2761_);
lean_closure_set(v___f_2768_, 2, v___y_2762_);
v___x_2769_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_box(0), v_mvarId_2759_, v___f_2768_, v___y_2763_, v___y_2764_, v___y_2765_, v___y_2766_);
if (lean_obj_tag(v___x_2769_) == 0)
{
return v___x_2769_;
}
else
{
lean_object* v_a_2770_; lean_object* v___x_2772_; uint8_t v_isShared_2773_; uint8_t v_isSharedCheck_2777_; 
v_a_2770_ = lean_ctor_get(v___x_2769_, 0);
v_isSharedCheck_2777_ = !lean_is_exclusive(v___x_2769_);
if (v_isSharedCheck_2777_ == 0)
{
v___x_2772_ = v___x_2769_;
v_isShared_2773_ = v_isSharedCheck_2777_;
goto v_resetjp_2771_;
}
else
{
lean_inc(v_a_2770_);
lean_dec(v___x_2769_);
v___x_2772_ = lean_box(0);
v_isShared_2773_ = v_isSharedCheck_2777_;
goto v_resetjp_2771_;
}
v_resetjp_2771_:
{
lean_object* v___x_2775_; 
if (v_isShared_2773_ == 0)
{
v___x_2775_ = v___x_2772_;
goto v_reusejp_2774_;
}
else
{
lean_object* v_reuseFailAlloc_2776_; 
v_reuseFailAlloc_2776_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2776_, 0, v_a_2770_);
v___x_2775_ = v_reuseFailAlloc_2776_;
goto v_reusejp_2774_;
}
v_reusejp_2774_:
{
return v___x_2775_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_RuleTac_applyForwardRule_spec__3___redArg___boxed(lean_object* v_mvarId_2778_, lean_object* v_x_2779_, lean_object* v___y_2780_, lean_object* v___y_2781_, lean_object* v___y_2782_, lean_object* v___y_2783_, lean_object* v___y_2784_, lean_object* v___y_2785_, lean_object* v___y_2786_){
_start:
{
lean_object* v_res_2787_; 
v_res_2787_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_RuleTac_applyForwardRule_spec__3___redArg(v_mvarId_2778_, v_x_2779_, v___y_2780_, v___y_2781_, v___y_2782_, v___y_2783_, v___y_2784_, v___y_2785_);
lean_dec(v___y_2785_);
lean_dec_ref(v___y_2784_);
lean_dec(v___y_2783_);
lean_dec_ref(v___y_2782_);
lean_dec(v___y_2781_);
lean_dec(v___y_2780_);
return v_res_2787_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_RuleTac_applyForwardRule_spec__3(lean_object* v_00_u03b1_2788_, lean_object* v_mvarId_2789_, lean_object* v_x_2790_, lean_object* v___y_2791_, lean_object* v___y_2792_, lean_object* v___y_2793_, lean_object* v___y_2794_, lean_object* v___y_2795_, lean_object* v___y_2796_){
_start:
{
lean_object* v___x_2798_; 
v___x_2798_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_RuleTac_applyForwardRule_spec__3___redArg(v_mvarId_2789_, v_x_2790_, v___y_2791_, v___y_2792_, v___y_2793_, v___y_2794_, v___y_2795_, v___y_2796_);
return v___x_2798_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_RuleTac_applyForwardRule_spec__3___boxed(lean_object* v_00_u03b1_2799_, lean_object* v_mvarId_2800_, lean_object* v_x_2801_, lean_object* v___y_2802_, lean_object* v___y_2803_, lean_object* v___y_2804_, lean_object* v___y_2805_, lean_object* v___y_2806_, lean_object* v___y_2807_, lean_object* v___y_2808_){
_start:
{
lean_object* v_res_2809_; 
v_res_2809_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_RuleTac_applyForwardRule_spec__3(v_00_u03b1_2799_, v_mvarId_2800_, v_x_2801_, v___y_2802_, v___y_2803_, v___y_2804_, v___y_2805_, v___y_2806_, v___y_2807_);
lean_dec(v___y_2807_);
lean_dec_ref(v___y_2806_);
lean_dec(v___y_2805_);
lean_dec_ref(v___y_2804_);
lean_dec(v___y_2803_);
lean_dec(v___y_2802_);
return v_res_2809_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_RuleTac_applyForwardRule_spec__2___redArg(lean_object* v_as_2810_, size_t v_i_2811_, size_t v_stop_2812_, lean_object* v_b_2813_, lean_object* v___y_2814_, lean_object* v___y_2815_, lean_object* v___y_2816_, lean_object* v___y_2817_){
_start:
{
uint8_t v___x_2819_; 
v___x_2819_ = lean_usize_dec_eq(v_i_2811_, v_stop_2812_);
if (v___x_2819_ == 0)
{
lean_object* v___x_2820_; lean_object* v___x_2821_; lean_object* v___x_2822_; 
v___x_2820_ = lean_array_uget_borrowed(v_as_2810_, v_i_2811_);
lean_inc(v___x_2820_);
v___x_2821_ = l_Lean_Expr_fvar___override(v___x_2820_);
v___x_2822_ = l_Lean_Meta_isProof(v___x_2821_, v___y_2814_, v___y_2815_, v___y_2816_, v___y_2817_);
if (lean_obj_tag(v___x_2822_) == 0)
{
lean_object* v_a_2823_; lean_object* v_a_2825_; uint8_t v___x_2829_; 
v_a_2823_ = lean_ctor_get(v___x_2822_, 0);
lean_inc(v_a_2823_);
lean_dec_ref_known(v___x_2822_, 1);
v___x_2829_ = lean_unbox(v_a_2823_);
lean_dec(v_a_2823_);
if (v___x_2829_ == 0)
{
v_a_2825_ = v_b_2813_;
goto v___jp_2824_;
}
else
{
lean_object* v___x_2830_; 
lean_inc(v___x_2820_);
v___x_2830_ = lean_array_push(v_b_2813_, v___x_2820_);
v_a_2825_ = v___x_2830_;
goto v___jp_2824_;
}
v___jp_2824_:
{
size_t v___x_2826_; size_t v___x_2827_; 
v___x_2826_ = ((size_t)1ULL);
v___x_2827_ = lean_usize_add(v_i_2811_, v___x_2826_);
v_i_2811_ = v___x_2827_;
v_b_2813_ = v_a_2825_;
goto _start;
}
}
else
{
lean_object* v_a_2831_; lean_object* v___x_2833_; uint8_t v_isShared_2834_; uint8_t v_isSharedCheck_2838_; 
lean_dec_ref(v_b_2813_);
v_a_2831_ = lean_ctor_get(v___x_2822_, 0);
v_isSharedCheck_2838_ = !lean_is_exclusive(v___x_2822_);
if (v_isSharedCheck_2838_ == 0)
{
v___x_2833_ = v___x_2822_;
v_isShared_2834_ = v_isSharedCheck_2838_;
goto v_resetjp_2832_;
}
else
{
lean_inc(v_a_2831_);
lean_dec(v___x_2822_);
v___x_2833_ = lean_box(0);
v_isShared_2834_ = v_isSharedCheck_2838_;
goto v_resetjp_2832_;
}
v_resetjp_2832_:
{
lean_object* v___x_2836_; 
if (v_isShared_2834_ == 0)
{
v___x_2836_ = v___x_2833_;
goto v_reusejp_2835_;
}
else
{
lean_object* v_reuseFailAlloc_2837_; 
v_reuseFailAlloc_2837_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2837_, 0, v_a_2831_);
v___x_2836_ = v_reuseFailAlloc_2837_;
goto v_reusejp_2835_;
}
v_reusejp_2835_:
{
return v___x_2836_;
}
}
}
}
else
{
lean_object* v___x_2839_; 
v___x_2839_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2839_, 0, v_b_2813_);
return v___x_2839_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_RuleTac_applyForwardRule_spec__2___redArg___boxed(lean_object* v_as_2840_, lean_object* v_i_2841_, lean_object* v_stop_2842_, lean_object* v_b_2843_, lean_object* v___y_2844_, lean_object* v___y_2845_, lean_object* v___y_2846_, lean_object* v___y_2847_, lean_object* v___y_2848_){
_start:
{
size_t v_i_boxed_2849_; size_t v_stop_boxed_2850_; lean_object* v_res_2851_; 
v_i_boxed_2849_ = lean_unbox_usize(v_i_2841_);
lean_dec(v_i_2841_);
v_stop_boxed_2850_ = lean_unbox_usize(v_stop_2842_);
lean_dec(v_stop_2842_);
v_res_2851_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_RuleTac_applyForwardRule_spec__2___redArg(v_as_2840_, v_i_boxed_2849_, v_stop_boxed_2850_, v_b_2843_, v___y_2844_, v___y_2845_, v___y_2846_, v___y_2847_);
lean_dec(v___y_2847_);
lean_dec_ref(v___y_2846_);
lean_dec(v___y_2845_);
lean_dec_ref(v___y_2844_);
lean_dec_ref(v_as_2840_);
return v_res_2851_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_applyForwardRule___lam__0(lean_object* v___x_2852_, lean_object* v___x_2853_, lean_object* v___y_2854_, size_t v___x_2855_, lean_object* v___y_2856_, lean_object* v___y_2857_, lean_object* v___y_2858_, lean_object* v___y_2859_, lean_object* v___y_2860_, lean_object* v___y_2861_){
_start:
{
lean_object* v___x_2863_; uint8_t v___x_2864_; 
v___x_2863_ = lean_mk_empty_array_with_capacity(v___x_2852_);
v___x_2864_ = lean_nat_dec_lt(v___x_2852_, v___x_2853_);
if (v___x_2864_ == 0)
{
lean_object* v___x_2865_; 
v___x_2865_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2865_, 0, v___x_2863_);
return v___x_2865_;
}
else
{
uint8_t v___x_2866_; 
v___x_2866_ = lean_nat_dec_le(v___x_2853_, v___x_2853_);
if (v___x_2866_ == 0)
{
if (v___x_2864_ == 0)
{
lean_object* v___x_2867_; 
v___x_2867_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2867_, 0, v___x_2863_);
return v___x_2867_;
}
else
{
size_t v___x_2868_; lean_object* v___x_2869_; 
v___x_2868_ = lean_usize_of_nat(v___x_2853_);
v___x_2869_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_RuleTac_applyForwardRule_spec__2___redArg(v___y_2854_, v___x_2855_, v___x_2868_, v___x_2863_, v___y_2858_, v___y_2859_, v___y_2860_, v___y_2861_);
return v___x_2869_;
}
}
else
{
size_t v___x_2870_; lean_object* v___x_2871_; 
v___x_2870_ = lean_usize_of_nat(v___x_2853_);
v___x_2871_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_RuleTac_applyForwardRule_spec__2___redArg(v___y_2854_, v___x_2855_, v___x_2870_, v___x_2863_, v___y_2858_, v___y_2859_, v___y_2860_, v___y_2861_);
return v___x_2871_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_applyForwardRule___lam__0___boxed(lean_object* v___x_2872_, lean_object* v___x_2873_, lean_object* v___y_2874_, lean_object* v___x_2875_, lean_object* v___y_2876_, lean_object* v___y_2877_, lean_object* v___y_2878_, lean_object* v___y_2879_, lean_object* v___y_2880_, lean_object* v___y_2881_, lean_object* v___y_2882_){
_start:
{
size_t v___x_17821__boxed_2883_; lean_object* v_res_2884_; 
v___x_17821__boxed_2883_ = lean_unbox_usize(v___x_2875_);
lean_dec(v___x_2875_);
v_res_2884_ = lp_aesop_Aesop_RuleTac_applyForwardRule___lam__0(v___x_2872_, v___x_2873_, v___y_2874_, v___x_17821__boxed_2883_, v___y_2876_, v___y_2877_, v___y_2878_, v___y_2879_, v___y_2880_, v___y_2881_);
lean_dec(v___y_2881_);
lean_dec_ref(v___y_2880_);
lean_dec(v___y_2879_);
lean_dec_ref(v___y_2878_);
lean_dec(v___y_2877_);
lean_dec(v___y_2876_);
lean_dec_ref(v___y_2874_);
lean_dec(v___x_2873_);
lean_dec(v___x_2872_);
return v_res_2884_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_applyForwardRule_spec__1(lean_object* v_as_2885_, size_t v_sz_2886_, size_t v_i_2887_, lean_object* v_b_2888_, lean_object* v___y_2889_, lean_object* v___y_2890_, lean_object* v___y_2891_, lean_object* v___y_2892_, lean_object* v___y_2893_, lean_object* v___y_2894_){
_start:
{
uint8_t v___x_2896_; 
v___x_2896_ = lean_usize_dec_lt(v_i_2887_, v_sz_2886_);
if (v___x_2896_ == 0)
{
lean_object* v___x_2897_; 
v___x_2897_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2897_, 0, v_b_2888_);
return v___x_2897_;
}
else
{
lean_object* v_snd_2898_; lean_object* v_snd_2899_; lean_object* v_fst_2900_; lean_object* v___x_2902_; uint8_t v_isShared_2903_; uint8_t v_isSharedCheck_2973_; 
v_snd_2898_ = lean_ctor_get(v_b_2888_, 1);
lean_inc(v_snd_2898_);
v_snd_2899_ = lean_ctor_get(v_snd_2898_, 1);
lean_inc(v_snd_2899_);
v_fst_2900_ = lean_ctor_get(v_b_2888_, 0);
v_isSharedCheck_2973_ = !lean_is_exclusive(v_b_2888_);
if (v_isSharedCheck_2973_ == 0)
{
lean_object* v_unused_2974_; 
v_unused_2974_ = lean_ctor_get(v_b_2888_, 1);
lean_dec(v_unused_2974_);
v___x_2902_ = v_b_2888_;
v_isShared_2903_ = v_isSharedCheck_2973_;
goto v_resetjp_2901_;
}
else
{
lean_inc(v_fst_2900_);
lean_dec(v_b_2888_);
v___x_2902_ = lean_box(0);
v_isShared_2903_ = v_isSharedCheck_2973_;
goto v_resetjp_2901_;
}
v_resetjp_2901_:
{
lean_object* v_fst_2904_; lean_object* v___x_2906_; uint8_t v_isShared_2907_; uint8_t v_isSharedCheck_2971_; 
v_fst_2904_ = lean_ctor_get(v_snd_2898_, 0);
v_isSharedCheck_2971_ = !lean_is_exclusive(v_snd_2898_);
if (v_isSharedCheck_2971_ == 0)
{
lean_object* v_unused_2972_; 
v_unused_2972_ = lean_ctor_get(v_snd_2898_, 1);
lean_dec(v_unused_2972_);
v___x_2906_ = v_snd_2898_;
v_isShared_2907_ = v_isSharedCheck_2971_;
goto v_resetjp_2905_;
}
else
{
lean_inc(v_fst_2904_);
lean_dec(v_snd_2898_);
v___x_2906_ = lean_box(0);
v_isShared_2907_ = v_isSharedCheck_2971_;
goto v_resetjp_2905_;
}
v_resetjp_2905_:
{
lean_object* v_array_2908_; lean_object* v_start_2909_; lean_object* v_stop_2910_; uint8_t v___x_2911_; 
v_array_2908_ = lean_ctor_get(v_snd_2899_, 0);
v_start_2909_ = lean_ctor_get(v_snd_2899_, 1);
v_stop_2910_ = lean_ctor_get(v_snd_2899_, 2);
v___x_2911_ = lean_nat_dec_lt(v_start_2909_, v_stop_2910_);
if (v___x_2911_ == 0)
{
lean_object* v___x_2913_; 
if (v_isShared_2907_ == 0)
{
v___x_2913_ = v___x_2906_;
goto v_reusejp_2912_;
}
else
{
lean_object* v_reuseFailAlloc_2918_; 
v_reuseFailAlloc_2918_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2918_, 0, v_fst_2904_);
lean_ctor_set(v_reuseFailAlloc_2918_, 1, v_snd_2899_);
v___x_2913_ = v_reuseFailAlloc_2918_;
goto v_reusejp_2912_;
}
v_reusejp_2912_:
{
lean_object* v___x_2915_; 
if (v_isShared_2903_ == 0)
{
lean_ctor_set(v___x_2902_, 1, v___x_2913_);
v___x_2915_ = v___x_2902_;
goto v_reusejp_2914_;
}
else
{
lean_object* v_reuseFailAlloc_2917_; 
v_reuseFailAlloc_2917_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2917_, 0, v_fst_2900_);
lean_ctor_set(v_reuseFailAlloc_2917_, 1, v___x_2913_);
v___x_2915_ = v_reuseFailAlloc_2917_;
goto v_reusejp_2914_;
}
v_reusejp_2914_:
{
lean_object* v___x_2916_; 
v___x_2916_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2916_, 0, v___x_2915_);
return v___x_2916_;
}
}
}
else
{
lean_object* v___x_2920_; uint8_t v_isShared_2921_; uint8_t v_isSharedCheck_2967_; 
lean_inc(v_stop_2910_);
lean_inc(v_start_2909_);
lean_inc_ref(v_array_2908_);
lean_del_object(v___x_2906_);
lean_del_object(v___x_2902_);
v_isSharedCheck_2967_ = !lean_is_exclusive(v_snd_2899_);
if (v_isSharedCheck_2967_ == 0)
{
lean_object* v_unused_2968_; lean_object* v_unused_2969_; lean_object* v_unused_2970_; 
v_unused_2968_ = lean_ctor_get(v_snd_2899_, 2);
lean_dec(v_unused_2968_);
v_unused_2969_ = lean_ctor_get(v_snd_2899_, 1);
lean_dec(v_unused_2969_);
v_unused_2970_ = lean_ctor_get(v_snd_2899_, 0);
lean_dec(v_unused_2970_);
v___x_2920_ = v_snd_2899_;
v_isShared_2921_ = v_isSharedCheck_2967_;
goto v_resetjp_2919_;
}
else
{
lean_dec(v_snd_2899_);
v___x_2920_ = lean_box(0);
v_isShared_2921_ = v_isSharedCheck_2967_;
goto v_resetjp_2919_;
}
v_resetjp_2919_:
{
lean_object* v_a_2922_; lean_object* v_snd_2923_; lean_object* v_fst_2924_; lean_object* v_fst_2925_; lean_object* v_snd_2926_; lean_object* v___x_2928_; uint8_t v_isShared_2929_; uint8_t v_isSharedCheck_2966_; 
v_a_2922_ = lean_array_uget_borrowed(v_as_2885_, v_i_2887_);
v_snd_2923_ = lean_ctor_get(v_a_2922_, 1);
lean_inc(v_snd_2923_);
v_fst_2924_ = lean_ctor_get(v_a_2922_, 0);
v_fst_2925_ = lean_ctor_get(v_snd_2923_, 0);
v_snd_2926_ = lean_ctor_get(v_snd_2923_, 1);
v_isSharedCheck_2966_ = !lean_is_exclusive(v_snd_2923_);
if (v_isSharedCheck_2966_ == 0)
{
v___x_2928_ = v_snd_2923_;
v_isShared_2929_ = v_isSharedCheck_2966_;
goto v_resetjp_2927_;
}
else
{
lean_inc(v_snd_2926_);
lean_inc(v_fst_2925_);
lean_dec(v_snd_2923_);
v___x_2928_ = lean_box(0);
v_isShared_2929_ = v_isSharedCheck_2966_;
goto v_resetjp_2927_;
}
v_resetjp_2927_:
{
lean_object* v___x_2930_; uint8_t v___x_2931_; uint8_t v___x_2932_; lean_object* v___x_2933_; lean_object* v___x_2934_; 
v___x_2930_ = lean_array_fget_borrowed(v_array_2908_, v_start_2909_);
v___x_2931_ = 0;
v___x_2932_ = 0;
lean_inc(v_fst_2924_);
lean_inc(v___x_2930_);
v___x_2933_ = lean_alloc_ctor(0, 3, 2);
lean_ctor_set(v___x_2933_, 0, v___x_2930_);
lean_ctor_set(v___x_2933_, 1, v_fst_2925_);
lean_ctor_set(v___x_2933_, 2, v_fst_2924_);
lean_ctor_set_uint8(v___x_2933_, sizeof(void*)*3, v___x_2931_);
lean_ctor_set_uint8(v___x_2933_, sizeof(void*)*3 + 1, v___x_2932_);
v___x_2934_ = lp_aesop_Aesop_RuleTac_assertForwardHyp(v_fst_2900_, v___x_2933_, v_snd_2926_, v___y_2889_, v___y_2890_, v___y_2891_, v___y_2892_, v___y_2893_, v___y_2894_);
if (lean_obj_tag(v___x_2934_) == 0)
{
lean_object* v_a_2935_; lean_object* v_fst_2936_; lean_object* v_snd_2937_; lean_object* v___x_2939_; uint8_t v_isShared_2940_; uint8_t v_isSharedCheck_2957_; 
v_a_2935_ = lean_ctor_get(v___x_2934_, 0);
lean_inc(v_a_2935_);
lean_dec_ref_known(v___x_2934_, 1);
v_fst_2936_ = lean_ctor_get(v_a_2935_, 0);
v_snd_2937_ = lean_ctor_get(v_a_2935_, 1);
v_isSharedCheck_2957_ = !lean_is_exclusive(v_a_2935_);
if (v_isSharedCheck_2957_ == 0)
{
v___x_2939_ = v_a_2935_;
v_isShared_2940_ = v_isSharedCheck_2957_;
goto v_resetjp_2938_;
}
else
{
lean_inc(v_snd_2937_);
lean_inc(v_fst_2936_);
lean_dec(v_a_2935_);
v___x_2939_ = lean_box(0);
v_isShared_2940_ = v_isSharedCheck_2957_;
goto v_resetjp_2938_;
}
v_resetjp_2938_:
{
lean_object* v___x_2941_; lean_object* v___x_2942_; lean_object* v___x_2944_; 
v___x_2941_ = lean_unsigned_to_nat(1u);
v___x_2942_ = lean_nat_add(v_start_2909_, v___x_2941_);
lean_dec(v_start_2909_);
if (v_isShared_2921_ == 0)
{
lean_ctor_set(v___x_2920_, 1, v___x_2942_);
v___x_2944_ = v___x_2920_;
goto v_reusejp_2943_;
}
else
{
lean_object* v_reuseFailAlloc_2956_; 
v_reuseFailAlloc_2956_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_2956_, 0, v_array_2908_);
lean_ctor_set(v_reuseFailAlloc_2956_, 1, v___x_2942_);
lean_ctor_set(v_reuseFailAlloc_2956_, 2, v_stop_2910_);
v___x_2944_ = v_reuseFailAlloc_2956_;
goto v_reusejp_2943_;
}
v_reusejp_2943_:
{
lean_object* v___x_2945_; lean_object* v___x_2946_; lean_object* v___x_2948_; 
v___x_2945_ = lean_box(0);
v___x_2946_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__0_spec__0___redArg(v_fst_2904_, v_fst_2936_, v___x_2945_);
if (v_isShared_2940_ == 0)
{
lean_ctor_set(v___x_2939_, 1, v___x_2944_);
lean_ctor_set(v___x_2939_, 0, v___x_2946_);
v___x_2948_ = v___x_2939_;
goto v_reusejp_2947_;
}
else
{
lean_object* v_reuseFailAlloc_2955_; 
v_reuseFailAlloc_2955_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2955_, 0, v___x_2946_);
lean_ctor_set(v_reuseFailAlloc_2955_, 1, v___x_2944_);
v___x_2948_ = v_reuseFailAlloc_2955_;
goto v_reusejp_2947_;
}
v_reusejp_2947_:
{
lean_object* v___x_2950_; 
if (v_isShared_2929_ == 0)
{
lean_ctor_set(v___x_2928_, 1, v___x_2948_);
lean_ctor_set(v___x_2928_, 0, v_snd_2937_);
v___x_2950_ = v___x_2928_;
goto v_reusejp_2949_;
}
else
{
lean_object* v_reuseFailAlloc_2954_; 
v_reuseFailAlloc_2954_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2954_, 0, v_snd_2937_);
lean_ctor_set(v_reuseFailAlloc_2954_, 1, v___x_2948_);
v___x_2950_ = v_reuseFailAlloc_2954_;
goto v_reusejp_2949_;
}
v_reusejp_2949_:
{
size_t v___x_2951_; size_t v___x_2952_; 
v___x_2951_ = ((size_t)1ULL);
v___x_2952_ = lean_usize_add(v_i_2887_, v___x_2951_);
v_i_2887_ = v___x_2952_;
v_b_2888_ = v___x_2950_;
goto _start;
}
}
}
}
}
else
{
lean_object* v_a_2958_; lean_object* v___x_2960_; uint8_t v_isShared_2961_; uint8_t v_isSharedCheck_2965_; 
lean_del_object(v___x_2928_);
lean_del_object(v___x_2920_);
lean_dec(v_stop_2910_);
lean_dec(v_start_2909_);
lean_dec_ref(v_array_2908_);
lean_dec(v_fst_2904_);
v_a_2958_ = lean_ctor_get(v___x_2934_, 0);
v_isSharedCheck_2965_ = !lean_is_exclusive(v___x_2934_);
if (v_isSharedCheck_2965_ == 0)
{
v___x_2960_ = v___x_2934_;
v_isShared_2961_ = v_isSharedCheck_2965_;
goto v_resetjp_2959_;
}
else
{
lean_inc(v_a_2958_);
lean_dec(v___x_2934_);
v___x_2960_ = lean_box(0);
v_isShared_2961_ = v_isSharedCheck_2965_;
goto v_resetjp_2959_;
}
v_resetjp_2959_:
{
lean_object* v___x_2963_; 
if (v_isShared_2961_ == 0)
{
v___x_2963_ = v___x_2960_;
goto v_reusejp_2962_;
}
else
{
lean_object* v_reuseFailAlloc_2964_; 
v_reuseFailAlloc_2964_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2964_, 0, v_a_2958_);
v___x_2963_ = v_reuseFailAlloc_2964_;
goto v_reusejp_2962_;
}
v_reusejp_2962_:
{
return v___x_2963_;
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
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_applyForwardRule_spec__1___boxed(lean_object* v_as_2975_, lean_object* v_sz_2976_, lean_object* v_i_2977_, lean_object* v_b_2978_, lean_object* v___y_2979_, lean_object* v___y_2980_, lean_object* v___y_2981_, lean_object* v___y_2982_, lean_object* v___y_2983_, lean_object* v___y_2984_, lean_object* v___y_2985_){
_start:
{
size_t v_sz_boxed_2986_; size_t v_i_boxed_2987_; lean_object* v_res_2988_; 
v_sz_boxed_2986_ = lean_unbox_usize(v_sz_2976_);
lean_dec(v_sz_2976_);
v_i_boxed_2987_ = lean_unbox_usize(v_i_2977_);
lean_dec(v_i_2977_);
v_res_2988_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_applyForwardRule_spec__1(v_as_2975_, v_sz_boxed_2986_, v_i_boxed_2987_, v_b_2978_, v___y_2979_, v___y_2980_, v___y_2981_, v___y_2982_, v___y_2983_, v___y_2984_);
lean_dec(v___y_2984_);
lean_dec_ref(v___y_2983_);
lean_dec(v___y_2982_);
lean_dec_ref(v___y_2981_);
lean_dec(v___y_2980_);
lean_dec(v___y_2979_);
lean_dec_ref(v_as_2975_);
return v_res_2988_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_RuleTac_applyForwardRule_spec__5(lean_object* v_x_2989_, lean_object* v_x_2990_){
_start:
{
if (lean_obj_tag(v_x_2990_) == 0)
{
return v_x_2989_;
}
else
{
lean_object* v_key_2991_; lean_object* v_tail_2992_; lean_object* v___x_2993_; 
v_key_2991_ = lean_ctor_get(v_x_2990_, 0);
lean_inc(v_key_2991_);
v_tail_2992_ = lean_ctor_get(v_x_2990_, 2);
lean_inc(v_tail_2992_);
lean_dec_ref_known(v_x_2990_, 3);
v___x_2993_ = lean_array_push(v_x_2989_, v_key_2991_);
v_x_2989_ = v___x_2993_;
v_x_2990_ = v_tail_2992_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_RuleTac_applyForwardRule_spec__6(lean_object* v_as_2995_, size_t v_i_2996_, size_t v_stop_2997_, lean_object* v_b_2998_){
_start:
{
uint8_t v___x_2999_; 
v___x_2999_ = lean_usize_dec_eq(v_i_2996_, v_stop_2997_);
if (v___x_2999_ == 0)
{
lean_object* v___x_3000_; lean_object* v___x_3001_; size_t v___x_3002_; size_t v___x_3003_; 
v___x_3000_ = lean_array_uget_borrowed(v_as_2995_, v_i_2996_);
lean_inc(v___x_3000_);
v___x_3001_ = lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_RuleTac_applyForwardRule_spec__5(v_b_2998_, v___x_3000_);
v___x_3002_ = ((size_t)1ULL);
v___x_3003_ = lean_usize_add(v_i_2996_, v___x_3002_);
v_i_2996_ = v___x_3003_;
v_b_2998_ = v___x_3001_;
goto _start;
}
else
{
return v_b_2998_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_RuleTac_applyForwardRule_spec__6___boxed(lean_object* v_as_3005_, lean_object* v_i_3006_, lean_object* v_stop_3007_, lean_object* v_b_3008_){
_start:
{
size_t v_i_boxed_3009_; size_t v_stop_boxed_3010_; lean_object* v_res_3011_; 
v_i_boxed_3009_ = lean_unbox_usize(v_i_3006_);
lean_dec(v_i_3006_);
v_stop_boxed_3010_ = lean_unbox_usize(v_stop_3007_);
lean_dec(v_stop_3007_);
v_res_3011_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_RuleTac_applyForwardRule_spec__6(v_as_3005_, v_i_boxed_3009_, v_stop_boxed_3010_, v_b_3008_);
lean_dec_ref(v_as_3005_);
return v_res_3011_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_RuleTac_applyForwardRule_spec__4(lean_object* v_as_3012_, size_t v_i_3013_, size_t v_stop_3014_, lean_object* v_b_3015_){
_start:
{
uint8_t v___x_3016_; 
v___x_3016_ = lean_usize_dec_eq(v_i_3013_, v_stop_3014_);
if (v___x_3016_ == 0)
{
lean_object* v___x_3017_; lean_object* v___x_3018_; lean_object* v___x_3019_; size_t v___x_3020_; size_t v___x_3021_; 
v___x_3017_ = lean_array_uget_borrowed(v_as_3012_, v_i_3013_);
v___x_3018_ = lean_box(0);
lean_inc(v___x_3017_);
v___x_3019_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__0_spec__0___redArg(v_b_3015_, v___x_3017_, v___x_3018_);
v___x_3020_ = ((size_t)1ULL);
v___x_3021_ = lean_usize_add(v_i_3013_, v___x_3020_);
v_i_3013_ = v___x_3021_;
v_b_3015_ = v___x_3019_;
goto _start;
}
else
{
return v_b_3015_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_RuleTac_applyForwardRule_spec__4___boxed(lean_object* v_as_3023_, lean_object* v_i_3024_, lean_object* v_stop_3025_, lean_object* v_b_3026_){
_start:
{
size_t v_i_boxed_3027_; size_t v_stop_boxed_3028_; lean_object* v_res_3029_; 
v_i_boxed_3027_ = lean_unbox_usize(v_i_3024_);
lean_dec(v_i_3024_);
v_stop_boxed_3028_ = lean_unbox_usize(v_stop_3025_);
lean_dec(v_stop_3025_);
v_res_3029_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_RuleTac_applyForwardRule_spec__4(v_as_3023_, v_i_boxed_3027_, v_stop_boxed_3028_, v_b_3026_);
lean_dec_ref(v_as_3023_);
return v_res_3029_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_RuleTac_applyForwardRule_spec__7___redArg(lean_object* v_msg_3030_, lean_object* v___y_3031_, lean_object* v___y_3032_, lean_object* v___y_3033_, lean_object* v___y_3034_){
_start:
{
lean_object* v_ref_3036_; lean_object* v___x_3037_; lean_object* v_a_3038_; lean_object* v___x_3040_; uint8_t v_isShared_3041_; uint8_t v_isSharedCheck_3046_; 
v_ref_3036_ = lean_ctor_get(v___y_3033_, 5);
v___x_3037_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_addTrace___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__3_spec__6(v_msg_3030_, v___y_3031_, v___y_3032_, v___y_3033_, v___y_3034_);
v_a_3038_ = lean_ctor_get(v___x_3037_, 0);
v_isSharedCheck_3046_ = !lean_is_exclusive(v___x_3037_);
if (v_isSharedCheck_3046_ == 0)
{
v___x_3040_ = v___x_3037_;
v_isShared_3041_ = v_isSharedCheck_3046_;
goto v_resetjp_3039_;
}
else
{
lean_inc(v_a_3038_);
lean_dec(v___x_3037_);
v___x_3040_ = lean_box(0);
v_isShared_3041_ = v_isSharedCheck_3046_;
goto v_resetjp_3039_;
}
v_resetjp_3039_:
{
lean_object* v___x_3042_; lean_object* v___x_3044_; 
lean_inc(v_ref_3036_);
v___x_3042_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3042_, 0, v_ref_3036_);
lean_ctor_set(v___x_3042_, 1, v_a_3038_);
if (v_isShared_3041_ == 0)
{
lean_ctor_set_tag(v___x_3040_, 1);
lean_ctor_set(v___x_3040_, 0, v___x_3042_);
v___x_3044_ = v___x_3040_;
goto v_reusejp_3043_;
}
else
{
lean_object* v_reuseFailAlloc_3045_; 
v_reuseFailAlloc_3045_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3045_, 0, v___x_3042_);
v___x_3044_ = v_reuseFailAlloc_3045_;
goto v_reusejp_3043_;
}
v_reusejp_3043_:
{
return v___x_3044_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_RuleTac_applyForwardRule_spec__7___redArg___boxed(lean_object* v_msg_3047_, lean_object* v___y_3048_, lean_object* v___y_3049_, lean_object* v___y_3050_, lean_object* v___y_3051_, lean_object* v___y_3052_){
_start:
{
lean_object* v_res_3053_; 
v_res_3053_ = lp_aesop_Lean_throwError___at___00Aesop_RuleTac_applyForwardRule_spec__7___redArg(v_msg_3047_, v___y_3048_, v___y_3049_, v___y_3050_, v___y_3051_);
lean_dec(v___y_3051_);
lean_dec_ref(v___y_3050_);
lean_dec(v___y_3049_);
lean_dec_ref(v___y_3048_);
return v_res_3053_;
}
}
static lean_object* _init_lp_aesop_Aesop_RuleTac_applyForwardRule___lam__1___closed__1(void){
_start:
{
lean_object* v___x_3055_; lean_object* v___x_3056_; 
v___x_3055_ = ((lean_object*)(lp_aesop_Aesop_RuleTac_applyForwardRule___lam__1___closed__0));
v___x_3056_ = l_Lean_stringToMessageData(v___x_3055_);
return v___x_3056_;
}
}
static lean_object* _init_lp_aesop_Aesop_RuleTac_applyForwardRule___lam__1___closed__3(void){
_start:
{
lean_object* v___x_3058_; lean_object* v___x_3059_; 
v___x_3058_ = ((lean_object*)(lp_aesop_Aesop_RuleTac_applyForwardRule___lam__1___closed__2));
v___x_3059_ = l_Lean_stringToMessageData(v___x_3058_);
return v___x_3059_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_applyForwardRule___lam__1(lean_object* v_maxDepth_x3f_3060_, lean_object* v_e_3061_, lean_object* v_patSubsts_x3f_3062_, lean_object* v_immediate_3063_, lean_object* v_goal_3064_, uint8_t v_clear_3065_, lean_object* v___y_3066_, lean_object* v___y_3067_, lean_object* v___y_3068_, lean_object* v___y_3069_, lean_object* v___y_3070_, lean_object* v___y_3071_){
_start:
{
lean_object* v___y_3074_; lean_object* v___y_3075_; uint8_t v___y_3076_; lean_object* v___y_3077_; lean_object* v___x_3080_; 
v___x_3080_ = lp_aesop_Aesop_getForwardHypData(v___y_3068_, v___y_3069_, v___y_3070_, v___y_3071_);
if (lean_obj_tag(v___x_3080_) == 0)
{
lean_object* v_a_3081_; lean_object* v___x_3082_; lean_object* v_addedFVars_3083_; size_t v___y_3085_; lean_object* v___y_3086_; lean_object* v___y_3087_; lean_object* v___y_3088_; size_t v___y_3089_; lean_object* v___y_3090_; lean_object* v___y_3091_; lean_object* v___y_3092_; lean_object* v___y_3093_; uint8_t v___y_3094_; lean_object* v___y_3095_; lean_object* v___y_3096_; lean_object* v___x_3129_; lean_object* v___x_3130_; lean_object* v___x_3131_; lean_object* v___x_3132_; 
v_a_3081_ = lean_ctor_get(v___x_3080_, 0);
lean_inc(v_a_3081_);
lean_dec_ref_known(v___x_3080_, 1);
v___x_3082_ = lean_unsigned_to_nat(0u);
v_addedFVars_3083_ = lean_obj_once(&lp_aesop_Aesop_RuleTac_ForwardM_instInhabitedState_default___closed__2, &lp_aesop_Aesop_RuleTac_ForwardM_instInhabitedState_default___closed__2_once, _init_lp_aesop_Aesop_RuleTac_ForwardM_instInhabitedState_default___closed__2);
v___x_3129_ = lean_obj_once(&lp_aesop_Aesop_RuleTac_ForwardM_instInhabitedState_default___closed__3, &lp_aesop_Aesop_RuleTac_ForwardM_instInhabitedState_default___closed__3_once, _init_lp_aesop_Aesop_RuleTac_ForwardM_instInhabitedState_default___closed__3);
v___x_3130_ = lean_st_mk_ref(v___x_3129_);
v___x_3131_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3131_, 0, v_maxDepth_x3f_3060_);
lean_ctor_set(v___x_3131_, 1, v_a_3081_);
lean_inc_ref(v_e_3061_);
v___x_3132_ = lp_aesop_Aesop_RuleTac_makeForwardHypProofs_x27(v_e_3061_, v_patSubsts_x3f_3062_, v_immediate_3063_, v___x_3131_, v___x_3130_, v___y_3066_, v___y_3067_, v___y_3068_, v___y_3069_, v___y_3070_, v___y_3071_);
lean_dec_ref_known(v___x_3131_, 2);
if (lean_obj_tag(v___x_3132_) == 0)
{
lean_object* v___x_3133_; lean_object* v_toAssert_3134_; lean_object* v_usedHyps_3135_; lean_object* v___x_3137_; uint8_t v_isShared_3138_; uint8_t v_isSharedCheck_3206_; 
lean_dec_ref_known(v___x_3132_, 1);
v___x_3133_ = lean_st_ref_get(v___x_3130_);
lean_dec(v___x_3130_);
v_toAssert_3134_ = lean_ctor_get(v___x_3133_, 0);
v_usedHyps_3135_ = lean_ctor_get(v___x_3133_, 1);
v_isSharedCheck_3206_ = !lean_is_exclusive(v___x_3133_);
if (v_isSharedCheck_3206_ == 0)
{
v___x_3137_ = v___x_3133_;
v_isShared_3138_ = v_isSharedCheck_3206_;
goto v_resetjp_3136_;
}
else
{
lean_inc(v_usedHyps_3135_);
lean_inc(v_toAssert_3134_);
lean_dec(v___x_3133_);
v___x_3137_ = lean_box(0);
v_isShared_3138_ = v_isSharedCheck_3206_;
goto v_resetjp_3136_;
}
v_resetjp_3136_:
{
lean_object* v___y_3140_; lean_object* v___y_3141_; lean_object* v___y_3142_; lean_object* v___y_3143_; lean_object* v___y_3144_; lean_object* v___y_3145_; lean_object* v___x_3190_; uint8_t v___x_3191_; 
v___x_3190_ = lean_array_get_size(v_toAssert_3134_);
v___x_3191_ = lean_nat_dec_eq(v___x_3190_, v___x_3082_);
if (v___x_3191_ == 0)
{
lean_dec_ref(v_e_3061_);
v___y_3140_ = v___y_3066_;
v___y_3141_ = v___y_3067_;
v___y_3142_ = v___y_3068_;
v___y_3143_ = v___y_3069_;
v___y_3144_ = v___y_3070_;
v___y_3145_ = v___y_3071_;
goto v___jp_3139_;
}
else
{
lean_object* v___x_3192_; lean_object* v___x_3193_; lean_object* v___x_3194_; lean_object* v___x_3195_; lean_object* v___x_3196_; lean_object* v___x_3197_; lean_object* v_a_3198_; lean_object* v___x_3200_; uint8_t v_isShared_3201_; uint8_t v_isSharedCheck_3205_; 
lean_del_object(v___x_3137_);
lean_dec_ref(v_usedHyps_3135_);
lean_dec_ref(v_toAssert_3134_);
lean_dec(v_goal_3064_);
v___x_3192_ = lean_obj_once(&lp_aesop_Aesop_RuleTac_applyForwardRule___lam__1___closed__1, &lp_aesop_Aesop_RuleTac_applyForwardRule___lam__1___closed__1_once, _init_lp_aesop_Aesop_RuleTac_applyForwardRule___lam__1___closed__1);
v___x_3193_ = l_Lean_MessageData_ofExpr(v_e_3061_);
v___x_3194_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3194_, 0, v___x_3192_);
lean_ctor_set(v___x_3194_, 1, v___x_3193_);
v___x_3195_ = lean_obj_once(&lp_aesop_Aesop_RuleTac_applyForwardRule___lam__1___closed__3, &lp_aesop_Aesop_RuleTac_applyForwardRule___lam__1___closed__3_once, _init_lp_aesop_Aesop_RuleTac_applyForwardRule___lam__1___closed__3);
v___x_3196_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3196_, 0, v___x_3194_);
lean_ctor_set(v___x_3196_, 1, v___x_3195_);
v___x_3197_ = lp_aesop_Lean_throwError___at___00Aesop_RuleTac_applyForwardRule_spec__7___redArg(v___x_3196_, v___y_3068_, v___y_3069_, v___y_3070_, v___y_3071_);
v_a_3198_ = lean_ctor_get(v___x_3197_, 0);
v_isSharedCheck_3205_ = !lean_is_exclusive(v___x_3197_);
if (v_isSharedCheck_3205_ == 0)
{
v___x_3200_ = v___x_3197_;
v_isShared_3201_ = v_isSharedCheck_3205_;
goto v_resetjp_3199_;
}
else
{
lean_inc(v_a_3198_);
lean_dec(v___x_3197_);
v___x_3200_ = lean_box(0);
v_isShared_3201_ = v_isSharedCheck_3205_;
goto v_resetjp_3199_;
}
v_resetjp_3199_:
{
lean_object* v___x_3203_; 
if (v_isShared_3201_ == 0)
{
v___x_3203_ = v___x_3200_;
goto v_reusejp_3202_;
}
else
{
lean_object* v_reuseFailAlloc_3204_; 
v_reuseFailAlloc_3204_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3204_, 0, v_a_3198_);
v___x_3203_ = v_reuseFailAlloc_3204_;
goto v_reusejp_3202_;
}
v_reusejp_3202_:
{
return v___x_3203_;
}
}
}
v___jp_3139_:
{
lean_object* v___x_3146_; lean_object* v___x_3147_; lean_object* v___x_3148_; lean_object* v_a_3149_; lean_object* v___x_3150_; lean_object* v___x_3151_; lean_object* v___x_3153_; 
v___x_3146_ = lean_array_get_size(v_toAssert_3134_);
v___x_3147_ = lp_aesop_Aesop_forwardHypPrefix;
v___x_3148_ = lp_aesop_Lean_Meta_getUnusedUserNames___at___00Aesop_RuleTac_applyForwardRule_spec__0___redArg(v___x_3146_, v___x_3147_, v___y_3142_);
v_a_3149_ = lean_ctor_get(v___x_3148_, 0);
lean_inc(v_a_3149_);
lean_dec_ref(v___x_3148_);
v___x_3150_ = lean_array_get_size(v_a_3149_);
v___x_3151_ = l_Array_toSubarray___redArg(v_a_3149_, v___x_3082_, v___x_3150_);
if (v_isShared_3138_ == 0)
{
lean_ctor_set(v___x_3137_, 1, v___x_3151_);
lean_ctor_set(v___x_3137_, 0, v_addedFVars_3083_);
v___x_3153_ = v___x_3137_;
goto v_reusejp_3152_;
}
else
{
lean_object* v_reuseFailAlloc_3189_; 
v_reuseFailAlloc_3189_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3189_, 0, v_addedFVars_3083_);
lean_ctor_set(v_reuseFailAlloc_3189_, 1, v___x_3151_);
v___x_3153_ = v_reuseFailAlloc_3189_;
goto v_reusejp_3152_;
}
v_reusejp_3152_:
{
lean_object* v___x_3154_; size_t v_sz_3155_; size_t v___x_3156_; lean_object* v___x_3157_; 
lean_inc(v_goal_3064_);
v___x_3154_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3154_, 0, v_goal_3064_);
lean_ctor_set(v___x_3154_, 1, v___x_3153_);
v_sz_3155_ = lean_array_size(v_toAssert_3134_);
v___x_3156_ = ((size_t)0ULL);
v___x_3157_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_applyForwardRule_spec__1(v_toAssert_3134_, v_sz_3155_, v___x_3156_, v___x_3154_, v___y_3140_, v___y_3141_, v___y_3142_, v___y_3143_, v___y_3144_, v___y_3145_);
lean_dec_ref(v_toAssert_3134_);
if (lean_obj_tag(v___x_3157_) == 0)
{
lean_object* v_a_3158_; lean_object* v___x_3160_; uint8_t v_isShared_3161_; uint8_t v_isSharedCheck_3180_; 
v_a_3158_ = lean_ctor_get(v___x_3157_, 0);
v_isSharedCheck_3180_ = !lean_is_exclusive(v___x_3157_);
if (v_isSharedCheck_3180_ == 0)
{
v___x_3160_ = v___x_3157_;
v_isShared_3161_ = v_isSharedCheck_3180_;
goto v_resetjp_3159_;
}
else
{
lean_inc(v_a_3158_);
lean_dec(v___x_3157_);
v___x_3160_ = lean_box(0);
v_isShared_3161_ = v_isSharedCheck_3180_;
goto v_resetjp_3159_;
}
v_resetjp_3159_:
{
lean_object* v_snd_3162_; lean_object* v_fst_3163_; lean_object* v_fst_3164_; uint8_t v___x_3165_; 
v_snd_3162_ = lean_ctor_get(v_a_3158_, 1);
lean_inc(v_snd_3162_);
v_fst_3163_ = lean_ctor_get(v_a_3158_, 0);
lean_inc(v_fst_3163_);
lean_dec(v_a_3158_);
v_fst_3164_ = lean_ctor_get(v_snd_3162_, 0);
lean_inc(v_fst_3164_);
lean_dec(v_snd_3162_);
v___x_3165_ = 0;
if (v_clear_3065_ == 0)
{
lean_object* v___x_3166_; lean_object* v___x_3168_; 
lean_dec_ref(v_usedHyps_3135_);
v___x_3166_ = lean_alloc_ctor(0, 4, 1);
lean_ctor_set(v___x_3166_, 0, v_goal_3064_);
lean_ctor_set(v___x_3166_, 1, v_fst_3163_);
lean_ctor_set(v___x_3166_, 2, v_fst_3164_);
lean_ctor_set(v___x_3166_, 3, v_addedFVars_3083_);
lean_ctor_set_uint8(v___x_3166_, sizeof(void*)*4, v___x_3165_);
if (v_isShared_3161_ == 0)
{
lean_ctor_set(v___x_3160_, 0, v___x_3166_);
v___x_3168_ = v___x_3160_;
goto v_reusejp_3167_;
}
else
{
lean_object* v_reuseFailAlloc_3169_; 
v_reuseFailAlloc_3169_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3169_, 0, v___x_3166_);
v___x_3168_ = v_reuseFailAlloc_3169_;
goto v_reusejp_3167_;
}
v_reusejp_3167_:
{
return v___x_3168_;
}
}
else
{
lean_object* v_size_3170_; lean_object* v_buckets_3171_; lean_object* v___x_3172_; lean_object* v___x_3173_; uint8_t v___x_3174_; 
lean_del_object(v___x_3160_);
v_size_3170_ = lean_ctor_get(v_usedHyps_3135_, 0);
lean_inc(v_size_3170_);
v_buckets_3171_ = lean_ctor_get(v_usedHyps_3135_, 1);
lean_inc_ref(v_buckets_3171_);
lean_dec_ref(v_usedHyps_3135_);
v___x_3172_ = lean_mk_empty_array_with_capacity(v_size_3170_);
lean_dec(v_size_3170_);
v___x_3173_ = lean_array_get_size(v_buckets_3171_);
v___x_3174_ = lean_nat_dec_lt(v___x_3082_, v___x_3173_);
if (v___x_3174_ == 0)
{
lean_dec_ref(v_buckets_3171_);
v___y_3085_ = v___x_3156_;
v___y_3086_ = v_fst_3164_;
v___y_3087_ = v___y_3140_;
v___y_3088_ = v___y_3142_;
v___y_3089_ = v___x_3156_;
v___y_3090_ = v___y_3144_;
v___y_3091_ = v___y_3145_;
v___y_3092_ = v___y_3143_;
v___y_3093_ = v_fst_3163_;
v___y_3094_ = v___x_3165_;
v___y_3095_ = v___y_3141_;
v___y_3096_ = v___x_3172_;
goto v___jp_3084_;
}
else
{
uint8_t v___x_3175_; 
v___x_3175_ = lean_nat_dec_le(v___x_3173_, v___x_3173_);
if (v___x_3175_ == 0)
{
if (v___x_3174_ == 0)
{
lean_dec_ref(v_buckets_3171_);
v___y_3085_ = v___x_3156_;
v___y_3086_ = v_fst_3164_;
v___y_3087_ = v___y_3140_;
v___y_3088_ = v___y_3142_;
v___y_3089_ = v___x_3156_;
v___y_3090_ = v___y_3144_;
v___y_3091_ = v___y_3145_;
v___y_3092_ = v___y_3143_;
v___y_3093_ = v_fst_3163_;
v___y_3094_ = v___x_3165_;
v___y_3095_ = v___y_3141_;
v___y_3096_ = v___x_3172_;
goto v___jp_3084_;
}
else
{
size_t v___x_3176_; lean_object* v___x_3177_; 
v___x_3176_ = lean_usize_of_nat(v___x_3173_);
v___x_3177_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_RuleTac_applyForwardRule_spec__6(v_buckets_3171_, v___x_3156_, v___x_3176_, v___x_3172_);
lean_dec_ref(v_buckets_3171_);
v___y_3085_ = v___x_3156_;
v___y_3086_ = v_fst_3164_;
v___y_3087_ = v___y_3140_;
v___y_3088_ = v___y_3142_;
v___y_3089_ = v___x_3156_;
v___y_3090_ = v___y_3144_;
v___y_3091_ = v___y_3145_;
v___y_3092_ = v___y_3143_;
v___y_3093_ = v_fst_3163_;
v___y_3094_ = v___x_3165_;
v___y_3095_ = v___y_3141_;
v___y_3096_ = v___x_3177_;
goto v___jp_3084_;
}
}
else
{
size_t v___x_3178_; lean_object* v___x_3179_; 
v___x_3178_ = lean_usize_of_nat(v___x_3173_);
v___x_3179_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_RuleTac_applyForwardRule_spec__6(v_buckets_3171_, v___x_3156_, v___x_3178_, v___x_3172_);
lean_dec_ref(v_buckets_3171_);
v___y_3085_ = v___x_3156_;
v___y_3086_ = v_fst_3164_;
v___y_3087_ = v___y_3140_;
v___y_3088_ = v___y_3142_;
v___y_3089_ = v___x_3156_;
v___y_3090_ = v___y_3144_;
v___y_3091_ = v___y_3145_;
v___y_3092_ = v___y_3143_;
v___y_3093_ = v_fst_3163_;
v___y_3094_ = v___x_3165_;
v___y_3095_ = v___y_3141_;
v___y_3096_ = v___x_3179_;
goto v___jp_3084_;
}
}
}
}
}
else
{
lean_object* v_a_3181_; lean_object* v___x_3183_; uint8_t v_isShared_3184_; uint8_t v_isSharedCheck_3188_; 
lean_dec_ref(v_usedHyps_3135_);
lean_dec(v_goal_3064_);
v_a_3181_ = lean_ctor_get(v___x_3157_, 0);
v_isSharedCheck_3188_ = !lean_is_exclusive(v___x_3157_);
if (v_isSharedCheck_3188_ == 0)
{
v___x_3183_ = v___x_3157_;
v_isShared_3184_ = v_isSharedCheck_3188_;
goto v_resetjp_3182_;
}
else
{
lean_inc(v_a_3181_);
lean_dec(v___x_3157_);
v___x_3183_ = lean_box(0);
v_isShared_3184_ = v_isSharedCheck_3188_;
goto v_resetjp_3182_;
}
v_resetjp_3182_:
{
lean_object* v___x_3186_; 
if (v_isShared_3184_ == 0)
{
v___x_3186_ = v___x_3183_;
goto v_reusejp_3185_;
}
else
{
lean_object* v_reuseFailAlloc_3187_; 
v_reuseFailAlloc_3187_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3187_, 0, v_a_3181_);
v___x_3186_ = v_reuseFailAlloc_3187_;
goto v_reusejp_3185_;
}
v_reusejp_3185_:
{
return v___x_3186_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_3207_; lean_object* v___x_3209_; uint8_t v_isShared_3210_; uint8_t v_isSharedCheck_3214_; 
lean_dec(v___x_3130_);
lean_dec(v_goal_3064_);
lean_dec_ref(v_e_3061_);
v_a_3207_ = lean_ctor_get(v___x_3132_, 0);
v_isSharedCheck_3214_ = !lean_is_exclusive(v___x_3132_);
if (v_isSharedCheck_3214_ == 0)
{
v___x_3209_ = v___x_3132_;
v_isShared_3210_ = v_isSharedCheck_3214_;
goto v_resetjp_3208_;
}
else
{
lean_inc(v_a_3207_);
lean_dec(v___x_3132_);
v___x_3209_ = lean_box(0);
v_isShared_3210_ = v_isSharedCheck_3214_;
goto v_resetjp_3208_;
}
v_resetjp_3208_:
{
lean_object* v___x_3212_; 
if (v_isShared_3210_ == 0)
{
v___x_3212_ = v___x_3209_;
goto v_reusejp_3211_;
}
else
{
lean_object* v_reuseFailAlloc_3213_; 
v_reuseFailAlloc_3213_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3213_, 0, v_a_3207_);
v___x_3212_ = v_reuseFailAlloc_3213_;
goto v_reusejp_3211_;
}
v_reusejp_3211_:
{
return v___x_3212_;
}
}
}
v___jp_3084_:
{
lean_object* v___x_3097_; lean_object* v___x_3098_; lean_object* v___f_3099_; lean_object* v___x_3100_; 
v___x_3097_ = lean_array_get_size(v___y_3096_);
v___x_3098_ = lean_box_usize(v___y_3085_);
v___f_3099_ = lean_alloc_closure((void*)(lp_aesop_Aesop_RuleTac_applyForwardRule___lam__0___boxed), 11, 4);
lean_closure_set(v___f_3099_, 0, v___x_3082_);
lean_closure_set(v___f_3099_, 1, v___x_3097_);
lean_closure_set(v___f_3099_, 2, v___y_3096_);
lean_closure_set(v___f_3099_, 3, v___x_3098_);
lean_inc(v___y_3093_);
v___x_3100_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_RuleTac_applyForwardRule_spec__3___redArg(v___y_3093_, v___f_3099_, v___y_3087_, v___y_3095_, v___y_3088_, v___y_3092_, v___y_3090_, v___y_3091_);
if (lean_obj_tag(v___x_3100_) == 0)
{
lean_object* v_a_3101_; lean_object* v___x_3102_; 
v_a_3101_ = lean_ctor_get(v___x_3100_, 0);
lean_inc(v_a_3101_);
lean_dec_ref_known(v___x_3100_, 1);
v___x_3102_ = lp_aesop_Aesop_tryClearManyS(v___y_3093_, v_a_3101_, v___y_3087_, v___y_3095_, v___y_3088_, v___y_3092_, v___y_3090_, v___y_3091_);
if (lean_obj_tag(v___x_3102_) == 0)
{
lean_object* v_a_3103_; lean_object* v_fst_3104_; lean_object* v_snd_3105_; lean_object* v___x_3106_; uint8_t v___x_3107_; 
v_a_3103_ = lean_ctor_get(v___x_3102_, 0);
lean_inc(v_a_3103_);
lean_dec_ref_known(v___x_3102_, 1);
v_fst_3104_ = lean_ctor_get(v_a_3103_, 0);
lean_inc(v_fst_3104_);
v_snd_3105_ = lean_ctor_get(v_a_3103_, 1);
lean_inc(v_snd_3105_);
lean_dec(v_a_3103_);
v___x_3106_ = lean_array_get_size(v_snd_3105_);
v___x_3107_ = lean_nat_dec_lt(v___x_3082_, v___x_3106_);
if (v___x_3107_ == 0)
{
lean_dec(v_snd_3105_);
v___y_3074_ = v___y_3086_;
v___y_3075_ = v_fst_3104_;
v___y_3076_ = v___y_3094_;
v___y_3077_ = v_addedFVars_3083_;
goto v___jp_3073_;
}
else
{
uint8_t v___x_3108_; 
v___x_3108_ = lean_nat_dec_le(v___x_3106_, v___x_3106_);
if (v___x_3108_ == 0)
{
if (v___x_3107_ == 0)
{
lean_dec(v_snd_3105_);
v___y_3074_ = v___y_3086_;
v___y_3075_ = v_fst_3104_;
v___y_3076_ = v___y_3094_;
v___y_3077_ = v_addedFVars_3083_;
goto v___jp_3073_;
}
else
{
size_t v___x_3109_; lean_object* v___x_3110_; 
v___x_3109_ = lean_usize_of_nat(v___x_3106_);
v___x_3110_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_RuleTac_applyForwardRule_spec__4(v_snd_3105_, v___y_3089_, v___x_3109_, v_addedFVars_3083_);
lean_dec(v_snd_3105_);
v___y_3074_ = v___y_3086_;
v___y_3075_ = v_fst_3104_;
v___y_3076_ = v___y_3094_;
v___y_3077_ = v___x_3110_;
goto v___jp_3073_;
}
}
else
{
size_t v___x_3111_; lean_object* v___x_3112_; 
v___x_3111_ = lean_usize_of_nat(v___x_3106_);
v___x_3112_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_RuleTac_applyForwardRule_spec__4(v_snd_3105_, v___y_3089_, v___x_3111_, v_addedFVars_3083_);
lean_dec(v_snd_3105_);
v___y_3074_ = v___y_3086_;
v___y_3075_ = v_fst_3104_;
v___y_3076_ = v___y_3094_;
v___y_3077_ = v___x_3112_;
goto v___jp_3073_;
}
}
}
else
{
lean_object* v_a_3113_; lean_object* v___x_3115_; uint8_t v_isShared_3116_; uint8_t v_isSharedCheck_3120_; 
lean_dec(v___y_3086_);
lean_dec(v_goal_3064_);
v_a_3113_ = lean_ctor_get(v___x_3102_, 0);
v_isSharedCheck_3120_ = !lean_is_exclusive(v___x_3102_);
if (v_isSharedCheck_3120_ == 0)
{
v___x_3115_ = v___x_3102_;
v_isShared_3116_ = v_isSharedCheck_3120_;
goto v_resetjp_3114_;
}
else
{
lean_inc(v_a_3113_);
lean_dec(v___x_3102_);
v___x_3115_ = lean_box(0);
v_isShared_3116_ = v_isSharedCheck_3120_;
goto v_resetjp_3114_;
}
v_resetjp_3114_:
{
lean_object* v___x_3118_; 
if (v_isShared_3116_ == 0)
{
v___x_3118_ = v___x_3115_;
goto v_reusejp_3117_;
}
else
{
lean_object* v_reuseFailAlloc_3119_; 
v_reuseFailAlloc_3119_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3119_, 0, v_a_3113_);
v___x_3118_ = v_reuseFailAlloc_3119_;
goto v_reusejp_3117_;
}
v_reusejp_3117_:
{
return v___x_3118_;
}
}
}
}
else
{
lean_object* v_a_3121_; lean_object* v___x_3123_; uint8_t v_isShared_3124_; uint8_t v_isSharedCheck_3128_; 
lean_dec(v___y_3093_);
lean_dec(v___y_3086_);
lean_dec(v_goal_3064_);
v_a_3121_ = lean_ctor_get(v___x_3100_, 0);
v_isSharedCheck_3128_ = !lean_is_exclusive(v___x_3100_);
if (v_isSharedCheck_3128_ == 0)
{
v___x_3123_ = v___x_3100_;
v_isShared_3124_ = v_isSharedCheck_3128_;
goto v_resetjp_3122_;
}
else
{
lean_inc(v_a_3121_);
lean_dec(v___x_3100_);
v___x_3123_ = lean_box(0);
v_isShared_3124_ = v_isSharedCheck_3128_;
goto v_resetjp_3122_;
}
v_resetjp_3122_:
{
lean_object* v___x_3126_; 
if (v_isShared_3124_ == 0)
{
v___x_3126_ = v___x_3123_;
goto v_reusejp_3125_;
}
else
{
lean_object* v_reuseFailAlloc_3127_; 
v_reuseFailAlloc_3127_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3127_, 0, v_a_3121_);
v___x_3126_ = v_reuseFailAlloc_3127_;
goto v_reusejp_3125_;
}
v_reusejp_3125_:
{
return v___x_3126_;
}
}
}
}
}
else
{
lean_object* v_a_3215_; lean_object* v___x_3217_; uint8_t v_isShared_3218_; uint8_t v_isSharedCheck_3222_; 
lean_dec(v_goal_3064_);
lean_dec_ref(v_immediate_3063_);
lean_dec_ref(v_e_3061_);
lean_dec(v_maxDepth_x3f_3060_);
v_a_3215_ = lean_ctor_get(v___x_3080_, 0);
v_isSharedCheck_3222_ = !lean_is_exclusive(v___x_3080_);
if (v_isSharedCheck_3222_ == 0)
{
v___x_3217_ = v___x_3080_;
v_isShared_3218_ = v_isSharedCheck_3222_;
goto v_resetjp_3216_;
}
else
{
lean_inc(v_a_3215_);
lean_dec(v___x_3080_);
v___x_3217_ = lean_box(0);
v_isShared_3218_ = v_isSharedCheck_3222_;
goto v_resetjp_3216_;
}
v_resetjp_3216_:
{
lean_object* v___x_3220_; 
if (v_isShared_3218_ == 0)
{
v___x_3220_ = v___x_3217_;
goto v_reusejp_3219_;
}
else
{
lean_object* v_reuseFailAlloc_3221_; 
v_reuseFailAlloc_3221_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3221_, 0, v_a_3215_);
v___x_3220_ = v_reuseFailAlloc_3221_;
goto v_reusejp_3219_;
}
v_reusejp_3219_:
{
return v___x_3220_;
}
}
}
v___jp_3073_:
{
lean_object* v___x_3078_; lean_object* v___x_3079_; 
v___x_3078_ = lean_alloc_ctor(0, 4, 1);
lean_ctor_set(v___x_3078_, 0, v_goal_3064_);
lean_ctor_set(v___x_3078_, 1, v___y_3075_);
lean_ctor_set(v___x_3078_, 2, v___y_3074_);
lean_ctor_set(v___x_3078_, 3, v___y_3077_);
lean_ctor_set_uint8(v___x_3078_, sizeof(void*)*4, v___y_3076_);
v___x_3079_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3079_, 0, v___x_3078_);
return v___x_3079_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_applyForwardRule___lam__1___boxed(lean_object* v_maxDepth_x3f_3223_, lean_object* v_e_3224_, lean_object* v_patSubsts_x3f_3225_, lean_object* v_immediate_3226_, lean_object* v_goal_3227_, lean_object* v_clear_3228_, lean_object* v___y_3229_, lean_object* v___y_3230_, lean_object* v___y_3231_, lean_object* v___y_3232_, lean_object* v___y_3233_, lean_object* v___y_3234_, lean_object* v___y_3235_){
_start:
{
uint8_t v_clear_boxed_3236_; lean_object* v_res_3237_; 
v_clear_boxed_3236_ = lean_unbox(v_clear_3228_);
v_res_3237_ = lp_aesop_Aesop_RuleTac_applyForwardRule___lam__1(v_maxDepth_x3f_3223_, v_e_3224_, v_patSubsts_x3f_3225_, v_immediate_3226_, v_goal_3227_, v_clear_boxed_3236_, v___y_3229_, v___y_3230_, v___y_3231_, v___y_3232_, v___y_3233_, v___y_3234_);
lean_dec(v___y_3234_);
lean_dec_ref(v___y_3233_);
lean_dec(v___y_3232_);
lean_dec_ref(v___y_3231_);
lean_dec(v___y_3230_);
lean_dec(v___y_3229_);
lean_dec(v_patSubsts_x3f_3225_);
return v_res_3237_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_applyForwardRule(lean_object* v_goal_3238_, lean_object* v_e_3239_, lean_object* v_patSubsts_x3f_3240_, lean_object* v_immediate_3241_, uint8_t v_clear_3242_, lean_object* v_maxDepth_x3f_3243_, lean_object* v_a_3244_, lean_object* v_a_3245_, lean_object* v_a_3246_, lean_object* v_a_3247_, lean_object* v_a_3248_, lean_object* v_a_3249_){
_start:
{
lean_object* v_keyedConfig_3251_; uint8_t v_trackZetaDelta_3252_; lean_object* v_zetaDeltaSet_3253_; lean_object* v_lctx_3254_; lean_object* v_localInstances_3255_; lean_object* v_defEqCtx_x3f_3256_; lean_object* v_synthPendingDepth_3257_; lean_object* v_customCanUnfoldPredicate_x3f_3258_; uint8_t v_univApprox_3259_; uint8_t v_inTypeClassResolution_3260_; uint8_t v_cacheInferType_3261_; lean_object* v___x_3262_; lean_object* v___f_3263_; uint8_t v___x_3264_; lean_object* v___x_3265_; lean_object* v___x_3266_; lean_object* v___x_3267_; 
v_keyedConfig_3251_ = lean_ctor_get(v_a_3246_, 0);
v_trackZetaDelta_3252_ = lean_ctor_get_uint8(v_a_3246_, sizeof(void*)*7);
v_zetaDeltaSet_3253_ = lean_ctor_get(v_a_3246_, 1);
v_lctx_3254_ = lean_ctor_get(v_a_3246_, 2);
v_localInstances_3255_ = lean_ctor_get(v_a_3246_, 3);
v_defEqCtx_x3f_3256_ = lean_ctor_get(v_a_3246_, 4);
v_synthPendingDepth_3257_ = lean_ctor_get(v_a_3246_, 5);
v_customCanUnfoldPredicate_x3f_3258_ = lean_ctor_get(v_a_3246_, 6);
v_univApprox_3259_ = lean_ctor_get_uint8(v_a_3246_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_3260_ = lean_ctor_get_uint8(v_a_3246_, sizeof(void*)*7 + 2);
v_cacheInferType_3261_ = lean_ctor_get_uint8(v_a_3246_, sizeof(void*)*7 + 3);
v___x_3262_ = lean_box(v_clear_3242_);
lean_inc(v_goal_3238_);
v___f_3263_ = lean_alloc_closure((void*)(lp_aesop_Aesop_RuleTac_applyForwardRule___lam__1___boxed), 13, 6);
lean_closure_set(v___f_3263_, 0, v_maxDepth_x3f_3243_);
lean_closure_set(v___f_3263_, 1, v_e_3239_);
lean_closure_set(v___f_3263_, 2, v_patSubsts_x3f_3240_);
lean_closure_set(v___f_3263_, 3, v_immediate_3241_);
lean_closure_set(v___f_3263_, 4, v_goal_3238_);
lean_closure_set(v___f_3263_, 5, v___x_3262_);
v___x_3264_ = 2;
lean_inc_ref(v_keyedConfig_3251_);
v___x_3265_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_3264_, v_keyedConfig_3251_);
lean_inc(v_customCanUnfoldPredicate_x3f_3258_);
lean_inc(v_synthPendingDepth_3257_);
lean_inc(v_defEqCtx_x3f_3256_);
lean_inc_ref(v_localInstances_3255_);
lean_inc_ref(v_lctx_3254_);
lean_inc(v_zetaDeltaSet_3253_);
v___x_3266_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_3266_, 0, v___x_3265_);
lean_ctor_set(v___x_3266_, 1, v_zetaDeltaSet_3253_);
lean_ctor_set(v___x_3266_, 2, v_lctx_3254_);
lean_ctor_set(v___x_3266_, 3, v_localInstances_3255_);
lean_ctor_set(v___x_3266_, 4, v_defEqCtx_x3f_3256_);
lean_ctor_set(v___x_3266_, 5, v_synthPendingDepth_3257_);
lean_ctor_set(v___x_3266_, 6, v_customCanUnfoldPredicate_x3f_3258_);
lean_ctor_set_uint8(v___x_3266_, sizeof(void*)*7, v_trackZetaDelta_3252_);
lean_ctor_set_uint8(v___x_3266_, sizeof(void*)*7 + 1, v_univApprox_3259_);
lean_ctor_set_uint8(v___x_3266_, sizeof(void*)*7 + 2, v_inTypeClassResolution_3260_);
lean_ctor_set_uint8(v___x_3266_, sizeof(void*)*7 + 3, v_cacheInferType_3261_);
v___x_3267_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_RuleTac_applyForwardRule_spec__3___redArg(v_goal_3238_, v___f_3263_, v_a_3244_, v_a_3245_, v___x_3266_, v_a_3247_, v_a_3248_, v_a_3249_);
lean_dec_ref_known(v___x_3266_, 7);
if (lean_obj_tag(v___x_3267_) == 0)
{
lean_object* v_a_3268_; lean_object* v___x_3270_; uint8_t v_isShared_3271_; uint8_t v_isSharedCheck_3275_; 
v_a_3268_ = lean_ctor_get(v___x_3267_, 0);
v_isSharedCheck_3275_ = !lean_is_exclusive(v___x_3267_);
if (v_isSharedCheck_3275_ == 0)
{
v___x_3270_ = v___x_3267_;
v_isShared_3271_ = v_isSharedCheck_3275_;
goto v_resetjp_3269_;
}
else
{
lean_inc(v_a_3268_);
lean_dec(v___x_3267_);
v___x_3270_ = lean_box(0);
v_isShared_3271_ = v_isSharedCheck_3275_;
goto v_resetjp_3269_;
}
v_resetjp_3269_:
{
lean_object* v___x_3273_; 
if (v_isShared_3271_ == 0)
{
v___x_3273_ = v___x_3270_;
goto v_reusejp_3272_;
}
else
{
lean_object* v_reuseFailAlloc_3274_; 
v_reuseFailAlloc_3274_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3274_, 0, v_a_3268_);
v___x_3273_ = v_reuseFailAlloc_3274_;
goto v_reusejp_3272_;
}
v_reusejp_3272_:
{
return v___x_3273_;
}
}
}
else
{
return v___x_3267_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_applyForwardRule___boxed(lean_object* v_goal_3276_, lean_object* v_e_3277_, lean_object* v_patSubsts_x3f_3278_, lean_object* v_immediate_3279_, lean_object* v_clear_3280_, lean_object* v_maxDepth_x3f_3281_, lean_object* v_a_3282_, lean_object* v_a_3283_, lean_object* v_a_3284_, lean_object* v_a_3285_, lean_object* v_a_3286_, lean_object* v_a_3287_, lean_object* v_a_3288_){
_start:
{
uint8_t v_clear_boxed_3289_; lean_object* v_res_3290_; 
v_clear_boxed_3289_ = lean_unbox(v_clear_3280_);
v_res_3290_ = lp_aesop_Aesop_RuleTac_applyForwardRule(v_goal_3276_, v_e_3277_, v_patSubsts_x3f_3278_, v_immediate_3279_, v_clear_boxed_3289_, v_maxDepth_x3f_3281_, v_a_3282_, v_a_3283_, v_a_3284_, v_a_3285_, v_a_3286_, v_a_3287_);
lean_dec(v_a_3287_);
lean_dec_ref(v_a_3286_);
lean_dec(v_a_3285_);
lean_dec_ref(v_a_3284_);
lean_dec(v_a_3283_);
lean_dec(v_a_3282_);
return v_res_3290_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_RuleTac_applyForwardRule_spec__2(lean_object* v_as_3291_, size_t v_i_3292_, size_t v_stop_3293_, lean_object* v_b_3294_, lean_object* v___y_3295_, lean_object* v___y_3296_, lean_object* v___y_3297_, lean_object* v___y_3298_, lean_object* v___y_3299_, lean_object* v___y_3300_){
_start:
{
lean_object* v___x_3302_; 
v___x_3302_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_RuleTac_applyForwardRule_spec__2___redArg(v_as_3291_, v_i_3292_, v_stop_3293_, v_b_3294_, v___y_3297_, v___y_3298_, v___y_3299_, v___y_3300_);
return v___x_3302_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_RuleTac_applyForwardRule_spec__2___boxed(lean_object* v_as_3303_, lean_object* v_i_3304_, lean_object* v_stop_3305_, lean_object* v_b_3306_, lean_object* v___y_3307_, lean_object* v___y_3308_, lean_object* v___y_3309_, lean_object* v___y_3310_, lean_object* v___y_3311_, lean_object* v___y_3312_, lean_object* v___y_3313_){
_start:
{
size_t v_i_boxed_3314_; size_t v_stop_boxed_3315_; lean_object* v_res_3316_; 
v_i_boxed_3314_ = lean_unbox_usize(v_i_3304_);
lean_dec(v_i_3304_);
v_stop_boxed_3315_ = lean_unbox_usize(v_stop_3305_);
lean_dec(v_stop_3305_);
v_res_3316_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_RuleTac_applyForwardRule_spec__2(v_as_3303_, v_i_boxed_3314_, v_stop_boxed_3315_, v_b_3306_, v___y_3307_, v___y_3308_, v___y_3309_, v___y_3310_, v___y_3311_, v___y_3312_);
lean_dec(v___y_3312_);
lean_dec_ref(v___y_3311_);
lean_dec(v___y_3310_);
lean_dec_ref(v___y_3309_);
lean_dec(v___y_3308_);
lean_dec(v___y_3307_);
lean_dec_ref(v_as_3303_);
return v_res_3316_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_RuleTac_applyForwardRule_spec__7(lean_object* v_00_u03b1_3317_, lean_object* v_msg_3318_, lean_object* v___y_3319_, lean_object* v___y_3320_, lean_object* v___y_3321_, lean_object* v___y_3322_, lean_object* v___y_3323_, lean_object* v___y_3324_){
_start:
{
lean_object* v___x_3326_; 
v___x_3326_ = lp_aesop_Lean_throwError___at___00Aesop_RuleTac_applyForwardRule_spec__7___redArg(v_msg_3318_, v___y_3321_, v___y_3322_, v___y_3323_, v___y_3324_);
return v___x_3326_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_RuleTac_applyForwardRule_spec__7___boxed(lean_object* v_00_u03b1_3327_, lean_object* v_msg_3328_, lean_object* v___y_3329_, lean_object* v___y_3330_, lean_object* v___y_3331_, lean_object* v___y_3332_, lean_object* v___y_3333_, lean_object* v___y_3334_, lean_object* v___y_3335_){
_start:
{
lean_object* v_res_3336_; 
v_res_3336_ = lp_aesop_Lean_throwError___at___00Aesop_RuleTac_applyForwardRule_spec__7(v_00_u03b1_3327_, v_msg_3328_, v___y_3329_, v___y_3330_, v___y_3331_, v___y_3332_, v___y_3333_, v___y_3334_);
lean_dec(v___y_3334_);
lean_dec_ref(v___y_3333_);
lean_dec(v___y_3332_);
lean_dec_ref(v___y_3331_);
lean_dec(v___y_3330_);
lean_dec(v___y_3329_);
return v_res_3336_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_forwardExpr___lam__0(lean_object* v_____x_3337_, lean_object* v___y_3338_, lean_object* v___y_3339_, lean_object* v___y_3340_, lean_object* v___y_3341_, lean_object* v___y_3342_){
_start:
{
lean_object* v_fst_3344_; lean_object* v_snd_3345_; lean_object* v___x_3347_; uint8_t v_isShared_3348_; uint8_t v_isSharedCheck_3359_; 
v_fst_3344_ = lean_ctor_get(v_____x_3337_, 0);
v_snd_3345_ = lean_ctor_get(v_____x_3337_, 1);
v_isSharedCheck_3359_ = !lean_is_exclusive(v_____x_3337_);
if (v_isSharedCheck_3359_ == 0)
{
v___x_3347_ = v_____x_3337_;
v_isShared_3348_ = v_isSharedCheck_3359_;
goto v_resetjp_3346_;
}
else
{
lean_inc(v_snd_3345_);
lean_inc(v_fst_3344_);
lean_dec(v_____x_3337_);
v___x_3347_ = lean_box(0);
v_isShared_3348_ = v_isSharedCheck_3359_;
goto v_resetjp_3346_;
}
v_resetjp_3346_:
{
lean_object* v___x_3349_; lean_object* v___x_3350_; lean_object* v___x_3351_; lean_object* v___x_3352_; lean_object* v___x_3353_; lean_object* v___x_3355_; 
v___x_3349_ = lean_unsigned_to_nat(1u);
v___x_3350_ = lean_mk_empty_array_with_capacity(v___x_3349_);
v___x_3351_ = lean_array_push(v___x_3350_, v_fst_3344_);
v___x_3352_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3352_, 0, v_snd_3345_);
v___x_3353_ = lean_box(0);
if (v_isShared_3348_ == 0)
{
lean_ctor_set(v___x_3347_, 1, v___x_3353_);
lean_ctor_set(v___x_3347_, 0, v___x_3352_);
v___x_3355_ = v___x_3347_;
goto v_reusejp_3354_;
}
else
{
lean_object* v_reuseFailAlloc_3358_; 
v_reuseFailAlloc_3358_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3358_, 0, v___x_3352_);
lean_ctor_set(v_reuseFailAlloc_3358_, 1, v___x_3353_);
v___x_3355_ = v_reuseFailAlloc_3358_;
goto v_reusejp_3354_;
}
v_reusejp_3354_:
{
lean_object* v___x_3356_; lean_object* v___x_3357_; 
v___x_3356_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3356_, 0, v___x_3351_);
lean_ctor_set(v___x_3356_, 1, v___x_3355_);
v___x_3357_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3357_, 0, v___x_3356_);
return v___x_3357_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_forwardExpr___lam__0___boxed(lean_object* v_____x_3360_, lean_object* v___y_3361_, lean_object* v___y_3362_, lean_object* v___y_3363_, lean_object* v___y_3364_, lean_object* v___y_3365_, lean_object* v___y_3366_){
_start:
{
lean_object* v_res_3367_; 
v_res_3367_ = lp_aesop_Aesop_RuleTac_forwardExpr___lam__0(v_____x_3360_, v___y_3361_, v___y_3362_, v___y_3363_, v___y_3364_, v___y_3365_);
lean_dec(v___y_3365_);
lean_dec_ref(v___y_3364_);
lean_dec(v___y_3363_);
lean_dec_ref(v___y_3362_);
lean_dec(v___y_3361_);
return v_res_3367_;
}
}
static lean_object* _init_lp_aesop_Aesop_RuleTac_forwardExpr___closed__0(void){
_start:
{
lean_object* v___x_3368_; 
v___x_3368_ = l_instMonadControlStateRefT_x27(lean_box(0), lean_box(0), lean_box(0));
return v___x_3368_;
}
}
static lean_object* _init_lp_aesop_Aesop_RuleTac_forwardExpr___closed__1(void){
_start:
{
lean_object* v___x_3369_; 
v___x_3369_ = l_instMonadEIO(lean_box(0));
return v___x_3369_;
}
}
static lean_object* _init_lp_aesop_Aesop_RuleTac_forwardExpr___closed__2(void){
_start:
{
lean_object* v___x_3370_; lean_object* v___x_3371_; 
v___x_3370_ = lean_obj_once(&lp_aesop_Aesop_RuleTac_forwardExpr___closed__1, &lp_aesop_Aesop_RuleTac_forwardExpr___closed__1_once, _init_lp_aesop_Aesop_RuleTac_forwardExpr___closed__1);
v___x_3371_ = l_StateRefT_x27_instMonad___redArg(v___x_3370_);
return v___x_3371_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_forwardExpr(lean_object* v_e_3401_, lean_object* v_immediate_3402_, uint8_t v_clear_3403_, lean_object* v_a_3404_, lean_object* v_a_3405_, lean_object* v_a_3406_, lean_object* v_a_3407_, lean_object* v_a_3408_, lean_object* v_a_3409_){
_start:
{
lean_object* v___x_3411_; lean_object* v___x_3412_; lean_object* v_toApplicative_3413_; lean_object* v_toFunctor_3414_; lean_object* v_toSeq_3415_; lean_object* v_toSeqLeft_3416_; lean_object* v_toSeqRight_3417_; lean_object* v___f_3418_; lean_object* v___f_3419_; lean_object* v___f_3420_; lean_object* v___f_3421_; lean_object* v___x_3422_; lean_object* v___f_3423_; lean_object* v___f_3424_; lean_object* v___f_3425_; lean_object* v___x_3426_; lean_object* v___x_3427_; lean_object* v___x_3428_; lean_object* v___x_3429_; lean_object* v___x_3430_; lean_object* v___f_3431_; lean_object* v___f_3432_; lean_object* v___x_3433_; lean_object* v_toApplicative_3434_; lean_object* v_toFunctor_3435_; lean_object* v_toSeq_3436_; lean_object* v_toSeqLeft_3437_; lean_object* v_toSeqRight_3438_; lean_object* v___f_3439_; lean_object* v___f_3440_; lean_object* v___x_3441_; lean_object* v___f_3442_; lean_object* v___f_3443_; lean_object* v___f_3444_; lean_object* v___x_3445_; lean_object* v___x_3446_; lean_object* v___x_3447_; lean_object* v_toApplicative_3448_; lean_object* v___x_3450_; uint8_t v_isShared_3451_; uint8_t v_isSharedCheck_3522_; 
v___x_3411_ = lean_obj_once(&lp_aesop_Aesop_RuleTac_forwardExpr___closed__0, &lp_aesop_Aesop_RuleTac_forwardExpr___closed__0_once, _init_lp_aesop_Aesop_RuleTac_forwardExpr___closed__0);
v___x_3412_ = lean_obj_once(&lp_aesop_Aesop_RuleTac_forwardExpr___closed__2, &lp_aesop_Aesop_RuleTac_forwardExpr___closed__2_once, _init_lp_aesop_Aesop_RuleTac_forwardExpr___closed__2);
v_toApplicative_3413_ = lean_ctor_get(v___x_3412_, 0);
v_toFunctor_3414_ = lean_ctor_get(v_toApplicative_3413_, 0);
v_toSeq_3415_ = lean_ctor_get(v_toApplicative_3413_, 2);
v_toSeqLeft_3416_ = lean_ctor_get(v_toApplicative_3413_, 3);
v_toSeqRight_3417_ = lean_ctor_get(v_toApplicative_3413_, 4);
v___f_3418_ = ((lean_object*)(lp_aesop_Aesop_RuleTac_forwardExpr___closed__3));
v___f_3419_ = ((lean_object*)(lp_aesop_Aesop_RuleTac_forwardExpr___closed__4));
lean_inc_ref_n(v_toFunctor_3414_, 2);
v___f_3420_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_3420_, 0, v_toFunctor_3414_);
v___f_3421_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_3421_, 0, v_toFunctor_3414_);
v___x_3422_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3422_, 0, v___f_3420_);
lean_ctor_set(v___x_3422_, 1, v___f_3421_);
lean_inc(v_toSeqRight_3417_);
v___f_3423_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_3423_, 0, v_toSeqRight_3417_);
lean_inc(v_toSeqLeft_3416_);
v___f_3424_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_3424_, 0, v_toSeqLeft_3416_);
lean_inc(v_toSeq_3415_);
v___f_3425_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_3425_, 0, v_toSeq_3415_);
v___x_3426_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_3426_, 0, v___x_3422_);
lean_ctor_set(v___x_3426_, 1, v___f_3418_);
lean_ctor_set(v___x_3426_, 2, v___f_3425_);
lean_ctor_set(v___x_3426_, 3, v___f_3424_);
lean_ctor_set(v___x_3426_, 4, v___f_3423_);
v___x_3427_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3427_, 0, v___x_3426_);
lean_ctor_set(v___x_3427_, 1, v___f_3419_);
v___x_3428_ = l_StateRefT_x27_instMonad___redArg(v___x_3427_);
v___x_3429_ = lean_alloc_closure((void*)(l_ReaderT_pure___boxed), 6, 3);
lean_closure_set(v___x_3429_, 0, lean_box(0));
lean_closure_set(v___x_3429_, 1, lean_box(0));
lean_closure_set(v___x_3429_, 2, v___x_3428_);
v___x_3430_ = l_instMonadControlTOfPure___redArg(v___x_3429_);
lean_inc_ref(v___x_3430_);
v___f_3431_ = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__3), 4, 2);
lean_closure_set(v___f_3431_, 0, v___x_3411_);
lean_closure_set(v___f_3431_, 1, v___x_3430_);
v___f_3432_ = lean_alloc_closure((void*)(l_instMonadControlTOfMonadControl___redArg___lam__4), 4, 2);
lean_closure_set(v___f_3432_, 0, v___x_3411_);
lean_closure_set(v___f_3432_, 1, v___x_3430_);
v___x_3433_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3433_, 0, v___f_3431_);
lean_ctor_set(v___x_3433_, 1, v___f_3432_);
v_toApplicative_3434_ = lean_ctor_get(v___x_3412_, 0);
v_toFunctor_3435_ = lean_ctor_get(v_toApplicative_3434_, 0);
v_toSeq_3436_ = lean_ctor_get(v_toApplicative_3434_, 2);
v_toSeqLeft_3437_ = lean_ctor_get(v_toApplicative_3434_, 3);
v_toSeqRight_3438_ = lean_ctor_get(v_toApplicative_3434_, 4);
lean_inc_ref_n(v_toFunctor_3435_, 2);
v___f_3439_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_3439_, 0, v_toFunctor_3435_);
v___f_3440_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_3440_, 0, v_toFunctor_3435_);
v___x_3441_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3441_, 0, v___f_3439_);
lean_ctor_set(v___x_3441_, 1, v___f_3440_);
lean_inc(v_toSeqRight_3438_);
v___f_3442_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_3442_, 0, v_toSeqRight_3438_);
lean_inc(v_toSeqLeft_3437_);
v___f_3443_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_3443_, 0, v_toSeqLeft_3437_);
lean_inc(v_toSeq_3436_);
v___f_3444_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_3444_, 0, v_toSeq_3436_);
v___x_3445_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_3445_, 0, v___x_3441_);
lean_ctor_set(v___x_3445_, 1, v___f_3418_);
lean_ctor_set(v___x_3445_, 2, v___f_3444_);
lean_ctor_set(v___x_3445_, 3, v___f_3443_);
lean_ctor_set(v___x_3445_, 4, v___f_3442_);
v___x_3446_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3446_, 0, v___x_3445_);
lean_ctor_set(v___x_3446_, 1, v___f_3419_);
v___x_3447_ = l_StateRefT_x27_instMonad___redArg(v___x_3446_);
v_toApplicative_3448_ = lean_ctor_get(v___x_3447_, 0);
v_isSharedCheck_3522_ = !lean_is_exclusive(v___x_3447_);
if (v_isSharedCheck_3522_ == 0)
{
lean_object* v_unused_3523_; 
v_unused_3523_ = lean_ctor_get(v___x_3447_, 1);
lean_dec(v_unused_3523_);
v___x_3450_ = v___x_3447_;
v_isShared_3451_ = v_isSharedCheck_3522_;
goto v_resetjp_3449_;
}
else
{
lean_inc(v_toApplicative_3448_);
lean_dec(v___x_3447_);
v___x_3450_ = lean_box(0);
v_isShared_3451_ = v_isSharedCheck_3522_;
goto v_resetjp_3449_;
}
v_resetjp_3449_:
{
lean_object* v_toFunctor_3452_; lean_object* v_toSeq_3453_; lean_object* v_toSeqLeft_3454_; lean_object* v_toSeqRight_3455_; lean_object* v___x_3457_; uint8_t v_isShared_3458_; uint8_t v_isSharedCheck_3520_; 
v_toFunctor_3452_ = lean_ctor_get(v_toApplicative_3448_, 0);
v_toSeq_3453_ = lean_ctor_get(v_toApplicative_3448_, 2);
v_toSeqLeft_3454_ = lean_ctor_get(v_toApplicative_3448_, 3);
v_toSeqRight_3455_ = lean_ctor_get(v_toApplicative_3448_, 4);
v_isSharedCheck_3520_ = !lean_is_exclusive(v_toApplicative_3448_);
if (v_isSharedCheck_3520_ == 0)
{
lean_object* v_unused_3521_; 
v_unused_3521_ = lean_ctor_get(v_toApplicative_3448_, 1);
lean_dec(v_unused_3521_);
v___x_3457_ = v_toApplicative_3448_;
v_isShared_3458_ = v_isSharedCheck_3520_;
goto v_resetjp_3456_;
}
else
{
lean_inc(v_toSeqRight_3455_);
lean_inc(v_toSeqLeft_3454_);
lean_inc(v_toSeq_3453_);
lean_inc(v_toFunctor_3452_);
lean_dec(v_toApplicative_3448_);
v___x_3457_ = lean_box(0);
v_isShared_3458_ = v_isSharedCheck_3520_;
goto v_resetjp_3456_;
}
v_resetjp_3456_:
{
lean_object* v___f_3459_; lean_object* v___f_3460_; lean_object* v___f_3461_; lean_object* v___f_3462_; lean_object* v___x_3463_; lean_object* v___f_3464_; lean_object* v___f_3465_; lean_object* v___f_3466_; lean_object* v___x_3468_; 
v___f_3459_ = ((lean_object*)(lp_aesop_Aesop_RuleTac_forwardExpr___closed__5));
v___f_3460_ = ((lean_object*)(lp_aesop_Aesop_RuleTac_forwardExpr___closed__6));
lean_inc_ref(v_toFunctor_3452_);
v___f_3461_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_3461_, 0, v_toFunctor_3452_);
v___f_3462_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_3462_, 0, v_toFunctor_3452_);
v___x_3463_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3463_, 0, v___f_3461_);
lean_ctor_set(v___x_3463_, 1, v___f_3462_);
v___f_3464_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_3464_, 0, v_toSeqRight_3455_);
v___f_3465_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_3465_, 0, v_toSeqLeft_3454_);
v___f_3466_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_3466_, 0, v_toSeq_3453_);
if (v_isShared_3458_ == 0)
{
lean_ctor_set(v___x_3457_, 4, v___f_3464_);
lean_ctor_set(v___x_3457_, 3, v___f_3465_);
lean_ctor_set(v___x_3457_, 2, v___f_3466_);
lean_ctor_set(v___x_3457_, 1, v___f_3459_);
lean_ctor_set(v___x_3457_, 0, v___x_3463_);
v___x_3468_ = v___x_3457_;
goto v_reusejp_3467_;
}
else
{
lean_object* v_reuseFailAlloc_3519_; 
v_reuseFailAlloc_3519_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3519_, 0, v___x_3463_);
lean_ctor_set(v_reuseFailAlloc_3519_, 1, v___f_3459_);
lean_ctor_set(v_reuseFailAlloc_3519_, 2, v___f_3466_);
lean_ctor_set(v_reuseFailAlloc_3519_, 3, v___f_3465_);
lean_ctor_set(v_reuseFailAlloc_3519_, 4, v___f_3464_);
v___x_3468_ = v_reuseFailAlloc_3519_;
goto v_reusejp_3467_;
}
v_reusejp_3467_:
{
lean_object* v___x_3470_; 
if (v_isShared_3451_ == 0)
{
lean_ctor_set(v___x_3450_, 1, v___f_3460_);
lean_ctor_set(v___x_3450_, 0, v___x_3468_);
v___x_3470_ = v___x_3450_;
goto v_reusejp_3469_;
}
else
{
lean_object* v_reuseFailAlloc_3518_; 
v_reuseFailAlloc_3518_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3518_, 0, v___x_3468_);
lean_ctor_set(v_reuseFailAlloc_3518_, 1, v___f_3460_);
v___x_3470_ = v_reuseFailAlloc_3518_;
goto v_reusejp_3469_;
}
v_reusejp_3469_:
{
lean_object* v___x_3471_; lean_object* v_options_3472_; lean_object* v_goal_3473_; lean_object* v_patternSubsts_x3f_3474_; lean_object* v_forwardMaxDepth_x3f_3475_; lean_object* v___f_3476_; lean_object* v___f_3477_; lean_object* v___x_3478_; lean_object* v___x_3479_; lean_object* v___x_3480_; lean_object* v___x_3481_; lean_object* v___x_246__overap_3482_; lean_object* v___x_3483_; 
lean_inc_ref(v___x_3470_);
v___x_3471_ = l_StateRefT_x27_instMonad___redArg(v___x_3470_);
v_options_3472_ = lean_ctor_get(v_a_3404_, 4);
lean_inc_ref(v_options_3472_);
v_goal_3473_ = lean_ctor_get(v_a_3404_, 0);
lean_inc_n(v_goal_3473_, 2);
v_patternSubsts_x3f_3474_ = lean_ctor_get(v_a_3404_, 3);
lean_inc(v_patternSubsts_x3f_3474_);
lean_dec_ref(v_a_3404_);
v_forwardMaxDepth_x3f_3475_ = lean_ctor_get(v_options_3472_, 1);
lean_inc(v_forwardMaxDepth_x3f_3475_);
lean_dec_ref(v_options_3472_);
v___f_3476_ = ((lean_object*)(lp_aesop_Aesop_RuleTac_forwardExpr___closed__7));
v___f_3477_ = ((lean_object*)(lp_aesop_Aesop_RuleTac_forwardExpr___closed__19));
v___x_3478_ = lean_box(v_clear_3403_);
v___x_3479_ = lean_alloc_closure((void*)(lp_aesop_Aesop_RuleTac_applyForwardRule___boxed), 13, 6);
lean_closure_set(v___x_3479_, 0, v_goal_3473_);
lean_closure_set(v___x_3479_, 1, v_e_3401_);
lean_closure_set(v___x_3479_, 2, v_patternSubsts_x3f_3474_);
lean_closure_set(v___x_3479_, 3, v_immediate_3402_);
lean_closure_set(v___x_3479_, 4, v___x_3478_);
lean_closure_set(v___x_3479_, 5, v_forwardMaxDepth_x3f_3475_);
lean_inc_ref(v___x_3471_);
v___x_3480_ = lp_aesop_Aesop_ScriptT_run___redArg(v___x_3471_, v___f_3477_, v___x_3479_);
v___x_3481_ = lean_alloc_closure((void*)(l_ReaderT_bind___boxed), 8, 7);
lean_closure_set(v___x_3481_, 0, lean_box(0));
lean_closure_set(v___x_3481_, 1, lean_box(0));
lean_closure_set(v___x_3481_, 2, v___x_3470_);
lean_closure_set(v___x_3481_, 3, lean_box(0));
lean_closure_set(v___x_3481_, 4, lean_box(0));
lean_closure_set(v___x_3481_, 5, v___x_3480_);
lean_closure_set(v___x_3481_, 6, v___f_3476_);
v___x_246__overap_3482_ = l_Lean_MVarId_withContext___redArg(v___x_3433_, v___x_3471_, v_goal_3473_, v___x_3481_);
lean_inc(v_a_3409_);
lean_inc_ref(v_a_3408_);
lean_inc(v_a_3407_);
lean_inc_ref(v_a_3406_);
lean_inc(v_a_3405_);
v___x_3483_ = lean_apply_6(v___x_246__overap_3482_, v_a_3405_, v_a_3406_, v_a_3407_, v_a_3408_, v_a_3409_, lean_box(0));
if (lean_obj_tag(v___x_3483_) == 0)
{
lean_object* v_a_3484_; lean_object* v_snd_3485_; lean_object* v_fst_3486_; lean_object* v_fst_3487_; lean_object* v_snd_3488_; lean_object* v___x_3489_; 
v_a_3484_ = lean_ctor_get(v___x_3483_, 0);
lean_inc(v_a_3484_);
lean_dec_ref_known(v___x_3483_, 1);
v_snd_3485_ = lean_ctor_get(v_a_3484_, 1);
lean_inc(v_snd_3485_);
v_fst_3486_ = lean_ctor_get(v_a_3484_, 0);
lean_inc(v_fst_3486_);
lean_dec(v_a_3484_);
v_fst_3487_ = lean_ctor_get(v_snd_3485_, 0);
lean_inc(v_fst_3487_);
v_snd_3488_ = lean_ctor_get(v_snd_3485_, 1);
lean_inc(v_snd_3488_);
lean_dec(v_snd_3485_);
v___x_3489_ = l_Lean_Meta_saveState___redArg(v_a_3407_, v_a_3409_);
if (lean_obj_tag(v___x_3489_) == 0)
{
lean_object* v_a_3490_; lean_object* v___x_3492_; uint8_t v_isShared_3493_; uint8_t v_isSharedCheck_3501_; 
v_a_3490_ = lean_ctor_get(v___x_3489_, 0);
v_isSharedCheck_3501_ = !lean_is_exclusive(v___x_3489_);
if (v_isSharedCheck_3501_ == 0)
{
v___x_3492_ = v___x_3489_;
v_isShared_3493_ = v_isSharedCheck_3501_;
goto v_resetjp_3491_;
}
else
{
lean_inc(v_a_3490_);
lean_dec(v___x_3489_);
v___x_3492_ = lean_box(0);
v_isShared_3493_ = v_isSharedCheck_3501_;
goto v_resetjp_3491_;
}
v_resetjp_3491_:
{
lean_object* v___x_3494_; lean_object* v___x_3495_; lean_object* v___x_3496_; lean_object* v___x_3497_; lean_object* v___x_3499_; 
v___x_3494_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_3494_, 0, v_fst_3486_);
lean_ctor_set(v___x_3494_, 1, v_a_3490_);
lean_ctor_set(v___x_3494_, 2, v_fst_3487_);
lean_ctor_set(v___x_3494_, 3, v_snd_3488_);
v___x_3495_ = lean_unsigned_to_nat(1u);
v___x_3496_ = lean_mk_empty_array_with_capacity(v___x_3495_);
v___x_3497_ = lean_array_push(v___x_3496_, v___x_3494_);
if (v_isShared_3493_ == 0)
{
lean_ctor_set(v___x_3492_, 0, v___x_3497_);
v___x_3499_ = v___x_3492_;
goto v_reusejp_3498_;
}
else
{
lean_object* v_reuseFailAlloc_3500_; 
v_reuseFailAlloc_3500_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3500_, 0, v___x_3497_);
v___x_3499_ = v_reuseFailAlloc_3500_;
goto v_reusejp_3498_;
}
v_reusejp_3498_:
{
return v___x_3499_;
}
}
}
else
{
lean_object* v_a_3502_; lean_object* v___x_3504_; uint8_t v_isShared_3505_; uint8_t v_isSharedCheck_3509_; 
lean_dec(v_snd_3488_);
lean_dec(v_fst_3487_);
lean_dec(v_fst_3486_);
v_a_3502_ = lean_ctor_get(v___x_3489_, 0);
v_isSharedCheck_3509_ = !lean_is_exclusive(v___x_3489_);
if (v_isSharedCheck_3509_ == 0)
{
v___x_3504_ = v___x_3489_;
v_isShared_3505_ = v_isSharedCheck_3509_;
goto v_resetjp_3503_;
}
else
{
lean_inc(v_a_3502_);
lean_dec(v___x_3489_);
v___x_3504_ = lean_box(0);
v_isShared_3505_ = v_isSharedCheck_3509_;
goto v_resetjp_3503_;
}
v_resetjp_3503_:
{
lean_object* v___x_3507_; 
if (v_isShared_3505_ == 0)
{
v___x_3507_ = v___x_3504_;
goto v_reusejp_3506_;
}
else
{
lean_object* v_reuseFailAlloc_3508_; 
v_reuseFailAlloc_3508_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3508_, 0, v_a_3502_);
v___x_3507_ = v_reuseFailAlloc_3508_;
goto v_reusejp_3506_;
}
v_reusejp_3506_:
{
return v___x_3507_;
}
}
}
}
else
{
lean_object* v_a_3510_; lean_object* v___x_3512_; uint8_t v_isShared_3513_; uint8_t v_isSharedCheck_3517_; 
v_a_3510_ = lean_ctor_get(v___x_3483_, 0);
v_isSharedCheck_3517_ = !lean_is_exclusive(v___x_3483_);
if (v_isSharedCheck_3517_ == 0)
{
v___x_3512_ = v___x_3483_;
v_isShared_3513_ = v_isSharedCheck_3517_;
goto v_resetjp_3511_;
}
else
{
lean_inc(v_a_3510_);
lean_dec(v___x_3483_);
v___x_3512_ = lean_box(0);
v_isShared_3513_ = v_isSharedCheck_3517_;
goto v_resetjp_3511_;
}
v_resetjp_3511_:
{
lean_object* v___x_3515_; 
if (v_isShared_3513_ == 0)
{
v___x_3515_ = v___x_3512_;
goto v_reusejp_3514_;
}
else
{
lean_object* v_reuseFailAlloc_3516_; 
v_reuseFailAlloc_3516_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3516_, 0, v_a_3510_);
v___x_3515_ = v_reuseFailAlloc_3516_;
goto v_reusejp_3514_;
}
v_reusejp_3514_:
{
return v___x_3515_;
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_forwardExpr___boxed(lean_object* v_e_3524_, lean_object* v_immediate_3525_, lean_object* v_clear_3526_, lean_object* v_a_3527_, lean_object* v_a_3528_, lean_object* v_a_3529_, lean_object* v_a_3530_, lean_object* v_a_3531_, lean_object* v_a_3532_, lean_object* v_a_3533_){
_start:
{
uint8_t v_clear_boxed_3534_; lean_object* v_res_3535_; 
v_clear_boxed_3534_ = lean_unbox(v_clear_3526_);
v_res_3535_ = lp_aesop_Aesop_RuleTac_forwardExpr(v_e_3524_, v_immediate_3525_, v_clear_boxed_3534_, v_a_3527_, v_a_3528_, v_a_3529_, v_a_3530_, v_a_3531_, v_a_3532_);
lean_dec(v_a_3532_);
lean_dec_ref(v_a_3531_);
lean_dec(v_a_3530_);
lean_dec_ref(v_a_3529_);
lean_dec(v_a_3528_);
return v_res_3535_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_forwardConst_spec__0___redArg(lean_object* v_x_3538_, lean_object* v___y_3539_, lean_object* v___y_3540_, lean_object* v___y_3541_, lean_object* v___y_3542_, lean_object* v___y_3543_){
_start:
{
lean_object* v___x_3545_; lean_object* v___x_3546_; lean_object* v___x_3547_; 
v___x_3545_ = ((lean_object*)(lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_forwardConst_spec__0___redArg___closed__0));
v___x_3546_ = lean_st_mk_ref(v___x_3545_);
lean_inc(v___y_3543_);
lean_inc_ref(v___y_3542_);
lean_inc(v___y_3541_);
lean_inc_ref(v___y_3540_);
lean_inc(v___y_3539_);
lean_inc(v___x_3546_);
v___x_3547_ = lean_apply_7(v_x_3538_, v___x_3546_, v___y_3539_, v___y_3540_, v___y_3541_, v___y_3542_, v___y_3543_, lean_box(0));
if (lean_obj_tag(v___x_3547_) == 0)
{
lean_object* v_a_3548_; lean_object* v___x_3550_; uint8_t v_isShared_3551_; uint8_t v_isSharedCheck_3557_; 
v_a_3548_ = lean_ctor_get(v___x_3547_, 0);
v_isSharedCheck_3557_ = !lean_is_exclusive(v___x_3547_);
if (v_isSharedCheck_3557_ == 0)
{
v___x_3550_ = v___x_3547_;
v_isShared_3551_ = v_isSharedCheck_3557_;
goto v_resetjp_3549_;
}
else
{
lean_inc(v_a_3548_);
lean_dec(v___x_3547_);
v___x_3550_ = lean_box(0);
v_isShared_3551_ = v_isSharedCheck_3557_;
goto v_resetjp_3549_;
}
v_resetjp_3549_:
{
lean_object* v___x_3552_; lean_object* v___x_3553_; lean_object* v___x_3555_; 
v___x_3552_ = lean_st_ref_get(v___x_3546_);
lean_dec(v___x_3546_);
v___x_3553_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3553_, 0, v_a_3548_);
lean_ctor_set(v___x_3553_, 1, v___x_3552_);
if (v_isShared_3551_ == 0)
{
lean_ctor_set(v___x_3550_, 0, v___x_3553_);
v___x_3555_ = v___x_3550_;
goto v_reusejp_3554_;
}
else
{
lean_object* v_reuseFailAlloc_3556_; 
v_reuseFailAlloc_3556_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3556_, 0, v___x_3553_);
v___x_3555_ = v_reuseFailAlloc_3556_;
goto v_reusejp_3554_;
}
v_reusejp_3554_:
{
return v___x_3555_;
}
}
}
else
{
lean_object* v_a_3558_; lean_object* v___x_3560_; uint8_t v_isShared_3561_; uint8_t v_isSharedCheck_3565_; 
lean_dec(v___x_3546_);
v_a_3558_ = lean_ctor_get(v___x_3547_, 0);
v_isSharedCheck_3565_ = !lean_is_exclusive(v___x_3547_);
if (v_isSharedCheck_3565_ == 0)
{
v___x_3560_ = v___x_3547_;
v_isShared_3561_ = v_isSharedCheck_3565_;
goto v_resetjp_3559_;
}
else
{
lean_inc(v_a_3558_);
lean_dec(v___x_3547_);
v___x_3560_ = lean_box(0);
v_isShared_3561_ = v_isSharedCheck_3565_;
goto v_resetjp_3559_;
}
v_resetjp_3559_:
{
lean_object* v___x_3563_; 
if (v_isShared_3561_ == 0)
{
v___x_3563_ = v___x_3560_;
goto v_reusejp_3562_;
}
else
{
lean_object* v_reuseFailAlloc_3564_; 
v_reuseFailAlloc_3564_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3564_, 0, v_a_3558_);
v___x_3563_ = v_reuseFailAlloc_3564_;
goto v_reusejp_3562_;
}
v_reusejp_3562_:
{
return v___x_3563_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_forwardConst_spec__0___redArg___boxed(lean_object* v_x_3566_, lean_object* v___y_3567_, lean_object* v___y_3568_, lean_object* v___y_3569_, lean_object* v___y_3570_, lean_object* v___y_3571_, lean_object* v___y_3572_){
_start:
{
lean_object* v_res_3573_; 
v_res_3573_ = lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_forwardConst_spec__0___redArg(v_x_3566_, v___y_3567_, v___y_3568_, v___y_3569_, v___y_3570_, v___y_3571_);
lean_dec(v___y_3571_);
lean_dec_ref(v___y_3570_);
lean_dec(v___y_3569_);
lean_dec_ref(v___y_3568_);
lean_dec(v___y_3567_);
return v_res_3573_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_forwardConst_spec__0(lean_object* v_00_u03b1_3574_, lean_object* v_x_3575_, lean_object* v___y_3576_, lean_object* v___y_3577_, lean_object* v___y_3578_, lean_object* v___y_3579_, lean_object* v___y_3580_){
_start:
{
lean_object* v___x_3582_; 
v___x_3582_ = lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_forwardConst_spec__0___redArg(v_x_3575_, v___y_3576_, v___y_3577_, v___y_3578_, v___y_3579_, v___y_3580_);
return v___x_3582_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_forwardConst_spec__0___boxed(lean_object* v_00_u03b1_3583_, lean_object* v_x_3584_, lean_object* v___y_3585_, lean_object* v___y_3586_, lean_object* v___y_3587_, lean_object* v___y_3588_, lean_object* v___y_3589_, lean_object* v___y_3590_){
_start:
{
lean_object* v_res_3591_; 
v_res_3591_ = lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_forwardConst_spec__0(v_00_u03b1_3583_, v_x_3584_, v___y_3585_, v___y_3586_, v___y_3587_, v___y_3588_, v___y_3589_);
lean_dec(v___y_3589_);
lean_dec_ref(v___y_3588_);
lean_dec(v___y_3587_);
lean_dec_ref(v___y_3586_);
lean_dec(v___y_3585_);
return v_res_3591_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_RuleTac_forwardConst_spec__1___redArg___lam__0(lean_object* v_x_3592_, lean_object* v___y_3593_, lean_object* v___y_3594_, lean_object* v___y_3595_, lean_object* v___y_3596_, lean_object* v___y_3597_){
_start:
{
lean_object* v___x_3599_; 
lean_inc(v___y_3593_);
v___x_3599_ = lean_apply_6(v_x_3592_, v___y_3593_, v___y_3594_, v___y_3595_, v___y_3596_, v___y_3597_, lean_box(0));
return v___x_3599_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_RuleTac_forwardConst_spec__1___redArg___lam__0___boxed(lean_object* v_x_3600_, lean_object* v___y_3601_, lean_object* v___y_3602_, lean_object* v___y_3603_, lean_object* v___y_3604_, lean_object* v___y_3605_, lean_object* v___y_3606_){
_start:
{
lean_object* v_res_3607_; 
v_res_3607_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_RuleTac_forwardConst_spec__1___redArg___lam__0(v_x_3600_, v___y_3601_, v___y_3602_, v___y_3603_, v___y_3604_, v___y_3605_);
lean_dec(v___y_3601_);
return v_res_3607_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_RuleTac_forwardConst_spec__1___redArg(lean_object* v_mvarId_3608_, lean_object* v_x_3609_, lean_object* v___y_3610_, lean_object* v___y_3611_, lean_object* v___y_3612_, lean_object* v___y_3613_, lean_object* v___y_3614_){
_start:
{
lean_object* v___f_3616_; lean_object* v___x_3617_; 
lean_inc(v___y_3610_);
v___f_3616_ = lean_alloc_closure((void*)(lp_aesop_Lean_MVarId_withContext___at___00Aesop_RuleTac_forwardConst_spec__1___redArg___lam__0___boxed), 7, 2);
lean_closure_set(v___f_3616_, 0, v_x_3609_);
lean_closure_set(v___f_3616_, 1, v___y_3610_);
v___x_3617_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_box(0), v_mvarId_3608_, v___f_3616_, v___y_3611_, v___y_3612_, v___y_3613_, v___y_3614_);
if (lean_obj_tag(v___x_3617_) == 0)
{
return v___x_3617_;
}
else
{
lean_object* v_a_3618_; lean_object* v___x_3620_; uint8_t v_isShared_3621_; uint8_t v_isSharedCheck_3625_; 
v_a_3618_ = lean_ctor_get(v___x_3617_, 0);
v_isSharedCheck_3625_ = !lean_is_exclusive(v___x_3617_);
if (v_isSharedCheck_3625_ == 0)
{
v___x_3620_ = v___x_3617_;
v_isShared_3621_ = v_isSharedCheck_3625_;
goto v_resetjp_3619_;
}
else
{
lean_inc(v_a_3618_);
lean_dec(v___x_3617_);
v___x_3620_ = lean_box(0);
v_isShared_3621_ = v_isSharedCheck_3625_;
goto v_resetjp_3619_;
}
v_resetjp_3619_:
{
lean_object* v___x_3623_; 
if (v_isShared_3621_ == 0)
{
v___x_3623_ = v___x_3620_;
goto v_reusejp_3622_;
}
else
{
lean_object* v_reuseFailAlloc_3624_; 
v_reuseFailAlloc_3624_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3624_, 0, v_a_3618_);
v___x_3623_ = v_reuseFailAlloc_3624_;
goto v_reusejp_3622_;
}
v_reusejp_3622_:
{
return v___x_3623_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_RuleTac_forwardConst_spec__1___redArg___boxed(lean_object* v_mvarId_3626_, lean_object* v_x_3627_, lean_object* v___y_3628_, lean_object* v___y_3629_, lean_object* v___y_3630_, lean_object* v___y_3631_, lean_object* v___y_3632_, lean_object* v___y_3633_){
_start:
{
lean_object* v_res_3634_; 
v_res_3634_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_RuleTac_forwardConst_spec__1___redArg(v_mvarId_3626_, v_x_3627_, v___y_3628_, v___y_3629_, v___y_3630_, v___y_3631_, v___y_3632_);
lean_dec(v___y_3632_);
lean_dec_ref(v___y_3631_);
lean_dec(v___y_3630_);
lean_dec_ref(v___y_3629_);
lean_dec(v___y_3628_);
return v_res_3634_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_RuleTac_forwardConst_spec__1(lean_object* v_00_u03b1_3635_, lean_object* v_mvarId_3636_, lean_object* v_x_3637_, lean_object* v___y_3638_, lean_object* v___y_3639_, lean_object* v___y_3640_, lean_object* v___y_3641_, lean_object* v___y_3642_){
_start:
{
lean_object* v___x_3644_; 
v___x_3644_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_RuleTac_forwardConst_spec__1___redArg(v_mvarId_3636_, v_x_3637_, v___y_3638_, v___y_3639_, v___y_3640_, v___y_3641_, v___y_3642_);
return v___x_3644_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_RuleTac_forwardConst_spec__1___boxed(lean_object* v_00_u03b1_3645_, lean_object* v_mvarId_3646_, lean_object* v_x_3647_, lean_object* v___y_3648_, lean_object* v___y_3649_, lean_object* v___y_3650_, lean_object* v___y_3651_, lean_object* v___y_3652_, lean_object* v___y_3653_){
_start:
{
lean_object* v_res_3654_; 
v_res_3654_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_RuleTac_forwardConst_spec__1(v_00_u03b1_3645_, v_mvarId_3646_, v_x_3647_, v___y_3648_, v___y_3649_, v___y_3650_, v___y_3651_, v___y_3652_);
lean_dec(v___y_3652_);
lean_dec_ref(v___y_3651_);
lean_dec(v___y_3650_);
lean_dec_ref(v___y_3649_);
lean_dec(v___y_3648_);
return v_res_3654_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_forwardConst___lam__0(lean_object* v___x_3655_, lean_object* v___y_3656_, lean_object* v___y_3657_, lean_object* v___y_3658_, lean_object* v___y_3659_, lean_object* v___y_3660_){
_start:
{
lean_object* v___x_3662_; 
v___x_3662_ = lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_forwardConst_spec__0___redArg(v___x_3655_, v___y_3656_, v___y_3657_, v___y_3658_, v___y_3659_, v___y_3660_);
if (lean_obj_tag(v___x_3662_) == 0)
{
lean_object* v_a_3663_; lean_object* v___x_3665_; uint8_t v_isShared_3666_; uint8_t v_isSharedCheck_3685_; 
v_a_3663_ = lean_ctor_get(v___x_3662_, 0);
v_isSharedCheck_3685_ = !lean_is_exclusive(v___x_3662_);
if (v_isSharedCheck_3685_ == 0)
{
v___x_3665_ = v___x_3662_;
v_isShared_3666_ = v_isSharedCheck_3685_;
goto v_resetjp_3664_;
}
else
{
lean_inc(v_a_3663_);
lean_dec(v___x_3662_);
v___x_3665_ = lean_box(0);
v_isShared_3666_ = v_isSharedCheck_3685_;
goto v_resetjp_3664_;
}
v_resetjp_3664_:
{
lean_object* v_fst_3667_; lean_object* v_snd_3668_; lean_object* v___x_3670_; uint8_t v_isShared_3671_; uint8_t v_isSharedCheck_3684_; 
v_fst_3667_ = lean_ctor_get(v_a_3663_, 0);
v_snd_3668_ = lean_ctor_get(v_a_3663_, 1);
v_isSharedCheck_3684_ = !lean_is_exclusive(v_a_3663_);
if (v_isSharedCheck_3684_ == 0)
{
v___x_3670_ = v_a_3663_;
v_isShared_3671_ = v_isSharedCheck_3684_;
goto v_resetjp_3669_;
}
else
{
lean_inc(v_snd_3668_);
lean_inc(v_fst_3667_);
lean_dec(v_a_3663_);
v___x_3670_ = lean_box(0);
v_isShared_3671_ = v_isSharedCheck_3684_;
goto v_resetjp_3669_;
}
v_resetjp_3669_:
{
lean_object* v___x_3672_; lean_object* v___x_3673_; lean_object* v___x_3674_; lean_object* v___x_3675_; lean_object* v___x_3676_; lean_object* v___x_3678_; 
v___x_3672_ = lean_unsigned_to_nat(1u);
v___x_3673_ = lean_mk_empty_array_with_capacity(v___x_3672_);
v___x_3674_ = lean_array_push(v___x_3673_, v_fst_3667_);
v___x_3675_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3675_, 0, v_snd_3668_);
v___x_3676_ = lean_box(0);
if (v_isShared_3671_ == 0)
{
lean_ctor_set(v___x_3670_, 1, v___x_3676_);
lean_ctor_set(v___x_3670_, 0, v___x_3675_);
v___x_3678_ = v___x_3670_;
goto v_reusejp_3677_;
}
else
{
lean_object* v_reuseFailAlloc_3683_; 
v_reuseFailAlloc_3683_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3683_, 0, v___x_3675_);
lean_ctor_set(v_reuseFailAlloc_3683_, 1, v___x_3676_);
v___x_3678_ = v_reuseFailAlloc_3683_;
goto v_reusejp_3677_;
}
v_reusejp_3677_:
{
lean_object* v___x_3679_; lean_object* v___x_3681_; 
v___x_3679_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3679_, 0, v___x_3674_);
lean_ctor_set(v___x_3679_, 1, v___x_3678_);
if (v_isShared_3666_ == 0)
{
lean_ctor_set(v___x_3665_, 0, v___x_3679_);
v___x_3681_ = v___x_3665_;
goto v_reusejp_3680_;
}
else
{
lean_object* v_reuseFailAlloc_3682_; 
v_reuseFailAlloc_3682_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3682_, 0, v___x_3679_);
v___x_3681_ = v_reuseFailAlloc_3682_;
goto v_reusejp_3680_;
}
v_reusejp_3680_:
{
return v___x_3681_;
}
}
}
}
}
else
{
lean_object* v_a_3686_; lean_object* v___x_3688_; uint8_t v_isShared_3689_; uint8_t v_isSharedCheck_3693_; 
v_a_3686_ = lean_ctor_get(v___x_3662_, 0);
v_isSharedCheck_3693_ = !lean_is_exclusive(v___x_3662_);
if (v_isSharedCheck_3693_ == 0)
{
v___x_3688_ = v___x_3662_;
v_isShared_3689_ = v_isSharedCheck_3693_;
goto v_resetjp_3687_;
}
else
{
lean_inc(v_a_3686_);
lean_dec(v___x_3662_);
v___x_3688_ = lean_box(0);
v_isShared_3689_ = v_isSharedCheck_3693_;
goto v_resetjp_3687_;
}
v_resetjp_3687_:
{
lean_object* v___x_3691_; 
if (v_isShared_3689_ == 0)
{
v___x_3691_ = v___x_3688_;
goto v_reusejp_3690_;
}
else
{
lean_object* v_reuseFailAlloc_3692_; 
v_reuseFailAlloc_3692_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3692_, 0, v_a_3686_);
v___x_3691_ = v_reuseFailAlloc_3692_;
goto v_reusejp_3690_;
}
v_reusejp_3690_:
{
return v___x_3691_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_forwardConst___lam__0___boxed(lean_object* v___x_3694_, lean_object* v___y_3695_, lean_object* v___y_3696_, lean_object* v___y_3697_, lean_object* v___y_3698_, lean_object* v___y_3699_, lean_object* v___y_3700_){
_start:
{
lean_object* v_res_3701_; 
v_res_3701_ = lp_aesop_Aesop_RuleTac_forwardConst___lam__0(v___x_3694_, v___y_3695_, v___y_3696_, v___y_3697_, v___y_3698_, v___y_3699_);
lean_dec(v___y_3699_);
lean_dec_ref(v___y_3698_);
lean_dec(v___y_3697_);
lean_dec_ref(v___y_3696_);
lean_dec(v___y_3695_);
return v_res_3701_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_forwardConst(lean_object* v_decl_3702_, lean_object* v_immediate_3703_, uint8_t v_clear_3704_, lean_object* v_input_3705_, lean_object* v_a_3706_, lean_object* v_a_3707_, lean_object* v_a_3708_, lean_object* v_a_3709_, lean_object* v_a_3710_){
_start:
{
lean_object* v___x_3712_; 
v___x_3712_ = l_Lean_Meta_mkConstWithFreshMVarLevels(v_decl_3702_, v_a_3707_, v_a_3708_, v_a_3709_, v_a_3710_);
if (lean_obj_tag(v___x_3712_) == 0)
{
lean_object* v_options_3713_; lean_object* v_a_3714_; lean_object* v_goal_3715_; lean_object* v_patternSubsts_x3f_3716_; lean_object* v_forwardMaxDepth_x3f_3717_; lean_object* v___x_3718_; lean_object* v___x_3719_; lean_object* v___f_3720_; lean_object* v___x_3721_; 
v_options_3713_ = lean_ctor_get(v_input_3705_, 4);
lean_inc_ref(v_options_3713_);
v_a_3714_ = lean_ctor_get(v___x_3712_, 0);
lean_inc(v_a_3714_);
lean_dec_ref_known(v___x_3712_, 1);
v_goal_3715_ = lean_ctor_get(v_input_3705_, 0);
lean_inc_n(v_goal_3715_, 2);
v_patternSubsts_x3f_3716_ = lean_ctor_get(v_input_3705_, 3);
lean_inc(v_patternSubsts_x3f_3716_);
lean_dec_ref(v_input_3705_);
v_forwardMaxDepth_x3f_3717_ = lean_ctor_get(v_options_3713_, 1);
lean_inc(v_forwardMaxDepth_x3f_3717_);
lean_dec_ref(v_options_3713_);
v___x_3718_ = lean_box(v_clear_3704_);
v___x_3719_ = lean_alloc_closure((void*)(lp_aesop_Aesop_RuleTac_applyForwardRule___boxed), 13, 6);
lean_closure_set(v___x_3719_, 0, v_goal_3715_);
lean_closure_set(v___x_3719_, 1, v_a_3714_);
lean_closure_set(v___x_3719_, 2, v_patternSubsts_x3f_3716_);
lean_closure_set(v___x_3719_, 3, v_immediate_3703_);
lean_closure_set(v___x_3719_, 4, v___x_3718_);
lean_closure_set(v___x_3719_, 5, v_forwardMaxDepth_x3f_3717_);
v___f_3720_ = lean_alloc_closure((void*)(lp_aesop_Aesop_RuleTac_forwardConst___lam__0___boxed), 7, 1);
lean_closure_set(v___f_3720_, 0, v___x_3719_);
v___x_3721_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_RuleTac_forwardConst_spec__1___redArg(v_goal_3715_, v___f_3720_, v_a_3706_, v_a_3707_, v_a_3708_, v_a_3709_, v_a_3710_);
if (lean_obj_tag(v___x_3721_) == 0)
{
lean_object* v_a_3722_; lean_object* v_snd_3723_; lean_object* v_fst_3724_; lean_object* v_fst_3725_; lean_object* v_snd_3726_; lean_object* v___x_3727_; 
v_a_3722_ = lean_ctor_get(v___x_3721_, 0);
lean_inc(v_a_3722_);
lean_dec_ref_known(v___x_3721_, 1);
v_snd_3723_ = lean_ctor_get(v_a_3722_, 1);
lean_inc(v_snd_3723_);
v_fst_3724_ = lean_ctor_get(v_a_3722_, 0);
lean_inc(v_fst_3724_);
lean_dec(v_a_3722_);
v_fst_3725_ = lean_ctor_get(v_snd_3723_, 0);
lean_inc(v_fst_3725_);
v_snd_3726_ = lean_ctor_get(v_snd_3723_, 1);
lean_inc(v_snd_3726_);
lean_dec(v_snd_3723_);
v___x_3727_ = l_Lean_Meta_saveState___redArg(v_a_3708_, v_a_3710_);
if (lean_obj_tag(v___x_3727_) == 0)
{
lean_object* v_a_3728_; lean_object* v___x_3730_; uint8_t v_isShared_3731_; uint8_t v_isSharedCheck_3739_; 
v_a_3728_ = lean_ctor_get(v___x_3727_, 0);
v_isSharedCheck_3739_ = !lean_is_exclusive(v___x_3727_);
if (v_isSharedCheck_3739_ == 0)
{
v___x_3730_ = v___x_3727_;
v_isShared_3731_ = v_isSharedCheck_3739_;
goto v_resetjp_3729_;
}
else
{
lean_inc(v_a_3728_);
lean_dec(v___x_3727_);
v___x_3730_ = lean_box(0);
v_isShared_3731_ = v_isSharedCheck_3739_;
goto v_resetjp_3729_;
}
v_resetjp_3729_:
{
lean_object* v___x_3732_; lean_object* v___x_3733_; lean_object* v___x_3734_; lean_object* v___x_3735_; lean_object* v___x_3737_; 
v___x_3732_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_3732_, 0, v_fst_3724_);
lean_ctor_set(v___x_3732_, 1, v_a_3728_);
lean_ctor_set(v___x_3732_, 2, v_fst_3725_);
lean_ctor_set(v___x_3732_, 3, v_snd_3726_);
v___x_3733_ = lean_unsigned_to_nat(1u);
v___x_3734_ = lean_mk_empty_array_with_capacity(v___x_3733_);
v___x_3735_ = lean_array_push(v___x_3734_, v___x_3732_);
if (v_isShared_3731_ == 0)
{
lean_ctor_set(v___x_3730_, 0, v___x_3735_);
v___x_3737_ = v___x_3730_;
goto v_reusejp_3736_;
}
else
{
lean_object* v_reuseFailAlloc_3738_; 
v_reuseFailAlloc_3738_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3738_, 0, v___x_3735_);
v___x_3737_ = v_reuseFailAlloc_3738_;
goto v_reusejp_3736_;
}
v_reusejp_3736_:
{
return v___x_3737_;
}
}
}
else
{
lean_object* v_a_3740_; lean_object* v___x_3742_; uint8_t v_isShared_3743_; uint8_t v_isSharedCheck_3747_; 
lean_dec(v_snd_3726_);
lean_dec(v_fst_3725_);
lean_dec(v_fst_3724_);
v_a_3740_ = lean_ctor_get(v___x_3727_, 0);
v_isSharedCheck_3747_ = !lean_is_exclusive(v___x_3727_);
if (v_isSharedCheck_3747_ == 0)
{
v___x_3742_ = v___x_3727_;
v_isShared_3743_ = v_isSharedCheck_3747_;
goto v_resetjp_3741_;
}
else
{
lean_inc(v_a_3740_);
lean_dec(v___x_3727_);
v___x_3742_ = lean_box(0);
v_isShared_3743_ = v_isSharedCheck_3747_;
goto v_resetjp_3741_;
}
v_resetjp_3741_:
{
lean_object* v___x_3745_; 
if (v_isShared_3743_ == 0)
{
v___x_3745_ = v___x_3742_;
goto v_reusejp_3744_;
}
else
{
lean_object* v_reuseFailAlloc_3746_; 
v_reuseFailAlloc_3746_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3746_, 0, v_a_3740_);
v___x_3745_ = v_reuseFailAlloc_3746_;
goto v_reusejp_3744_;
}
v_reusejp_3744_:
{
return v___x_3745_;
}
}
}
}
else
{
lean_object* v_a_3748_; lean_object* v___x_3750_; uint8_t v_isShared_3751_; uint8_t v_isSharedCheck_3755_; 
v_a_3748_ = lean_ctor_get(v___x_3721_, 0);
v_isSharedCheck_3755_ = !lean_is_exclusive(v___x_3721_);
if (v_isSharedCheck_3755_ == 0)
{
v___x_3750_ = v___x_3721_;
v_isShared_3751_ = v_isSharedCheck_3755_;
goto v_resetjp_3749_;
}
else
{
lean_inc(v_a_3748_);
lean_dec(v___x_3721_);
v___x_3750_ = lean_box(0);
v_isShared_3751_ = v_isSharedCheck_3755_;
goto v_resetjp_3749_;
}
v_resetjp_3749_:
{
lean_object* v___x_3753_; 
if (v_isShared_3751_ == 0)
{
v___x_3753_ = v___x_3750_;
goto v_reusejp_3752_;
}
else
{
lean_object* v_reuseFailAlloc_3754_; 
v_reuseFailAlloc_3754_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3754_, 0, v_a_3748_);
v___x_3753_ = v_reuseFailAlloc_3754_;
goto v_reusejp_3752_;
}
v_reusejp_3752_:
{
return v___x_3753_;
}
}
}
}
else
{
lean_object* v_a_3756_; lean_object* v___x_3758_; uint8_t v_isShared_3759_; uint8_t v_isSharedCheck_3763_; 
lean_dec_ref(v_input_3705_);
lean_dec_ref(v_immediate_3703_);
v_a_3756_ = lean_ctor_get(v___x_3712_, 0);
v_isSharedCheck_3763_ = !lean_is_exclusive(v___x_3712_);
if (v_isSharedCheck_3763_ == 0)
{
v___x_3758_ = v___x_3712_;
v_isShared_3759_ = v_isSharedCheck_3763_;
goto v_resetjp_3757_;
}
else
{
lean_inc(v_a_3756_);
lean_dec(v___x_3712_);
v___x_3758_ = lean_box(0);
v_isShared_3759_ = v_isSharedCheck_3763_;
goto v_resetjp_3757_;
}
v_resetjp_3757_:
{
lean_object* v___x_3761_; 
if (v_isShared_3759_ == 0)
{
v___x_3761_ = v___x_3758_;
goto v_reusejp_3760_;
}
else
{
lean_object* v_reuseFailAlloc_3762_; 
v_reuseFailAlloc_3762_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3762_, 0, v_a_3756_);
v___x_3761_ = v_reuseFailAlloc_3762_;
goto v_reusejp_3760_;
}
v_reusejp_3760_:
{
return v___x_3761_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_forwardConst___boxed(lean_object* v_decl_3764_, lean_object* v_immediate_3765_, lean_object* v_clear_3766_, lean_object* v_input_3767_, lean_object* v_a_3768_, lean_object* v_a_3769_, lean_object* v_a_3770_, lean_object* v_a_3771_, lean_object* v_a_3772_, lean_object* v_a_3773_){
_start:
{
uint8_t v_clear_boxed_3774_; lean_object* v_res_3775_; 
v_clear_boxed_3774_ = lean_unbox(v_clear_3766_);
v_res_3775_ = lp_aesop_Aesop_RuleTac_forwardConst(v_decl_3764_, v_immediate_3765_, v_clear_boxed_3774_, v_input_3767_, v_a_3768_, v_a_3769_, v_a_3770_, v_a_3771_, v_a_3772_);
lean_dec(v_a_3772_);
lean_dec_ref(v_a_3771_);
lean_dec(v_a_3770_);
lean_dec_ref(v_a_3769_);
lean_dec(v_a_3768_);
return v_res_3775_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_forwardTerm___lam__0(lean_object* v___x_3776_, lean_object* v___y_3777_, lean_object* v___y_3778_, lean_object* v___y_3779_, lean_object* v___y_3780_, lean_object* v___y_3781_){
_start:
{
lean_object* v___x_3783_; 
v___x_3783_ = lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_forwardConst_spec__0___redArg(v___x_3776_, v___y_3777_, v___y_3778_, v___y_3779_, v___y_3780_, v___y_3781_);
if (lean_obj_tag(v___x_3783_) == 0)
{
lean_object* v_a_3784_; lean_object* v___x_3786_; uint8_t v_isShared_3787_; uint8_t v_isSharedCheck_3806_; 
v_a_3784_ = lean_ctor_get(v___x_3783_, 0);
v_isSharedCheck_3806_ = !lean_is_exclusive(v___x_3783_);
if (v_isSharedCheck_3806_ == 0)
{
v___x_3786_ = v___x_3783_;
v_isShared_3787_ = v_isSharedCheck_3806_;
goto v_resetjp_3785_;
}
else
{
lean_inc(v_a_3784_);
lean_dec(v___x_3783_);
v___x_3786_ = lean_box(0);
v_isShared_3787_ = v_isSharedCheck_3806_;
goto v_resetjp_3785_;
}
v_resetjp_3785_:
{
lean_object* v_fst_3788_; lean_object* v_snd_3789_; lean_object* v___x_3791_; uint8_t v_isShared_3792_; uint8_t v_isSharedCheck_3805_; 
v_fst_3788_ = lean_ctor_get(v_a_3784_, 0);
v_snd_3789_ = lean_ctor_get(v_a_3784_, 1);
v_isSharedCheck_3805_ = !lean_is_exclusive(v_a_3784_);
if (v_isSharedCheck_3805_ == 0)
{
v___x_3791_ = v_a_3784_;
v_isShared_3792_ = v_isSharedCheck_3805_;
goto v_resetjp_3790_;
}
else
{
lean_inc(v_snd_3789_);
lean_inc(v_fst_3788_);
lean_dec(v_a_3784_);
v___x_3791_ = lean_box(0);
v_isShared_3792_ = v_isSharedCheck_3805_;
goto v_resetjp_3790_;
}
v_resetjp_3790_:
{
lean_object* v___x_3793_; lean_object* v___x_3794_; lean_object* v___x_3795_; lean_object* v___x_3796_; lean_object* v___x_3797_; lean_object* v___x_3799_; 
v___x_3793_ = lean_unsigned_to_nat(1u);
v___x_3794_ = lean_mk_empty_array_with_capacity(v___x_3793_);
v___x_3795_ = lean_array_push(v___x_3794_, v_fst_3788_);
v___x_3796_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3796_, 0, v_snd_3789_);
v___x_3797_ = lean_box(0);
if (v_isShared_3792_ == 0)
{
lean_ctor_set(v___x_3791_, 1, v___x_3797_);
lean_ctor_set(v___x_3791_, 0, v___x_3796_);
v___x_3799_ = v___x_3791_;
goto v_reusejp_3798_;
}
else
{
lean_object* v_reuseFailAlloc_3804_; 
v_reuseFailAlloc_3804_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3804_, 0, v___x_3796_);
lean_ctor_set(v_reuseFailAlloc_3804_, 1, v___x_3797_);
v___x_3799_ = v_reuseFailAlloc_3804_;
goto v_reusejp_3798_;
}
v_reusejp_3798_:
{
lean_object* v___x_3800_; lean_object* v___x_3802_; 
v___x_3800_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3800_, 0, v___x_3795_);
lean_ctor_set(v___x_3800_, 1, v___x_3799_);
if (v_isShared_3787_ == 0)
{
lean_ctor_set(v___x_3786_, 0, v___x_3800_);
v___x_3802_ = v___x_3786_;
goto v_reusejp_3801_;
}
else
{
lean_object* v_reuseFailAlloc_3803_; 
v_reuseFailAlloc_3803_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3803_, 0, v___x_3800_);
v___x_3802_ = v_reuseFailAlloc_3803_;
goto v_reusejp_3801_;
}
v_reusejp_3801_:
{
return v___x_3802_;
}
}
}
}
}
else
{
lean_object* v_a_3807_; lean_object* v___x_3809_; uint8_t v_isShared_3810_; uint8_t v_isSharedCheck_3814_; 
v_a_3807_ = lean_ctor_get(v___x_3783_, 0);
v_isSharedCheck_3814_ = !lean_is_exclusive(v___x_3783_);
if (v_isSharedCheck_3814_ == 0)
{
v___x_3809_ = v___x_3783_;
v_isShared_3810_ = v_isSharedCheck_3814_;
goto v_resetjp_3808_;
}
else
{
lean_inc(v_a_3807_);
lean_dec(v___x_3783_);
v___x_3809_ = lean_box(0);
v_isShared_3810_ = v_isSharedCheck_3814_;
goto v_resetjp_3808_;
}
v_resetjp_3808_:
{
lean_object* v___x_3812_; 
if (v_isShared_3810_ == 0)
{
v___x_3812_ = v___x_3809_;
goto v_reusejp_3811_;
}
else
{
lean_object* v_reuseFailAlloc_3813_; 
v_reuseFailAlloc_3813_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3813_, 0, v_a_3807_);
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
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_forwardTerm___lam__0___boxed(lean_object* v___x_3815_, lean_object* v___y_3816_, lean_object* v___y_3817_, lean_object* v___y_3818_, lean_object* v___y_3819_, lean_object* v___y_3820_, lean_object* v___y_3821_){
_start:
{
lean_object* v_res_3822_; 
v_res_3822_ = lp_aesop_Aesop_RuleTac_forwardTerm___lam__0(v___x_3815_, v___y_3816_, v___y_3817_, v___y_3818_, v___y_3819_, v___y_3820_);
lean_dec(v___y_3820_);
lean_dec_ref(v___y_3819_);
lean_dec(v___y_3818_);
lean_dec_ref(v___y_3817_);
lean_dec(v___y_3816_);
return v_res_3822_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_forwardTerm___lam__1(lean_object* v_goal_3823_, lean_object* v_stx_3824_, lean_object* v_options_3825_, lean_object* v_patternSubsts_x3f_3826_, lean_object* v_immediate_3827_, uint8_t v_clear_3828_, lean_object* v___y_3829_, lean_object* v___y_3830_, lean_object* v___y_3831_, lean_object* v___y_3832_, lean_object* v___y_3833_){
_start:
{
lean_object* v___x_3835_; 
lean_inc(v_goal_3823_);
v___x_3835_ = lp_aesop_Aesop_elabRuleTermForApplyLikeMetaM(v_goal_3823_, v_stx_3824_, v___y_3830_, v___y_3831_, v___y_3832_, v___y_3833_);
if (lean_obj_tag(v___x_3835_) == 0)
{
lean_object* v_a_3836_; lean_object* v_forwardMaxDepth_x3f_3837_; lean_object* v___x_3838_; lean_object* v___x_3839_; lean_object* v___f_3840_; lean_object* v___x_3841_; 
v_a_3836_ = lean_ctor_get(v___x_3835_, 0);
lean_inc(v_a_3836_);
lean_dec_ref_known(v___x_3835_, 1);
v_forwardMaxDepth_x3f_3837_ = lean_ctor_get(v_options_3825_, 1);
lean_inc(v_forwardMaxDepth_x3f_3837_);
lean_dec_ref(v_options_3825_);
v___x_3838_ = lean_box(v_clear_3828_);
lean_inc(v_goal_3823_);
v___x_3839_ = lean_alloc_closure((void*)(lp_aesop_Aesop_RuleTac_applyForwardRule___boxed), 13, 6);
lean_closure_set(v___x_3839_, 0, v_goal_3823_);
lean_closure_set(v___x_3839_, 1, v_a_3836_);
lean_closure_set(v___x_3839_, 2, v_patternSubsts_x3f_3826_);
lean_closure_set(v___x_3839_, 3, v_immediate_3827_);
lean_closure_set(v___x_3839_, 4, v___x_3838_);
lean_closure_set(v___x_3839_, 5, v_forwardMaxDepth_x3f_3837_);
v___f_3840_ = lean_alloc_closure((void*)(lp_aesop_Aesop_RuleTac_forwardTerm___lam__0___boxed), 7, 1);
lean_closure_set(v___f_3840_, 0, v___x_3839_);
v___x_3841_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_RuleTac_forwardConst_spec__1___redArg(v_goal_3823_, v___f_3840_, v___y_3829_, v___y_3830_, v___y_3831_, v___y_3832_, v___y_3833_);
if (lean_obj_tag(v___x_3841_) == 0)
{
lean_object* v_a_3842_; lean_object* v_snd_3843_; lean_object* v_fst_3844_; lean_object* v_fst_3845_; lean_object* v_snd_3846_; lean_object* v___x_3847_; 
v_a_3842_ = lean_ctor_get(v___x_3841_, 0);
lean_inc(v_a_3842_);
lean_dec_ref_known(v___x_3841_, 1);
v_snd_3843_ = lean_ctor_get(v_a_3842_, 1);
lean_inc(v_snd_3843_);
v_fst_3844_ = lean_ctor_get(v_a_3842_, 0);
lean_inc(v_fst_3844_);
lean_dec(v_a_3842_);
v_fst_3845_ = lean_ctor_get(v_snd_3843_, 0);
lean_inc(v_fst_3845_);
v_snd_3846_ = lean_ctor_get(v_snd_3843_, 1);
lean_inc(v_snd_3846_);
lean_dec(v_snd_3843_);
v___x_3847_ = l_Lean_Meta_saveState___redArg(v___y_3831_, v___y_3833_);
if (lean_obj_tag(v___x_3847_) == 0)
{
lean_object* v_a_3848_; lean_object* v___x_3850_; uint8_t v_isShared_3851_; uint8_t v_isSharedCheck_3859_; 
v_a_3848_ = lean_ctor_get(v___x_3847_, 0);
v_isSharedCheck_3859_ = !lean_is_exclusive(v___x_3847_);
if (v_isSharedCheck_3859_ == 0)
{
v___x_3850_ = v___x_3847_;
v_isShared_3851_ = v_isSharedCheck_3859_;
goto v_resetjp_3849_;
}
else
{
lean_inc(v_a_3848_);
lean_dec(v___x_3847_);
v___x_3850_ = lean_box(0);
v_isShared_3851_ = v_isSharedCheck_3859_;
goto v_resetjp_3849_;
}
v_resetjp_3849_:
{
lean_object* v___x_3852_; lean_object* v___x_3853_; lean_object* v___x_3854_; lean_object* v___x_3855_; lean_object* v___x_3857_; 
v___x_3852_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_3852_, 0, v_fst_3844_);
lean_ctor_set(v___x_3852_, 1, v_a_3848_);
lean_ctor_set(v___x_3852_, 2, v_fst_3845_);
lean_ctor_set(v___x_3852_, 3, v_snd_3846_);
v___x_3853_ = lean_unsigned_to_nat(1u);
v___x_3854_ = lean_mk_empty_array_with_capacity(v___x_3853_);
v___x_3855_ = lean_array_push(v___x_3854_, v___x_3852_);
if (v_isShared_3851_ == 0)
{
lean_ctor_set(v___x_3850_, 0, v___x_3855_);
v___x_3857_ = v___x_3850_;
goto v_reusejp_3856_;
}
else
{
lean_object* v_reuseFailAlloc_3858_; 
v_reuseFailAlloc_3858_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3858_, 0, v___x_3855_);
v___x_3857_ = v_reuseFailAlloc_3858_;
goto v_reusejp_3856_;
}
v_reusejp_3856_:
{
return v___x_3857_;
}
}
}
else
{
lean_object* v_a_3860_; lean_object* v___x_3862_; uint8_t v_isShared_3863_; uint8_t v_isSharedCheck_3867_; 
lean_dec(v_snd_3846_);
lean_dec(v_fst_3845_);
lean_dec(v_fst_3844_);
v_a_3860_ = lean_ctor_get(v___x_3847_, 0);
v_isSharedCheck_3867_ = !lean_is_exclusive(v___x_3847_);
if (v_isSharedCheck_3867_ == 0)
{
v___x_3862_ = v___x_3847_;
v_isShared_3863_ = v_isSharedCheck_3867_;
goto v_resetjp_3861_;
}
else
{
lean_inc(v_a_3860_);
lean_dec(v___x_3847_);
v___x_3862_ = lean_box(0);
v_isShared_3863_ = v_isSharedCheck_3867_;
goto v_resetjp_3861_;
}
v_resetjp_3861_:
{
lean_object* v___x_3865_; 
if (v_isShared_3863_ == 0)
{
v___x_3865_ = v___x_3862_;
goto v_reusejp_3864_;
}
else
{
lean_object* v_reuseFailAlloc_3866_; 
v_reuseFailAlloc_3866_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3866_, 0, v_a_3860_);
v___x_3865_ = v_reuseFailAlloc_3866_;
goto v_reusejp_3864_;
}
v_reusejp_3864_:
{
return v___x_3865_;
}
}
}
}
else
{
lean_object* v_a_3868_; lean_object* v___x_3870_; uint8_t v_isShared_3871_; uint8_t v_isSharedCheck_3875_; 
v_a_3868_ = lean_ctor_get(v___x_3841_, 0);
v_isSharedCheck_3875_ = !lean_is_exclusive(v___x_3841_);
if (v_isSharedCheck_3875_ == 0)
{
v___x_3870_ = v___x_3841_;
v_isShared_3871_ = v_isSharedCheck_3875_;
goto v_resetjp_3869_;
}
else
{
lean_inc(v_a_3868_);
lean_dec(v___x_3841_);
v___x_3870_ = lean_box(0);
v_isShared_3871_ = v_isSharedCheck_3875_;
goto v_resetjp_3869_;
}
v_resetjp_3869_:
{
lean_object* v___x_3873_; 
if (v_isShared_3871_ == 0)
{
v___x_3873_ = v___x_3870_;
goto v_reusejp_3872_;
}
else
{
lean_object* v_reuseFailAlloc_3874_; 
v_reuseFailAlloc_3874_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3874_, 0, v_a_3868_);
v___x_3873_ = v_reuseFailAlloc_3874_;
goto v_reusejp_3872_;
}
v_reusejp_3872_:
{
return v___x_3873_;
}
}
}
}
else
{
lean_object* v_a_3876_; lean_object* v___x_3878_; uint8_t v_isShared_3879_; uint8_t v_isSharedCheck_3883_; 
lean_dec_ref(v_immediate_3827_);
lean_dec(v_patternSubsts_x3f_3826_);
lean_dec_ref(v_options_3825_);
lean_dec(v_goal_3823_);
v_a_3876_ = lean_ctor_get(v___x_3835_, 0);
v_isSharedCheck_3883_ = !lean_is_exclusive(v___x_3835_);
if (v_isSharedCheck_3883_ == 0)
{
v___x_3878_ = v___x_3835_;
v_isShared_3879_ = v_isSharedCheck_3883_;
goto v_resetjp_3877_;
}
else
{
lean_inc(v_a_3876_);
lean_dec(v___x_3835_);
v___x_3878_ = lean_box(0);
v_isShared_3879_ = v_isSharedCheck_3883_;
goto v_resetjp_3877_;
}
v_resetjp_3877_:
{
lean_object* v___x_3881_; 
if (v_isShared_3879_ == 0)
{
v___x_3881_ = v___x_3878_;
goto v_reusejp_3880_;
}
else
{
lean_object* v_reuseFailAlloc_3882_; 
v_reuseFailAlloc_3882_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3882_, 0, v_a_3876_);
v___x_3881_ = v_reuseFailAlloc_3882_;
goto v_reusejp_3880_;
}
v_reusejp_3880_:
{
return v___x_3881_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_forwardTerm___lam__1___boxed(lean_object* v_goal_3884_, lean_object* v_stx_3885_, lean_object* v_options_3886_, lean_object* v_patternSubsts_x3f_3887_, lean_object* v_immediate_3888_, lean_object* v_clear_3889_, lean_object* v___y_3890_, lean_object* v___y_3891_, lean_object* v___y_3892_, lean_object* v___y_3893_, lean_object* v___y_3894_, lean_object* v___y_3895_){
_start:
{
uint8_t v_clear_boxed_3896_; lean_object* v_res_3897_; 
v_clear_boxed_3896_ = lean_unbox(v_clear_3889_);
v_res_3897_ = lp_aesop_Aesop_RuleTac_forwardTerm___lam__1(v_goal_3884_, v_stx_3885_, v_options_3886_, v_patternSubsts_x3f_3887_, v_immediate_3888_, v_clear_boxed_3896_, v___y_3890_, v___y_3891_, v___y_3892_, v___y_3893_, v___y_3894_);
lean_dec(v___y_3894_);
lean_dec_ref(v___y_3893_);
lean_dec(v___y_3892_);
lean_dec_ref(v___y_3891_);
lean_dec(v___y_3890_);
return v_res_3897_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_forwardTerm(lean_object* v_stx_3898_, lean_object* v_immediate_3899_, uint8_t v_clear_3900_, lean_object* v_input_3901_, lean_object* v_a_3902_, lean_object* v_a_3903_, lean_object* v_a_3904_, lean_object* v_a_3905_, lean_object* v_a_3906_){
_start:
{
lean_object* v_goal_3908_; lean_object* v_patternSubsts_x3f_3909_; lean_object* v_options_3910_; lean_object* v___x_3911_; lean_object* v___f_3912_; lean_object* v___x_3913_; 
v_goal_3908_ = lean_ctor_get(v_input_3901_, 0);
lean_inc_n(v_goal_3908_, 2);
v_patternSubsts_x3f_3909_ = lean_ctor_get(v_input_3901_, 3);
lean_inc(v_patternSubsts_x3f_3909_);
v_options_3910_ = lean_ctor_get(v_input_3901_, 4);
lean_inc_ref(v_options_3910_);
lean_dec_ref(v_input_3901_);
v___x_3911_ = lean_box(v_clear_3900_);
v___f_3912_ = lean_alloc_closure((void*)(lp_aesop_Aesop_RuleTac_forwardTerm___lam__1___boxed), 12, 6);
lean_closure_set(v___f_3912_, 0, v_goal_3908_);
lean_closure_set(v___f_3912_, 1, v_stx_3898_);
lean_closure_set(v___f_3912_, 2, v_options_3910_);
lean_closure_set(v___f_3912_, 3, v_patternSubsts_x3f_3909_);
lean_closure_set(v___f_3912_, 4, v_immediate_3899_);
lean_closure_set(v___f_3912_, 5, v___x_3911_);
v___x_3913_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_RuleTac_forwardConst_spec__1___redArg(v_goal_3908_, v___f_3912_, v_a_3902_, v_a_3903_, v_a_3904_, v_a_3905_, v_a_3906_);
return v___x_3913_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_forwardTerm___boxed(lean_object* v_stx_3914_, lean_object* v_immediate_3915_, lean_object* v_clear_3916_, lean_object* v_input_3917_, lean_object* v_a_3918_, lean_object* v_a_3919_, lean_object* v_a_3920_, lean_object* v_a_3921_, lean_object* v_a_3922_, lean_object* v_a_3923_){
_start:
{
uint8_t v_clear_boxed_3924_; lean_object* v_res_3925_; 
v_clear_boxed_3924_ = lean_unbox(v_clear_3916_);
v_res_3925_ = lp_aesop_Aesop_RuleTac_forwardTerm(v_stx_3914_, v_immediate_3915_, v_clear_boxed_3924_, v_input_3917_, v_a_3918_, v_a_3919_, v_a_3920_, v_a_3921_, v_a_3922_);
lean_dec(v_a_3922_);
lean_dec_ref(v_a_3921_);
lean_dec(v_a_3920_);
lean_dec_ref(v_a_3919_);
lean_dec(v_a_3918_);
return v_res_3925_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_forward(lean_object* v_t_3926_, lean_object* v_immediate_3927_, uint8_t v_clear_3928_, lean_object* v_a_3929_, lean_object* v_a_3930_, lean_object* v_a_3931_, lean_object* v_a_3932_, lean_object* v_a_3933_, lean_object* v_a_3934_){
_start:
{
if (lean_obj_tag(v_t_3926_) == 0)
{
lean_object* v_decl_3936_; lean_object* v___x_3937_; 
v_decl_3936_ = lean_ctor_get(v_t_3926_, 0);
lean_inc(v_decl_3936_);
lean_dec_ref_known(v_t_3926_, 1);
v___x_3937_ = lp_aesop_Aesop_RuleTac_forwardConst(v_decl_3936_, v_immediate_3927_, v_clear_3928_, v_a_3929_, v_a_3930_, v_a_3931_, v_a_3932_, v_a_3933_, v_a_3934_);
return v___x_3937_;
}
else
{
lean_object* v_term_3938_; lean_object* v___x_3939_; 
v_term_3938_ = lean_ctor_get(v_t_3926_, 0);
lean_inc(v_term_3938_);
lean_dec_ref_known(v_t_3926_, 1);
v___x_3939_ = lp_aesop_Aesop_RuleTac_forwardTerm(v_term_3938_, v_immediate_3927_, v_clear_3928_, v_a_3929_, v_a_3930_, v_a_3931_, v_a_3932_, v_a_3933_, v_a_3934_);
return v___x_3939_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_forward___boxed(lean_object* v_t_3940_, lean_object* v_immediate_3941_, lean_object* v_clear_3942_, lean_object* v_a_3943_, lean_object* v_a_3944_, lean_object* v_a_3945_, lean_object* v_a_3946_, lean_object* v_a_3947_, lean_object* v_a_3948_, lean_object* v_a_3949_){
_start:
{
uint8_t v_clear_boxed_3950_; lean_object* v_res_3951_; 
v_clear_boxed_3950_ = lean_unbox(v_clear_3942_);
v_res_3951_ = lp_aesop_Aesop_RuleTac_forward(v_t_3940_, v_immediate_3941_, v_clear_boxed_3950_, v_a_3943_, v_a_3944_, v_a_3945_, v_a_3946_, v_a_3947_, v_a_3948_);
lean_dec(v_a_3948_);
lean_dec_ref(v_a_3947_);
lean_dec(v_a_3946_);
lean_dec_ref(v_a_3945_);
lean_dec(v_a_3944_);
return v_res_3951_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_forwardMatches___lam__0(lean_object* v_goal_3952_, lean_object* v_fst_3953_, lean_object* v_fst_3954_, lean_object* v_fst_3955_, uint8_t v_anySuccess_3956_, lean_object* v_snd_3957_, lean_object* v_____r_3958_, lean_object* v___y_3959_, lean_object* v___y_3960_, lean_object* v___y_3961_, lean_object* v___y_3962_, lean_object* v___y_3963_){
_start:
{
lean_object* v___x_3965_; lean_object* v___x_3966_; lean_object* v___x_3967_; lean_object* v___x_3968_; lean_object* v___x_3969_; lean_object* v___x_3970_; lean_object* v___x_3971_; lean_object* v___x_3972_; lean_object* v___x_3973_; 
v___x_3965_ = lean_alloc_ctor(0, 4, 1);
lean_ctor_set(v___x_3965_, 0, v_goal_3952_);
lean_ctor_set(v___x_3965_, 1, v_fst_3953_);
lean_ctor_set(v___x_3965_, 2, v_fst_3954_);
lean_ctor_set(v___x_3965_, 3, v_fst_3955_);
lean_ctor_set_uint8(v___x_3965_, sizeof(void*)*4, v_anySuccess_3956_);
v___x_3966_ = lean_unsigned_to_nat(1u);
v___x_3967_ = lean_mk_empty_array_with_capacity(v___x_3966_);
v___x_3968_ = lean_array_push(v___x_3967_, v___x_3965_);
v___x_3969_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3969_, 0, v_snd_3957_);
v___x_3970_ = lean_box(0);
v___x_3971_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3971_, 0, v___x_3969_);
lean_ctor_set(v___x_3971_, 1, v___x_3970_);
v___x_3972_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3972_, 0, v___x_3968_);
lean_ctor_set(v___x_3972_, 1, v___x_3971_);
v___x_3973_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3973_, 0, v___x_3972_);
return v___x_3973_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_forwardMatches___lam__0___boxed(lean_object* v_goal_3974_, lean_object* v_fst_3975_, lean_object* v_fst_3976_, lean_object* v_fst_3977_, lean_object* v_anySuccess_3978_, lean_object* v_snd_3979_, lean_object* v_____r_3980_, lean_object* v___y_3981_, lean_object* v___y_3982_, lean_object* v___y_3983_, lean_object* v___y_3984_, lean_object* v___y_3985_, lean_object* v___y_3986_){
_start:
{
uint8_t v_anySuccess_boxed_3987_; lean_object* v_res_3988_; 
v_anySuccess_boxed_3987_ = lean_unbox(v_anySuccess_3978_);
v_res_3988_ = lp_aesop_Aesop_RuleTac_forwardMatches___lam__0(v_goal_3974_, v_fst_3975_, v_fst_3976_, v_fst_3977_, v_anySuccess_boxed_3987_, v_snd_3979_, v_____r_3980_, v___y_3981_, v___y_3982_, v___y_3983_, v___y_3984_, v___y_3985_);
lean_dec(v___y_3985_);
lean_dec_ref(v___y_3984_);
lean_dec(v___y_3983_);
lean_dec_ref(v___y_3982_);
lean_dec(v___y_3981_);
return v_res_3988_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_RuleTac_forwardMatches_spec__2___redArg(lean_object* v_msg_3989_, lean_object* v___y_3990_, lean_object* v___y_3991_, lean_object* v___y_3992_, lean_object* v___y_3993_){
_start:
{
lean_object* v_ref_3995_; lean_object* v___x_3996_; lean_object* v_a_3997_; lean_object* v___x_3999_; uint8_t v_isShared_4000_; uint8_t v_isSharedCheck_4005_; 
v_ref_3995_ = lean_ctor_get(v___y_3992_, 5);
v___x_3996_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_addTrace___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__3_spec__6(v_msg_3989_, v___y_3990_, v___y_3991_, v___y_3992_, v___y_3993_);
v_a_3997_ = lean_ctor_get(v___x_3996_, 0);
v_isSharedCheck_4005_ = !lean_is_exclusive(v___x_3996_);
if (v_isSharedCheck_4005_ == 0)
{
v___x_3999_ = v___x_3996_;
v_isShared_4000_ = v_isSharedCheck_4005_;
goto v_resetjp_3998_;
}
else
{
lean_inc(v_a_3997_);
lean_dec(v___x_3996_);
v___x_3999_ = lean_box(0);
v_isShared_4000_ = v_isSharedCheck_4005_;
goto v_resetjp_3998_;
}
v_resetjp_3998_:
{
lean_object* v___x_4001_; lean_object* v___x_4003_; 
lean_inc(v_ref_3995_);
v___x_4001_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4001_, 0, v_ref_3995_);
lean_ctor_set(v___x_4001_, 1, v_a_3997_);
if (v_isShared_4000_ == 0)
{
lean_ctor_set_tag(v___x_3999_, 1);
lean_ctor_set(v___x_3999_, 0, v___x_4001_);
v___x_4003_ = v___x_3999_;
goto v_reusejp_4002_;
}
else
{
lean_object* v_reuseFailAlloc_4004_; 
v_reuseFailAlloc_4004_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4004_, 0, v___x_4001_);
v___x_4003_ = v_reuseFailAlloc_4004_;
goto v_reusejp_4002_;
}
v_reusejp_4002_:
{
return v___x_4003_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_RuleTac_forwardMatches_spec__2___redArg___boxed(lean_object* v_msg_4006_, lean_object* v___y_4007_, lean_object* v___y_4008_, lean_object* v___y_4009_, lean_object* v___y_4010_, lean_object* v___y_4011_){
_start:
{
lean_object* v_res_4012_; 
v_res_4012_ = lp_aesop_Lean_throwError___at___00Aesop_RuleTac_forwardMatches_spec__2___redArg(v_msg_4006_, v___y_4007_, v___y_4008_, v___y_4009_, v___y_4010_);
lean_dec(v___y_4010_);
lean_dec_ref(v___y_4009_);
lean_dec(v___y_4008_);
lean_dec_ref(v___y_4007_);
return v_res_4012_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_forwardMatches_spec__0(lean_object* v_as_4013_, size_t v_sz_4014_, size_t v_i_4015_, lean_object* v_b_4016_, lean_object* v___y_4017_, lean_object* v___y_4018_, lean_object* v___y_4019_, lean_object* v___y_4020_, lean_object* v___y_4021_){
_start:
{
lean_object* v_a_4024_; uint8_t v___x_4028_; 
v___x_4028_ = lean_usize_dec_lt(v_i_4015_, v_sz_4014_);
if (v___x_4028_ == 0)
{
lean_object* v___x_4029_; 
v___x_4029_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4029_, 0, v_b_4016_);
return v___x_4029_;
}
else
{
lean_object* v_fst_4030_; lean_object* v_snd_4031_; lean_object* v_a_4032_; lean_object* v___x_4033_; lean_object* v___x_4034_; lean_object* v___x_4035_; 
v_fst_4030_ = lean_ctor_get(v_b_4016_, 0);
lean_inc_n(v_fst_4030_, 2);
v_snd_4031_ = lean_ctor_get(v_b_4016_, 1);
lean_inc(v_snd_4031_);
lean_dec_ref(v_b_4016_);
v_a_4032_ = lean_array_uget_borrowed(v_as_4013_, v_i_4015_);
v___x_4033_ = lean_box(v___x_4028_);
lean_inc(v_a_4032_);
v___x_4034_ = lean_alloc_closure((void*)(lp_aesop_Aesop_ForwardRuleMatch_apply___boxed), 10, 3);
lean_closure_set(v___x_4034_, 0, v_fst_4030_);
lean_closure_set(v___x_4034_, 1, v_a_4032_);
lean_closure_set(v___x_4034_, 2, v___x_4033_);
v___x_4035_ = lp_aesop_Aesop_ScriptT_run___at___00Aesop_RuleTac_forwardConst_spec__0___redArg(v___x_4034_, v___y_4017_, v___y_4018_, v___y_4019_, v___y_4020_, v___y_4021_);
if (lean_obj_tag(v___x_4035_) == 0)
{
lean_object* v_a_4036_; lean_object* v_snd_4037_; lean_object* v_snd_4038_; lean_object* v_fst_4039_; 
v_a_4036_ = lean_ctor_get(v___x_4035_, 0);
lean_inc(v_a_4036_);
lean_dec_ref_known(v___x_4035_, 1);
v_snd_4037_ = lean_ctor_get(v_snd_4031_, 1);
lean_inc(v_snd_4037_);
v_snd_4038_ = lean_ctor_get(v_snd_4037_, 1);
lean_inc(v_snd_4038_);
v_fst_4039_ = lean_ctor_get(v_a_4036_, 0);
if (lean_obj_tag(v_fst_4039_) == 1)
{
lean_object* v_val_4040_; lean_object* v_snd_4041_; lean_object* v_snd_4042_; lean_object* v_fst_4043_; lean_object* v_fst_4044_; lean_object* v___x_4046_; uint8_t v_isShared_4047_; uint8_t v_isSharedCheck_4083_; 
lean_dec(v_fst_4030_);
v_val_4040_ = lean_ctor_get(v_fst_4039_, 0);
lean_inc(v_val_4040_);
v_snd_4041_ = lean_ctor_get(v_val_4040_, 1);
lean_inc(v_snd_4041_);
v_snd_4042_ = lean_ctor_get(v_a_4036_, 1);
lean_inc(v_snd_4042_);
lean_dec(v_a_4036_);
v_fst_4043_ = lean_ctor_get(v_snd_4031_, 0);
lean_inc(v_fst_4043_);
lean_dec(v_snd_4031_);
v_fst_4044_ = lean_ctor_get(v_snd_4037_, 0);
v_isSharedCheck_4083_ = !lean_is_exclusive(v_snd_4037_);
if (v_isSharedCheck_4083_ == 0)
{
lean_object* v_unused_4084_; 
v_unused_4084_ = lean_ctor_get(v_snd_4037_, 1);
lean_dec(v_unused_4084_);
v___x_4046_ = v_snd_4037_;
v_isShared_4047_ = v_isSharedCheck_4083_;
goto v_resetjp_4045_;
}
else
{
lean_inc(v_fst_4044_);
lean_dec(v_snd_4037_);
v___x_4046_ = lean_box(0);
v_isShared_4047_ = v_isSharedCheck_4083_;
goto v_resetjp_4045_;
}
v_resetjp_4045_:
{
lean_object* v_snd_4048_; lean_object* v___x_4050_; uint8_t v_isShared_4051_; uint8_t v_isSharedCheck_4081_; 
v_snd_4048_ = lean_ctor_get(v_snd_4038_, 1);
v_isSharedCheck_4081_ = !lean_is_exclusive(v_snd_4038_);
if (v_isSharedCheck_4081_ == 0)
{
lean_object* v_unused_4082_; 
v_unused_4082_ = lean_ctor_get(v_snd_4038_, 0);
lean_dec(v_unused_4082_);
v___x_4050_ = v_snd_4038_;
v_isShared_4051_ = v_isSharedCheck_4081_;
goto v_resetjp_4049_;
}
else
{
lean_inc(v_snd_4048_);
lean_dec(v_snd_4038_);
v___x_4050_ = lean_box(0);
v_isShared_4051_ = v_isSharedCheck_4081_;
goto v_resetjp_4049_;
}
v_resetjp_4049_:
{
lean_object* v_fst_4052_; lean_object* v___x_4054_; uint8_t v_isShared_4055_; uint8_t v_isSharedCheck_4079_; 
v_fst_4052_ = lean_ctor_get(v_val_4040_, 0);
v_isSharedCheck_4079_ = !lean_is_exclusive(v_val_4040_);
if (v_isSharedCheck_4079_ == 0)
{
lean_object* v_unused_4080_; 
v_unused_4080_ = lean_ctor_get(v_val_4040_, 1);
lean_dec(v_unused_4080_);
v___x_4054_ = v_val_4040_;
v_isShared_4055_ = v_isSharedCheck_4079_;
goto v_resetjp_4053_;
}
else
{
lean_inc(v_fst_4052_);
lean_dec(v_val_4040_);
v___x_4054_ = lean_box(0);
v_isShared_4055_ = v_isSharedCheck_4079_;
goto v_resetjp_4053_;
}
v_resetjp_4053_:
{
lean_object* v_fst_4056_; lean_object* v_snd_4057_; lean_object* v___x_4059_; uint8_t v_isShared_4060_; uint8_t v_isSharedCheck_4078_; 
v_fst_4056_ = lean_ctor_get(v_snd_4041_, 0);
v_snd_4057_ = lean_ctor_get(v_snd_4041_, 1);
v_isSharedCheck_4078_ = !lean_is_exclusive(v_snd_4041_);
if (v_isSharedCheck_4078_ == 0)
{
v___x_4059_ = v_snd_4041_;
v_isShared_4060_ = v_isSharedCheck_4078_;
goto v_resetjp_4058_;
}
else
{
lean_inc(v_snd_4057_);
lean_inc(v_fst_4056_);
lean_dec(v_snd_4041_);
v___x_4059_ = lean_box(0);
v_isShared_4060_ = v_isSharedCheck_4078_;
goto v_resetjp_4058_;
}
v_resetjp_4058_:
{
lean_object* v___x_4061_; lean_object* v___x_4062_; lean_object* v___x_4063_; lean_object* v___x_4064_; lean_object* v___x_4065_; lean_object* v___x_4067_; 
v___x_4061_ = lean_box(0);
v___x_4062_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__0_spec__0___redArg(v_fst_4043_, v_fst_4056_, v___x_4061_);
v___x_4063_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_insertManyIfNewUnit___at___00__private_Aesop_RuleTac_Forward_0__Aesop_RuleTac_makeForwardHypProofs_loop_spec__0(v_fst_4044_, v_snd_4057_);
lean_dec(v_snd_4057_);
v___x_4064_ = l_Array_append___redArg(v_snd_4048_, v_snd_4042_);
lean_dec(v_snd_4042_);
v___x_4065_ = lean_box(v___x_4028_);
if (v_isShared_4060_ == 0)
{
lean_ctor_set(v___x_4059_, 1, v___x_4064_);
lean_ctor_set(v___x_4059_, 0, v___x_4065_);
v___x_4067_ = v___x_4059_;
goto v_reusejp_4066_;
}
else
{
lean_object* v_reuseFailAlloc_4077_; 
v_reuseFailAlloc_4077_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4077_, 0, v___x_4065_);
lean_ctor_set(v_reuseFailAlloc_4077_, 1, v___x_4064_);
v___x_4067_ = v_reuseFailAlloc_4077_;
goto v_reusejp_4066_;
}
v_reusejp_4066_:
{
lean_object* v___x_4069_; 
if (v_isShared_4055_ == 0)
{
lean_ctor_set(v___x_4054_, 1, v___x_4067_);
lean_ctor_set(v___x_4054_, 0, v___x_4063_);
v___x_4069_ = v___x_4054_;
goto v_reusejp_4068_;
}
else
{
lean_object* v_reuseFailAlloc_4076_; 
v_reuseFailAlloc_4076_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4076_, 0, v___x_4063_);
lean_ctor_set(v_reuseFailAlloc_4076_, 1, v___x_4067_);
v___x_4069_ = v_reuseFailAlloc_4076_;
goto v_reusejp_4068_;
}
v_reusejp_4068_:
{
lean_object* v___x_4071_; 
if (v_isShared_4051_ == 0)
{
lean_ctor_set(v___x_4050_, 1, v___x_4069_);
lean_ctor_set(v___x_4050_, 0, v___x_4062_);
v___x_4071_ = v___x_4050_;
goto v_reusejp_4070_;
}
else
{
lean_object* v_reuseFailAlloc_4075_; 
v_reuseFailAlloc_4075_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4075_, 0, v___x_4062_);
lean_ctor_set(v_reuseFailAlloc_4075_, 1, v___x_4069_);
v___x_4071_ = v_reuseFailAlloc_4075_;
goto v_reusejp_4070_;
}
v_reusejp_4070_:
{
lean_object* v___x_4073_; 
if (v_isShared_4047_ == 0)
{
lean_ctor_set(v___x_4046_, 1, v___x_4071_);
lean_ctor_set(v___x_4046_, 0, v_fst_4052_);
v___x_4073_ = v___x_4046_;
goto v_reusejp_4072_;
}
else
{
lean_object* v_reuseFailAlloc_4074_; 
v_reuseFailAlloc_4074_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4074_, 0, v_fst_4052_);
lean_ctor_set(v_reuseFailAlloc_4074_, 1, v___x_4071_);
v___x_4073_ = v_reuseFailAlloc_4074_;
goto v_reusejp_4072_;
}
v_reusejp_4072_:
{
v_a_4024_ = v___x_4073_;
goto v___jp_4023_;
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
lean_object* v___x_4086_; uint8_t v_isShared_4087_; uint8_t v_isSharedCheck_4118_; 
v_isSharedCheck_4118_ = !lean_is_exclusive(v_a_4036_);
if (v_isSharedCheck_4118_ == 0)
{
lean_object* v_unused_4119_; lean_object* v_unused_4120_; 
v_unused_4119_ = lean_ctor_get(v_a_4036_, 1);
lean_dec(v_unused_4119_);
v_unused_4120_ = lean_ctor_get(v_a_4036_, 0);
lean_dec(v_unused_4120_);
v___x_4086_ = v_a_4036_;
v_isShared_4087_ = v_isSharedCheck_4118_;
goto v_resetjp_4085_;
}
else
{
lean_dec(v_a_4036_);
v___x_4086_ = lean_box(0);
v_isShared_4087_ = v_isSharedCheck_4118_;
goto v_resetjp_4085_;
}
v_resetjp_4085_:
{
lean_object* v_fst_4088_; lean_object* v___x_4090_; uint8_t v_isShared_4091_; uint8_t v_isSharedCheck_4116_; 
v_fst_4088_ = lean_ctor_get(v_snd_4031_, 0);
v_isSharedCheck_4116_ = !lean_is_exclusive(v_snd_4031_);
if (v_isSharedCheck_4116_ == 0)
{
lean_object* v_unused_4117_; 
v_unused_4117_ = lean_ctor_get(v_snd_4031_, 1);
lean_dec(v_unused_4117_);
v___x_4090_ = v_snd_4031_;
v_isShared_4091_ = v_isSharedCheck_4116_;
goto v_resetjp_4089_;
}
else
{
lean_inc(v_fst_4088_);
lean_dec(v_snd_4031_);
v___x_4090_ = lean_box(0);
v_isShared_4091_ = v_isSharedCheck_4116_;
goto v_resetjp_4089_;
}
v_resetjp_4089_:
{
lean_object* v_fst_4092_; lean_object* v___x_4094_; uint8_t v_isShared_4095_; uint8_t v_isSharedCheck_4114_; 
v_fst_4092_ = lean_ctor_get(v_snd_4037_, 0);
v_isSharedCheck_4114_ = !lean_is_exclusive(v_snd_4037_);
if (v_isSharedCheck_4114_ == 0)
{
lean_object* v_unused_4115_; 
v_unused_4115_ = lean_ctor_get(v_snd_4037_, 1);
lean_dec(v_unused_4115_);
v___x_4094_ = v_snd_4037_;
v_isShared_4095_ = v_isSharedCheck_4114_;
goto v_resetjp_4093_;
}
else
{
lean_inc(v_fst_4092_);
lean_dec(v_snd_4037_);
v___x_4094_ = lean_box(0);
v_isShared_4095_ = v_isSharedCheck_4114_;
goto v_resetjp_4093_;
}
v_resetjp_4093_:
{
lean_object* v_fst_4096_; lean_object* v_snd_4097_; lean_object* v___x_4099_; uint8_t v_isShared_4100_; uint8_t v_isSharedCheck_4113_; 
v_fst_4096_ = lean_ctor_get(v_snd_4038_, 0);
v_snd_4097_ = lean_ctor_get(v_snd_4038_, 1);
v_isSharedCheck_4113_ = !lean_is_exclusive(v_snd_4038_);
if (v_isSharedCheck_4113_ == 0)
{
v___x_4099_ = v_snd_4038_;
v_isShared_4100_ = v_isSharedCheck_4113_;
goto v_resetjp_4098_;
}
else
{
lean_inc(v_snd_4097_);
lean_inc(v_fst_4096_);
lean_dec(v_snd_4038_);
v___x_4099_ = lean_box(0);
v_isShared_4100_ = v_isSharedCheck_4113_;
goto v_resetjp_4098_;
}
v_resetjp_4098_:
{
lean_object* v___x_4102_; 
if (v_isShared_4100_ == 0)
{
v___x_4102_ = v___x_4099_;
goto v_reusejp_4101_;
}
else
{
lean_object* v_reuseFailAlloc_4112_; 
v_reuseFailAlloc_4112_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4112_, 0, v_fst_4096_);
lean_ctor_set(v_reuseFailAlloc_4112_, 1, v_snd_4097_);
v___x_4102_ = v_reuseFailAlloc_4112_;
goto v_reusejp_4101_;
}
v_reusejp_4101_:
{
lean_object* v___x_4104_; 
if (v_isShared_4095_ == 0)
{
lean_ctor_set(v___x_4094_, 1, v___x_4102_);
v___x_4104_ = v___x_4094_;
goto v_reusejp_4103_;
}
else
{
lean_object* v_reuseFailAlloc_4111_; 
v_reuseFailAlloc_4111_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4111_, 0, v_fst_4092_);
lean_ctor_set(v_reuseFailAlloc_4111_, 1, v___x_4102_);
v___x_4104_ = v_reuseFailAlloc_4111_;
goto v_reusejp_4103_;
}
v_reusejp_4103_:
{
lean_object* v___x_4106_; 
if (v_isShared_4091_ == 0)
{
lean_ctor_set(v___x_4090_, 1, v___x_4104_);
v___x_4106_ = v___x_4090_;
goto v_reusejp_4105_;
}
else
{
lean_object* v_reuseFailAlloc_4110_; 
v_reuseFailAlloc_4110_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4110_, 0, v_fst_4088_);
lean_ctor_set(v_reuseFailAlloc_4110_, 1, v___x_4104_);
v___x_4106_ = v_reuseFailAlloc_4110_;
goto v_reusejp_4105_;
}
v_reusejp_4105_:
{
lean_object* v___x_4108_; 
if (v_isShared_4087_ == 0)
{
lean_ctor_set(v___x_4086_, 1, v___x_4106_);
lean_ctor_set(v___x_4086_, 0, v_fst_4030_);
v___x_4108_ = v___x_4086_;
goto v_reusejp_4107_;
}
else
{
lean_object* v_reuseFailAlloc_4109_; 
v_reuseFailAlloc_4109_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4109_, 0, v_fst_4030_);
lean_ctor_set(v_reuseFailAlloc_4109_, 1, v___x_4106_);
v___x_4108_ = v_reuseFailAlloc_4109_;
goto v_reusejp_4107_;
}
v_reusejp_4107_:
{
v_a_4024_ = v___x_4108_;
goto v___jp_4023_;
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
lean_object* v_a_4121_; lean_object* v___x_4123_; uint8_t v_isShared_4124_; uint8_t v_isSharedCheck_4128_; 
lean_dec(v_snd_4031_);
lean_dec(v_fst_4030_);
v_a_4121_ = lean_ctor_get(v___x_4035_, 0);
v_isSharedCheck_4128_ = !lean_is_exclusive(v___x_4035_);
if (v_isSharedCheck_4128_ == 0)
{
v___x_4123_ = v___x_4035_;
v_isShared_4124_ = v_isSharedCheck_4128_;
goto v_resetjp_4122_;
}
else
{
lean_inc(v_a_4121_);
lean_dec(v___x_4035_);
v___x_4123_ = lean_box(0);
v_isShared_4124_ = v_isSharedCheck_4128_;
goto v_resetjp_4122_;
}
v_resetjp_4122_:
{
lean_object* v___x_4126_; 
if (v_isShared_4124_ == 0)
{
v___x_4126_ = v___x_4123_;
goto v_reusejp_4125_;
}
else
{
lean_object* v_reuseFailAlloc_4127_; 
v_reuseFailAlloc_4127_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4127_, 0, v_a_4121_);
v___x_4126_ = v_reuseFailAlloc_4127_;
goto v_reusejp_4125_;
}
v_reusejp_4125_:
{
return v___x_4126_;
}
}
}
}
v___jp_4023_:
{
size_t v___x_4025_; size_t v___x_4026_; 
v___x_4025_ = ((size_t)1ULL);
v___x_4026_ = lean_usize_add(v_i_4015_, v___x_4025_);
v_i_4015_ = v___x_4026_;
v_b_4016_ = v_a_4024_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_forwardMatches_spec__0___boxed(lean_object* v_as_4129_, lean_object* v_sz_4130_, lean_object* v_i_4131_, lean_object* v_b_4132_, lean_object* v___y_4133_, lean_object* v___y_4134_, lean_object* v___y_4135_, lean_object* v___y_4136_, lean_object* v___y_4137_, lean_object* v___y_4138_){
_start:
{
size_t v_sz_boxed_4139_; size_t v_i_boxed_4140_; lean_object* v_res_4141_; 
v_sz_boxed_4139_ = lean_unbox_usize(v_sz_4130_);
lean_dec(v_sz_4130_);
v_i_boxed_4140_ = lean_unbox_usize(v_i_4131_);
lean_dec(v_i_4131_);
v_res_4141_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_forwardMatches_spec__0(v_as_4129_, v_sz_boxed_4139_, v_i_boxed_4140_, v_b_4132_, v___y_4133_, v___y_4134_, v___y_4135_, v___y_4136_, v___y_4137_);
lean_dec(v___y_4137_);
lean_dec_ref(v___y_4136_);
lean_dec(v___y_4135_);
lean_dec_ref(v___y_4134_);
lean_dec(v___y_4133_);
lean_dec_ref(v_as_4129_);
return v_res_4141_;
}
}
static lean_object* _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_forwardMatches_spec__1___closed__1(void){
_start:
{
lean_object* v___x_4143_; lean_object* v___x_4144_; 
v___x_4143_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_forwardMatches_spec__1___closed__0));
v___x_4144_ = l_Lean_stringToMessageData(v___x_4143_);
return v___x_4144_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_forwardMatches_spec__1(size_t v_sz_4159_, size_t v_i_4160_, lean_object* v_bs_4161_){
_start:
{
uint8_t v___x_4162_; 
v___x_4162_ = lean_usize_dec_lt(v_i_4160_, v_sz_4159_);
if (v___x_4162_ == 0)
{
return v_bs_4161_;
}
else
{
lean_object* v_v_4163_; lean_object* v_rule_4164_; lean_object* v_name_4165_; lean_object* v_match_4166_; lean_object* v___x_4168_; uint8_t v_isShared_4169_; uint8_t v_isSharedCheck_4219_; 
v_v_4163_ = lean_array_uget(v_bs_4161_, v_i_4160_);
v_rule_4164_ = lean_ctor_get(v_v_4163_, 0);
lean_inc_ref(v_rule_4164_);
v_name_4165_ = lean_ctor_get(v_rule_4164_, 1);
v_match_4166_ = lean_ctor_get(v_v_4163_, 1);
v_isSharedCheck_4219_ = !lean_is_exclusive(v_v_4163_);
if (v_isSharedCheck_4219_ == 0)
{
lean_object* v_unused_4220_; 
v_unused_4220_ = lean_ctor_get(v_v_4163_, 0);
lean_dec(v_unused_4220_);
v___x_4168_ = v_v_4163_;
v_isShared_4169_ = v_isSharedCheck_4219_;
goto v_resetjp_4167_;
}
else
{
lean_inc(v_match_4166_);
lean_dec(v_v_4163_);
v___x_4168_ = lean_box(0);
v_isShared_4169_ = v_isSharedCheck_4219_;
goto v_resetjp_4167_;
}
v_resetjp_4167_:
{
lean_object* v_name_4170_; uint8_t v_builder_4171_; uint8_t v_phase_4172_; uint8_t v_scope_4173_; lean_object* v___x_4174_; lean_object* v_bs_x27_4175_; lean_object* v___y_4177_; lean_object* v___y_4178_; lean_object* v___y_4179_; lean_object* v___y_4197_; lean_object* v___y_4198_; lean_object* v___y_4199_; lean_object* v___y_4205_; 
v_name_4170_ = lean_ctor_get(v_name_4165_, 0);
v_builder_4171_ = lean_ctor_get_uint8(v_name_4165_, sizeof(void*)*1 + 8);
v_phase_4172_ = lean_ctor_get_uint8(v_name_4165_, sizeof(void*)*1 + 9);
v_scope_4173_ = lean_ctor_get_uint8(v_name_4165_, sizeof(void*)*1 + 10);
v___x_4174_ = lean_unsigned_to_nat(0u);
v_bs_x27_4175_ = lean_array_uset(v_bs_4161_, v_i_4160_, v___x_4174_);
switch(v_phase_4172_)
{
case 0:
{
lean_object* v___x_4216_; 
v___x_4216_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_forwardMatches_spec__1___closed__13));
v___y_4205_ = v___x_4216_;
goto v___jp_4204_;
}
case 1:
{
lean_object* v___x_4217_; 
v___x_4217_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_forwardMatches_spec__1___closed__14));
v___y_4205_ = v___x_4217_;
goto v___jp_4204_;
}
default: 
{
lean_object* v___x_4218_; 
v___x_4218_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_forwardMatches_spec__1___closed__15));
v___y_4205_ = v___x_4218_;
goto v___jp_4204_;
}
}
v___jp_4176_:
{
lean_object* v___x_4180_; lean_object* v___x_4181_; lean_object* v___x_4182_; lean_object* v___x_4183_; lean_object* v___x_4184_; lean_object* v___x_4185_; lean_object* v___x_4186_; lean_object* v___x_4188_; 
v___x_4180_ = lean_string_append(v___y_4178_, v___y_4179_);
v___x_4181_ = lean_string_append(v___x_4180_, v___y_4177_);
lean_inc(v_name_4170_);
v___x_4182_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_name_4170_, v___x_4162_);
v___x_4183_ = lean_string_append(v___x_4181_, v___x_4182_);
lean_dec_ref(v___x_4182_);
v___x_4184_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_4184_, 0, v___x_4183_);
v___x_4185_ = l_Lean_MessageData_ofFormat(v___x_4184_);
v___x_4186_ = lean_obj_once(&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_forwardMatches_spec__1___closed__1, &lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_forwardMatches_spec__1___closed__1_once, _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_forwardMatches_spec__1___closed__1);
if (v_isShared_4169_ == 0)
{
lean_ctor_set_tag(v___x_4168_, 7);
lean_ctor_set(v___x_4168_, 1, v___x_4186_);
lean_ctor_set(v___x_4168_, 0, v___x_4185_);
v___x_4188_ = v___x_4168_;
goto v_reusejp_4187_;
}
else
{
lean_object* v_reuseFailAlloc_4195_; 
v_reuseFailAlloc_4195_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4195_, 0, v___x_4185_);
lean_ctor_set(v_reuseFailAlloc_4195_, 1, v___x_4186_);
v___x_4188_ = v_reuseFailAlloc_4195_;
goto v_reusejp_4187_;
}
v_reusejp_4187_:
{
lean_object* v___x_4189_; lean_object* v___x_4190_; size_t v___x_4191_; size_t v___x_4192_; lean_object* v___x_4193_; 
v___x_4189_ = lp_aesop_Aesop_CompleteMatch_toMessageData(v_rule_4164_, v_match_4166_);
lean_dec_ref(v_match_4166_);
v___x_4190_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_4190_, 0, v___x_4188_);
lean_ctor_set(v___x_4190_, 1, v___x_4189_);
v___x_4191_ = ((size_t)1ULL);
v___x_4192_ = lean_usize_add(v_i_4160_, v___x_4191_);
v___x_4193_ = lean_array_uset(v_bs_x27_4175_, v_i_4160_, v___x_4190_);
v_i_4160_ = v___x_4192_;
v_bs_4161_ = v___x_4193_;
goto _start;
}
}
v___jp_4196_:
{
lean_object* v___x_4200_; lean_object* v___x_4201_; 
v___x_4200_ = lean_string_append(v___y_4198_, v___y_4199_);
v___x_4201_ = lean_string_append(v___x_4200_, v___y_4197_);
if (v_scope_4173_ == 0)
{
lean_object* v___x_4202_; 
v___x_4202_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_forwardMatches_spec__1___closed__2));
v___y_4177_ = v___y_4197_;
v___y_4178_ = v___x_4201_;
v___y_4179_ = v___x_4202_;
goto v___jp_4176_;
}
else
{
lean_object* v___x_4203_; 
v___x_4203_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_forwardMatches_spec__1___closed__3));
v___y_4177_ = v___y_4197_;
v___y_4178_ = v___x_4201_;
v___y_4179_ = v___x_4203_;
goto v___jp_4176_;
}
}
v___jp_4204_:
{
lean_object* v___x_4206_; lean_object* v___x_4207_; 
v___x_4206_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_forwardMatches_spec__1___closed__4));
lean_inc_ref(v___y_4205_);
v___x_4207_ = lean_string_append(v___y_4205_, v___x_4206_);
switch(v_builder_4171_)
{
case 0:
{
lean_object* v___x_4208_; 
v___x_4208_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_forwardMatches_spec__1___closed__5));
v___y_4197_ = v___x_4206_;
v___y_4198_ = v___x_4207_;
v___y_4199_ = v___x_4208_;
goto v___jp_4196_;
}
case 1:
{
lean_object* v___x_4209_; 
v___x_4209_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_forwardMatches_spec__1___closed__6));
v___y_4197_ = v___x_4206_;
v___y_4198_ = v___x_4207_;
v___y_4199_ = v___x_4209_;
goto v___jp_4196_;
}
case 2:
{
lean_object* v___x_4210_; 
v___x_4210_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_forwardMatches_spec__1___closed__7));
v___y_4197_ = v___x_4206_;
v___y_4198_ = v___x_4207_;
v___y_4199_ = v___x_4210_;
goto v___jp_4196_;
}
case 3:
{
lean_object* v___x_4211_; 
v___x_4211_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_forwardMatches_spec__1___closed__8));
v___y_4197_ = v___x_4206_;
v___y_4198_ = v___x_4207_;
v___y_4199_ = v___x_4211_;
goto v___jp_4196_;
}
case 4:
{
lean_object* v___x_4212_; 
v___x_4212_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_forwardMatches_spec__1___closed__9));
v___y_4197_ = v___x_4206_;
v___y_4198_ = v___x_4207_;
v___y_4199_ = v___x_4212_;
goto v___jp_4196_;
}
case 5:
{
lean_object* v___x_4213_; 
v___x_4213_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_forwardMatches_spec__1___closed__10));
v___y_4197_ = v___x_4206_;
v___y_4198_ = v___x_4207_;
v___y_4199_ = v___x_4213_;
goto v___jp_4196_;
}
case 6:
{
lean_object* v___x_4214_; 
v___x_4214_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_forwardMatches_spec__1___closed__11));
v___y_4197_ = v___x_4206_;
v___y_4198_ = v___x_4207_;
v___y_4199_ = v___x_4214_;
goto v___jp_4196_;
}
default: 
{
lean_object* v___x_4215_; 
v___x_4215_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_forwardMatches_spec__1___closed__12));
v___y_4197_ = v___x_4206_;
v___y_4198_ = v___x_4207_;
v___y_4199_ = v___x_4215_;
goto v___jp_4196_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_forwardMatches_spec__1___boxed(lean_object* v_sz_4221_, lean_object* v_i_4222_, lean_object* v_bs_4223_){
_start:
{
size_t v_sz_boxed_4224_; size_t v_i_boxed_4225_; lean_object* v_res_4226_; 
v_sz_boxed_4224_ = lean_unbox_usize(v_sz_4221_);
lean_dec(v_sz_4221_);
v_i_boxed_4225_ = lean_unbox_usize(v_i_4222_);
lean_dec(v_i_4222_);
v_res_4226_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_forwardMatches_spec__1(v_sz_boxed_4224_, v_i_boxed_4225_, v_bs_4223_);
return v_res_4226_;
}
}
static lean_object* _init_lp_aesop_Aesop_RuleTac_forwardMatches___closed__1(void){
_start:
{
lean_object* v___x_4231_; lean_object* v_addedFVars_4232_; lean_object* v___x_4233_; 
v___x_4231_ = ((lean_object*)(lp_aesop_Aesop_RuleTac_forwardMatches___closed__0));
v_addedFVars_4232_ = lean_obj_once(&lp_aesop_Aesop_RuleTac_ForwardM_instInhabitedState_default___closed__2, &lp_aesop_Aesop_RuleTac_ForwardM_instInhabitedState_default___closed__2_once, _init_lp_aesop_Aesop_RuleTac_ForwardM_instInhabitedState_default___closed__2);
v___x_4233_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4233_, 0, v_addedFVars_4232_);
lean_ctor_set(v___x_4233_, 1, v___x_4231_);
return v___x_4233_;
}
}
static lean_object* _init_lp_aesop_Aesop_RuleTac_forwardMatches___closed__2(void){
_start:
{
lean_object* v___x_4234_; lean_object* v_addedFVars_4235_; lean_object* v___x_4236_; 
v___x_4234_ = lean_obj_once(&lp_aesop_Aesop_RuleTac_forwardMatches___closed__1, &lp_aesop_Aesop_RuleTac_forwardMatches___closed__1_once, _init_lp_aesop_Aesop_RuleTac_forwardMatches___closed__1);
v_addedFVars_4235_ = lean_obj_once(&lp_aesop_Aesop_RuleTac_ForwardM_instInhabitedState_default___closed__2, &lp_aesop_Aesop_RuleTac_ForwardM_instInhabitedState_default___closed__2_once, _init_lp_aesop_Aesop_RuleTac_ForwardM_instInhabitedState_default___closed__2);
v___x_4236_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4236_, 0, v_addedFVars_4235_);
lean_ctor_set(v___x_4236_, 1, v___x_4234_);
return v___x_4236_;
}
}
static lean_object* _init_lp_aesop_Aesop_RuleTac_forwardMatches___closed__4(void){
_start:
{
lean_object* v___x_4238_; lean_object* v___x_4239_; 
v___x_4238_ = ((lean_object*)(lp_aesop_Aesop_RuleTac_forwardMatches___closed__3));
v___x_4239_ = l_Lean_stringToMessageData(v___x_4238_);
return v___x_4239_;
}
}
static lean_object* _init_lp_aesop_Aesop_RuleTac_forwardMatches___closed__7(void){
_start:
{
lean_object* v___x_4243_; lean_object* v___x_4244_; 
v___x_4243_ = ((lean_object*)(lp_aesop_Aesop_RuleTac_forwardMatches___closed__6));
v___x_4244_ = l_Lean_MessageData_ofFormat(v___x_4243_);
return v___x_4244_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_forwardMatches(lean_object* v_ms_4245_, lean_object* v_a_4246_, lean_object* v_a_4247_, lean_object* v_a_4248_, lean_object* v_a_4249_, lean_object* v_a_4250_, lean_object* v_a_4251_){
_start:
{
lean_object* v___y_4254_; lean_object* v_goal_4281_; uint8_t v_anySuccess_4282_; lean_object* v___x_4283_; lean_object* v___x_4284_; size_t v_sz_4285_; size_t v___x_4286_; lean_object* v___x_4287_; 
v_goal_4281_ = lean_ctor_get(v_a_4246_, 0);
lean_inc_n(v_goal_4281_, 2);
lean_dec_ref(v_a_4246_);
v_anySuccess_4282_ = 0;
v___x_4283_ = lean_obj_once(&lp_aesop_Aesop_RuleTac_forwardMatches___closed__2, &lp_aesop_Aesop_RuleTac_forwardMatches___closed__2_once, _init_lp_aesop_Aesop_RuleTac_forwardMatches___closed__2);
v___x_4284_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4284_, 0, v_goal_4281_);
lean_ctor_set(v___x_4284_, 1, v___x_4283_);
v_sz_4285_ = lean_array_size(v_ms_4245_);
v___x_4286_ = ((size_t)0ULL);
v___x_4287_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_RuleTac_forwardMatches_spec__0(v_ms_4245_, v_sz_4285_, v___x_4286_, v___x_4284_, v_a_4247_, v_a_4248_, v_a_4249_, v_a_4250_, v_a_4251_);
if (lean_obj_tag(v___x_4287_) == 0)
{
lean_object* v_a_4288_; lean_object* v_snd_4289_; lean_object* v_snd_4290_; lean_object* v_snd_4291_; lean_object* v_fst_4292_; uint8_t v___x_4293_; 
v_a_4288_ = lean_ctor_get(v___x_4287_, 0);
lean_inc(v_a_4288_);
lean_dec_ref_known(v___x_4287_, 1);
v_snd_4289_ = lean_ctor_get(v_a_4288_, 1);
lean_inc(v_snd_4289_);
v_snd_4290_ = lean_ctor_get(v_snd_4289_, 1);
lean_inc(v_snd_4290_);
v_snd_4291_ = lean_ctor_get(v_snd_4290_, 1);
lean_inc(v_snd_4291_);
v_fst_4292_ = lean_ctor_get(v_snd_4291_, 0);
v___x_4293_ = lean_unbox(v_fst_4292_);
if (v___x_4293_ == 0)
{
lean_object* v___x_4295_; uint8_t v_isShared_4296_; uint8_t v_isSharedCheck_4315_; 
lean_dec(v_snd_4290_);
lean_dec(v_snd_4289_);
lean_dec(v_a_4288_);
lean_dec(v_goal_4281_);
v_isSharedCheck_4315_ = !lean_is_exclusive(v_snd_4291_);
if (v_isSharedCheck_4315_ == 0)
{
lean_object* v_unused_4316_; lean_object* v_unused_4317_; 
v_unused_4316_ = lean_ctor_get(v_snd_4291_, 1);
lean_dec(v_unused_4316_);
v_unused_4317_ = lean_ctor_get(v_snd_4291_, 0);
lean_dec(v_unused_4317_);
v___x_4295_ = v_snd_4291_;
v_isShared_4296_ = v_isSharedCheck_4315_;
goto v_resetjp_4294_;
}
else
{
lean_dec(v_snd_4291_);
v___x_4295_ = lean_box(0);
v_isShared_4296_ = v_isSharedCheck_4315_;
goto v_resetjp_4294_;
}
v_resetjp_4294_:
{
lean_object* v___x_4297_; lean_object* v___x_4298_; lean_object* v___x_4299_; lean_object* v___x_4300_; lean_object* v___x_4301_; lean_object* v___x_4302_; lean_object* v___x_4304_; 
v___x_4297_ = lean_obj_once(&lp_aesop_Aesop_RuleTac_forwardMatches___closed__4, &lp_aesop_Aesop_RuleTac_forwardMatches___closed__4_once, _init_lp_aesop_Aesop_RuleTac_forwardMatches___closed__4);
v___x_4298_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_RuleTac_forwardMatches_spec__1(v_sz_4285_, v___x_4286_, v_ms_4245_);
v___x_4299_ = lean_array_to_list(v___x_4298_);
v___x_4300_ = lean_obj_once(&lp_aesop_Aesop_RuleTac_forwardMatches___closed__7, &lp_aesop_Aesop_RuleTac_forwardMatches___closed__7_once, _init_lp_aesop_Aesop_RuleTac_forwardMatches___closed__7);
v___x_4301_ = l_Lean_MessageData_joinSep(v___x_4299_, v___x_4300_);
v___x_4302_ = l_Lean_indentD(v___x_4301_);
if (v_isShared_4296_ == 0)
{
lean_ctor_set_tag(v___x_4295_, 7);
lean_ctor_set(v___x_4295_, 1, v___x_4302_);
lean_ctor_set(v___x_4295_, 0, v___x_4297_);
v___x_4304_ = v___x_4295_;
goto v_reusejp_4303_;
}
else
{
lean_object* v_reuseFailAlloc_4314_; 
v_reuseFailAlloc_4314_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4314_, 0, v___x_4297_);
lean_ctor_set(v_reuseFailAlloc_4314_, 1, v___x_4302_);
v___x_4304_ = v_reuseFailAlloc_4314_;
goto v_reusejp_4303_;
}
v_reusejp_4303_:
{
lean_object* v___x_4305_; lean_object* v_a_4306_; lean_object* v___x_4308_; uint8_t v_isShared_4309_; uint8_t v_isSharedCheck_4313_; 
v___x_4305_ = lp_aesop_Lean_throwError___at___00Aesop_RuleTac_forwardMatches_spec__2___redArg(v___x_4304_, v_a_4248_, v_a_4249_, v_a_4250_, v_a_4251_);
v_a_4306_ = lean_ctor_get(v___x_4305_, 0);
v_isSharedCheck_4313_ = !lean_is_exclusive(v___x_4305_);
if (v_isSharedCheck_4313_ == 0)
{
v___x_4308_ = v___x_4305_;
v_isShared_4309_ = v_isSharedCheck_4313_;
goto v_resetjp_4307_;
}
else
{
lean_inc(v_a_4306_);
lean_dec(v___x_4305_);
v___x_4308_ = lean_box(0);
v_isShared_4309_ = v_isSharedCheck_4313_;
goto v_resetjp_4307_;
}
v_resetjp_4307_:
{
lean_object* v___x_4311_; 
if (v_isShared_4309_ == 0)
{
v___x_4311_ = v___x_4308_;
goto v_reusejp_4310_;
}
else
{
lean_object* v_reuseFailAlloc_4312_; 
v_reuseFailAlloc_4312_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4312_, 0, v_a_4306_);
v___x_4311_ = v_reuseFailAlloc_4312_;
goto v_reusejp_4310_;
}
v_reusejp_4310_:
{
return v___x_4311_;
}
}
}
}
}
else
{
lean_object* v_fst_4318_; lean_object* v_fst_4319_; lean_object* v_fst_4320_; lean_object* v_snd_4321_; lean_object* v___x_4322_; lean_object* v___x_4323_; 
lean_dec_ref(v_ms_4245_);
v_fst_4318_ = lean_ctor_get(v_a_4288_, 0);
lean_inc(v_fst_4318_);
lean_dec(v_a_4288_);
v_fst_4319_ = lean_ctor_get(v_snd_4289_, 0);
lean_inc(v_fst_4319_);
lean_dec(v_snd_4289_);
v_fst_4320_ = lean_ctor_get(v_snd_4290_, 0);
lean_inc(v_fst_4320_);
lean_dec(v_snd_4290_);
v_snd_4321_ = lean_ctor_get(v_snd_4291_, 1);
lean_inc(v_snd_4321_);
lean_dec(v_snd_4291_);
v___x_4322_ = lean_box(0);
v___x_4323_ = lp_aesop_Aesop_RuleTac_forwardMatches___lam__0(v_goal_4281_, v_fst_4318_, v_fst_4319_, v_fst_4320_, v_anySuccess_4282_, v_snd_4321_, v___x_4322_, v_a_4247_, v_a_4248_, v_a_4249_, v_a_4250_, v_a_4251_);
v___y_4254_ = v___x_4323_;
goto v___jp_4253_;
}
}
else
{
lean_object* v_a_4324_; lean_object* v___x_4326_; uint8_t v_isShared_4327_; uint8_t v_isSharedCheck_4331_; 
lean_dec(v_goal_4281_);
lean_dec_ref(v_ms_4245_);
v_a_4324_ = lean_ctor_get(v___x_4287_, 0);
v_isSharedCheck_4331_ = !lean_is_exclusive(v___x_4287_);
if (v_isSharedCheck_4331_ == 0)
{
v___x_4326_ = v___x_4287_;
v_isShared_4327_ = v_isSharedCheck_4331_;
goto v_resetjp_4325_;
}
else
{
lean_inc(v_a_4324_);
lean_dec(v___x_4287_);
v___x_4326_ = lean_box(0);
v_isShared_4327_ = v_isSharedCheck_4331_;
goto v_resetjp_4325_;
}
v_resetjp_4325_:
{
lean_object* v___x_4329_; 
if (v_isShared_4327_ == 0)
{
v___x_4329_ = v___x_4326_;
goto v_reusejp_4328_;
}
else
{
lean_object* v_reuseFailAlloc_4330_; 
v_reuseFailAlloc_4330_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4330_, 0, v_a_4324_);
v___x_4329_ = v_reuseFailAlloc_4330_;
goto v_reusejp_4328_;
}
v_reusejp_4328_:
{
return v___x_4329_;
}
}
}
v___jp_4253_:
{
lean_object* v_a_4255_; lean_object* v_snd_4256_; lean_object* v_fst_4257_; lean_object* v_fst_4258_; lean_object* v_snd_4259_; lean_object* v___x_4260_; 
v_a_4255_ = lean_ctor_get(v___y_4254_, 0);
lean_inc(v_a_4255_);
lean_dec_ref(v___y_4254_);
v_snd_4256_ = lean_ctor_get(v_a_4255_, 1);
lean_inc(v_snd_4256_);
v_fst_4257_ = lean_ctor_get(v_a_4255_, 0);
lean_inc(v_fst_4257_);
lean_dec(v_a_4255_);
v_fst_4258_ = lean_ctor_get(v_snd_4256_, 0);
lean_inc(v_fst_4258_);
v_snd_4259_ = lean_ctor_get(v_snd_4256_, 1);
lean_inc(v_snd_4259_);
lean_dec(v_snd_4256_);
v___x_4260_ = l_Lean_Meta_saveState___redArg(v_a_4249_, v_a_4251_);
if (lean_obj_tag(v___x_4260_) == 0)
{
lean_object* v_a_4261_; lean_object* v___x_4263_; uint8_t v_isShared_4264_; uint8_t v_isSharedCheck_4272_; 
v_a_4261_ = lean_ctor_get(v___x_4260_, 0);
v_isSharedCheck_4272_ = !lean_is_exclusive(v___x_4260_);
if (v_isSharedCheck_4272_ == 0)
{
v___x_4263_ = v___x_4260_;
v_isShared_4264_ = v_isSharedCheck_4272_;
goto v_resetjp_4262_;
}
else
{
lean_inc(v_a_4261_);
lean_dec(v___x_4260_);
v___x_4263_ = lean_box(0);
v_isShared_4264_ = v_isSharedCheck_4272_;
goto v_resetjp_4262_;
}
v_resetjp_4262_:
{
lean_object* v___x_4265_; lean_object* v___x_4266_; lean_object* v___x_4267_; lean_object* v___x_4268_; lean_object* v___x_4270_; 
v___x_4265_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_4265_, 0, v_fst_4257_);
lean_ctor_set(v___x_4265_, 1, v_a_4261_);
lean_ctor_set(v___x_4265_, 2, v_fst_4258_);
lean_ctor_set(v___x_4265_, 3, v_snd_4259_);
v___x_4266_ = lean_unsigned_to_nat(1u);
v___x_4267_ = lean_mk_empty_array_with_capacity(v___x_4266_);
v___x_4268_ = lean_array_push(v___x_4267_, v___x_4265_);
if (v_isShared_4264_ == 0)
{
lean_ctor_set(v___x_4263_, 0, v___x_4268_);
v___x_4270_ = v___x_4263_;
goto v_reusejp_4269_;
}
else
{
lean_object* v_reuseFailAlloc_4271_; 
v_reuseFailAlloc_4271_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4271_, 0, v___x_4268_);
v___x_4270_ = v_reuseFailAlloc_4271_;
goto v_reusejp_4269_;
}
v_reusejp_4269_:
{
return v___x_4270_;
}
}
}
else
{
lean_object* v_a_4273_; lean_object* v___x_4275_; uint8_t v_isShared_4276_; uint8_t v_isSharedCheck_4280_; 
lean_dec(v_snd_4259_);
lean_dec(v_fst_4258_);
lean_dec(v_fst_4257_);
v_a_4273_ = lean_ctor_get(v___x_4260_, 0);
v_isSharedCheck_4280_ = !lean_is_exclusive(v___x_4260_);
if (v_isSharedCheck_4280_ == 0)
{
v___x_4275_ = v___x_4260_;
v_isShared_4276_ = v_isSharedCheck_4280_;
goto v_resetjp_4274_;
}
else
{
lean_inc(v_a_4273_);
lean_dec(v___x_4260_);
v___x_4275_ = lean_box(0);
v_isShared_4276_ = v_isSharedCheck_4280_;
goto v_resetjp_4274_;
}
v_resetjp_4274_:
{
lean_object* v___x_4278_; 
if (v_isShared_4276_ == 0)
{
v___x_4278_ = v___x_4275_;
goto v_reusejp_4277_;
}
else
{
lean_object* v_reuseFailAlloc_4279_; 
v_reuseFailAlloc_4279_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4279_, 0, v_a_4273_);
v___x_4278_ = v_reuseFailAlloc_4279_;
goto v_reusejp_4277_;
}
v_reusejp_4277_:
{
return v___x_4278_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_forwardMatches___boxed(lean_object* v_ms_4332_, lean_object* v_a_4333_, lean_object* v_a_4334_, lean_object* v_a_4335_, lean_object* v_a_4336_, lean_object* v_a_4337_, lean_object* v_a_4338_, lean_object* v_a_4339_){
_start:
{
lean_object* v_res_4340_; 
v_res_4340_ = lp_aesop_Aesop_RuleTac_forwardMatches(v_ms_4332_, v_a_4333_, v_a_4334_, v_a_4335_, v_a_4336_, v_a_4337_, v_a_4338_);
lean_dec(v_a_4338_);
lean_dec_ref(v_a_4337_);
lean_dec(v_a_4336_);
lean_dec_ref(v_a_4335_);
lean_dec(v_a_4334_);
return v_res_4340_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_RuleTac_forwardMatches_spec__2(lean_object* v_00_u03b1_4341_, lean_object* v_msg_4342_, lean_object* v___y_4343_, lean_object* v___y_4344_, lean_object* v___y_4345_, lean_object* v___y_4346_, lean_object* v___y_4347_){
_start:
{
lean_object* v___x_4349_; 
v___x_4349_ = lp_aesop_Lean_throwError___at___00Aesop_RuleTac_forwardMatches_spec__2___redArg(v_msg_4342_, v___y_4344_, v___y_4345_, v___y_4346_, v___y_4347_);
return v___x_4349_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_RuleTac_forwardMatches_spec__2___boxed(lean_object* v_00_u03b1_4350_, lean_object* v_msg_4351_, lean_object* v___y_4352_, lean_object* v___y_4353_, lean_object* v___y_4354_, lean_object* v___y_4355_, lean_object* v___y_4356_, lean_object* v___y_4357_){
_start:
{
lean_object* v_res_4358_; 
v_res_4358_ = lp_aesop_Lean_throwError___at___00Aesop_RuleTac_forwardMatches_spec__2(v_00_u03b1_4350_, v_msg_4351_, v___y_4352_, v___y_4353_, v___y_4354_, v___y_4355_, v___y_4356_);
lean_dec(v___y_4356_);
lean_dec_ref(v___y_4355_);
lean_dec(v___y_4354_);
lean_dec_ref(v___y_4353_);
lean_dec(v___y_4352_);
return v_res_4358_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_forwardMatch(lean_object* v_m_4359_, lean_object* v_a_4360_, lean_object* v_a_4361_, lean_object* v_a_4362_, lean_object* v_a_4363_, lean_object* v_a_4364_, lean_object* v_a_4365_){
_start:
{
lean_object* v___x_4367_; lean_object* v___x_4368_; lean_object* v___x_4369_; lean_object* v___x_4370_; 
v___x_4367_ = lean_unsigned_to_nat(1u);
v___x_4368_ = lean_mk_empty_array_with_capacity(v___x_4367_);
v___x_4369_ = lean_array_push(v___x_4368_, v_m_4359_);
v___x_4370_ = lp_aesop_Aesop_RuleTac_forwardMatches(v___x_4369_, v_a_4360_, v_a_4361_, v_a_4362_, v_a_4363_, v_a_4364_, v_a_4365_);
return v___x_4370_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleTac_forwardMatch___boxed(lean_object* v_m_4371_, lean_object* v_a_4372_, lean_object* v_a_4373_, lean_object* v_a_4374_, lean_object* v_a_4375_, lean_object* v_a_4376_, lean_object* v_a_4377_, lean_object* v_a_4378_){
_start:
{
lean_object* v_res_4379_; 
v_res_4379_ = lp_aesop_Aesop_RuleTac_forwardMatch(v_m_4371_, v_a_4372_, v_a_4373_, v_a_4374_, v_a_4375_, v_a_4376_, v_a_4377_);
lean_dec(v_a_4377_);
lean_dec_ref(v_a_4376_);
lean_dec(v_a_4375_);
lean_dec_ref(v_a_4374_);
lean_dec(v_a_4373_);
return v_res_4379_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Forward_Match(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_RuleTac_Forward_Basic(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Lean_Meta_UnusedNames(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_CollectFVars(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Tactic_Apply(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_RuleTac_Forward(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Forward_Match(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_RuleTac_Forward_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Lean_Meta_UnusedNames(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_CollectFVars(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Tactic_Apply(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_aesop_Aesop_RuleTac_ForwardM_instInhabitedContext_default = _init_lp_aesop_Aesop_RuleTac_ForwardM_instInhabitedContext_default();
lean_mark_persistent(lp_aesop_Aesop_RuleTac_ForwardM_instInhabitedContext_default);
lp_aesop_Aesop_RuleTac_ForwardM_instInhabitedContext = _init_lp_aesop_Aesop_RuleTac_ForwardM_instInhabitedContext();
lean_mark_persistent(lp_aesop_Aesop_RuleTac_ForwardM_instInhabitedContext);
lp_aesop_Aesop_RuleTac_ForwardM_instInhabitedState_default = _init_lp_aesop_Aesop_RuleTac_ForwardM_instInhabitedState_default();
lean_mark_persistent(lp_aesop_Aesop_RuleTac_ForwardM_instInhabitedState_default);
lp_aesop_Aesop_RuleTac_ForwardM_instInhabitedState = _init_lp_aesop_Aesop_RuleTac_ForwardM_instInhabitedState();
lean_mark_persistent(lp_aesop_Aesop_RuleTac_ForwardM_instInhabitedState);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_RuleTac_Forward(uint8_t builtin) {
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
lean_object* initialize_aesop_Aesop_Forward_Match(uint8_t builtin);
lean_object* initialize_aesop_Aesop_RuleTac_Forward_Basic(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Lean_Meta_UnusedNames(uint8_t builtin);
lean_object* initialize_Lean_Meta_CollectFVars(uint8_t builtin);
lean_object* initialize_Lean_Meta_Tactic_Apply(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_RuleTac_Forward(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Forward_Match(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_RuleTac_Forward_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Lean_Meta_UnusedNames(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_CollectFVars(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Tactic_Apply(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_RuleTac_Forward(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_RuleTac_Forward(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_RuleTac_Forward(builtin);
}
#ifdef __cplusplus
}
#endif
