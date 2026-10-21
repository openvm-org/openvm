// Lean compiler output
// Module: Aesop.Forward.Match
// Imports: public import Init public meta import Init import Lean.Meta.Tactic.Apply public import Aesop.Forward.Match.Types public import Aesop.Rule public import Aesop.RuleTac.ElabRuleTerm public import Aesop.Script.SpecificTactics import Aesop.RuleTac.Forward.Basic import Batteries.Lean.Meta.UnusedNames
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
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
size_t lean_usize_add(size_t, size_t);
lean_object* l_Lean_Expr_consumeMData(lean_object*);
uint8_t lp_aesop_Aesop_instBEqPhaseName_beq(uint8_t, uint8_t);
uint8_t lp_aesop_Aesop_instBEqScopeName_beq(uint8_t, uint8_t);
uint8_t lean_name_eq(lean_object*, lean_object*);
uint8_t lean_uint64_dec_eq(uint64_t, uint64_t);
uint8_t lp_aesop_Aesop_instBEqBuilderName_beq(uint8_t, uint8_t);
uint8_t l_Lean_Expr_containsFVar(lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* lean_st_ref_take(lean_object*);
double lean_float_of_nat(lean_object*);
lean_object* l_Lean_PersistentArray_push___redArg(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
uint8_t lean_expr_eqv(lean_object*, lean_object*);
uint64_t l_Lean_instHashableLevelMVarId_hash(lean_object*);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_usize_to_nat(size_t);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_instBEqLevelMVarId_beq(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkCollisionNode___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_usize_shift_right(size_t, size_t);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntries(lean_object*, lean_object*);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_mul(size_t, size_t);
uint8_t lean_usize_dec_le(size_t, size_t);
lean_object* l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
extern lean_object* lp_aesop_Aesop_instInhabitedForwardRuleMatch_default;
lean_object* lean_array_get_borrowed(lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_Expr_mvarId_x21(lean_object*);
uint64_t l_Lean_instHashableMVarId_hash(lean_object*);
uint8_t l_Lean_instBEqMVarId_beq(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_ForwardRulePriority_penalty_x3f(lean_object*);
size_t lean_array_size(lean_object*);
uint64_t lean_uint64_shift_right(uint64_t, uint64_t);
uint64_t lean_uint64_xor(uint64_t, uint64_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* lean_nat_div(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_PersistentArray_toArray___redArg(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_mkPanicMessageWithDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__2___boxed(lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__6(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_instInhabited(lean_object*);
lean_object* l_instInhabitedOfMonad___redArg(lean_object*, lean_object*);
lean_object* lean_panic_fn_borrowed(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_Substitution_empty(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_Substitution_mergeCompatible(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_Substitution_find_x3f(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_Substitution_findLevel_x3f(lean_object*, lean_object*);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* l_Lean_MessageData_ofList(lean_object*);
lean_object* lp_batteries_Lean_LocalContext_getUnusedUserName(lean_object*, lean_object*);
lean_object* l_Lean_Expr_fvar___override(lean_object*);
lean_object* l_Lean_Meta_isProof(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(lean_object*, uint8_t);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
extern lean_object* lp_aesop_Aesop_TraceOption_forward;
lean_object* l_Lean_mkAppN(lean_object*, lean_object*);
lean_object* l_Lean_Meta_abstractMVars(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_toSubarray___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Meta_synthAppInstances(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofLevel(lean_object*);
lean_object* l_Lean_indentExpr(lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
lean_object* l_Lean_Meta_ConfigWithKey_setTransparency(uint8_t, lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_forallMetaTelescope(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkConstWithFreshMVarLevels(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_elabRuleTermForApplyLikeMetaM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_withFullElaboration___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_TermElabM_run___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_collectLevelMVars(lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withNewMCtxDepthImp(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_append(lean_object*, lean_object*);
uint8_t l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(lean_object*, lean_object*, lean_object*);
lean_object* lean_io_mono_nanos_now();
double lean_float_div(double, double);
extern lean_object* l_Lean_trace_profiler;
lean_object* l_Lean_PersistentArray_append___redArg(lean_object*, lean_object*);
double lean_float_sub(double, double);
uint8_t lean_float_decLt(double, double);
extern lean_object* l_Lean_trace_profiler_useHeartbeats;
extern lean_object* l_Lean_trace_profiler_threshold;
lean_object* lean_io_get_num_heartbeats();
lean_object* lp_aesop_Aesop_assertHypothesisS(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instMonadEIO(lean_object*);
lean_object* l_StateRefT_x27_instMonad___redArg(lean_object*);
lean_object* l_Lean_Core_instMonadCoreM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Core_instMonadCoreM___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instFunctorOfMonad___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instFunctorOfMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_instMonadMetaM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_instMonadMetaM___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lp_aesop_Aesop_ForwardRule_destruct(lean_object*);
lean_object* lp_aesop_Aesop_tryClearManyS(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_aesop_Aesop_TraceOption_forwardDebug;
lean_object* lp_aesop_Aesop_isHypRedundantReducibleRigid(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
lean_object* lp_aesop_Aesop_ForwardRulePriority_successProbability_x3f(lean_object*);
extern lean_object* lp_aesop_Aesop_forwardHypPrefix;
LEAN_EXPORT uint8_t lp_aesop_Aesop_elabForwardRuleTerm___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_elabForwardRuleTerm___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_elabForwardRuleTerm___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_elabForwardRuleTerm___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_elabForwardRuleTerm___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_elabForwardRuleTerm___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_elabForwardRuleTerm___closed__0 = (const lean_object*)&lp_aesop_Aesop_elabForwardRuleTerm___closed__0_value;
static const lean_array_object lp_aesop_Aesop_elabForwardRuleTerm___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_elabForwardRuleTerm___closed__1 = (const lean_object*)&lp_aesop_Aesop_elabForwardRuleTerm___closed__1_value;
static const lean_ctor_object lp_aesop_Aesop_elabForwardRuleTerm___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*8 + 16, .m_other = 8, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_elabForwardRuleTerm___closed__0_value),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_elabForwardRuleTerm___closed__1_value),LEAN_SCALAR_PTR_LITERAL(1, 1, 1, 1, 0, 0, 0, 0),LEAN_SCALAR_PTR_LITERAL(1, 0, 1, 0, 0, 0, 0, 0)}};
static const lean_object* lp_aesop_Aesop_elabForwardRuleTerm___closed__2 = (const lean_object*)&lp_aesop_Aesop_elabForwardRuleTerm___closed__2_value;
static const lean_ctor_object lp_aesop_Aesop_elabForwardRuleTerm___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*7 + 0, .m_other = 7, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop_Aesop_elabForwardRuleTerm___closed__3 = (const lean_object*)&lp_aesop_Aesop_elabForwardRuleTerm___closed__3_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_elabForwardRuleTerm(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_elabForwardRuleTerm___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_aesop_Aesop_Match_initial___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_Match_initial___closed__0 = (const lean_object*)&lp_aesop_Aesop_Match_initial___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Match_initial(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Match_initial___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Match_addHypOrPatSubst(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Match_addHypOrPatSubst___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Match_containsHyp_spec__0(lean_object*, lean_object*, size_t, size_t);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Match_containsHyp_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Match_containsHyp(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Match_containsHyp___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Option_instBEq_beq___at___00Aesop_Match_containsPatSubst_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Option_instBEq_beq___at___00Aesop_Match_containsPatSubst_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Array_isEqvAux___at___00Aesop_Match_containsPatSubst_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_isEqvAux___at___00Aesop_Match_containsPatSubst_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Match_containsPatSubst_spec__2(lean_object*, lean_object*, size_t, size_t);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Match_containsPatSubst_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_Match_containsPatSubst(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Match_containsPatSubst___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Array_isEqvAux___at___00Aesop_Match_containsPatSubst_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_isEqvAux___at___00Aesop_Match_containsPatSubst_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_panic___at___00Aesop_CompleteMatch_reconstructArgs_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__0, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_panic___at___00Aesop_CompleteMatch_reconstructArgs_spec__0___closed__0 = (const lean_object*)&lp_aesop_panic___at___00Aesop_CompleteMatch_reconstructArgs_spec__0___closed__0_value;
static const lean_closure_object lp_aesop_panic___at___00Aesop_CompleteMatch_reconstructArgs_spec__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__1___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_panic___at___00Aesop_CompleteMatch_reconstructArgs_spec__0___closed__1 = (const lean_object*)&lp_aesop_panic___at___00Aesop_CompleteMatch_reconstructArgs_spec__0___closed__1_value;
static const lean_closure_object lp_aesop_panic___at___00Aesop_CompleteMatch_reconstructArgs_spec__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__2___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_panic___at___00Aesop_CompleteMatch_reconstructArgs_spec__0___closed__2 = (const lean_object*)&lp_aesop_panic___at___00Aesop_CompleteMatch_reconstructArgs_spec__0___closed__2_value;
static const lean_closure_object lp_aesop_panic___at___00Aesop_CompleteMatch_reconstructArgs_spec__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__3, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_panic___at___00Aesop_CompleteMatch_reconstructArgs_spec__0___closed__3 = (const lean_object*)&lp_aesop_panic___at___00Aesop_CompleteMatch_reconstructArgs_spec__0___closed__3_value;
static const lean_closure_object lp_aesop_panic___at___00Aesop_CompleteMatch_reconstructArgs_spec__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__4___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_panic___at___00Aesop_CompleteMatch_reconstructArgs_spec__0___closed__4 = (const lean_object*)&lp_aesop_panic___at___00Aesop_CompleteMatch_reconstructArgs_spec__0___closed__4_value;
static const lean_closure_object lp_aesop_panic___at___00Aesop_CompleteMatch_reconstructArgs_spec__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__5___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_panic___at___00Aesop_CompleteMatch_reconstructArgs_spec__0___closed__5 = (const lean_object*)&lp_aesop_panic___at___00Aesop_CompleteMatch_reconstructArgs_spec__0___closed__5_value;
static const lean_closure_object lp_aesop_panic___at___00Aesop_CompleteMatch_reconstructArgs_spec__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__6, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_panic___at___00Aesop_CompleteMatch_reconstructArgs_spec__0___closed__6 = (const lean_object*)&lp_aesop_panic___at___00Aesop_CompleteMatch_reconstructArgs_spec__0___closed__6_value;
static lean_once_cell_t lp_aesop_panic___at___00Aesop_CompleteMatch_reconstructArgs_spec__0___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_panic___at___00Aesop_CompleteMatch_reconstructArgs_spec__0___closed__7;
LEAN_EXPORT lean_object* lp_aesop_panic___at___00Aesop_CompleteMatch_reconstructArgs_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_CompleteMatch_reconstructArgs_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_CompleteMatch_reconstructArgs_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_CompleteMatch_reconstructArgs_spec__1(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_CompleteMatch_reconstructArgs_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_CompleteMatch_reconstructArgs_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_CompleteMatch_reconstructArgs_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_CompleteMatch_reconstructArgs___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "Aesop.Forward.Match"};
static const lean_object* lp_aesop_Aesop_CompleteMatch_reconstructArgs___closed__0 = (const lean_object*)&lp_aesop_Aesop_CompleteMatch_reconstructArgs___closed__0_value;
static const lean_string_object lp_aesop_Aesop_CompleteMatch_reconstructArgs___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 36, .m_capacity = 36, .m_length = 35, .m_data = "Aesop.CompleteMatch.reconstructArgs"};
static const lean_object* lp_aesop_Aesop_CompleteMatch_reconstructArgs___closed__1 = (const lean_object*)&lp_aesop_Aesop_CompleteMatch_reconstructArgs___closed__1_value;
static const lean_string_object lp_aesop_Aesop_CompleteMatch_reconstructArgs___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 69, .m_capacity = 69, .m_length = 68, .m_data = "assertion violation: m.clusterMatches.size == r.slotClusters.size\n  "};
static const lean_object* lp_aesop_Aesop_CompleteMatch_reconstructArgs___closed__2 = (const lean_object*)&lp_aesop_Aesop_CompleteMatch_reconstructArgs___closed__2_value;
static lean_once_cell_t lp_aesop_Aesop_CompleteMatch_reconstructArgs___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_CompleteMatch_reconstructArgs___closed__3;
LEAN_EXPORT lean_object* lp_aesop_Aesop_CompleteMatch_reconstructArgs(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_CompleteMatch_reconstructArgs___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_CompleteMatch_reconstructArgs_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_CompleteMatch_reconstructArgs_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_CompleteMatch_reconstructArgs_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_CompleteMatch_reconstructArgs_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CompleteMatch_toMessageData_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "_"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CompleteMatch_toMessageData_spec__0___closed__0 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CompleteMatch_toMessageData_spec__0___closed__0_value;
static lean_once_cell_t lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CompleteMatch_toMessageData_spec__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CompleteMatch_toMessageData_spec__0___closed__1;
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CompleteMatch_toMessageData_spec__0(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CompleteMatch_toMessageData_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_List_mapTR_loop___at___00Aesop_CompleteMatch_toMessageData_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_CompleteMatch_toMessageData(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_CompleteMatch_toMessageData___boxed(lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = " "};
static const lean_object* lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__0 = (const lean_object*)&lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__1;
static const lean_string_object lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "global"};
static const lean_object* lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__2 = (const lean_object*)&lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__2_value;
static const lean_string_object lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "local"};
static const lean_object* lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__3 = (const lean_object*)&lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__3_value;
static const lean_string_object lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "|"};
static const lean_object* lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__4 = (const lean_object*)&lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__4_value;
static const lean_string_object lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "apply"};
static const lean_object* lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__5 = (const lean_object*)&lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__5_value;
static const lean_string_object lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "cases"};
static const lean_object* lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__6 = (const lean_object*)&lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__6_value;
static const lean_string_object lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "constructors"};
static const lean_object* lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__7 = (const lean_object*)&lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__7_value;
static const lean_string_object lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "destruct"};
static const lean_object* lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__8 = (const lean_object*)&lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__8_value;
static const lean_string_object lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "forward"};
static const lean_object* lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__9 = (const lean_object*)&lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__9_value;
static const lean_string_object lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "simp"};
static const lean_object* lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__10 = (const lean_object*)&lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__10_value;
static const lean_string_object lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "tactic"};
static const lean_object* lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__11 = (const lean_object*)&lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__11_value;
static const lean_string_object lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "unfold"};
static const lean_object* lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__12 = (const lean_object*)&lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__12_value;
static const lean_string_object lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "norm"};
static const lean_object* lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__13 = (const lean_object*)&lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__13_value;
static const lean_string_object lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "safe"};
static const lean_object* lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__14 = (const lean_object*)&lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__14_value;
static const lean_string_object lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "unsafe"};
static const lean_object* lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__15 = (const lean_object*)&lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__15_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0(lean_object*);
static const lean_closure_object lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___closed__0 = (const lean_object*)&lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_ForwardRuleMatch_instToMessageData = (const lean_object*)&lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_foldHypsM___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_foldHypsM___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_foldHypsM___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_foldHypsM___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_foldHypsM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_foldHyps_spec__0_spec__0___redArg(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_foldHyps_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_foldHyps_spec__0_spec__1___redArg(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_foldHyps_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_foldHyps_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_foldHyps_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_foldHyps___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_foldHyps___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_foldHyps(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_foldHyps___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_foldHyps_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_foldHyps_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_foldHyps_spec__0_spec__0(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_foldHyps_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_foldHyps_spec__0_spec__1(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_foldHyps_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_ForwardRuleMatch_anyHyp_spec__0(lean_object*, lean_object*, size_t, size_t);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_ForwardRuleMatch_anyHyp_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_ForwardRuleMatch_anyHyp_spec__1(lean_object*, lean_object*, size_t, size_t);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_ForwardRuleMatch_anyHyp_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_ForwardRuleMatch_anyHyp(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_anyHyp___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_getPropHyps___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_getPropHyps___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_getPropHyps_spec__0_spec__0___redArg(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_getPropHyps_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_getPropHyps_spec__0_spec__1___redArg(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_getPropHyps_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_getPropHyps_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_getPropHyps_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_ForwardRuleMatch_getPropHyps___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_ForwardRuleMatch_getPropHyps___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_ForwardRuleMatch_getPropHyps___closed__0 = (const lean_object*)&lp_aesop_Aesop_ForwardRuleMatch_getPropHyps___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_getPropHyps(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_getPropHyps___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_getPropHyps_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_getPropHyps_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_getPropHyps_spec__0_spec__0(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_getPropHyps_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_getPropHyps_spec__0_spec__1(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_getPropHyps_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_instantiateMVars___at___00Aesop_ForwardRuleMatch_getProof_spec__2___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_instantiateMVars___at___00Aesop_ForwardRuleMatch_getProof_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_instantiateMVars___at___00Aesop_ForwardRuleMatch_getProof_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_instantiateMVars___at___00Aesop_ForwardRuleMatch_getProof_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_withNewMCtxDepth___at___00Aesop_ForwardRuleMatch_getProof_spec__9___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_withNewMCtxDepth___at___00Aesop_ForwardRuleMatch_getProof_spec__9___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_withNewMCtxDepth___at___00Aesop_ForwardRuleMatch_getProof_spec__9(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_withNewMCtxDepth___at___00Aesop_ForwardRuleMatch_getProof_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_ForwardRuleMatch_getProof_spec__10___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_ForwardRuleMatch_getProof_spec__10___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_ForwardRuleMatch_getProof_spec__10(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_ForwardRuleMatch_getProof_spec__10___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_ForwardRuleMatch_getProof_spec__11___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_ForwardRuleMatch_getProof_spec__11___redArg___closed__0;
static lean_once_cell_t lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_ForwardRuleMatch_getProof_spec__11___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_ForwardRuleMatch_getProof_spec__11___redArg___closed__1;
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_ForwardRuleMatch_getProof_spec__11___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_ForwardRuleMatch_getProof_spec__11___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_ForwardRuleMatch_getProof_spec__11(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_ForwardRuleMatch_getProof_spec__11___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_Option_get___at___00Aesop_ForwardRuleMatch_getProof_spec__12(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Aesop_ForwardRuleMatch_getProof_spec__12___boxed(lean_object*, lean_object*);
static const lean_ctor_object lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__0___closed__0 = (const lean_object*)&lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__0___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_ForwardRuleMatch_getProof_spec__7(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_ForwardRuleMatch_getProof_spec__7___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_ForwardRuleMatch_getProof_spec__3___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_ForwardRuleMatch_getProof_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_addTrace___at___00Aesop_ForwardRuleMatch_getProof_spec__4_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_addTrace___at___00Aesop_ForwardRuleMatch_getProof_spec__4_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_ForwardRuleMatch_getProof_spec__8___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_ForwardRuleMatch_getProof_spec__8___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Lean_addTrace___at___00Aesop_ForwardRuleMatch_getProof_spec__4___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_aesop_Lean_addTrace___at___00Aesop_ForwardRuleMatch_getProof_spec__4___closed__0;
static const lean_string_object lp_aesop_Lean_addTrace___at___00Aesop_ForwardRuleMatch_getProof_spec__4___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_aesop_Lean_addTrace___at___00Aesop_ForwardRuleMatch_getProof_spec__4___closed__1 = (const lean_object*)&lp_aesop_Lean_addTrace___at___00Aesop_ForwardRuleMatch_getProof_spec__4___closed__1_value;
static const lean_array_object lp_aesop_Lean_addTrace___at___00Aesop_ForwardRuleMatch_getProof_spec__4___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Lean_addTrace___at___00Aesop_ForwardRuleMatch_getProof_spec__4___closed__2 = (const lean_object*)&lp_aesop_Lean_addTrace___at___00Aesop_ForwardRuleMatch_getProof_spec__4___closed__2_value;
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00Aesop_ForwardRuleMatch_getProof_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00Aesop_ForwardRuleMatch_getProof_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_ForwardRuleMatch_getProof_spec__1_spec__2_spec__9_spec__23_spec__26___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_ForwardRuleMatch_getProof_spec__1_spec__2_spec__9_spec__23___redArg(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_ForwardRuleMatch_getProof_spec__1_spec__2_spec__9___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_ForwardRuleMatch_getProof_spec__1_spec__2_spec__9___redArg___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_ForwardRuleMatch_getProof_spec__1_spec__2_spec__9___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_ForwardRuleMatch_getProof_spec__1_spec__2_spec__9_spec__24___redArg(size_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_ForwardRuleMatch_getProof_spec__1_spec__2_spec__9_spec__24___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_ForwardRuleMatch_getProof_spec__1_spec__2_spec__9___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_ForwardRuleMatch_getProof_spec__1_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_assignLevelMVar___at___00Aesop_ForwardRuleMatch_getProof_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_assignLevelMVar___at___00Aesop_ForwardRuleMatch_getProof_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_ForwardRuleMatch_getProof_spec__5(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_ForwardRuleMatch_getProof_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_ForwardRuleMatch_getProof_spec__0_spec__0_spec__6_spec__19_spec__22___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_ForwardRuleMatch_getProof_spec__0_spec__0_spec__6_spec__19___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_ForwardRuleMatch_getProof_spec__0_spec__0_spec__6___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_ForwardRuleMatch_getProof_spec__0_spec__0_spec__6_spec__20___redArg(size_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_ForwardRuleMatch_getProof_spec__0_spec__0_spec__6_spec__20___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_ForwardRuleMatch_getProof_spec__0_spec__0_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_ForwardRuleMatch_getProof_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_assign___at___00Aesop_ForwardRuleMatch_getProof_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_assign___at___00Aesop_ForwardRuleMatch_getProof_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_ForwardRuleMatch_getProof_spec__6(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_ForwardRuleMatch_getProof_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "result: "};
static const lean_object* lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__0 = (const lean_object*)&lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__1;
static const lean_string_object lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "instance synthesis failed"};
static const lean_object* lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__2 = (const lean_object*)&lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__2_value;
static lean_once_cell_t lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__3;
static const lean_string_object lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "aesop"};
static const lean_object* lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__4 = (const lean_object*)&lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__4_value;
static const lean_ctor_object lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(29, 147, 5, 234, 46, 7, 240, 183)}};
static const lean_object* lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__5 = (const lean_object*)&lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__5_value;
static const lean_string_object lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "levels: "};
static const lean_object* lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__6 = (const lean_object*)&lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__6_value;
static lean_once_cell_t lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__7;
static const lean_string_object lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "args:   "};
static const lean_object* lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__8 = (const lean_object*)&lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__8_value;
static lean_once_cell_t lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__9;
static const lean_string_object lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "rule term"};
static const lean_object* lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__10 = (const lean_object*)&lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__10_value;
static lean_once_cell_t lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__11;
static const lean_string_object lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "\nwith type"};
static const lean_object* lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__12 = (const lean_object*)&lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__12_value;
static lean_once_cell_t lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__13;
static const lean_string_object lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "\n was expected to have "};
static const lean_object* lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__14 = (const lean_object*)&lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__14_value;
static lean_once_cell_t lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__15;
static const lean_string_object lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 31, .m_capacity = 31, .m_length = 30, .m_data = " level metavariables, but has "};
static const lean_object* lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__16 = (const lean_object*)&lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__16_value;
static lean_once_cell_t lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__17;
static const lean_string_object lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = " arguments, but has "};
static const lean_object* lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__18 = (const lean_object*)&lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__18_value;
static lean_once_cell_t lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__19;
static lean_once_cell_t lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__20;
static lean_once_cell_t lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__21;
static const lean_array_object lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__22 = (const lean_object*)&lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__22_value;
static lean_once_cell_t lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__23;
static const lean_string_object lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "term: "};
static const lean_object* lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__24 = (const lean_object*)&lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__24_value;
static lean_once_cell_t lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__25_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__25;
static const lean_string_object lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = " : "};
static const lean_object* lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__26 = (const lean_object*)&lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__26_value;
static lean_once_cell_t lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__27_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__27;
static const lean_string_object lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "rule: "};
static const lean_object* lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__28 = (const lean_object*)&lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__28_value;
static lean_once_cell_t lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__29_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__29;
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13_spec__16_spec__19(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13_spec__16_spec__19___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13_spec__16(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13_spec__16___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13_spec__18(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13_spec__18___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13_spec__19(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13_spec__19___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13_spec__17___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13_spec__17___redArg___boxed(lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 54, .m_capacity = 54, .m_length = 53, .m_data = "<exception thrown while producing trace node message>"};
static const lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13___closed__0 = (const lean_object*)&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13___closed__0_value;
static lean_once_cell_t lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13___closed__1;
static lean_once_cell_t lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13___closed__2;
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_ForwardRuleMatch_getProof___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ":\n"};
static const lean_object* lp_aesop_Aesop_ForwardRuleMatch_getProof___closed__0 = (const lean_object*)&lp_aesop_Aesop_ForwardRuleMatch_getProof___closed__0_value;
static const lean_string_object lp_aesop_Aesop_ForwardRuleMatch_getProof___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 47, .m_capacity = 47, .m_length = 46, .m_data = "while constructing a new hyp for forward rule "};
static const lean_object* lp_aesop_Aesop_ForwardRuleMatch_getProof___closed__1 = (const lean_object*)&lp_aesop_Aesop_ForwardRuleMatch_getProof___closed__1_value;
static const lean_closure_object lp_aesop_Aesop_ForwardRuleMatch_getProof___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__0___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_ForwardRuleMatch_getProof___closed__2 = (const lean_object*)&lp_aesop_Aesop_ForwardRuleMatch_getProof___closed__2_value;
static const lean_string_object lp_aesop_Aesop_ForwardRuleMatch_getProof___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 42, .m_capacity = 42, .m_length = 41, .m_data = "proof construction for forward rule match"};
static const lean_object* lp_aesop_Aesop_ForwardRuleMatch_getProof___closed__3 = (const lean_object*)&lp_aesop_Aesop_ForwardRuleMatch_getProof___closed__3_value;
static lean_once_cell_t lp_aesop_Aesop_ForwardRuleMatch_getProof___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_ForwardRuleMatch_getProof___closed__4;
static lean_once_cell_t lp_aesop_Aesop_ForwardRuleMatch_getProof___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_ForwardRuleMatch_getProof___closed__5;
static const lean_string_object lp_aesop_Aesop_ForwardRuleMatch_getProof___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_aesop_Aesop_ForwardRuleMatch_getProof___closed__6 = (const lean_object*)&lp_aesop_Aesop_ForwardRuleMatch_getProof___closed__6_value;
static const lean_ctor_object lp_aesop_Aesop_ForwardRuleMatch_getProof___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_ForwardRuleMatch_getProof___closed__6_value),LEAN_SCALAR_PTR_LITERAL(212, 145, 141, 177, 67, 149, 127, 197)}};
static const lean_object* lp_aesop_Aesop_ForwardRuleMatch_getProof___closed__7 = (const lean_object*)&lp_aesop_Aesop_ForwardRuleMatch_getProof___closed__7_value;
static lean_once_cell_t lp_aesop_Aesop_ForwardRuleMatch_getProof___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_aesop_Aesop_ForwardRuleMatch_getProof___closed__8;
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_getProof(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_getProof___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_assign___at___00Aesop_ForwardRuleMatch_getProof_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_assign___at___00Aesop_ForwardRuleMatch_getProof_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_assignLevelMVar___at___00Aesop_ForwardRuleMatch_getProof_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_assignLevelMVar___at___00Aesop_ForwardRuleMatch_getProof_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_ForwardRuleMatch_getProof_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_ForwardRuleMatch_getProof_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_ForwardRuleMatch_getProof_spec__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_ForwardRuleMatch_getProof_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13_spec__17(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13_spec__17___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_ForwardRuleMatch_getProof_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_ForwardRuleMatch_getProof_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_ForwardRuleMatch_getProof_spec__0_spec__0_spec__6(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_ForwardRuleMatch_getProof_spec__0_spec__0_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_ForwardRuleMatch_getProof_spec__1_spec__2_spec__9(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_ForwardRuleMatch_getProof_spec__1_spec__2_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_ForwardRuleMatch_getProof_spec__0_spec__0_spec__6_spec__19(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_ForwardRuleMatch_getProof_spec__0_spec__0_spec__6_spec__20(lean_object*, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_ForwardRuleMatch_getProof_spec__0_spec__0_spec__6_spec__20___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_ForwardRuleMatch_getProof_spec__1_spec__2_spec__9_spec__23(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_ForwardRuleMatch_getProof_spec__1_spec__2_spec__9_spec__24(lean_object*, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_ForwardRuleMatch_getProof_spec__1_spec__2_spec__9_spec__24___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_ForwardRuleMatch_getProof_spec__0_spec__0_spec__6_spec__19_spec__22(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_ForwardRuleMatch_getProof_spec__1_spec__2_spec__9_spec__23_spec__26(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_getUnusedUserName___at___00Aesop_ForwardRuleMatch_apply_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_getUnusedUserName___at___00Aesop_ForwardRuleMatch_apply_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_getUnusedUserName___at___00Aesop_ForwardRuleMatch_apply_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_getUnusedUserName___at___00Aesop_ForwardRuleMatch_apply_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_panic___at___00Aesop_ForwardRuleMatch_apply_spec__1___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_panic___at___00Aesop_ForwardRuleMatch_apply_spec__1___closed__0;
static const lean_closure_object lp_aesop_panic___at___00Aesop_ForwardRuleMatch_apply_spec__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadCoreM___lam__0___boxed, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_panic___at___00Aesop_ForwardRuleMatch_apply_spec__1___closed__1 = (const lean_object*)&lp_aesop_panic___at___00Aesop_ForwardRuleMatch_apply_spec__1___closed__1_value;
static const lean_closure_object lp_aesop_panic___at___00Aesop_ForwardRuleMatch_apply_spec__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadCoreM___lam__1___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_panic___at___00Aesop_ForwardRuleMatch_apply_spec__1___closed__2 = (const lean_object*)&lp_aesop_panic___at___00Aesop_ForwardRuleMatch_apply_spec__1___closed__2_value;
static const lean_closure_object lp_aesop_panic___at___00Aesop_ForwardRuleMatch_apply_spec__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instMonadMetaM___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_panic___at___00Aesop_ForwardRuleMatch_apply_spec__1___closed__3 = (const lean_object*)&lp_aesop_panic___at___00Aesop_ForwardRuleMatch_apply_spec__1___closed__3_value;
static const lean_closure_object lp_aesop_panic___at___00Aesop_ForwardRuleMatch_apply_spec__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instMonadMetaM___lam__1___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_panic___at___00Aesop_ForwardRuleMatch_apply_spec__1___closed__4 = (const lean_object*)&lp_aesop_panic___at___00Aesop_ForwardRuleMatch_apply_spec__1___closed__4_value;
LEAN_EXPORT lean_object* lp_aesop_panic___at___00Aesop_ForwardRuleMatch_apply_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_panic___at___00Aesop_ForwardRuleMatch_apply_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_ForwardRuleMatch_apply_spec__2___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_ForwardRuleMatch_apply_spec__2___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_ForwardRuleMatch_apply_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_ForwardRuleMatch_apply_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_ForwardRuleMatch_apply_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_ForwardRuleMatch_apply_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_ForwardRuleMatch_apply_spec__5___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_ForwardRuleMatch_apply_spec__5___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_ForwardRuleMatch_apply_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_ForwardRuleMatch_apply_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_apply___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_apply___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00Aesop_ForwardRuleMatch_apply_spec__4___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00Aesop_ForwardRuleMatch_apply_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_ForwardRuleMatch_apply_spec__3___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_ForwardRuleMatch_apply_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_ForwardRuleMatch_apply___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "already exists: "};
static const lean_object* lp_aesop_Aesop_ForwardRuleMatch_apply___lam__1___closed__0 = (const lean_object*)&lp_aesop_Aesop_ForwardRuleMatch_apply___lam__1___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_ForwardRuleMatch_apply___lam__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_ForwardRuleMatch_apply___lam__1___closed__1;
static const lean_string_object lp_aesop_Aesop_ForwardRuleMatch_apply___lam__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "false"};
static const lean_object* lp_aesop_Aesop_ForwardRuleMatch_apply___lam__1___closed__2 = (const lean_object*)&lp_aesop_Aesop_ForwardRuleMatch_apply___lam__1___closed__2_value;
static const lean_string_object lp_aesop_Aesop_ForwardRuleMatch_apply___lam__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "true"};
static const lean_object* lp_aesop_Aesop_ForwardRuleMatch_apply___lam__1___closed__3 = (const lean_object*)&lp_aesop_Aesop_ForwardRuleMatch_apply___lam__1___closed__3_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_apply___lam__1(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_apply___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_apply___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_apply___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_apply_spec__6_spec__7___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_apply_spec__6_spec__7___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_apply_spec__6_spec__8(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_apply_spec__6_spec__8___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_apply_spec__6_spec__6___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_apply_spec__6_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_apply_spec__6(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_apply_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_ForwardRuleMatch_apply___lam__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = "Aesop.ForwardRuleMatch.apply"};
static const lean_object* lp_aesop_Aesop_ForwardRuleMatch_apply___lam__3___closed__0 = (const lean_object*)&lp_aesop_Aesop_ForwardRuleMatch_apply___lam__3___closed__0_value;
static const lean_string_object lp_aesop_Aesop_ForwardRuleMatch_apply___lam__3___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 34, .m_capacity = 34, .m_length = 33, .m_data = "unreachable code has been reached"};
static const lean_object* lp_aesop_Aesop_ForwardRuleMatch_apply___lam__3___closed__1 = (const lean_object*)&lp_aesop_Aesop_ForwardRuleMatch_apply___lam__3___closed__1_value;
static lean_once_cell_t lp_aesop_Aesop_ForwardRuleMatch_apply___lam__3___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_ForwardRuleMatch_apply___lam__3___closed__2;
static const lean_array_object lp_aesop_Aesop_ForwardRuleMatch_apply___lam__3___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_ForwardRuleMatch_apply___lam__3___closed__3 = (const lean_object*)&lp_aesop_Aesop_ForwardRuleMatch_apply___lam__3___closed__3_value;
static const lean_string_object lp_aesop_Aesop_ForwardRuleMatch_apply___lam__3___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 33, .m_capacity = 33, .m_length = 32, .m_data = "check whether hyp already exists"};
static const lean_object* lp_aesop_Aesop_ForwardRuleMatch_apply___lam__3___closed__4 = (const lean_object*)&lp_aesop_Aesop_ForwardRuleMatch_apply___lam__3___closed__4_value;
static lean_once_cell_t lp_aesop_Aesop_ForwardRuleMatch_apply___lam__3___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_ForwardRuleMatch_apply___lam__3___closed__5;
static lean_once_cell_t lp_aesop_Aesop_ForwardRuleMatch_apply___lam__3___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_ForwardRuleMatch_apply___lam__3___closed__6;
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_apply___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_apply___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_apply___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_apply___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_apply_spec__7_spec__10(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_apply_spec__7_spec__10___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_apply_spec__7(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_apply_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_ForwardRuleMatch_apply___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "apply complete match"};
static const lean_object* lp_aesop_Aesop_ForwardRuleMatch_apply___closed__0 = (const lean_object*)&lp_aesop_Aesop_ForwardRuleMatch_apply___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_ForwardRuleMatch_apply___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_ForwardRuleMatch_apply___closed__1;
static lean_once_cell_t lp_aesop_Aesop_ForwardRuleMatch_apply___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_ForwardRuleMatch_apply___closed__2;
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_apply(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_apply___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_ForwardRuleMatch_apply_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_ForwardRuleMatch_apply_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00Aesop_ForwardRuleMatch_apply_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00Aesop_ForwardRuleMatch_apply_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_apply_spec__6_spec__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_apply_spec__6_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_apply_spec__6_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_apply_spec__6_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__1_spec__3_spec__4_spec__8___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__1_spec__3_spec__4___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__1_spec__3___redArg(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__1_spec__2___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__1_spec__2___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__1_spec__4___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__0_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__2(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_aesop___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__3___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__3___redArg___closed__0 = (const lean_object*)&lp_aesop___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__3___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__3___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__4___redArg(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop___private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f___redArg___closed__0;
static lean_once_cell_t lp_aesop___private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f___redArg___closed__1;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__4(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__0_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__1_spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__1_spec__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__1_spec__4(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__1_spec__3_spec__4(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__1_spec__3_spec__4_spec__8(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_forwardRuleMatchesToNormRules_x3f___lam__0(lean_object*);
static const lean_closure_object lp_aesop_Aesop_forwardRuleMatchesToNormRules_x3f___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_forwardRuleMatchesToNormRules_x3f___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_forwardRuleMatchesToNormRules_x3f___closed__0 = (const lean_object*)&lp_aesop_Aesop_forwardRuleMatchesToNormRules_x3f___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_forwardRuleMatchesToNormRules_x3f(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_forwardRuleMatchesToNormRules_x3f___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_forwardRuleMatchesToSafeRules_x3f___lam__0(lean_object*);
static const lean_closure_object lp_aesop_Aesop_forwardRuleMatchesToSafeRules_x3f___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_forwardRuleMatchesToSafeRules_x3f___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_forwardRuleMatchesToSafeRules_x3f___closed__0 = (const lean_object*)&lp_aesop_Aesop_forwardRuleMatchesToSafeRules_x3f___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_forwardRuleMatchesToSafeRules_x3f(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_forwardRuleMatchesToSafeRules_x3f___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_forwardRuleMatchesToUnsafeRules_x3f___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_forwardRuleMatchesToUnsafeRules_x3f___lam__0___boxed(lean_object*);
static const lean_closure_object lp_aesop_Aesop_forwardRuleMatchesToUnsafeRules_x3f___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_forwardRuleMatchesToUnsafeRules_x3f___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_forwardRuleMatchesToUnsafeRules_x3f___closed__0 = (const lean_object*)&lp_aesop_Aesop_forwardRuleMatchesToUnsafeRules_x3f___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_forwardRuleMatchesToUnsafeRules_x3f(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_forwardRuleMatchesToUnsafeRules_x3f___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_elabForwardRuleTerm___lam__0(lean_object* v_x_1_){
_start:
{
uint8_t v___x_2_; 
v___x_2_ = 0;
return v___x_2_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_elabForwardRuleTerm___lam__0___boxed(lean_object* v_x_3_){
_start:
{
uint8_t v_res_4_; lean_object* v_r_5_; 
v_res_4_ = lp_aesop_Aesop_elabForwardRuleTerm___lam__0(v_x_3_);
lean_dec(v_x_3_);
v_r_5_ = lean_box(v_res_4_);
return v_r_5_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_elabForwardRuleTerm___lam__1(lean_object* v_goal_6_, lean_object* v_term_7_, lean_object* v___y_8_, lean_object* v___y_9_, lean_object* v___y_10_, lean_object* v___y_11_, lean_object* v___y_12_, lean_object* v___y_13_){
_start:
{
lean_object* v___x_15_; 
v___x_15_ = lp_aesop_Aesop_elabRuleTermForApplyLikeMetaM(v_goal_6_, v_term_7_, v___y_10_, v___y_11_, v___y_12_, v___y_13_);
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_elabForwardRuleTerm___lam__1___boxed(lean_object* v_goal_16_, lean_object* v_term_17_, lean_object* v___y_18_, lean_object* v___y_19_, lean_object* v___y_20_, lean_object* v___y_21_, lean_object* v___y_22_, lean_object* v___y_23_, lean_object* v___y_24_){
_start:
{
lean_object* v_res_25_; 
v_res_25_ = lp_aesop_Aesop_elabForwardRuleTerm___lam__1(v_goal_16_, v_term_17_, v___y_18_, v___y_19_, v___y_20_, v___y_21_, v___y_22_, v___y_23_);
lean_dec(v___y_23_);
lean_dec_ref(v___y_22_);
lean_dec(v___y_21_);
lean_dec_ref(v___y_20_);
lean_dec(v___y_19_);
lean_dec_ref(v___y_18_);
return v_res_25_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_elabForwardRuleTerm(lean_object* v_goal_40_, lean_object* v_x_41_, lean_object* v_a_42_, lean_object* v_a_43_, lean_object* v_a_44_, lean_object* v_a_45_){
_start:
{
if (lean_obj_tag(v_x_41_) == 0)
{
lean_object* v_decl_47_; lean_object* v___x_48_; 
lean_dec(v_goal_40_);
v_decl_47_ = lean_ctor_get(v_x_41_, 0);
lean_inc(v_decl_47_);
lean_dec_ref_known(v_x_41_, 1);
v___x_48_ = l_Lean_Meta_mkConstWithFreshMVarLevels(v_decl_47_, v_a_42_, v_a_43_, v_a_44_, v_a_45_);
return v___x_48_;
}
else
{
lean_object* v_term_49_; lean_object* v___f_50_; lean_object* v___x_51_; lean_object* v___x_52_; lean_object* v___x_53_; lean_object* v___x_54_; 
v_term_49_ = lean_ctor_get(v_x_41_, 0);
lean_inc(v_term_49_);
lean_dec_ref_known(v_x_41_, 1);
v___f_50_ = lean_alloc_closure((void*)(lp_aesop_Aesop_elabForwardRuleTerm___lam__1___boxed), 9, 2);
lean_closure_set(v___f_50_, 0, v_goal_40_);
lean_closure_set(v___f_50_, 1, v_term_49_);
v___x_51_ = lean_alloc_closure((void*)(lp_aesop_Aesop_withFullElaboration___boxed), 9, 2);
lean_closure_set(v___x_51_, 0, lean_box(0));
lean_closure_set(v___x_51_, 1, v___f_50_);
v___x_52_ = ((lean_object*)(lp_aesop_Aesop_elabForwardRuleTerm___closed__2));
v___x_53_ = ((lean_object*)(lp_aesop_Aesop_elabForwardRuleTerm___closed__3));
v___x_54_ = l_Lean_Elab_Term_TermElabM_run___redArg(v___x_51_, v___x_52_, v___x_53_, v_a_42_, v_a_43_, v_a_44_, v_a_45_);
if (lean_obj_tag(v___x_54_) == 0)
{
lean_object* v_a_55_; lean_object* v___x_57_; uint8_t v_isShared_58_; uint8_t v_isSharedCheck_63_; 
v_a_55_ = lean_ctor_get(v___x_54_, 0);
v_isSharedCheck_63_ = !lean_is_exclusive(v___x_54_);
if (v_isSharedCheck_63_ == 0)
{
v___x_57_ = v___x_54_;
v_isShared_58_ = v_isSharedCheck_63_;
goto v_resetjp_56_;
}
else
{
lean_inc(v_a_55_);
lean_dec(v___x_54_);
v___x_57_ = lean_box(0);
v_isShared_58_ = v_isSharedCheck_63_;
goto v_resetjp_56_;
}
v_resetjp_56_:
{
lean_object* v_fst_59_; lean_object* v___x_61_; 
v_fst_59_ = lean_ctor_get(v_a_55_, 0);
lean_inc(v_fst_59_);
lean_dec(v_a_55_);
if (v_isShared_58_ == 0)
{
lean_ctor_set(v___x_57_, 0, v_fst_59_);
v___x_61_ = v___x_57_;
goto v_reusejp_60_;
}
else
{
lean_object* v_reuseFailAlloc_62_; 
v_reuseFailAlloc_62_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_62_, 0, v_fst_59_);
v___x_61_ = v_reuseFailAlloc_62_;
goto v_reusejp_60_;
}
v_reusejp_60_:
{
return v___x_61_;
}
}
}
else
{
lean_object* v_a_64_; lean_object* v___x_66_; uint8_t v_isShared_67_; uint8_t v_isSharedCheck_71_; 
v_a_64_ = lean_ctor_get(v___x_54_, 0);
v_isSharedCheck_71_ = !lean_is_exclusive(v___x_54_);
if (v_isSharedCheck_71_ == 0)
{
v___x_66_ = v___x_54_;
v_isShared_67_ = v_isSharedCheck_71_;
goto v_resetjp_65_;
}
else
{
lean_inc(v_a_64_);
lean_dec(v___x_54_);
v___x_66_ = lean_box(0);
v_isShared_67_ = v_isSharedCheck_71_;
goto v_resetjp_65_;
}
v_resetjp_65_:
{
lean_object* v___x_69_; 
if (v_isShared_67_ == 0)
{
v___x_69_ = v___x_66_;
goto v_reusejp_68_;
}
else
{
lean_object* v_reuseFailAlloc_70_; 
v_reuseFailAlloc_70_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_70_, 0, v_a_64_);
v___x_69_ = v_reuseFailAlloc_70_;
goto v_reusejp_68_;
}
v_reusejp_68_:
{
return v___x_69_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_elabForwardRuleTerm___boxed(lean_object* v_goal_72_, lean_object* v_x_73_, lean_object* v_a_74_, lean_object* v_a_75_, lean_object* v_a_76_, lean_object* v_a_77_, lean_object* v_a_78_){
_start:
{
lean_object* v_res_79_; 
v_res_79_ = lp_aesop_Aesop_elabForwardRuleTerm(v_goal_72_, v_x_73_, v_a_74_, v_a_75_, v_a_76_, v_a_77_);
lean_dec(v_a_77_);
lean_dec_ref(v_a_76_);
lean_dec(v_a_75_);
lean_dec_ref(v_a_74_);
return v_res_79_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Match_initial(lean_object* v_subst_82_, uint8_t v_isPatSubst_83_, lean_object* v_forwardDeps_84_, lean_object* v_conclusionDeps_85_){
_start:
{
lean_object* v___y_87_; 
if (v_isPatSubst_83_ == 0)
{
lean_object* v___x_90_; 
v___x_90_ = ((lean_object*)(lp_aesop_Aesop_Match_initial___closed__0));
v___y_87_ = v___x_90_;
goto v___jp_86_;
}
else
{
lean_object* v___x_91_; lean_object* v___x_92_; lean_object* v___x_93_; 
v___x_91_ = lean_unsigned_to_nat(1u);
v___x_92_ = lean_mk_empty_array_with_capacity(v___x_91_);
lean_inc_ref(v_subst_82_);
v___x_93_ = lean_array_push(v___x_92_, v_subst_82_);
v___y_87_ = v___x_93_;
goto v___jp_86_;
}
v___jp_86_:
{
lean_object* v___x_88_; lean_object* v___x_89_; 
v___x_88_ = lean_unsigned_to_nat(0u);
v___x_89_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_89_, 0, v_subst_82_);
lean_ctor_set(v___x_89_, 1, v___y_87_);
lean_ctor_set(v___x_89_, 2, v___x_88_);
lean_ctor_set(v___x_89_, 3, v_forwardDeps_84_);
lean_ctor_set(v___x_89_, 4, v_conclusionDeps_85_);
return v___x_89_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Match_initial___boxed(lean_object* v_subst_94_, lean_object* v_isPatSubst_95_, lean_object* v_forwardDeps_96_, lean_object* v_conclusionDeps_97_){
_start:
{
uint8_t v_isPatSubst_boxed_98_; lean_object* v_res_99_; 
v_isPatSubst_boxed_98_ = lean_unbox(v_isPatSubst_95_);
v_res_99_ = lp_aesop_Aesop_Match_initial(v_subst_94_, v_isPatSubst_boxed_98_, v_forwardDeps_96_, v_conclusionDeps_97_);
return v_res_99_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Match_addHypOrPatSubst(lean_object* v_subst_100_, uint8_t v_isPatSubst_101_, lean_object* v_forwardDeps_102_, lean_object* v_m_103_){
_start:
{
lean_object* v_subst_104_; lean_object* v_patInstSubsts_105_; lean_object* v_level_106_; lean_object* v_conclusionDeps_107_; lean_object* v___x_109_; uint8_t v_isShared_110_; uint8_t v_isSharedCheck_120_; 
v_subst_104_ = lean_ctor_get(v_m_103_, 0);
v_patInstSubsts_105_ = lean_ctor_get(v_m_103_, 1);
v_level_106_ = lean_ctor_get(v_m_103_, 2);
v_conclusionDeps_107_ = lean_ctor_get(v_m_103_, 4);
v_isSharedCheck_120_ = !lean_is_exclusive(v_m_103_);
if (v_isSharedCheck_120_ == 0)
{
lean_object* v_unused_121_; 
v_unused_121_ = lean_ctor_get(v_m_103_, 3);
lean_dec(v_unused_121_);
v___x_109_ = v_m_103_;
v_isShared_110_ = v_isSharedCheck_120_;
goto v_resetjp_108_;
}
else
{
lean_inc(v_conclusionDeps_107_);
lean_inc(v_level_106_);
lean_inc(v_patInstSubsts_105_);
lean_inc(v_subst_104_);
lean_dec(v_m_103_);
v___x_109_ = lean_box(0);
v_isShared_110_ = v_isSharedCheck_120_;
goto v_resetjp_108_;
}
v_resetjp_108_:
{
lean_object* v___x_111_; lean_object* v___y_113_; 
v___x_111_ = lp_aesop_Aesop_Substitution_mergeCompatible(v_subst_104_, v_subst_100_);
if (v_isPatSubst_101_ == 0)
{
lean_dec_ref(v_subst_100_);
v___y_113_ = v_patInstSubsts_105_;
goto v___jp_112_;
}
else
{
lean_object* v___x_119_; 
v___x_119_ = lean_array_push(v_patInstSubsts_105_, v_subst_100_);
v___y_113_ = v___x_119_;
goto v___jp_112_;
}
v___jp_112_:
{
lean_object* v___x_114_; lean_object* v___x_115_; lean_object* v___x_117_; 
v___x_114_ = lean_unsigned_to_nat(1u);
v___x_115_ = lean_nat_add(v_level_106_, v___x_114_);
lean_dec(v_level_106_);
if (v_isShared_110_ == 0)
{
lean_ctor_set(v___x_109_, 3, v_forwardDeps_102_);
lean_ctor_set(v___x_109_, 2, v___x_115_);
lean_ctor_set(v___x_109_, 1, v___y_113_);
lean_ctor_set(v___x_109_, 0, v___x_111_);
v___x_117_ = v___x_109_;
goto v_reusejp_116_;
}
else
{
lean_object* v_reuseFailAlloc_118_; 
v_reuseFailAlloc_118_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_118_, 0, v___x_111_);
lean_ctor_set(v_reuseFailAlloc_118_, 1, v___y_113_);
lean_ctor_set(v_reuseFailAlloc_118_, 2, v___x_115_);
lean_ctor_set(v_reuseFailAlloc_118_, 3, v_forwardDeps_102_);
lean_ctor_set(v_reuseFailAlloc_118_, 4, v_conclusionDeps_107_);
v___x_117_ = v_reuseFailAlloc_118_;
goto v_reusejp_116_;
}
v_reusejp_116_:
{
return v___x_117_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Match_addHypOrPatSubst___boxed(lean_object* v_subst_122_, lean_object* v_isPatSubst_123_, lean_object* v_forwardDeps_124_, lean_object* v_m_125_){
_start:
{
uint8_t v_isPatSubst_boxed_126_; lean_object* v_res_127_; 
v_isPatSubst_boxed_126_ = lean_unbox(v_isPatSubst_123_);
v_res_127_ = lp_aesop_Aesop_Match_addHypOrPatSubst(v_subst_122_, v_isPatSubst_boxed_126_, v_forwardDeps_124_, v_m_125_);
return v_res_127_;
}
}
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Match_containsHyp_spec__0(lean_object* v_hyp_128_, lean_object* v_as_129_, size_t v_i_130_, size_t v_stop_131_){
_start:
{
uint8_t v___x_132_; 
v___x_132_ = lean_usize_dec_eq(v_i_130_, v_stop_131_);
if (v___x_132_ == 0)
{
uint8_t v___x_133_; uint8_t v___y_135_; lean_object* v___x_139_; 
v___x_133_ = 1;
v___x_139_ = lean_array_uget_borrowed(v_as_129_, v_i_130_);
if (lean_obj_tag(v___x_139_) == 0)
{
v___y_135_ = v___x_132_;
goto v___jp_134_;
}
else
{
lean_object* v_val_140_; uint8_t v___x_141_; 
v_val_140_ = lean_ctor_get(v___x_139_, 0);
v___x_141_ = l_Lean_Expr_containsFVar(v_val_140_, v_hyp_128_);
v___y_135_ = v___x_141_;
goto v___jp_134_;
}
v___jp_134_:
{
if (v___y_135_ == 0)
{
size_t v___x_136_; size_t v___x_137_; 
v___x_136_ = ((size_t)1ULL);
v___x_137_ = lean_usize_add(v_i_130_, v___x_136_);
v_i_130_ = v___x_137_;
goto _start;
}
else
{
return v___x_133_;
}
}
}
else
{
uint8_t v___x_142_; 
v___x_142_ = 0;
return v___x_142_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Match_containsHyp_spec__0___boxed(lean_object* v_hyp_143_, lean_object* v_as_144_, lean_object* v_i_145_, lean_object* v_stop_146_){
_start:
{
size_t v_i_boxed_147_; size_t v_stop_boxed_148_; uint8_t v_res_149_; lean_object* v_r_150_; 
v_i_boxed_147_ = lean_unbox_usize(v_i_145_);
lean_dec(v_i_145_);
v_stop_boxed_148_ = lean_unbox_usize(v_stop_146_);
lean_dec(v_stop_146_);
v_res_149_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Match_containsHyp_spec__0(v_hyp_143_, v_as_144_, v_i_boxed_147_, v_stop_boxed_148_);
lean_dec_ref(v_as_144_);
lean_dec(v_hyp_143_);
v_r_150_ = lean_box(v_res_149_);
return v_r_150_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Match_containsHyp(lean_object* v_hyp_151_, lean_object* v_m_152_){
_start:
{
lean_object* v_subst_153_; lean_object* v_premises_154_; lean_object* v___x_155_; lean_object* v___x_156_; uint8_t v___x_157_; 
v_subst_153_ = lean_ctor_get(v_m_152_, 0);
v_premises_154_ = lean_ctor_get(v_subst_153_, 0);
v___x_155_ = lean_unsigned_to_nat(0u);
v___x_156_ = lean_array_get_size(v_premises_154_);
v___x_157_ = lean_nat_dec_lt(v___x_155_, v___x_156_);
if (v___x_157_ == 0)
{
return v___x_157_;
}
else
{
if (v___x_157_ == 0)
{
return v___x_157_;
}
else
{
size_t v___x_158_; size_t v___x_159_; uint8_t v___x_160_; 
v___x_158_ = ((size_t)0ULL);
v___x_159_ = lean_usize_of_nat(v___x_156_);
v___x_160_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Match_containsHyp_spec__0(v_hyp_151_, v_premises_154_, v___x_158_, v___x_159_);
return v___x_160_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Match_containsHyp___boxed(lean_object* v_hyp_161_, lean_object* v_m_162_){
_start:
{
uint8_t v_res_163_; lean_object* v_r_164_; 
v_res_163_ = lp_aesop_Aesop_Match_containsHyp(v_hyp_161_, v_m_162_);
lean_dec_ref(v_m_162_);
lean_dec(v_hyp_161_);
v_r_164_ = lean_box(v_res_163_);
return v_r_164_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Option_instBEq_beq___at___00Aesop_Match_containsPatSubst_spec__0(lean_object* v_x_165_, lean_object* v_x_166_){
_start:
{
if (lean_obj_tag(v_x_165_) == 0)
{
if (lean_obj_tag(v_x_166_) == 0)
{
uint8_t v___x_167_; 
v___x_167_ = 1;
return v___x_167_;
}
else
{
uint8_t v___x_168_; 
v___x_168_ = 0;
return v___x_168_;
}
}
else
{
if (lean_obj_tag(v_x_166_) == 0)
{
uint8_t v___x_169_; 
v___x_169_ = 0;
return v___x_169_;
}
else
{
lean_object* v_val_170_; lean_object* v_val_171_; uint8_t v___x_172_; 
v_val_170_ = lean_ctor_get(v_x_165_, 0);
v_val_171_ = lean_ctor_get(v_x_166_, 0);
v___x_172_ = lean_expr_eqv(v_val_170_, v_val_171_);
return v___x_172_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Option_instBEq_beq___at___00Aesop_Match_containsPatSubst_spec__0___boxed(lean_object* v_x_173_, lean_object* v_x_174_){
_start:
{
uint8_t v_res_175_; lean_object* v_r_176_; 
v_res_175_ = lp_aesop_Option_instBEq_beq___at___00Aesop_Match_containsPatSubst_spec__0(v_x_173_, v_x_174_);
lean_dec(v_x_174_);
lean_dec(v_x_173_);
v_r_176_ = lean_box(v_res_175_);
return v_r_176_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Array_isEqvAux___at___00Aesop_Match_containsPatSubst_spec__1___redArg(lean_object* v_xs_177_, lean_object* v_ys_178_, lean_object* v_x_179_){
_start:
{
lean_object* v_zero_180_; uint8_t v_isZero_181_; 
v_zero_180_ = lean_unsigned_to_nat(0u);
v_isZero_181_ = lean_nat_dec_eq(v_x_179_, v_zero_180_);
if (v_isZero_181_ == 1)
{
lean_dec(v_x_179_);
return v_isZero_181_;
}
else
{
lean_object* v_one_182_; lean_object* v_n_183_; lean_object* v___x_184_; lean_object* v___x_185_; uint8_t v___x_186_; 
v_one_182_ = lean_unsigned_to_nat(1u);
v_n_183_ = lean_nat_sub(v_x_179_, v_one_182_);
lean_dec(v_x_179_);
v___x_184_ = lean_array_fget_borrowed(v_xs_177_, v_n_183_);
v___x_185_ = lean_array_fget_borrowed(v_ys_178_, v_n_183_);
v___x_186_ = lp_aesop_Option_instBEq_beq___at___00Aesop_Match_containsPatSubst_spec__0(v___x_184_, v___x_185_);
if (v___x_186_ == 0)
{
lean_dec(v_n_183_);
return v___x_186_;
}
else
{
v_x_179_ = v_n_183_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_isEqvAux___at___00Aesop_Match_containsPatSubst_spec__1___redArg___boxed(lean_object* v_xs_188_, lean_object* v_ys_189_, lean_object* v_x_190_){
_start:
{
uint8_t v_res_191_; lean_object* v_r_192_; 
v_res_191_ = lp_aesop_Array_isEqvAux___at___00Aesop_Match_containsPatSubst_spec__1___redArg(v_xs_188_, v_ys_189_, v_x_190_);
lean_dec_ref(v_ys_189_);
lean_dec_ref(v_xs_188_);
v_r_192_ = lean_box(v_res_191_);
return v_r_192_;
}
}
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Match_containsPatSubst_spec__2(lean_object* v_subst_193_, lean_object* v_as_194_, size_t v_i_195_, size_t v_stop_196_){
_start:
{
uint8_t v___x_197_; 
v___x_197_ = lean_usize_dec_eq(v_i_195_, v_stop_196_);
if (v___x_197_ == 0)
{
lean_object* v___x_198_; lean_object* v_premises_199_; lean_object* v_premises_200_; uint8_t v___x_201_; uint8_t v___y_203_; lean_object* v___x_207_; lean_object* v___x_208_; uint8_t v___x_209_; 
v___x_198_ = lean_array_uget_borrowed(v_as_194_, v_i_195_);
v_premises_199_ = lean_ctor_get(v___x_198_, 0);
v_premises_200_ = lean_ctor_get(v_subst_193_, 0);
v___x_201_ = 1;
v___x_207_ = lean_array_get_size(v_premises_199_);
v___x_208_ = lean_array_get_size(v_premises_200_);
v___x_209_ = lean_nat_dec_eq(v___x_207_, v___x_208_);
if (v___x_209_ == 0)
{
v___y_203_ = v___x_197_;
goto v___jp_202_;
}
else
{
uint8_t v___x_210_; 
v___x_210_ = lp_aesop_Array_isEqvAux___at___00Aesop_Match_containsPatSubst_spec__1___redArg(v_premises_199_, v_premises_200_, v___x_207_);
v___y_203_ = v___x_210_;
goto v___jp_202_;
}
v___jp_202_:
{
if (v___y_203_ == 0)
{
size_t v___x_204_; size_t v___x_205_; 
v___x_204_ = ((size_t)1ULL);
v___x_205_ = lean_usize_add(v_i_195_, v___x_204_);
v_i_195_ = v___x_205_;
goto _start;
}
else
{
return v___x_201_;
}
}
}
else
{
uint8_t v___x_211_; 
v___x_211_ = 0;
return v___x_211_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Match_containsPatSubst_spec__2___boxed(lean_object* v_subst_212_, lean_object* v_as_213_, lean_object* v_i_214_, lean_object* v_stop_215_){
_start:
{
size_t v_i_boxed_216_; size_t v_stop_boxed_217_; uint8_t v_res_218_; lean_object* v_r_219_; 
v_i_boxed_216_ = lean_unbox_usize(v_i_214_);
lean_dec(v_i_214_);
v_stop_boxed_217_ = lean_unbox_usize(v_stop_215_);
lean_dec(v_stop_215_);
v_res_218_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Match_containsPatSubst_spec__2(v_subst_212_, v_as_213_, v_i_boxed_216_, v_stop_boxed_217_);
lean_dec_ref(v_as_213_);
lean_dec_ref(v_subst_212_);
v_r_219_ = lean_box(v_res_218_);
return v_r_219_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Match_containsPatSubst(lean_object* v_subst_220_, lean_object* v_m_221_){
_start:
{
lean_object* v_patInstSubsts_222_; lean_object* v___x_223_; lean_object* v___x_224_; uint8_t v___x_225_; 
v_patInstSubsts_222_ = lean_ctor_get(v_m_221_, 1);
v___x_223_ = lean_unsigned_to_nat(0u);
v___x_224_ = lean_array_get_size(v_patInstSubsts_222_);
v___x_225_ = lean_nat_dec_lt(v___x_223_, v___x_224_);
if (v___x_225_ == 0)
{
return v___x_225_;
}
else
{
if (v___x_225_ == 0)
{
return v___x_225_;
}
else
{
size_t v___x_226_; size_t v___x_227_; uint8_t v___x_228_; 
v___x_226_ = ((size_t)0ULL);
v___x_227_ = lean_usize_of_nat(v___x_224_);
v___x_228_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_Match_containsPatSubst_spec__2(v_subst_220_, v_patInstSubsts_222_, v___x_226_, v___x_227_);
return v___x_228_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Match_containsPatSubst___boxed(lean_object* v_subst_229_, lean_object* v_m_230_){
_start:
{
uint8_t v_res_231_; lean_object* v_r_232_; 
v_res_231_ = lp_aesop_Aesop_Match_containsPatSubst(v_subst_229_, v_m_230_);
lean_dec_ref(v_m_230_);
lean_dec_ref(v_subst_229_);
v_r_232_ = lean_box(v_res_231_);
return v_r_232_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Array_isEqvAux___at___00Aesop_Match_containsPatSubst_spec__1(lean_object* v_xs_233_, lean_object* v_ys_234_, lean_object* v_hsz_235_, lean_object* v_x_236_, lean_object* v_x_237_){
_start:
{
uint8_t v___x_238_; 
v___x_238_ = lp_aesop_Array_isEqvAux___at___00Aesop_Match_containsPatSubst_spec__1___redArg(v_xs_233_, v_ys_234_, v_x_236_);
return v___x_238_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_isEqvAux___at___00Aesop_Match_containsPatSubst_spec__1___boxed(lean_object* v_xs_239_, lean_object* v_ys_240_, lean_object* v_hsz_241_, lean_object* v_x_242_, lean_object* v_x_243_){
_start:
{
uint8_t v_res_244_; lean_object* v_r_245_; 
v_res_244_ = lp_aesop_Array_isEqvAux___at___00Aesop_Match_containsPatSubst_spec__1(v_xs_239_, v_ys_240_, v_hsz_241_, v_x_242_, v_x_243_);
lean_dec_ref(v_ys_240_);
lean_dec_ref(v_xs_239_);
v_r_245_ = lean_box(v_res_244_);
return v_r_245_;
}
}
static lean_object* _init_lp_aesop_panic___at___00Aesop_CompleteMatch_reconstructArgs_spec__0___closed__7(void){
_start:
{
lean_object* v___x_253_; 
v___x_253_ = l_Array_instInhabited(lean_box(0));
return v___x_253_;
}
}
LEAN_EXPORT lean_object* lp_aesop_panic___at___00Aesop_CompleteMatch_reconstructArgs_spec__0(lean_object* v_msg_254_){
_start:
{
lean_object* v___f_255_; lean_object* v___f_256_; lean_object* v___f_257_; lean_object* v___f_258_; lean_object* v___f_259_; lean_object* v___f_260_; lean_object* v___f_261_; lean_object* v___x_262_; lean_object* v___x_263_; lean_object* v___x_264_; lean_object* v___x_265_; lean_object* v___x_266_; lean_object* v___x_267_; lean_object* v___x_268_; 
v___f_255_ = ((lean_object*)(lp_aesop_panic___at___00Aesop_CompleteMatch_reconstructArgs_spec__0___closed__0));
v___f_256_ = ((lean_object*)(lp_aesop_panic___at___00Aesop_CompleteMatch_reconstructArgs_spec__0___closed__1));
v___f_257_ = ((lean_object*)(lp_aesop_panic___at___00Aesop_CompleteMatch_reconstructArgs_spec__0___closed__2));
v___f_258_ = ((lean_object*)(lp_aesop_panic___at___00Aesop_CompleteMatch_reconstructArgs_spec__0___closed__3));
v___f_259_ = ((lean_object*)(lp_aesop_panic___at___00Aesop_CompleteMatch_reconstructArgs_spec__0___closed__4));
v___f_260_ = ((lean_object*)(lp_aesop_panic___at___00Aesop_CompleteMatch_reconstructArgs_spec__0___closed__5));
v___f_261_ = ((lean_object*)(lp_aesop_panic___at___00Aesop_CompleteMatch_reconstructArgs_spec__0___closed__6));
v___x_262_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_262_, 0, v___f_255_);
lean_ctor_set(v___x_262_, 1, v___f_256_);
v___x_263_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_263_, 0, v___x_262_);
lean_ctor_set(v___x_263_, 1, v___f_257_);
lean_ctor_set(v___x_263_, 2, v___f_258_);
lean_ctor_set(v___x_263_, 3, v___f_259_);
lean_ctor_set(v___x_263_, 4, v___f_260_);
v___x_264_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_264_, 0, v___x_263_);
lean_ctor_set(v___x_264_, 1, v___f_261_);
v___x_265_ = lean_obj_once(&lp_aesop_panic___at___00Aesop_CompleteMatch_reconstructArgs_spec__0___closed__7, &lp_aesop_panic___at___00Aesop_CompleteMatch_reconstructArgs_spec__0___closed__7_once, _init_lp_aesop_panic___at___00Aesop_CompleteMatch_reconstructArgs_spec__0___closed__7);
v___x_266_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_266_, 0, v___x_265_);
lean_ctor_set(v___x_266_, 1, v___x_265_);
v___x_267_ = l_instInhabitedOfMonad___redArg(v___x_264_, v___x_266_);
v___x_268_ = lean_panic_fn_borrowed(v___x_267_, v_msg_254_);
lean_dec(v___x_267_);
return v___x_268_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_CompleteMatch_reconstructArgs_spec__3___redArg(lean_object* v___x_269_, lean_object* v_range_270_, lean_object* v_b_271_, lean_object* v_i_272_){
_start:
{
lean_object* v_stop_273_; lean_object* v_step_274_; uint8_t v___x_275_; 
v_stop_273_ = lean_ctor_get(v_range_270_, 1);
v_step_274_ = lean_ctor_get(v_range_270_, 2);
v___x_275_ = lean_nat_dec_lt(v_i_272_, v_stop_273_);
if (v___x_275_ == 0)
{
lean_dec(v_i_272_);
return v_b_271_;
}
else
{
lean_object* v___x_276_; lean_object* v___x_277_; lean_object* v___x_278_; 
v___x_276_ = lp_aesop_Aesop_Substitution_findLevel_x3f(v_i_272_, v___x_269_);
v___x_277_ = lean_array_push(v_b_271_, v___x_276_);
v___x_278_ = lean_nat_add(v_i_272_, v_step_274_);
lean_dec(v_i_272_);
v_b_271_ = v___x_277_;
v_i_272_ = v___x_278_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_CompleteMatch_reconstructArgs_spec__3___redArg___boxed(lean_object* v___x_280_, lean_object* v_range_281_, lean_object* v_b_282_, lean_object* v_i_283_){
_start:
{
lean_object* v_res_284_; 
v_res_284_ = lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_CompleteMatch_reconstructArgs_spec__3___redArg(v___x_280_, v_range_281_, v_b_282_, v_i_283_);
lean_dec_ref(v_range_281_);
lean_dec_ref(v___x_280_);
return v_res_284_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_CompleteMatch_reconstructArgs_spec__1(lean_object* v_as_285_, size_t v_sz_286_, size_t v_i_287_, lean_object* v_b_288_){
_start:
{
uint8_t v___x_289_; 
v___x_289_ = lean_usize_dec_lt(v_i_287_, v_sz_286_);
if (v___x_289_ == 0)
{
return v_b_288_;
}
else
{
lean_object* v_a_290_; lean_object* v_subst_291_; lean_object* v___x_292_; size_t v___x_293_; size_t v___x_294_; 
v_a_290_ = lean_array_uget_borrowed(v_as_285_, v_i_287_);
v_subst_291_ = lean_ctor_get(v_a_290_, 0);
lean_inc_ref(v_subst_291_);
v___x_292_ = lp_aesop_Aesop_Substitution_mergeCompatible(v_subst_291_, v_b_288_);
lean_dec_ref(v_b_288_);
v___x_293_ = ((size_t)1ULL);
v___x_294_ = lean_usize_add(v_i_287_, v___x_293_);
v_i_287_ = v___x_294_;
v_b_288_ = v___x_292_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_CompleteMatch_reconstructArgs_spec__1___boxed(lean_object* v_as_296_, lean_object* v_sz_297_, lean_object* v_i_298_, lean_object* v_b_299_){
_start:
{
size_t v_sz_boxed_300_; size_t v_i_boxed_301_; lean_object* v_res_302_; 
v_sz_boxed_300_ = lean_unbox_usize(v_sz_297_);
lean_dec(v_sz_297_);
v_i_boxed_301_ = lean_unbox_usize(v_i_298_);
lean_dec(v_i_298_);
v_res_302_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_CompleteMatch_reconstructArgs_spec__1(v_as_296_, v_sz_boxed_300_, v_i_boxed_301_, v_b_299_);
lean_dec_ref(v_as_296_);
return v_res_302_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_CompleteMatch_reconstructArgs_spec__2___redArg(lean_object* v___x_303_, lean_object* v_range_304_, lean_object* v_b_305_, lean_object* v_i_306_){
_start:
{
lean_object* v_stop_307_; lean_object* v_step_308_; uint8_t v___x_309_; 
v_stop_307_ = lean_ctor_get(v_range_304_, 1);
v_step_308_ = lean_ctor_get(v_range_304_, 2);
v___x_309_ = lean_nat_dec_lt(v_i_306_, v_stop_307_);
if (v___x_309_ == 0)
{
lean_dec(v_i_306_);
return v_b_305_;
}
else
{
lean_object* v___x_310_; lean_object* v___x_311_; lean_object* v___x_312_; 
v___x_310_ = lp_aesop_Aesop_Substitution_find_x3f(v_i_306_, v___x_303_);
v___x_311_ = lean_array_push(v_b_305_, v___x_310_);
v___x_312_ = lean_nat_add(v_i_306_, v_step_308_);
lean_dec(v_i_306_);
v_b_305_ = v___x_311_;
v_i_306_ = v___x_312_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_CompleteMatch_reconstructArgs_spec__2___redArg___boxed(lean_object* v___x_314_, lean_object* v_range_315_, lean_object* v_b_316_, lean_object* v_i_317_){
_start:
{
lean_object* v_res_318_; 
v_res_318_ = lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_CompleteMatch_reconstructArgs_spec__2___redArg(v___x_314_, v_range_315_, v_b_316_, v_i_317_);
lean_dec_ref(v_range_315_);
lean_dec_ref(v___x_314_);
return v_res_318_;
}
}
static lean_object* _init_lp_aesop_Aesop_CompleteMatch_reconstructArgs___closed__3(void){
_start:
{
lean_object* v___x_322_; lean_object* v___x_323_; lean_object* v___x_324_; lean_object* v___x_325_; lean_object* v___x_326_; lean_object* v___x_327_; 
v___x_322_ = ((lean_object*)(lp_aesop_Aesop_CompleteMatch_reconstructArgs___closed__2));
v___x_323_ = lean_unsigned_to_nat(2u);
v___x_324_ = lean_unsigned_to_nat(76u);
v___x_325_ = ((lean_object*)(lp_aesop_Aesop_CompleteMatch_reconstructArgs___closed__1));
v___x_326_ = ((lean_object*)(lp_aesop_Aesop_CompleteMatch_reconstructArgs___closed__0));
v___x_327_ = l_mkPanicMessageWithDecl(v___x_326_, v___x_325_, v___x_324_, v___x_323_, v___x_322_);
return v___x_327_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_CompleteMatch_reconstructArgs(lean_object* v_r_328_, lean_object* v_m_329_){
_start:
{
lean_object* v_toForwardRuleInfo_330_; lean_object* v_numPremises_331_; lean_object* v_numLevelParams_332_; lean_object* v_slotClusters_333_; lean_object* v___x_334_; lean_object* v___x_335_; uint8_t v___x_336_; 
v_toForwardRuleInfo_330_ = lean_ctor_get(v_r_328_, 0);
lean_inc_ref(v_toForwardRuleInfo_330_);
lean_dec_ref(v_r_328_);
v_numPremises_331_ = lean_ctor_get(v_toForwardRuleInfo_330_, 0);
lean_inc(v_numPremises_331_);
v_numLevelParams_332_ = lean_ctor_get(v_toForwardRuleInfo_330_, 1);
lean_inc(v_numLevelParams_332_);
v_slotClusters_333_ = lean_ctor_get(v_toForwardRuleInfo_330_, 2);
lean_inc_ref(v_slotClusters_333_);
lean_dec_ref(v_toForwardRuleInfo_330_);
v___x_334_ = lean_array_get_size(v_m_329_);
v___x_335_ = lean_array_get_size(v_slotClusters_333_);
lean_dec_ref(v_slotClusters_333_);
v___x_336_ = lean_nat_dec_eq(v___x_334_, v___x_335_);
if (v___x_336_ == 0)
{
lean_object* v___x_337_; lean_object* v___x_338_; 
lean_dec(v_numLevelParams_332_);
lean_dec(v_numPremises_331_);
v___x_337_ = lean_obj_once(&lp_aesop_Aesop_CompleteMatch_reconstructArgs___closed__3, &lp_aesop_Aesop_CompleteMatch_reconstructArgs___closed__3_once, _init_lp_aesop_Aesop_CompleteMatch_reconstructArgs___closed__3);
v___x_338_ = lp_aesop_panic___at___00Aesop_CompleteMatch_reconstructArgs_spec__0(v___x_337_);
return v___x_338_;
}
else
{
lean_object* v_subst_339_; size_t v_sz_340_; size_t v___x_341_; lean_object* v___x_342_; lean_object* v___x_343_; lean_object* v___x_344_; lean_object* v___x_345_; lean_object* v___x_346_; lean_object* v___x_347_; lean_object* v___x_348_; lean_object* v___x_349_; lean_object* v___x_350_; lean_object* v___x_351_; 
lean_inc(v_numLevelParams_332_);
lean_inc(v_numPremises_331_);
v_subst_339_ = lp_aesop_Aesop_Substitution_empty(v_numPremises_331_, v_numLevelParams_332_);
v_sz_340_ = lean_array_size(v_m_329_);
v___x_341_ = ((size_t)0ULL);
v___x_342_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_CompleteMatch_reconstructArgs_spec__1(v_m_329_, v_sz_340_, v___x_341_, v_subst_339_);
v___x_343_ = lean_mk_empty_array_with_capacity(v_numPremises_331_);
v___x_344_ = lean_unsigned_to_nat(0u);
v___x_345_ = lean_unsigned_to_nat(1u);
v___x_346_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_346_, 0, v___x_344_);
lean_ctor_set(v___x_346_, 1, v_numPremises_331_);
lean_ctor_set(v___x_346_, 2, v___x_345_);
v___x_347_ = lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_CompleteMatch_reconstructArgs_spec__2___redArg(v___x_342_, v___x_346_, v___x_343_, v___x_344_);
lean_dec_ref_known(v___x_346_, 3);
v___x_348_ = lean_mk_empty_array_with_capacity(v_numLevelParams_332_);
v___x_349_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_349_, 0, v___x_344_);
lean_ctor_set(v___x_349_, 1, v_numLevelParams_332_);
lean_ctor_set(v___x_349_, 2, v___x_345_);
v___x_350_ = lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_CompleteMatch_reconstructArgs_spec__3___redArg(v___x_342_, v___x_349_, v___x_348_, v___x_344_);
lean_dec_ref_known(v___x_349_, 3);
lean_dec_ref(v___x_342_);
v___x_351_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_351_, 0, v___x_347_);
lean_ctor_set(v___x_351_, 1, v___x_350_);
return v___x_351_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_CompleteMatch_reconstructArgs___boxed(lean_object* v_r_352_, lean_object* v_m_353_){
_start:
{
lean_object* v_res_354_; 
v_res_354_ = lp_aesop_Aesop_CompleteMatch_reconstructArgs(v_r_352_, v_m_353_);
lean_dec_ref(v_m_353_);
return v_res_354_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_CompleteMatch_reconstructArgs_spec__2(lean_object* v___x_355_, lean_object* v_range_356_, lean_object* v_b_357_, lean_object* v_i_358_, lean_object* v_hs_359_, lean_object* v_hl_360_){
_start:
{
lean_object* v___x_361_; 
v___x_361_ = lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_CompleteMatch_reconstructArgs_spec__2___redArg(v___x_355_, v_range_356_, v_b_357_, v_i_358_);
return v___x_361_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_CompleteMatch_reconstructArgs_spec__2___boxed(lean_object* v___x_362_, lean_object* v_range_363_, lean_object* v_b_364_, lean_object* v_i_365_, lean_object* v_hs_366_, lean_object* v_hl_367_){
_start:
{
lean_object* v_res_368_; 
v_res_368_ = lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_CompleteMatch_reconstructArgs_spec__2(v___x_362_, v_range_363_, v_b_364_, v_i_365_, v_hs_366_, v_hl_367_);
lean_dec_ref(v_range_363_);
lean_dec_ref(v___x_362_);
return v_res_368_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_CompleteMatch_reconstructArgs_spec__3(lean_object* v___x_369_, lean_object* v_range_370_, lean_object* v_b_371_, lean_object* v_i_372_, lean_object* v_hs_373_, lean_object* v_hl_374_){
_start:
{
lean_object* v___x_375_; 
v___x_375_ = lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_CompleteMatch_reconstructArgs_spec__3___redArg(v___x_369_, v_range_370_, v_b_371_, v_i_372_);
return v___x_375_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_CompleteMatch_reconstructArgs_spec__3___boxed(lean_object* v___x_376_, lean_object* v_range_377_, lean_object* v_b_378_, lean_object* v_i_379_, lean_object* v_hs_380_, lean_object* v_hl_381_){
_start:
{
lean_object* v_res_382_; 
v_res_382_ = lp_aesop___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Aesop_CompleteMatch_reconstructArgs_spec__3(v___x_376_, v_range_377_, v_b_378_, v_i_379_, v_hs_380_, v_hl_381_);
lean_dec_ref(v_range_377_);
lean_dec_ref(v___x_376_);
return v_res_382_;
}
}
static lean_object* _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CompleteMatch_toMessageData_spec__0___closed__1(void){
_start:
{
lean_object* v___x_384_; lean_object* v___x_385_; 
v___x_384_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CompleteMatch_toMessageData_spec__0___closed__0));
v___x_385_ = l_Lean_stringToMessageData(v___x_384_);
return v___x_385_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CompleteMatch_toMessageData_spec__0(size_t v_sz_386_, size_t v_i_387_, lean_object* v_bs_388_){
_start:
{
uint8_t v___x_389_; 
v___x_389_ = lean_usize_dec_lt(v_i_387_, v_sz_386_);
if (v___x_389_ == 0)
{
return v_bs_388_;
}
else
{
lean_object* v_v_390_; lean_object* v___x_391_; lean_object* v_bs_x27_392_; lean_object* v___y_394_; 
v_v_390_ = lean_array_uget(v_bs_388_, v_i_387_);
v___x_391_ = lean_unsigned_to_nat(0u);
v_bs_x27_392_ = lean_array_uset(v_bs_388_, v_i_387_, v___x_391_);
if (lean_obj_tag(v_v_390_) == 0)
{
lean_object* v___x_399_; 
v___x_399_ = lean_obj_once(&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CompleteMatch_toMessageData_spec__0___closed__1, &lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CompleteMatch_toMessageData_spec__0___closed__1_once, _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CompleteMatch_toMessageData_spec__0___closed__1);
v___y_394_ = v___x_399_;
goto v___jp_393_;
}
else
{
lean_object* v_val_400_; lean_object* v___x_401_; 
v_val_400_ = lean_ctor_get(v_v_390_, 0);
lean_inc(v_val_400_);
lean_dec_ref_known(v_v_390_, 1);
v___x_401_ = l_Lean_MessageData_ofExpr(v_val_400_);
v___y_394_ = v___x_401_;
goto v___jp_393_;
}
v___jp_393_:
{
size_t v___x_395_; size_t v___x_396_; lean_object* v___x_397_; 
v___x_395_ = ((size_t)1ULL);
v___x_396_ = lean_usize_add(v_i_387_, v___x_395_);
v___x_397_ = lean_array_uset(v_bs_x27_392_, v_i_387_, v___y_394_);
v_i_387_ = v___x_396_;
v_bs_388_ = v___x_397_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CompleteMatch_toMessageData_spec__0___boxed(lean_object* v_sz_402_, lean_object* v_i_403_, lean_object* v_bs_404_){
_start:
{
size_t v_sz_boxed_405_; size_t v_i_boxed_406_; lean_object* v_res_407_; 
v_sz_boxed_405_ = lean_unbox_usize(v_sz_402_);
lean_dec(v_sz_402_);
v_i_boxed_406_ = lean_unbox_usize(v_i_403_);
lean_dec(v_i_403_);
v_res_407_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CompleteMatch_toMessageData_spec__0(v_sz_boxed_405_, v_i_boxed_406_, v_bs_404_);
return v_res_407_;
}
}
LEAN_EXPORT lean_object* lp_aesop_List_mapTR_loop___at___00Aesop_CompleteMatch_toMessageData_spec__1(lean_object* v_a_408_, lean_object* v_a_409_){
_start:
{
if (lean_obj_tag(v_a_408_) == 0)
{
lean_object* v___x_410_; 
v___x_410_ = l_List_reverse___redArg(v_a_409_);
return v___x_410_;
}
else
{
lean_object* v_head_411_; lean_object* v_tail_412_; lean_object* v___x_414_; uint8_t v_isShared_415_; uint8_t v_isSharedCheck_420_; 
v_head_411_ = lean_ctor_get(v_a_408_, 0);
v_tail_412_ = lean_ctor_get(v_a_408_, 1);
v_isSharedCheck_420_ = !lean_is_exclusive(v_a_408_);
if (v_isSharedCheck_420_ == 0)
{
v___x_414_ = v_a_408_;
v_isShared_415_ = v_isSharedCheck_420_;
goto v_resetjp_413_;
}
else
{
lean_inc(v_tail_412_);
lean_inc(v_head_411_);
lean_dec(v_a_408_);
v___x_414_ = lean_box(0);
v_isShared_415_ = v_isSharedCheck_420_;
goto v_resetjp_413_;
}
v_resetjp_413_:
{
lean_object* v___x_417_; 
if (v_isShared_415_ == 0)
{
lean_ctor_set(v___x_414_, 1, v_a_409_);
v___x_417_ = v___x_414_;
goto v_reusejp_416_;
}
else
{
lean_object* v_reuseFailAlloc_419_; 
v_reuseFailAlloc_419_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_419_, 0, v_head_411_);
lean_ctor_set(v_reuseFailAlloc_419_, 1, v_a_409_);
v___x_417_ = v_reuseFailAlloc_419_;
goto v_reusejp_416_;
}
v_reusejp_416_:
{
v_a_408_ = v_tail_412_;
v_a_409_ = v___x_417_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_CompleteMatch_toMessageData(lean_object* v_r_421_, lean_object* v_m_422_){
_start:
{
lean_object* v___x_423_; lean_object* v_fst_424_; size_t v_sz_425_; size_t v___x_426_; lean_object* v___x_427_; lean_object* v___x_428_; lean_object* v___x_429_; lean_object* v___x_430_; lean_object* v___x_431_; 
v___x_423_ = lp_aesop_Aesop_CompleteMatch_reconstructArgs(v_r_421_, v_m_422_);
v_fst_424_ = lean_ctor_get(v___x_423_, 0);
lean_inc(v_fst_424_);
lean_dec_ref(v___x_423_);
v_sz_425_ = lean_array_size(v_fst_424_);
v___x_426_ = ((size_t)0ULL);
v___x_427_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CompleteMatch_toMessageData_spec__0(v_sz_425_, v___x_426_, v_fst_424_);
v___x_428_ = lean_array_to_list(v___x_427_);
v___x_429_ = lean_box(0);
v___x_430_ = lp_aesop_List_mapTR_loop___at___00Aesop_CompleteMatch_toMessageData_spec__1(v___x_428_, v___x_429_);
v___x_431_ = l_Lean_MessageData_ofList(v___x_430_);
return v___x_431_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_CompleteMatch_toMessageData___boxed(lean_object* v_r_432_, lean_object* v_m_433_){
_start:
{
lean_object* v_res_434_; 
v_res_434_ = lp_aesop_Aesop_CompleteMatch_toMessageData(v_r_432_, v_m_433_);
lean_dec_ref(v_m_433_);
return v_res_434_;
}
}
static lean_object* _init_lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__1(void){
_start:
{
lean_object* v___x_436_; lean_object* v___x_437_; 
v___x_436_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__0));
v___x_437_ = l_Lean_stringToMessageData(v___x_436_);
return v___x_437_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0(lean_object* v_m_452_){
_start:
{
lean_object* v_rule_453_; lean_object* v_name_454_; lean_object* v_match_455_; lean_object* v___x_457_; uint8_t v_isShared_458_; uint8_t v_isSharedCheck_503_; 
v_rule_453_ = lean_ctor_get(v_m_452_, 0);
lean_inc_ref(v_rule_453_);
v_name_454_ = lean_ctor_get(v_rule_453_, 1);
v_match_455_ = lean_ctor_get(v_m_452_, 1);
v_isSharedCheck_503_ = !lean_is_exclusive(v_m_452_);
if (v_isSharedCheck_503_ == 0)
{
lean_object* v_unused_504_; 
v_unused_504_ = lean_ctor_get(v_m_452_, 0);
lean_dec(v_unused_504_);
v___x_457_ = v_m_452_;
v_isShared_458_ = v_isSharedCheck_503_;
goto v_resetjp_456_;
}
else
{
lean_inc(v_match_455_);
lean_dec(v_m_452_);
v___x_457_ = lean_box(0);
v_isShared_458_ = v_isSharedCheck_503_;
goto v_resetjp_456_;
}
v_resetjp_456_:
{
lean_object* v_name_459_; uint8_t v_builder_460_; uint8_t v_phase_461_; uint8_t v_scope_462_; lean_object* v___y_464_; lean_object* v___y_465_; lean_object* v___y_466_; lean_object* v___y_481_; lean_object* v___y_482_; lean_object* v___y_483_; lean_object* v___y_489_; 
v_name_459_ = lean_ctor_get(v_name_454_, 0);
v_builder_460_ = lean_ctor_get_uint8(v_name_454_, sizeof(void*)*1 + 8);
v_phase_461_ = lean_ctor_get_uint8(v_name_454_, sizeof(void*)*1 + 9);
v_scope_462_ = lean_ctor_get_uint8(v_name_454_, sizeof(void*)*1 + 10);
switch(v_phase_461_)
{
case 0:
{
lean_object* v___x_500_; 
v___x_500_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__13));
v___y_489_ = v___x_500_;
goto v___jp_488_;
}
case 1:
{
lean_object* v___x_501_; 
v___x_501_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__14));
v___y_489_ = v___x_501_;
goto v___jp_488_;
}
default: 
{
lean_object* v___x_502_; 
v___x_502_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__15));
v___y_489_ = v___x_502_;
goto v___jp_488_;
}
}
v___jp_463_:
{
lean_object* v___x_467_; lean_object* v___x_468_; uint8_t v___x_469_; lean_object* v___x_470_; lean_object* v___x_471_; lean_object* v___x_472_; lean_object* v___x_473_; lean_object* v___x_474_; lean_object* v___x_476_; 
v___x_467_ = lean_string_append(v___y_464_, v___y_466_);
v___x_468_ = lean_string_append(v___x_467_, v___y_465_);
v___x_469_ = 1;
lean_inc(v_name_459_);
v___x_470_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_name_459_, v___x_469_);
v___x_471_ = lean_string_append(v___x_468_, v___x_470_);
lean_dec_ref(v___x_470_);
v___x_472_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_472_, 0, v___x_471_);
v___x_473_ = l_Lean_MessageData_ofFormat(v___x_472_);
v___x_474_ = lean_obj_once(&lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__1, &lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__1_once, _init_lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__1);
if (v_isShared_458_ == 0)
{
lean_ctor_set_tag(v___x_457_, 7);
lean_ctor_set(v___x_457_, 1, v___x_474_);
lean_ctor_set(v___x_457_, 0, v___x_473_);
v___x_476_ = v___x_457_;
goto v_reusejp_475_;
}
else
{
lean_object* v_reuseFailAlloc_479_; 
v_reuseFailAlloc_479_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_479_, 0, v___x_473_);
lean_ctor_set(v_reuseFailAlloc_479_, 1, v___x_474_);
v___x_476_ = v_reuseFailAlloc_479_;
goto v_reusejp_475_;
}
v_reusejp_475_:
{
lean_object* v___x_477_; lean_object* v___x_478_; 
v___x_477_ = lp_aesop_Aesop_CompleteMatch_toMessageData(v_rule_453_, v_match_455_);
lean_dec_ref(v_match_455_);
v___x_478_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_478_, 0, v___x_476_);
lean_ctor_set(v___x_478_, 1, v___x_477_);
return v___x_478_;
}
}
v___jp_480_:
{
lean_object* v___x_484_; lean_object* v___x_485_; 
v___x_484_ = lean_string_append(v___y_481_, v___y_483_);
v___x_485_ = lean_string_append(v___x_484_, v___y_482_);
if (v_scope_462_ == 0)
{
lean_object* v___x_486_; 
v___x_486_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__2));
v___y_464_ = v___x_485_;
v___y_465_ = v___y_482_;
v___y_466_ = v___x_486_;
goto v___jp_463_;
}
else
{
lean_object* v___x_487_; 
v___x_487_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__3));
v___y_464_ = v___x_485_;
v___y_465_ = v___y_482_;
v___y_466_ = v___x_487_;
goto v___jp_463_;
}
}
v___jp_488_:
{
lean_object* v___x_490_; lean_object* v___x_491_; 
v___x_490_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__4));
lean_inc_ref(v___y_489_);
v___x_491_ = lean_string_append(v___y_489_, v___x_490_);
switch(v_builder_460_)
{
case 0:
{
lean_object* v___x_492_; 
v___x_492_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__5));
v___y_481_ = v___x_491_;
v___y_482_ = v___x_490_;
v___y_483_ = v___x_492_;
goto v___jp_480_;
}
case 1:
{
lean_object* v___x_493_; 
v___x_493_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__6));
v___y_481_ = v___x_491_;
v___y_482_ = v___x_490_;
v___y_483_ = v___x_493_;
goto v___jp_480_;
}
case 2:
{
lean_object* v___x_494_; 
v___x_494_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__7));
v___y_481_ = v___x_491_;
v___y_482_ = v___x_490_;
v___y_483_ = v___x_494_;
goto v___jp_480_;
}
case 3:
{
lean_object* v___x_495_; 
v___x_495_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__8));
v___y_481_ = v___x_491_;
v___y_482_ = v___x_490_;
v___y_483_ = v___x_495_;
goto v___jp_480_;
}
case 4:
{
lean_object* v___x_496_; 
v___x_496_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__9));
v___y_481_ = v___x_491_;
v___y_482_ = v___x_490_;
v___y_483_ = v___x_496_;
goto v___jp_480_;
}
case 5:
{
lean_object* v___x_497_; 
v___x_497_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__10));
v___y_481_ = v___x_491_;
v___y_482_ = v___x_490_;
v___y_483_ = v___x_497_;
goto v___jp_480_;
}
case 6:
{
lean_object* v___x_498_; 
v___x_498_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__11));
v___y_481_ = v___x_491_;
v___y_482_ = v___x_490_;
v___y_483_ = v___x_498_;
goto v___jp_480_;
}
default: 
{
lean_object* v___x_499_; 
v___x_499_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__12));
v___y_481_ = v___x_491_;
v___y_482_ = v___x_490_;
v___y_483_ = v___x_499_;
goto v___jp_480_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_foldHypsM___redArg___lam__0(lean_object* v_f_507_, lean_object* v_toPure_508_, lean_object* v_x_509_, lean_object* v_x_510_){
_start:
{
if (lean_obj_tag(v_x_510_) == 1)
{
lean_object* v_val_511_; lean_object* v___x_512_; 
v_val_511_ = lean_ctor_get(v_x_510_, 0);
v___x_512_ = l_Lean_Expr_consumeMData(v_val_511_);
if (lean_obj_tag(v___x_512_) == 1)
{
lean_object* v_fvarId_513_; lean_object* v___x_514_; 
lean_dec(v_toPure_508_);
v_fvarId_513_ = lean_ctor_get(v___x_512_, 0);
lean_inc(v_fvarId_513_);
lean_dec_ref_known(v___x_512_, 1);
v___x_514_ = lean_apply_2(v_f_507_, v_x_509_, v_fvarId_513_);
return v___x_514_;
}
else
{
lean_object* v___x_515_; 
lean_dec_ref(v___x_512_);
lean_dec(v_f_507_);
v___x_515_ = lean_apply_2(v_toPure_508_, lean_box(0), v_x_509_);
return v___x_515_;
}
}
else
{
lean_object* v___x_516_; 
lean_dec(v_f_507_);
v___x_516_ = lean_apply_2(v_toPure_508_, lean_box(0), v_x_509_);
return v___x_516_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_foldHypsM___redArg___lam__0___boxed(lean_object* v_f_517_, lean_object* v_toPure_518_, lean_object* v_x_519_, lean_object* v_x_520_){
_start:
{
lean_object* v_res_521_; 
v_res_521_ = lp_aesop_Aesop_ForwardRuleMatch_foldHypsM___redArg___lam__0(v_f_517_, v_toPure_518_, v_x_519_, v_x_520_);
lean_dec(v_x_520_);
return v_res_521_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_foldHypsM___redArg___lam__1(lean_object* v_toPure_522_, lean_object* v_inst_523_, lean_object* v___f_524_, lean_object* v_s_525_, lean_object* v_cm_526_){
_start:
{
lean_object* v_subst_527_; lean_object* v_premises_528_; lean_object* v___x_529_; lean_object* v___x_530_; uint8_t v___x_531_; 
v_subst_527_ = lean_ctor_get(v_cm_526_, 0);
lean_inc_ref(v_subst_527_);
lean_dec_ref(v_cm_526_);
v_premises_528_ = lean_ctor_get(v_subst_527_, 0);
lean_inc_ref(v_premises_528_);
lean_dec_ref(v_subst_527_);
v___x_529_ = lean_unsigned_to_nat(0u);
v___x_530_ = lean_array_get_size(v_premises_528_);
v___x_531_ = lean_nat_dec_lt(v___x_529_, v___x_530_);
if (v___x_531_ == 0)
{
lean_object* v___x_532_; 
lean_dec_ref(v_premises_528_);
lean_dec(v___f_524_);
lean_dec_ref(v_inst_523_);
v___x_532_ = lean_apply_2(v_toPure_522_, lean_box(0), v_s_525_);
return v___x_532_;
}
else
{
uint8_t v___x_533_; 
v___x_533_ = lean_nat_dec_le(v___x_530_, v___x_530_);
if (v___x_533_ == 0)
{
if (v___x_531_ == 0)
{
lean_object* v___x_534_; 
lean_dec_ref(v_premises_528_);
lean_dec(v___f_524_);
lean_dec_ref(v_inst_523_);
v___x_534_ = lean_apply_2(v_toPure_522_, lean_box(0), v_s_525_);
return v___x_534_;
}
else
{
size_t v___x_535_; size_t v___x_536_; lean_object* v___x_537_; 
lean_dec(v_toPure_522_);
v___x_535_ = ((size_t)0ULL);
v___x_536_ = lean_usize_of_nat(v___x_530_);
v___x_537_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v_inst_523_, v___f_524_, v_premises_528_, v___x_535_, v___x_536_, v_s_525_);
return v___x_537_;
}
}
else
{
size_t v___x_538_; size_t v___x_539_; lean_object* v___x_540_; 
lean_dec(v_toPure_522_);
v___x_538_ = ((size_t)0ULL);
v___x_539_ = lean_usize_of_nat(v___x_530_);
v___x_540_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v_inst_523_, v___f_524_, v_premises_528_, v___x_538_, v___x_539_, v_s_525_);
return v___x_540_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_foldHypsM___redArg(lean_object* v_inst_541_, lean_object* v_f_542_, lean_object* v_init_543_, lean_object* v_m_544_){
_start:
{
lean_object* v_toApplicative_545_; lean_object* v_toPure_546_; lean_object* v_match_547_; lean_object* v___x_548_; lean_object* v___x_549_; uint8_t v___x_550_; 
v_toApplicative_545_ = lean_ctor_get(v_inst_541_, 0);
v_toPure_546_ = lean_ctor_get(v_toApplicative_545_, 1);
v_match_547_ = lean_ctor_get(v_m_544_, 1);
lean_inc_ref(v_match_547_);
lean_dec_ref(v_m_544_);
v___x_548_ = lean_unsigned_to_nat(0u);
v___x_549_ = lean_array_get_size(v_match_547_);
v___x_550_ = lean_nat_dec_lt(v___x_548_, v___x_549_);
if (v___x_550_ == 0)
{
lean_object* v___x_551_; 
lean_inc(v_toPure_546_);
lean_dec_ref(v_match_547_);
lean_dec(v_f_542_);
lean_dec_ref(v_inst_541_);
v___x_551_ = lean_apply_2(v_toPure_546_, lean_box(0), v_init_543_);
return v___x_551_;
}
else
{
lean_object* v___f_552_; lean_object* v___f_553_; uint8_t v___x_554_; 
lean_inc_n(v_toPure_546_, 2);
v___f_552_ = lean_alloc_closure((void*)(lp_aesop_Aesop_ForwardRuleMatch_foldHypsM___redArg___lam__0___boxed), 4, 2);
lean_closure_set(v___f_552_, 0, v_f_542_);
lean_closure_set(v___f_552_, 1, v_toPure_546_);
lean_inc_ref(v_inst_541_);
v___f_553_ = lean_alloc_closure((void*)(lp_aesop_Aesop_ForwardRuleMatch_foldHypsM___redArg___lam__1), 5, 3);
lean_closure_set(v___f_553_, 0, v_toPure_546_);
lean_closure_set(v___f_553_, 1, v_inst_541_);
lean_closure_set(v___f_553_, 2, v___f_552_);
v___x_554_ = lean_nat_dec_le(v___x_549_, v___x_549_);
if (v___x_554_ == 0)
{
if (v___x_550_ == 0)
{
lean_object* v___x_555_; 
lean_inc(v_toPure_546_);
lean_dec_ref(v___f_553_);
lean_dec_ref(v_match_547_);
lean_dec_ref(v_inst_541_);
v___x_555_ = lean_apply_2(v_toPure_546_, lean_box(0), v_init_543_);
return v___x_555_;
}
else
{
size_t v___x_556_; size_t v___x_557_; lean_object* v___x_558_; 
v___x_556_ = ((size_t)0ULL);
v___x_557_ = lean_usize_of_nat(v___x_549_);
v___x_558_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v_inst_541_, v___f_553_, v_match_547_, v___x_556_, v___x_557_, v_init_543_);
return v___x_558_;
}
}
else
{
size_t v___x_559_; size_t v___x_560_; lean_object* v___x_561_; 
v___x_559_ = ((size_t)0ULL);
v___x_560_ = lean_usize_of_nat(v___x_549_);
v___x_561_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v_inst_541_, v___f_553_, v_match_547_, v___x_559_, v___x_560_, v_init_543_);
return v___x_561_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_foldHypsM(lean_object* v_M_562_, lean_object* v_00_u03c3_563_, lean_object* v_inst_564_, lean_object* v_f_565_, lean_object* v_init_566_, lean_object* v_m_567_){
_start:
{
lean_object* v___x_568_; 
v___x_568_ = lp_aesop_Aesop_ForwardRuleMatch_foldHypsM___redArg(v_inst_564_, v_f_565_, v_init_566_, v_m_567_);
return v___x_568_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_foldHyps_spec__0_spec__0___redArg(lean_object* v_f_569_, lean_object* v_as_570_, size_t v_i_571_, size_t v_stop_572_, lean_object* v_b_573_){
_start:
{
lean_object* v___y_575_; uint8_t v___x_579_; 
v___x_579_ = lean_usize_dec_eq(v_i_571_, v_stop_572_);
if (v___x_579_ == 0)
{
lean_object* v___x_580_; 
v___x_580_ = lean_array_uget_borrowed(v_as_570_, v_i_571_);
if (lean_obj_tag(v___x_580_) == 1)
{
lean_object* v_val_581_; lean_object* v___x_582_; 
v_val_581_ = lean_ctor_get(v___x_580_, 0);
v___x_582_ = l_Lean_Expr_consumeMData(v_val_581_);
if (lean_obj_tag(v___x_582_) == 1)
{
lean_object* v_fvarId_583_; lean_object* v___x_584_; 
v_fvarId_583_ = lean_ctor_get(v___x_582_, 0);
lean_inc(v_fvarId_583_);
lean_dec_ref_known(v___x_582_, 1);
lean_inc(v_f_569_);
v___x_584_ = lean_apply_2(v_f_569_, v_b_573_, v_fvarId_583_);
v___y_575_ = v___x_584_;
goto v___jp_574_;
}
else
{
lean_dec_ref(v___x_582_);
v___y_575_ = v_b_573_;
goto v___jp_574_;
}
}
else
{
v___y_575_ = v_b_573_;
goto v___jp_574_;
}
}
else
{
lean_dec(v_f_569_);
return v_b_573_;
}
v___jp_574_:
{
size_t v___x_576_; size_t v___x_577_; 
v___x_576_ = ((size_t)1ULL);
v___x_577_ = lean_usize_add(v_i_571_, v___x_576_);
v_i_571_ = v___x_577_;
v_b_573_ = v___y_575_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_foldHyps_spec__0_spec__0___redArg___boxed(lean_object* v_f_585_, lean_object* v_as_586_, lean_object* v_i_587_, lean_object* v_stop_588_, lean_object* v_b_589_){
_start:
{
size_t v_i_boxed_590_; size_t v_stop_boxed_591_; lean_object* v_res_592_; 
v_i_boxed_590_ = lean_unbox_usize(v_i_587_);
lean_dec(v_i_587_);
v_stop_boxed_591_ = lean_unbox_usize(v_stop_588_);
lean_dec(v_stop_588_);
v_res_592_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_foldHyps_spec__0_spec__0___redArg(v_f_585_, v_as_586_, v_i_boxed_590_, v_stop_boxed_591_, v_b_589_);
lean_dec_ref(v_as_586_);
return v_res_592_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_foldHyps_spec__0_spec__1___redArg(lean_object* v_f_593_, lean_object* v_as_594_, size_t v_i_595_, size_t v_stop_596_, lean_object* v_b_597_){
_start:
{
lean_object* v___y_599_; uint8_t v___x_603_; 
v___x_603_ = lean_usize_dec_eq(v_i_595_, v_stop_596_);
if (v___x_603_ == 0)
{
lean_object* v___x_604_; lean_object* v_subst_605_; lean_object* v_premises_606_; lean_object* v___x_607_; lean_object* v___x_608_; uint8_t v___x_609_; 
v___x_604_ = lean_array_uget_borrowed(v_as_594_, v_i_595_);
v_subst_605_ = lean_ctor_get(v___x_604_, 0);
v_premises_606_ = lean_ctor_get(v_subst_605_, 0);
v___x_607_ = lean_unsigned_to_nat(0u);
v___x_608_ = lean_array_get_size(v_premises_606_);
v___x_609_ = lean_nat_dec_lt(v___x_607_, v___x_608_);
if (v___x_609_ == 0)
{
v___y_599_ = v_b_597_;
goto v___jp_598_;
}
else
{
uint8_t v___x_610_; 
v___x_610_ = lean_nat_dec_le(v___x_608_, v___x_608_);
if (v___x_610_ == 0)
{
if (v___x_609_ == 0)
{
v___y_599_ = v_b_597_;
goto v___jp_598_;
}
else
{
size_t v___x_611_; size_t v___x_612_; lean_object* v___x_613_; 
v___x_611_ = ((size_t)0ULL);
v___x_612_ = lean_usize_of_nat(v___x_608_);
lean_inc(v_f_593_);
v___x_613_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_foldHyps_spec__0_spec__0___redArg(v_f_593_, v_premises_606_, v___x_611_, v___x_612_, v_b_597_);
v___y_599_ = v___x_613_;
goto v___jp_598_;
}
}
else
{
size_t v___x_614_; size_t v___x_615_; lean_object* v___x_616_; 
v___x_614_ = ((size_t)0ULL);
v___x_615_ = lean_usize_of_nat(v___x_608_);
lean_inc(v_f_593_);
v___x_616_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_foldHyps_spec__0_spec__0___redArg(v_f_593_, v_premises_606_, v___x_614_, v___x_615_, v_b_597_);
v___y_599_ = v___x_616_;
goto v___jp_598_;
}
}
}
else
{
lean_dec(v_f_593_);
return v_b_597_;
}
v___jp_598_:
{
size_t v___x_600_; size_t v___x_601_; 
v___x_600_ = ((size_t)1ULL);
v___x_601_ = lean_usize_add(v_i_595_, v___x_600_);
v_i_595_ = v___x_601_;
v_b_597_ = v___y_599_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_foldHyps_spec__0_spec__1___redArg___boxed(lean_object* v_f_617_, lean_object* v_as_618_, lean_object* v_i_619_, lean_object* v_stop_620_, lean_object* v_b_621_){
_start:
{
size_t v_i_boxed_622_; size_t v_stop_boxed_623_; lean_object* v_res_624_; 
v_i_boxed_622_ = lean_unbox_usize(v_i_619_);
lean_dec(v_i_619_);
v_stop_boxed_623_ = lean_unbox_usize(v_stop_620_);
lean_dec(v_stop_620_);
v_res_624_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_foldHyps_spec__0_spec__1___redArg(v_f_617_, v_as_618_, v_i_boxed_622_, v_stop_boxed_623_, v_b_621_);
lean_dec_ref(v_as_618_);
return v_res_624_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_foldHyps_spec__0___redArg(lean_object* v_f_625_, lean_object* v_init_626_, lean_object* v_m_627_){
_start:
{
lean_object* v_match_628_; lean_object* v___x_629_; lean_object* v___x_630_; uint8_t v___x_631_; 
v_match_628_ = lean_ctor_get(v_m_627_, 1);
v___x_629_ = lean_unsigned_to_nat(0u);
v___x_630_ = lean_array_get_size(v_match_628_);
v___x_631_ = lean_nat_dec_lt(v___x_629_, v___x_630_);
if (v___x_631_ == 0)
{
lean_dec(v_f_625_);
return v_init_626_;
}
else
{
uint8_t v___x_632_; 
v___x_632_ = lean_nat_dec_le(v___x_630_, v___x_630_);
if (v___x_632_ == 0)
{
if (v___x_631_ == 0)
{
lean_dec(v_f_625_);
return v_init_626_;
}
else
{
size_t v___x_633_; size_t v___x_634_; lean_object* v___x_635_; 
v___x_633_ = ((size_t)0ULL);
v___x_634_ = lean_usize_of_nat(v___x_630_);
v___x_635_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_foldHyps_spec__0_spec__1___redArg(v_f_625_, v_match_628_, v___x_633_, v___x_634_, v_init_626_);
return v___x_635_;
}
}
else
{
size_t v___x_636_; size_t v___x_637_; lean_object* v___x_638_; 
v___x_636_ = ((size_t)0ULL);
v___x_637_ = lean_usize_of_nat(v___x_630_);
v___x_638_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_foldHyps_spec__0_spec__1___redArg(v_f_625_, v_match_628_, v___x_636_, v___x_637_, v_init_626_);
return v___x_638_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_foldHyps_spec__0___redArg___boxed(lean_object* v_f_639_, lean_object* v_init_640_, lean_object* v_m_641_){
_start:
{
lean_object* v_res_642_; 
v_res_642_ = lp_aesop_Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_foldHyps_spec__0___redArg(v_f_639_, v_init_640_, v_m_641_);
lean_dec_ref(v_m_641_);
return v_res_642_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_foldHyps___redArg(lean_object* v_f_643_, lean_object* v_init_644_, lean_object* v_m_645_){
_start:
{
lean_object* v___x_646_; 
v___x_646_ = lp_aesop_Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_foldHyps_spec__0___redArg(v_f_643_, v_init_644_, v_m_645_);
return v___x_646_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_foldHyps___redArg___boxed(lean_object* v_f_647_, lean_object* v_init_648_, lean_object* v_m_649_){
_start:
{
lean_object* v_res_650_; 
v_res_650_ = lp_aesop_Aesop_ForwardRuleMatch_foldHyps___redArg(v_f_647_, v_init_648_, v_m_649_);
lean_dec_ref(v_m_649_);
return v_res_650_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_foldHyps(lean_object* v_00_u03c3_651_, lean_object* v_f_652_, lean_object* v_init_653_, lean_object* v_m_654_){
_start:
{
lean_object* v___x_655_; 
v___x_655_ = lp_aesop_Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_foldHyps_spec__0___redArg(v_f_652_, v_init_653_, v_m_654_);
return v___x_655_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_foldHyps___boxed(lean_object* v_00_u03c3_656_, lean_object* v_f_657_, lean_object* v_init_658_, lean_object* v_m_659_){
_start:
{
lean_object* v_res_660_; 
v_res_660_ = lp_aesop_Aesop_ForwardRuleMatch_foldHyps(v_00_u03c3_656_, v_f_657_, v_init_658_, v_m_659_);
lean_dec_ref(v_m_659_);
return v_res_660_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_foldHyps_spec__0(lean_object* v_00_u03c3_661_, lean_object* v_f_662_, lean_object* v_init_663_, lean_object* v_m_664_){
_start:
{
lean_object* v___x_665_; 
v___x_665_ = lp_aesop_Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_foldHyps_spec__0___redArg(v_f_662_, v_init_663_, v_m_664_);
return v___x_665_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_foldHyps_spec__0___boxed(lean_object* v_00_u03c3_666_, lean_object* v_f_667_, lean_object* v_init_668_, lean_object* v_m_669_){
_start:
{
lean_object* v_res_670_; 
v_res_670_ = lp_aesop_Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_foldHyps_spec__0(v_00_u03c3_666_, v_f_667_, v_init_668_, v_m_669_);
lean_dec_ref(v_m_669_);
return v_res_670_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_foldHyps_spec__0_spec__0(lean_object* v_00_u03c3_671_, lean_object* v_f_672_, lean_object* v_as_673_, size_t v_i_674_, size_t v_stop_675_, lean_object* v_b_676_){
_start:
{
lean_object* v___x_677_; 
v___x_677_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_foldHyps_spec__0_spec__0___redArg(v_f_672_, v_as_673_, v_i_674_, v_stop_675_, v_b_676_);
return v___x_677_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_foldHyps_spec__0_spec__0___boxed(lean_object* v_00_u03c3_678_, lean_object* v_f_679_, lean_object* v_as_680_, lean_object* v_i_681_, lean_object* v_stop_682_, lean_object* v_b_683_){
_start:
{
size_t v_i_boxed_684_; size_t v_stop_boxed_685_; lean_object* v_res_686_; 
v_i_boxed_684_ = lean_unbox_usize(v_i_681_);
lean_dec(v_i_681_);
v_stop_boxed_685_ = lean_unbox_usize(v_stop_682_);
lean_dec(v_stop_682_);
v_res_686_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_foldHyps_spec__0_spec__0(v_00_u03c3_678_, v_f_679_, v_as_680_, v_i_boxed_684_, v_stop_boxed_685_, v_b_683_);
lean_dec_ref(v_as_680_);
return v_res_686_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_foldHyps_spec__0_spec__1(lean_object* v_00_u03c3_687_, lean_object* v_f_688_, lean_object* v_as_689_, size_t v_i_690_, size_t v_stop_691_, lean_object* v_b_692_){
_start:
{
lean_object* v___x_693_; 
v___x_693_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_foldHyps_spec__0_spec__1___redArg(v_f_688_, v_as_689_, v_i_690_, v_stop_691_, v_b_692_);
return v___x_693_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_foldHyps_spec__0_spec__1___boxed(lean_object* v_00_u03c3_694_, lean_object* v_f_695_, lean_object* v_as_696_, lean_object* v_i_697_, lean_object* v_stop_698_, lean_object* v_b_699_){
_start:
{
size_t v_i_boxed_700_; size_t v_stop_boxed_701_; lean_object* v_res_702_; 
v_i_boxed_700_ = lean_unbox_usize(v_i_697_);
lean_dec(v_i_697_);
v_stop_boxed_701_ = lean_unbox_usize(v_stop_698_);
lean_dec(v_stop_698_);
v_res_702_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_foldHyps_spec__0_spec__1(v_00_u03c3_694_, v_f_695_, v_as_696_, v_i_boxed_700_, v_stop_boxed_701_, v_b_699_);
lean_dec_ref(v_as_696_);
return v_res_702_;
}
}
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_ForwardRuleMatch_anyHyp_spec__0(lean_object* v_f_703_, lean_object* v_as_704_, size_t v_i_705_, size_t v_stop_706_){
_start:
{
uint8_t v___x_707_; 
v___x_707_ = lean_usize_dec_eq(v_i_705_, v_stop_706_);
if (v___x_707_ == 0)
{
uint8_t v___x_708_; uint8_t v___y_710_; lean_object* v___x_714_; 
v___x_708_ = 1;
v___x_714_ = lean_array_uget_borrowed(v_as_704_, v_i_705_);
if (lean_obj_tag(v___x_714_) == 1)
{
lean_object* v_val_715_; lean_object* v___x_716_; 
v_val_715_ = lean_ctor_get(v___x_714_, 0);
v___x_716_ = l_Lean_Expr_consumeMData(v_val_715_);
if (lean_obj_tag(v___x_716_) == 1)
{
lean_object* v_fvarId_717_; lean_object* v___x_718_; uint8_t v___x_719_; 
v_fvarId_717_ = lean_ctor_get(v___x_716_, 0);
lean_inc(v_fvarId_717_);
lean_dec_ref_known(v___x_716_, 1);
lean_inc_ref(v_f_703_);
v___x_718_ = lean_apply_1(v_f_703_, v_fvarId_717_);
v___x_719_ = lean_unbox(v___x_718_);
v___y_710_ = v___x_719_;
goto v___jp_709_;
}
else
{
lean_dec_ref(v___x_716_);
v___y_710_ = v___x_707_;
goto v___jp_709_;
}
}
else
{
v___y_710_ = v___x_707_;
goto v___jp_709_;
}
v___jp_709_:
{
if (v___y_710_ == 0)
{
size_t v___x_711_; size_t v___x_712_; 
v___x_711_ = ((size_t)1ULL);
v___x_712_ = lean_usize_add(v_i_705_, v___x_711_);
v_i_705_ = v___x_712_;
goto _start;
}
else
{
lean_dec_ref(v_f_703_);
return v___x_708_;
}
}
}
else
{
uint8_t v___x_720_; 
lean_dec_ref(v_f_703_);
v___x_720_ = 0;
return v___x_720_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_ForwardRuleMatch_anyHyp_spec__0___boxed(lean_object* v_f_721_, lean_object* v_as_722_, lean_object* v_i_723_, lean_object* v_stop_724_){
_start:
{
size_t v_i_boxed_725_; size_t v_stop_boxed_726_; uint8_t v_res_727_; lean_object* v_r_728_; 
v_i_boxed_725_ = lean_unbox_usize(v_i_723_);
lean_dec(v_i_723_);
v_stop_boxed_726_ = lean_unbox_usize(v_stop_724_);
lean_dec(v_stop_724_);
v_res_727_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_ForwardRuleMatch_anyHyp_spec__0(v_f_721_, v_as_722_, v_i_boxed_725_, v_stop_boxed_726_);
lean_dec_ref(v_as_722_);
v_r_728_ = lean_box(v_res_727_);
return v_r_728_;
}
}
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_ForwardRuleMatch_anyHyp_spec__1(lean_object* v_f_729_, lean_object* v_as_730_, size_t v_i_731_, size_t v_stop_732_){
_start:
{
uint8_t v___x_733_; 
v___x_733_ = lean_usize_dec_eq(v_i_731_, v_stop_732_);
if (v___x_733_ == 0)
{
lean_object* v___x_734_; lean_object* v_subst_735_; lean_object* v_premises_736_; uint8_t v___x_737_; uint8_t v___y_739_; lean_object* v___x_743_; lean_object* v___x_744_; uint8_t v___x_745_; 
v___x_734_ = lean_array_uget_borrowed(v_as_730_, v_i_731_);
v_subst_735_ = lean_ctor_get(v___x_734_, 0);
v_premises_736_ = lean_ctor_get(v_subst_735_, 0);
v___x_737_ = 1;
v___x_743_ = lean_unsigned_to_nat(0u);
v___x_744_ = lean_array_get_size(v_premises_736_);
v___x_745_ = lean_nat_dec_lt(v___x_743_, v___x_744_);
if (v___x_745_ == 0)
{
v___y_739_ = v___x_733_;
goto v___jp_738_;
}
else
{
if (v___x_745_ == 0)
{
v___y_739_ = v___x_733_;
goto v___jp_738_;
}
else
{
size_t v___x_746_; size_t v___x_747_; uint8_t v___x_748_; 
v___x_746_ = ((size_t)0ULL);
v___x_747_ = lean_usize_of_nat(v___x_744_);
lean_inc_ref(v_f_729_);
v___x_748_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_ForwardRuleMatch_anyHyp_spec__0(v_f_729_, v_premises_736_, v___x_746_, v___x_747_);
v___y_739_ = v___x_748_;
goto v___jp_738_;
}
}
v___jp_738_:
{
if (v___y_739_ == 0)
{
size_t v___x_740_; size_t v___x_741_; 
v___x_740_ = ((size_t)1ULL);
v___x_741_ = lean_usize_add(v_i_731_, v___x_740_);
v_i_731_ = v___x_741_;
goto _start;
}
else
{
lean_dec_ref(v_f_729_);
return v___x_737_;
}
}
}
else
{
uint8_t v___x_749_; 
lean_dec_ref(v_f_729_);
v___x_749_ = 0;
return v___x_749_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_ForwardRuleMatch_anyHyp_spec__1___boxed(lean_object* v_f_750_, lean_object* v_as_751_, lean_object* v_i_752_, lean_object* v_stop_753_){
_start:
{
size_t v_i_boxed_754_; size_t v_stop_boxed_755_; uint8_t v_res_756_; lean_object* v_r_757_; 
v_i_boxed_754_ = lean_unbox_usize(v_i_752_);
lean_dec(v_i_752_);
v_stop_boxed_755_ = lean_unbox_usize(v_stop_753_);
lean_dec(v_stop_753_);
v_res_756_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_ForwardRuleMatch_anyHyp_spec__1(v_f_750_, v_as_751_, v_i_boxed_754_, v_stop_boxed_755_);
lean_dec_ref(v_as_751_);
v_r_757_ = lean_box(v_res_756_);
return v_r_757_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_ForwardRuleMatch_anyHyp(lean_object* v_m_758_, lean_object* v_f_759_){
_start:
{
lean_object* v_match_760_; lean_object* v___x_761_; lean_object* v___x_762_; uint8_t v___x_763_; 
v_match_760_ = lean_ctor_get(v_m_758_, 1);
v___x_761_ = lean_unsigned_to_nat(0u);
v___x_762_ = lean_array_get_size(v_match_760_);
v___x_763_ = lean_nat_dec_lt(v___x_761_, v___x_762_);
if (v___x_763_ == 0)
{
lean_dec_ref(v_f_759_);
return v___x_763_;
}
else
{
if (v___x_763_ == 0)
{
lean_dec_ref(v_f_759_);
return v___x_763_;
}
else
{
size_t v___x_764_; size_t v___x_765_; uint8_t v___x_766_; 
v___x_764_ = ((size_t)0ULL);
v___x_765_ = lean_usize_of_nat(v___x_762_);
v___x_766_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Aesop_ForwardRuleMatch_anyHyp_spec__1(v_f_759_, v_match_760_, v___x_764_, v___x_765_);
return v___x_766_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_anyHyp___boxed(lean_object* v_m_767_, lean_object* v_f_768_){
_start:
{
uint8_t v_res_769_; lean_object* v_r_770_; 
v_res_769_ = lp_aesop_Aesop_ForwardRuleMatch_anyHyp(v_m_767_, v_f_768_);
lean_dec_ref(v_m_767_);
v_r_770_ = lean_box(v_res_769_);
return v_r_770_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_getPropHyps___lam__0(lean_object* v_hs_771_, lean_object* v_h_772_, lean_object* v___y_773_, lean_object* v___y_774_, lean_object* v___y_775_, lean_object* v___y_776_){
_start:
{
lean_object* v___x_778_; lean_object* v___x_779_; 
lean_inc(v_h_772_);
v___x_778_ = l_Lean_Expr_fvar___override(v_h_772_);
v___x_779_ = l_Lean_Meta_isProof(v___x_778_, v___y_773_, v___y_774_, v___y_775_, v___y_776_);
if (lean_obj_tag(v___x_779_) == 0)
{
lean_object* v_a_780_; lean_object* v___x_782_; uint8_t v_isShared_783_; uint8_t v_isSharedCheck_792_; 
v_a_780_ = lean_ctor_get(v___x_779_, 0);
v_isSharedCheck_792_ = !lean_is_exclusive(v___x_779_);
if (v_isSharedCheck_792_ == 0)
{
v___x_782_ = v___x_779_;
v_isShared_783_ = v_isSharedCheck_792_;
goto v_resetjp_781_;
}
else
{
lean_inc(v_a_780_);
lean_dec(v___x_779_);
v___x_782_ = lean_box(0);
v_isShared_783_ = v_isSharedCheck_792_;
goto v_resetjp_781_;
}
v_resetjp_781_:
{
uint8_t v___x_784_; 
v___x_784_ = lean_unbox(v_a_780_);
lean_dec(v_a_780_);
if (v___x_784_ == 0)
{
lean_object* v___x_786_; 
lean_dec(v_h_772_);
if (v_isShared_783_ == 0)
{
lean_ctor_set(v___x_782_, 0, v_hs_771_);
v___x_786_ = v___x_782_;
goto v_reusejp_785_;
}
else
{
lean_object* v_reuseFailAlloc_787_; 
v_reuseFailAlloc_787_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_787_, 0, v_hs_771_);
v___x_786_ = v_reuseFailAlloc_787_;
goto v_reusejp_785_;
}
v_reusejp_785_:
{
return v___x_786_;
}
}
else
{
lean_object* v___x_788_; lean_object* v___x_790_; 
v___x_788_ = lean_array_push(v_hs_771_, v_h_772_);
if (v_isShared_783_ == 0)
{
lean_ctor_set(v___x_782_, 0, v___x_788_);
v___x_790_ = v___x_782_;
goto v_reusejp_789_;
}
else
{
lean_object* v_reuseFailAlloc_791_; 
v_reuseFailAlloc_791_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_791_, 0, v___x_788_);
v___x_790_ = v_reuseFailAlloc_791_;
goto v_reusejp_789_;
}
v_reusejp_789_:
{
return v___x_790_;
}
}
}
}
else
{
lean_object* v_a_793_; lean_object* v___x_795_; uint8_t v_isShared_796_; uint8_t v_isSharedCheck_800_; 
lean_dec(v_h_772_);
lean_dec_ref(v_hs_771_);
v_a_793_ = lean_ctor_get(v___x_779_, 0);
v_isSharedCheck_800_ = !lean_is_exclusive(v___x_779_);
if (v_isSharedCheck_800_ == 0)
{
v___x_795_ = v___x_779_;
v_isShared_796_ = v_isSharedCheck_800_;
goto v_resetjp_794_;
}
else
{
lean_inc(v_a_793_);
lean_dec(v___x_779_);
v___x_795_ = lean_box(0);
v_isShared_796_ = v_isSharedCheck_800_;
goto v_resetjp_794_;
}
v_resetjp_794_:
{
lean_object* v___x_798_; 
if (v_isShared_796_ == 0)
{
v___x_798_ = v___x_795_;
goto v_reusejp_797_;
}
else
{
lean_object* v_reuseFailAlloc_799_; 
v_reuseFailAlloc_799_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_799_, 0, v_a_793_);
v___x_798_ = v_reuseFailAlloc_799_;
goto v_reusejp_797_;
}
v_reusejp_797_:
{
return v___x_798_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_getPropHyps___lam__0___boxed(lean_object* v_hs_801_, lean_object* v_h_802_, lean_object* v___y_803_, lean_object* v___y_804_, lean_object* v___y_805_, lean_object* v___y_806_, lean_object* v___y_807_){
_start:
{
lean_object* v_res_808_; 
v_res_808_ = lp_aesop_Aesop_ForwardRuleMatch_getPropHyps___lam__0(v_hs_801_, v_h_802_, v___y_803_, v___y_804_, v___y_805_, v___y_806_);
lean_dec(v___y_806_);
lean_dec_ref(v___y_805_);
lean_dec(v___y_804_);
lean_dec_ref(v___y_803_);
return v_res_808_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_getPropHyps_spec__0_spec__0___redArg(lean_object* v_f_809_, lean_object* v_as_810_, size_t v_i_811_, size_t v_stop_812_, lean_object* v_b_813_, lean_object* v___y_814_, lean_object* v___y_815_, lean_object* v___y_816_, lean_object* v___y_817_){
_start:
{
lean_object* v_a_820_; uint8_t v___x_824_; 
v___x_824_ = lean_usize_dec_eq(v_i_811_, v_stop_812_);
if (v___x_824_ == 0)
{
lean_object* v___x_825_; 
v___x_825_ = lean_array_uget_borrowed(v_as_810_, v_i_811_);
if (lean_obj_tag(v___x_825_) == 1)
{
lean_object* v_val_826_; lean_object* v___x_827_; 
v_val_826_ = lean_ctor_get(v___x_825_, 0);
v___x_827_ = l_Lean_Expr_consumeMData(v_val_826_);
if (lean_obj_tag(v___x_827_) == 1)
{
lean_object* v_fvarId_828_; lean_object* v___x_829_; 
v_fvarId_828_ = lean_ctor_get(v___x_827_, 0);
lean_inc(v_fvarId_828_);
lean_dec_ref_known(v___x_827_, 1);
lean_inc_ref(v_f_809_);
lean_inc(v___y_817_);
lean_inc_ref(v___y_816_);
lean_inc(v___y_815_);
lean_inc_ref(v___y_814_);
v___x_829_ = lean_apply_7(v_f_809_, v_b_813_, v_fvarId_828_, v___y_814_, v___y_815_, v___y_816_, v___y_817_, lean_box(0));
if (lean_obj_tag(v___x_829_) == 0)
{
lean_object* v_a_830_; 
v_a_830_ = lean_ctor_get(v___x_829_, 0);
lean_inc(v_a_830_);
lean_dec_ref_known(v___x_829_, 1);
v_a_820_ = v_a_830_;
goto v___jp_819_;
}
else
{
lean_dec_ref(v_f_809_);
return v___x_829_;
}
}
else
{
lean_dec_ref(v___x_827_);
v_a_820_ = v_b_813_;
goto v___jp_819_;
}
}
else
{
v_a_820_ = v_b_813_;
goto v___jp_819_;
}
}
else
{
lean_object* v___x_831_; 
lean_dec_ref(v_f_809_);
v___x_831_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_831_, 0, v_b_813_);
return v___x_831_;
}
v___jp_819_:
{
size_t v___x_821_; size_t v___x_822_; 
v___x_821_ = ((size_t)1ULL);
v___x_822_ = lean_usize_add(v_i_811_, v___x_821_);
v_i_811_ = v___x_822_;
v_b_813_ = v_a_820_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_getPropHyps_spec__0_spec__0___redArg___boxed(lean_object* v_f_832_, lean_object* v_as_833_, lean_object* v_i_834_, lean_object* v_stop_835_, lean_object* v_b_836_, lean_object* v___y_837_, lean_object* v___y_838_, lean_object* v___y_839_, lean_object* v___y_840_, lean_object* v___y_841_){
_start:
{
size_t v_i_boxed_842_; size_t v_stop_boxed_843_; lean_object* v_res_844_; 
v_i_boxed_842_ = lean_unbox_usize(v_i_834_);
lean_dec(v_i_834_);
v_stop_boxed_843_ = lean_unbox_usize(v_stop_835_);
lean_dec(v_stop_835_);
v_res_844_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_getPropHyps_spec__0_spec__0___redArg(v_f_832_, v_as_833_, v_i_boxed_842_, v_stop_boxed_843_, v_b_836_, v___y_837_, v___y_838_, v___y_839_, v___y_840_);
lean_dec(v___y_840_);
lean_dec_ref(v___y_839_);
lean_dec(v___y_838_);
lean_dec_ref(v___y_837_);
lean_dec_ref(v_as_833_);
return v_res_844_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_getPropHyps_spec__0_spec__1___redArg(lean_object* v_f_845_, lean_object* v_as_846_, size_t v_i_847_, size_t v_stop_848_, lean_object* v_b_849_, lean_object* v___y_850_, lean_object* v___y_851_, lean_object* v___y_852_, lean_object* v___y_853_){
_start:
{
lean_object* v_a_856_; lean_object* v___y_861_; uint8_t v___x_863_; 
v___x_863_ = lean_usize_dec_eq(v_i_847_, v_stop_848_);
if (v___x_863_ == 0)
{
lean_object* v___x_864_; lean_object* v_subst_865_; lean_object* v_premises_866_; lean_object* v___x_867_; lean_object* v___x_868_; uint8_t v___x_869_; 
v___x_864_ = lean_array_uget_borrowed(v_as_846_, v_i_847_);
v_subst_865_ = lean_ctor_get(v___x_864_, 0);
v_premises_866_ = lean_ctor_get(v_subst_865_, 0);
v___x_867_ = lean_unsigned_to_nat(0u);
v___x_868_ = lean_array_get_size(v_premises_866_);
v___x_869_ = lean_nat_dec_lt(v___x_867_, v___x_868_);
if (v___x_869_ == 0)
{
v_a_856_ = v_b_849_;
goto v___jp_855_;
}
else
{
uint8_t v___x_870_; 
v___x_870_ = lean_nat_dec_le(v___x_868_, v___x_868_);
if (v___x_870_ == 0)
{
if (v___x_869_ == 0)
{
v_a_856_ = v_b_849_;
goto v___jp_855_;
}
else
{
size_t v___x_871_; size_t v___x_872_; lean_object* v___x_873_; 
v___x_871_ = ((size_t)0ULL);
v___x_872_ = lean_usize_of_nat(v___x_868_);
lean_inc_ref(v_f_845_);
v___x_873_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_getPropHyps_spec__0_spec__0___redArg(v_f_845_, v_premises_866_, v___x_871_, v___x_872_, v_b_849_, v___y_850_, v___y_851_, v___y_852_, v___y_853_);
v___y_861_ = v___x_873_;
goto v___jp_860_;
}
}
else
{
size_t v___x_874_; size_t v___x_875_; lean_object* v___x_876_; 
v___x_874_ = ((size_t)0ULL);
v___x_875_ = lean_usize_of_nat(v___x_868_);
lean_inc_ref(v_f_845_);
v___x_876_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_getPropHyps_spec__0_spec__0___redArg(v_f_845_, v_premises_866_, v___x_874_, v___x_875_, v_b_849_, v___y_850_, v___y_851_, v___y_852_, v___y_853_);
v___y_861_ = v___x_876_;
goto v___jp_860_;
}
}
}
else
{
lean_object* v___x_877_; 
lean_dec_ref(v_f_845_);
v___x_877_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_877_, 0, v_b_849_);
return v___x_877_;
}
v___jp_855_:
{
size_t v___x_857_; size_t v___x_858_; 
v___x_857_ = ((size_t)1ULL);
v___x_858_ = lean_usize_add(v_i_847_, v___x_857_);
v_i_847_ = v___x_858_;
v_b_849_ = v_a_856_;
goto _start;
}
v___jp_860_:
{
if (lean_obj_tag(v___y_861_) == 0)
{
lean_object* v_a_862_; 
v_a_862_ = lean_ctor_get(v___y_861_, 0);
lean_inc(v_a_862_);
lean_dec_ref_known(v___y_861_, 1);
v_a_856_ = v_a_862_;
goto v___jp_855_;
}
else
{
lean_dec_ref(v_f_845_);
return v___y_861_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_getPropHyps_spec__0_spec__1___redArg___boxed(lean_object* v_f_878_, lean_object* v_as_879_, lean_object* v_i_880_, lean_object* v_stop_881_, lean_object* v_b_882_, lean_object* v___y_883_, lean_object* v___y_884_, lean_object* v___y_885_, lean_object* v___y_886_, lean_object* v___y_887_){
_start:
{
size_t v_i_boxed_888_; size_t v_stop_boxed_889_; lean_object* v_res_890_; 
v_i_boxed_888_ = lean_unbox_usize(v_i_880_);
lean_dec(v_i_880_);
v_stop_boxed_889_ = lean_unbox_usize(v_stop_881_);
lean_dec(v_stop_881_);
v_res_890_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_getPropHyps_spec__0_spec__1___redArg(v_f_878_, v_as_879_, v_i_boxed_888_, v_stop_boxed_889_, v_b_882_, v___y_883_, v___y_884_, v___y_885_, v___y_886_);
lean_dec(v___y_886_);
lean_dec_ref(v___y_885_);
lean_dec(v___y_884_);
lean_dec_ref(v___y_883_);
lean_dec_ref(v_as_879_);
return v_res_890_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_getPropHyps_spec__0___redArg(lean_object* v_f_891_, lean_object* v_init_892_, lean_object* v_m_893_, lean_object* v___y_894_, lean_object* v___y_895_, lean_object* v___y_896_, lean_object* v___y_897_){
_start:
{
lean_object* v_match_899_; lean_object* v___x_900_; lean_object* v___x_901_; uint8_t v___x_902_; 
v_match_899_ = lean_ctor_get(v_m_893_, 1);
v___x_900_ = lean_unsigned_to_nat(0u);
v___x_901_ = lean_array_get_size(v_match_899_);
v___x_902_ = lean_nat_dec_lt(v___x_900_, v___x_901_);
if (v___x_902_ == 0)
{
lean_object* v___x_903_; 
lean_dec_ref(v_f_891_);
v___x_903_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_903_, 0, v_init_892_);
return v___x_903_;
}
else
{
uint8_t v___x_904_; 
v___x_904_ = lean_nat_dec_le(v___x_901_, v___x_901_);
if (v___x_904_ == 0)
{
if (v___x_902_ == 0)
{
lean_object* v___x_905_; 
lean_dec_ref(v_f_891_);
v___x_905_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_905_, 0, v_init_892_);
return v___x_905_;
}
else
{
size_t v___x_906_; size_t v___x_907_; lean_object* v___x_908_; 
v___x_906_ = ((size_t)0ULL);
v___x_907_ = lean_usize_of_nat(v___x_901_);
v___x_908_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_getPropHyps_spec__0_spec__1___redArg(v_f_891_, v_match_899_, v___x_906_, v___x_907_, v_init_892_, v___y_894_, v___y_895_, v___y_896_, v___y_897_);
return v___x_908_;
}
}
else
{
size_t v___x_909_; size_t v___x_910_; lean_object* v___x_911_; 
v___x_909_ = ((size_t)0ULL);
v___x_910_ = lean_usize_of_nat(v___x_901_);
v___x_911_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_getPropHyps_spec__0_spec__1___redArg(v_f_891_, v_match_899_, v___x_909_, v___x_910_, v_init_892_, v___y_894_, v___y_895_, v___y_896_, v___y_897_);
return v___x_911_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_getPropHyps_spec__0___redArg___boxed(lean_object* v_f_912_, lean_object* v_init_913_, lean_object* v_m_914_, lean_object* v___y_915_, lean_object* v___y_916_, lean_object* v___y_917_, lean_object* v___y_918_, lean_object* v___y_919_){
_start:
{
lean_object* v_res_920_; 
v_res_920_ = lp_aesop_Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_getPropHyps_spec__0___redArg(v_f_912_, v_init_913_, v_m_914_, v___y_915_, v___y_916_, v___y_917_, v___y_918_);
lean_dec(v___y_918_);
lean_dec_ref(v___y_917_);
lean_dec(v___y_916_);
lean_dec_ref(v___y_915_);
lean_dec_ref(v_m_914_);
return v_res_920_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_getPropHyps(lean_object* v_m_922_, lean_object* v_a_923_, lean_object* v_a_924_, lean_object* v_a_925_, lean_object* v_a_926_){
_start:
{
lean_object* v_rule_928_; lean_object* v_toForwardRuleInfo_929_; lean_object* v_numPremises_930_; lean_object* v___f_931_; lean_object* v___x_932_; lean_object* v___x_933_; 
v_rule_928_ = lean_ctor_get(v_m_922_, 0);
v_toForwardRuleInfo_929_ = lean_ctor_get(v_rule_928_, 0);
v_numPremises_930_ = lean_ctor_get(v_toForwardRuleInfo_929_, 0);
v___f_931_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_getPropHyps___closed__0));
v___x_932_ = lean_mk_empty_array_with_capacity(v_numPremises_930_);
v___x_933_ = lp_aesop_Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_getPropHyps_spec__0___redArg(v___f_931_, v___x_932_, v_m_922_, v_a_923_, v_a_924_, v_a_925_, v_a_926_);
return v___x_933_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_getPropHyps___boxed(lean_object* v_m_934_, lean_object* v_a_935_, lean_object* v_a_936_, lean_object* v_a_937_, lean_object* v_a_938_, lean_object* v_a_939_){
_start:
{
lean_object* v_res_940_; 
v_res_940_ = lp_aesop_Aesop_ForwardRuleMatch_getPropHyps(v_m_934_, v_a_935_, v_a_936_, v_a_937_, v_a_938_);
lean_dec(v_a_938_);
lean_dec_ref(v_a_937_);
lean_dec(v_a_936_);
lean_dec_ref(v_a_935_);
lean_dec_ref(v_m_934_);
return v_res_940_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_getPropHyps_spec__0(lean_object* v_00_u03c3_941_, lean_object* v_f_942_, lean_object* v_init_943_, lean_object* v_m_944_, lean_object* v___y_945_, lean_object* v___y_946_, lean_object* v___y_947_, lean_object* v___y_948_){
_start:
{
lean_object* v___x_950_; 
v___x_950_ = lp_aesop_Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_getPropHyps_spec__0___redArg(v_f_942_, v_init_943_, v_m_944_, v___y_945_, v___y_946_, v___y_947_, v___y_948_);
return v___x_950_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_getPropHyps_spec__0___boxed(lean_object* v_00_u03c3_951_, lean_object* v_f_952_, lean_object* v_init_953_, lean_object* v_m_954_, lean_object* v___y_955_, lean_object* v___y_956_, lean_object* v___y_957_, lean_object* v___y_958_, lean_object* v___y_959_){
_start:
{
lean_object* v_res_960_; 
v_res_960_ = lp_aesop_Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_getPropHyps_spec__0(v_00_u03c3_951_, v_f_952_, v_init_953_, v_m_954_, v___y_955_, v___y_956_, v___y_957_, v___y_958_);
lean_dec(v___y_958_);
lean_dec_ref(v___y_957_);
lean_dec(v___y_956_);
lean_dec_ref(v___y_955_);
lean_dec_ref(v_m_954_);
return v_res_960_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_getPropHyps_spec__0_spec__0(lean_object* v_00_u03c3_961_, lean_object* v_f_962_, lean_object* v_as_963_, size_t v_i_964_, size_t v_stop_965_, lean_object* v_b_966_, lean_object* v___y_967_, lean_object* v___y_968_, lean_object* v___y_969_, lean_object* v___y_970_){
_start:
{
lean_object* v___x_972_; 
v___x_972_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_getPropHyps_spec__0_spec__0___redArg(v_f_962_, v_as_963_, v_i_964_, v_stop_965_, v_b_966_, v___y_967_, v___y_968_, v___y_969_, v___y_970_);
return v___x_972_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_getPropHyps_spec__0_spec__0___boxed(lean_object* v_00_u03c3_973_, lean_object* v_f_974_, lean_object* v_as_975_, lean_object* v_i_976_, lean_object* v_stop_977_, lean_object* v_b_978_, lean_object* v___y_979_, lean_object* v___y_980_, lean_object* v___y_981_, lean_object* v___y_982_, lean_object* v___y_983_){
_start:
{
size_t v_i_boxed_984_; size_t v_stop_boxed_985_; lean_object* v_res_986_; 
v_i_boxed_984_ = lean_unbox_usize(v_i_976_);
lean_dec(v_i_976_);
v_stop_boxed_985_ = lean_unbox_usize(v_stop_977_);
lean_dec(v_stop_977_);
v_res_986_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_getPropHyps_spec__0_spec__0(v_00_u03c3_973_, v_f_974_, v_as_975_, v_i_boxed_984_, v_stop_boxed_985_, v_b_978_, v___y_979_, v___y_980_, v___y_981_, v___y_982_);
lean_dec(v___y_982_);
lean_dec_ref(v___y_981_);
lean_dec(v___y_980_);
lean_dec_ref(v___y_979_);
lean_dec_ref(v_as_975_);
return v_res_986_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_getPropHyps_spec__0_spec__1(lean_object* v_00_u03c3_987_, lean_object* v_f_988_, lean_object* v_as_989_, size_t v_i_990_, size_t v_stop_991_, lean_object* v_b_992_, lean_object* v___y_993_, lean_object* v___y_994_, lean_object* v___y_995_, lean_object* v___y_996_){
_start:
{
lean_object* v___x_998_; 
v___x_998_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_getPropHyps_spec__0_spec__1___redArg(v_f_988_, v_as_989_, v_i_990_, v_stop_991_, v_b_992_, v___y_993_, v___y_994_, v___y_995_, v___y_996_);
return v___x_998_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_getPropHyps_spec__0_spec__1___boxed(lean_object* v_00_u03c3_999_, lean_object* v_f_1000_, lean_object* v_as_1001_, lean_object* v_i_1002_, lean_object* v_stop_1003_, lean_object* v_b_1004_, lean_object* v___y_1005_, lean_object* v___y_1006_, lean_object* v___y_1007_, lean_object* v___y_1008_, lean_object* v___y_1009_){
_start:
{
size_t v_i_boxed_1010_; size_t v_stop_boxed_1011_; lean_object* v_res_1012_; 
v_i_boxed_1010_ = lean_unbox_usize(v_i_1002_);
lean_dec(v_i_1002_);
v_stop_boxed_1011_ = lean_unbox_usize(v_stop_1003_);
lean_dec(v_stop_1003_);
v_res_1012_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_ForwardRuleMatch_foldHypsM___at___00Aesop_ForwardRuleMatch_getPropHyps_spec__0_spec__1(v_00_u03c3_999_, v_f_1000_, v_as_1001_, v_i_boxed_1010_, v_stop_boxed_1011_, v_b_1004_, v___y_1005_, v___y_1006_, v___y_1007_, v___y_1008_);
lean_dec(v___y_1008_);
lean_dec_ref(v___y_1007_);
lean_dec(v___y_1006_);
lean_dec_ref(v___y_1005_);
lean_dec_ref(v_as_1001_);
return v_res_1012_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_instantiateMVars___at___00Aesop_ForwardRuleMatch_getProof_spec__2___redArg(lean_object* v_e_1013_, lean_object* v___y_1014_){
_start:
{
uint8_t v___x_1016_; 
v___x_1016_ = l_Lean_Expr_hasMVar(v_e_1013_);
if (v___x_1016_ == 0)
{
lean_object* v___x_1017_; 
v___x_1017_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1017_, 0, v_e_1013_);
return v___x_1017_;
}
else
{
lean_object* v___x_1018_; lean_object* v_mctx_1019_; lean_object* v___x_1020_; lean_object* v_fst_1021_; lean_object* v_snd_1022_; lean_object* v___x_1023_; lean_object* v_cache_1024_; lean_object* v_zetaDeltaFVarIds_1025_; lean_object* v_postponed_1026_; lean_object* v_diag_1027_; lean_object* v___x_1029_; uint8_t v_isShared_1030_; uint8_t v_isSharedCheck_1036_; 
v___x_1018_ = lean_st_ref_get(v___y_1014_);
v_mctx_1019_ = lean_ctor_get(v___x_1018_, 0);
lean_inc_ref(v_mctx_1019_);
lean_dec(v___x_1018_);
v___x_1020_ = l_Lean_instantiateMVarsCore(v_mctx_1019_, v_e_1013_);
v_fst_1021_ = lean_ctor_get(v___x_1020_, 0);
lean_inc(v_fst_1021_);
v_snd_1022_ = lean_ctor_get(v___x_1020_, 1);
lean_inc(v_snd_1022_);
lean_dec_ref(v___x_1020_);
v___x_1023_ = lean_st_ref_take(v___y_1014_);
v_cache_1024_ = lean_ctor_get(v___x_1023_, 1);
v_zetaDeltaFVarIds_1025_ = lean_ctor_get(v___x_1023_, 2);
v_postponed_1026_ = lean_ctor_get(v___x_1023_, 3);
v_diag_1027_ = lean_ctor_get(v___x_1023_, 4);
v_isSharedCheck_1036_ = !lean_is_exclusive(v___x_1023_);
if (v_isSharedCheck_1036_ == 0)
{
lean_object* v_unused_1037_; 
v_unused_1037_ = lean_ctor_get(v___x_1023_, 0);
lean_dec(v_unused_1037_);
v___x_1029_ = v___x_1023_;
v_isShared_1030_ = v_isSharedCheck_1036_;
goto v_resetjp_1028_;
}
else
{
lean_inc(v_diag_1027_);
lean_inc(v_postponed_1026_);
lean_inc(v_zetaDeltaFVarIds_1025_);
lean_inc(v_cache_1024_);
lean_dec(v___x_1023_);
v___x_1029_ = lean_box(0);
v_isShared_1030_ = v_isSharedCheck_1036_;
goto v_resetjp_1028_;
}
v_resetjp_1028_:
{
lean_object* v___x_1032_; 
if (v_isShared_1030_ == 0)
{
lean_ctor_set(v___x_1029_, 0, v_snd_1022_);
v___x_1032_ = v___x_1029_;
goto v_reusejp_1031_;
}
else
{
lean_object* v_reuseFailAlloc_1035_; 
v_reuseFailAlloc_1035_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1035_, 0, v_snd_1022_);
lean_ctor_set(v_reuseFailAlloc_1035_, 1, v_cache_1024_);
lean_ctor_set(v_reuseFailAlloc_1035_, 2, v_zetaDeltaFVarIds_1025_);
lean_ctor_set(v_reuseFailAlloc_1035_, 3, v_postponed_1026_);
lean_ctor_set(v_reuseFailAlloc_1035_, 4, v_diag_1027_);
v___x_1032_ = v_reuseFailAlloc_1035_;
goto v_reusejp_1031_;
}
v_reusejp_1031_:
{
lean_object* v___x_1033_; lean_object* v___x_1034_; 
v___x_1033_ = lean_st_ref_set(v___y_1014_, v___x_1032_);
v___x_1034_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1034_, 0, v_fst_1021_);
return v___x_1034_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_instantiateMVars___at___00Aesop_ForwardRuleMatch_getProof_spec__2___redArg___boxed(lean_object* v_e_1038_, lean_object* v___y_1039_, lean_object* v___y_1040_){
_start:
{
lean_object* v_res_1041_; 
v_res_1041_ = lp_aesop_Lean_instantiateMVars___at___00Aesop_ForwardRuleMatch_getProof_spec__2___redArg(v_e_1038_, v___y_1039_);
lean_dec(v___y_1039_);
return v_res_1041_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_instantiateMVars___at___00Aesop_ForwardRuleMatch_getProof_spec__2(lean_object* v_e_1042_, lean_object* v___y_1043_, lean_object* v___y_1044_, lean_object* v___y_1045_, lean_object* v___y_1046_){
_start:
{
lean_object* v___x_1048_; 
v___x_1048_ = lp_aesop_Lean_instantiateMVars___at___00Aesop_ForwardRuleMatch_getProof_spec__2___redArg(v_e_1042_, v___y_1044_);
return v___x_1048_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_instantiateMVars___at___00Aesop_ForwardRuleMatch_getProof_spec__2___boxed(lean_object* v_e_1049_, lean_object* v___y_1050_, lean_object* v___y_1051_, lean_object* v___y_1052_, lean_object* v___y_1053_, lean_object* v___y_1054_){
_start:
{
lean_object* v_res_1055_; 
v_res_1055_ = lp_aesop_Lean_instantiateMVars___at___00Aesop_ForwardRuleMatch_getProof_spec__2(v_e_1049_, v___y_1050_, v___y_1051_, v___y_1052_, v___y_1053_);
lean_dec(v___y_1053_);
lean_dec_ref(v___y_1052_);
lean_dec(v___y_1051_);
lean_dec_ref(v___y_1050_);
return v_res_1055_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_withNewMCtxDepth___at___00Aesop_ForwardRuleMatch_getProof_spec__9___redArg(lean_object* v_k_1056_, uint8_t v_allowLevelAssignments_1057_, lean_object* v___y_1058_, lean_object* v___y_1059_, lean_object* v___y_1060_, lean_object* v___y_1061_){
_start:
{
lean_object* v___x_1063_; 
v___x_1063_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withNewMCtxDepthImp(lean_box(0), v_allowLevelAssignments_1057_, v_k_1056_, v___y_1058_, v___y_1059_, v___y_1060_, v___y_1061_);
if (lean_obj_tag(v___x_1063_) == 0)
{
lean_object* v_a_1064_; lean_object* v___x_1066_; uint8_t v_isShared_1067_; uint8_t v_isSharedCheck_1071_; 
v_a_1064_ = lean_ctor_get(v___x_1063_, 0);
v_isSharedCheck_1071_ = !lean_is_exclusive(v___x_1063_);
if (v_isSharedCheck_1071_ == 0)
{
v___x_1066_ = v___x_1063_;
v_isShared_1067_ = v_isSharedCheck_1071_;
goto v_resetjp_1065_;
}
else
{
lean_inc(v_a_1064_);
lean_dec(v___x_1063_);
v___x_1066_ = lean_box(0);
v_isShared_1067_ = v_isSharedCheck_1071_;
goto v_resetjp_1065_;
}
v_resetjp_1065_:
{
lean_object* v___x_1069_; 
if (v_isShared_1067_ == 0)
{
v___x_1069_ = v___x_1066_;
goto v_reusejp_1068_;
}
else
{
lean_object* v_reuseFailAlloc_1070_; 
v_reuseFailAlloc_1070_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1070_, 0, v_a_1064_);
v___x_1069_ = v_reuseFailAlloc_1070_;
goto v_reusejp_1068_;
}
v_reusejp_1068_:
{
return v___x_1069_;
}
}
}
else
{
lean_object* v_a_1072_; lean_object* v___x_1074_; uint8_t v_isShared_1075_; uint8_t v_isSharedCheck_1079_; 
v_a_1072_ = lean_ctor_get(v___x_1063_, 0);
v_isSharedCheck_1079_ = !lean_is_exclusive(v___x_1063_);
if (v_isSharedCheck_1079_ == 0)
{
v___x_1074_ = v___x_1063_;
v_isShared_1075_ = v_isSharedCheck_1079_;
goto v_resetjp_1073_;
}
else
{
lean_inc(v_a_1072_);
lean_dec(v___x_1063_);
v___x_1074_ = lean_box(0);
v_isShared_1075_ = v_isSharedCheck_1079_;
goto v_resetjp_1073_;
}
v_resetjp_1073_:
{
lean_object* v___x_1077_; 
if (v_isShared_1075_ == 0)
{
v___x_1077_ = v___x_1074_;
goto v_reusejp_1076_;
}
else
{
lean_object* v_reuseFailAlloc_1078_; 
v_reuseFailAlloc_1078_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1078_, 0, v_a_1072_);
v___x_1077_ = v_reuseFailAlloc_1078_;
goto v_reusejp_1076_;
}
v_reusejp_1076_:
{
return v___x_1077_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_withNewMCtxDepth___at___00Aesop_ForwardRuleMatch_getProof_spec__9___redArg___boxed(lean_object* v_k_1080_, lean_object* v_allowLevelAssignments_1081_, lean_object* v___y_1082_, lean_object* v___y_1083_, lean_object* v___y_1084_, lean_object* v___y_1085_, lean_object* v___y_1086_){
_start:
{
uint8_t v_allowLevelAssignments_boxed_1087_; lean_object* v_res_1088_; 
v_allowLevelAssignments_boxed_1087_ = lean_unbox(v_allowLevelAssignments_1081_);
v_res_1088_ = lp_aesop_Lean_Meta_withNewMCtxDepth___at___00Aesop_ForwardRuleMatch_getProof_spec__9___redArg(v_k_1080_, v_allowLevelAssignments_boxed_1087_, v___y_1082_, v___y_1083_, v___y_1084_, v___y_1085_);
lean_dec(v___y_1085_);
lean_dec_ref(v___y_1084_);
lean_dec(v___y_1083_);
lean_dec_ref(v___y_1082_);
return v_res_1088_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_withNewMCtxDepth___at___00Aesop_ForwardRuleMatch_getProof_spec__9(lean_object* v_00_u03b1_1089_, lean_object* v_k_1090_, uint8_t v_allowLevelAssignments_1091_, lean_object* v___y_1092_, lean_object* v___y_1093_, lean_object* v___y_1094_, lean_object* v___y_1095_){
_start:
{
lean_object* v___x_1097_; 
v___x_1097_ = lp_aesop_Lean_Meta_withNewMCtxDepth___at___00Aesop_ForwardRuleMatch_getProof_spec__9___redArg(v_k_1090_, v_allowLevelAssignments_1091_, v___y_1092_, v___y_1093_, v___y_1094_, v___y_1095_);
return v___x_1097_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_withNewMCtxDepth___at___00Aesop_ForwardRuleMatch_getProof_spec__9___boxed(lean_object* v_00_u03b1_1098_, lean_object* v_k_1099_, lean_object* v_allowLevelAssignments_1100_, lean_object* v___y_1101_, lean_object* v___y_1102_, lean_object* v___y_1103_, lean_object* v___y_1104_, lean_object* v___y_1105_){
_start:
{
uint8_t v_allowLevelAssignments_boxed_1106_; lean_object* v_res_1107_; 
v_allowLevelAssignments_boxed_1106_ = lean_unbox(v_allowLevelAssignments_1100_);
v_res_1107_ = lp_aesop_Lean_Meta_withNewMCtxDepth___at___00Aesop_ForwardRuleMatch_getProof_spec__9(v_00_u03b1_1098_, v_k_1099_, v_allowLevelAssignments_boxed_1106_, v___y_1101_, v___y_1102_, v___y_1103_, v___y_1104_);
lean_dec(v___y_1104_);
lean_dec_ref(v___y_1103_);
lean_dec(v___y_1102_);
lean_dec_ref(v___y_1101_);
return v_res_1107_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_ForwardRuleMatch_getProof_spec__10___redArg(lean_object* v_mvarId_1108_, lean_object* v_x_1109_, lean_object* v___y_1110_, lean_object* v___y_1111_, lean_object* v___y_1112_, lean_object* v___y_1113_){
_start:
{
lean_object* v___x_1115_; 
v___x_1115_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_box(0), v_mvarId_1108_, v_x_1109_, v___y_1110_, v___y_1111_, v___y_1112_, v___y_1113_);
if (lean_obj_tag(v___x_1115_) == 0)
{
lean_object* v_a_1116_; lean_object* v___x_1118_; uint8_t v_isShared_1119_; uint8_t v_isSharedCheck_1123_; 
v_a_1116_ = lean_ctor_get(v___x_1115_, 0);
v_isSharedCheck_1123_ = !lean_is_exclusive(v___x_1115_);
if (v_isSharedCheck_1123_ == 0)
{
v___x_1118_ = v___x_1115_;
v_isShared_1119_ = v_isSharedCheck_1123_;
goto v_resetjp_1117_;
}
else
{
lean_inc(v_a_1116_);
lean_dec(v___x_1115_);
v___x_1118_ = lean_box(0);
v_isShared_1119_ = v_isSharedCheck_1123_;
goto v_resetjp_1117_;
}
v_resetjp_1117_:
{
lean_object* v___x_1121_; 
if (v_isShared_1119_ == 0)
{
v___x_1121_ = v___x_1118_;
goto v_reusejp_1120_;
}
else
{
lean_object* v_reuseFailAlloc_1122_; 
v_reuseFailAlloc_1122_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1122_, 0, v_a_1116_);
v___x_1121_ = v_reuseFailAlloc_1122_;
goto v_reusejp_1120_;
}
v_reusejp_1120_:
{
return v___x_1121_;
}
}
}
else
{
lean_object* v_a_1124_; lean_object* v___x_1126_; uint8_t v_isShared_1127_; uint8_t v_isSharedCheck_1131_; 
v_a_1124_ = lean_ctor_get(v___x_1115_, 0);
v_isSharedCheck_1131_ = !lean_is_exclusive(v___x_1115_);
if (v_isSharedCheck_1131_ == 0)
{
v___x_1126_ = v___x_1115_;
v_isShared_1127_ = v_isSharedCheck_1131_;
goto v_resetjp_1125_;
}
else
{
lean_inc(v_a_1124_);
lean_dec(v___x_1115_);
v___x_1126_ = lean_box(0);
v_isShared_1127_ = v_isSharedCheck_1131_;
goto v_resetjp_1125_;
}
v_resetjp_1125_:
{
lean_object* v___x_1129_; 
if (v_isShared_1127_ == 0)
{
v___x_1129_ = v___x_1126_;
goto v_reusejp_1128_;
}
else
{
lean_object* v_reuseFailAlloc_1130_; 
v_reuseFailAlloc_1130_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1130_, 0, v_a_1124_);
v___x_1129_ = v_reuseFailAlloc_1130_;
goto v_reusejp_1128_;
}
v_reusejp_1128_:
{
return v___x_1129_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_ForwardRuleMatch_getProof_spec__10___redArg___boxed(lean_object* v_mvarId_1132_, lean_object* v_x_1133_, lean_object* v___y_1134_, lean_object* v___y_1135_, lean_object* v___y_1136_, lean_object* v___y_1137_, lean_object* v___y_1138_){
_start:
{
lean_object* v_res_1139_; 
v_res_1139_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_ForwardRuleMatch_getProof_spec__10___redArg(v_mvarId_1132_, v_x_1133_, v___y_1134_, v___y_1135_, v___y_1136_, v___y_1137_);
lean_dec(v___y_1137_);
lean_dec_ref(v___y_1136_);
lean_dec(v___y_1135_);
lean_dec_ref(v___y_1134_);
return v_res_1139_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_ForwardRuleMatch_getProof_spec__10(lean_object* v_00_u03b1_1140_, lean_object* v_mvarId_1141_, lean_object* v_x_1142_, lean_object* v___y_1143_, lean_object* v___y_1144_, lean_object* v___y_1145_, lean_object* v___y_1146_){
_start:
{
lean_object* v___x_1148_; 
v___x_1148_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_ForwardRuleMatch_getProof_spec__10___redArg(v_mvarId_1141_, v_x_1142_, v___y_1143_, v___y_1144_, v___y_1145_, v___y_1146_);
return v___x_1148_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_ForwardRuleMatch_getProof_spec__10___boxed(lean_object* v_00_u03b1_1149_, lean_object* v_mvarId_1150_, lean_object* v_x_1151_, lean_object* v___y_1152_, lean_object* v___y_1153_, lean_object* v___y_1154_, lean_object* v___y_1155_, lean_object* v___y_1156_){
_start:
{
lean_object* v_res_1157_; 
v_res_1157_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_ForwardRuleMatch_getProof_spec__10(v_00_u03b1_1149_, v_mvarId_1150_, v_x_1151_, v___y_1152_, v___y_1153_, v___y_1154_, v___y_1155_);
lean_dec(v___y_1155_);
lean_dec_ref(v___y_1154_);
lean_dec(v___y_1153_);
lean_dec_ref(v___y_1152_);
return v_res_1157_;
}
}
static lean_object* _init_lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_ForwardRuleMatch_getProof_spec__11___redArg___closed__0(void){
_start:
{
lean_object* v___x_1158_; lean_object* v___x_1159_; lean_object* v___x_1160_; 
v___x_1158_ = lean_unsigned_to_nat(32u);
v___x_1159_ = lean_mk_empty_array_with_capacity(v___x_1158_);
v___x_1160_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1160_, 0, v___x_1159_);
return v___x_1160_;
}
}
static lean_object* _init_lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_ForwardRuleMatch_getProof_spec__11___redArg___closed__1(void){
_start:
{
size_t v___x_1161_; lean_object* v___x_1162_; lean_object* v___x_1163_; lean_object* v___x_1164_; lean_object* v___x_1165_; lean_object* v___x_1166_; 
v___x_1161_ = ((size_t)5ULL);
v___x_1162_ = lean_unsigned_to_nat(0u);
v___x_1163_ = lean_unsigned_to_nat(32u);
v___x_1164_ = lean_mk_empty_array_with_capacity(v___x_1163_);
v___x_1165_ = lean_obj_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_ForwardRuleMatch_getProof_spec__11___redArg___closed__0, &lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_ForwardRuleMatch_getProof_spec__11___redArg___closed__0_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_ForwardRuleMatch_getProof_spec__11___redArg___closed__0);
v___x_1166_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_1166_, 0, v___x_1165_);
lean_ctor_set(v___x_1166_, 1, v___x_1164_);
lean_ctor_set(v___x_1166_, 2, v___x_1162_);
lean_ctor_set(v___x_1166_, 3, v___x_1162_);
lean_ctor_set_usize(v___x_1166_, 4, v___x_1161_);
return v___x_1166_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_ForwardRuleMatch_getProof_spec__11___redArg(lean_object* v___y_1167_){
_start:
{
lean_object* v___x_1169_; lean_object* v_traceState_1170_; lean_object* v_traces_1171_; lean_object* v___x_1172_; lean_object* v_traceState_1173_; lean_object* v_env_1174_; lean_object* v_nextMacroScope_1175_; lean_object* v_ngen_1176_; lean_object* v_auxDeclNGen_1177_; lean_object* v_cache_1178_; lean_object* v_messages_1179_; lean_object* v_infoState_1180_; lean_object* v_snapshotTasks_1181_; lean_object* v___x_1183_; uint8_t v_isShared_1184_; uint8_t v_isSharedCheck_1200_; 
v___x_1169_ = lean_st_ref_get(v___y_1167_);
v_traceState_1170_ = lean_ctor_get(v___x_1169_, 4);
lean_inc_ref(v_traceState_1170_);
lean_dec(v___x_1169_);
v_traces_1171_ = lean_ctor_get(v_traceState_1170_, 0);
lean_inc_ref(v_traces_1171_);
lean_dec_ref(v_traceState_1170_);
v___x_1172_ = lean_st_ref_take(v___y_1167_);
v_traceState_1173_ = lean_ctor_get(v___x_1172_, 4);
v_env_1174_ = lean_ctor_get(v___x_1172_, 0);
v_nextMacroScope_1175_ = lean_ctor_get(v___x_1172_, 1);
v_ngen_1176_ = lean_ctor_get(v___x_1172_, 2);
v_auxDeclNGen_1177_ = lean_ctor_get(v___x_1172_, 3);
v_cache_1178_ = lean_ctor_get(v___x_1172_, 5);
v_messages_1179_ = lean_ctor_get(v___x_1172_, 6);
v_infoState_1180_ = lean_ctor_get(v___x_1172_, 7);
v_snapshotTasks_1181_ = lean_ctor_get(v___x_1172_, 8);
v_isSharedCheck_1200_ = !lean_is_exclusive(v___x_1172_);
if (v_isSharedCheck_1200_ == 0)
{
v___x_1183_ = v___x_1172_;
v_isShared_1184_ = v_isSharedCheck_1200_;
goto v_resetjp_1182_;
}
else
{
lean_inc(v_snapshotTasks_1181_);
lean_inc(v_infoState_1180_);
lean_inc(v_messages_1179_);
lean_inc(v_cache_1178_);
lean_inc(v_traceState_1173_);
lean_inc(v_auxDeclNGen_1177_);
lean_inc(v_ngen_1176_);
lean_inc(v_nextMacroScope_1175_);
lean_inc(v_env_1174_);
lean_dec(v___x_1172_);
v___x_1183_ = lean_box(0);
v_isShared_1184_ = v_isSharedCheck_1200_;
goto v_resetjp_1182_;
}
v_resetjp_1182_:
{
uint64_t v_tid_1185_; lean_object* v___x_1187_; uint8_t v_isShared_1188_; uint8_t v_isSharedCheck_1198_; 
v_tid_1185_ = lean_ctor_get_uint64(v_traceState_1173_, sizeof(void*)*1);
v_isSharedCheck_1198_ = !lean_is_exclusive(v_traceState_1173_);
if (v_isSharedCheck_1198_ == 0)
{
lean_object* v_unused_1199_; 
v_unused_1199_ = lean_ctor_get(v_traceState_1173_, 0);
lean_dec(v_unused_1199_);
v___x_1187_ = v_traceState_1173_;
v_isShared_1188_ = v_isSharedCheck_1198_;
goto v_resetjp_1186_;
}
else
{
lean_dec(v_traceState_1173_);
v___x_1187_ = lean_box(0);
v_isShared_1188_ = v_isSharedCheck_1198_;
goto v_resetjp_1186_;
}
v_resetjp_1186_:
{
lean_object* v___x_1189_; lean_object* v___x_1191_; 
v___x_1189_ = lean_obj_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_ForwardRuleMatch_getProof_spec__11___redArg___closed__1, &lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_ForwardRuleMatch_getProof_spec__11___redArg___closed__1_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_ForwardRuleMatch_getProof_spec__11___redArg___closed__1);
if (v_isShared_1188_ == 0)
{
lean_ctor_set(v___x_1187_, 0, v___x_1189_);
v___x_1191_ = v___x_1187_;
goto v_reusejp_1190_;
}
else
{
lean_object* v_reuseFailAlloc_1197_; 
v_reuseFailAlloc_1197_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_1197_, 0, v___x_1189_);
lean_ctor_set_uint64(v_reuseFailAlloc_1197_, sizeof(void*)*1, v_tid_1185_);
v___x_1191_ = v_reuseFailAlloc_1197_;
goto v_reusejp_1190_;
}
v_reusejp_1190_:
{
lean_object* v___x_1193_; 
if (v_isShared_1184_ == 0)
{
lean_ctor_set(v___x_1183_, 4, v___x_1191_);
v___x_1193_ = v___x_1183_;
goto v_reusejp_1192_;
}
else
{
lean_object* v_reuseFailAlloc_1196_; 
v_reuseFailAlloc_1196_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_1196_, 0, v_env_1174_);
lean_ctor_set(v_reuseFailAlloc_1196_, 1, v_nextMacroScope_1175_);
lean_ctor_set(v_reuseFailAlloc_1196_, 2, v_ngen_1176_);
lean_ctor_set(v_reuseFailAlloc_1196_, 3, v_auxDeclNGen_1177_);
lean_ctor_set(v_reuseFailAlloc_1196_, 4, v___x_1191_);
lean_ctor_set(v_reuseFailAlloc_1196_, 5, v_cache_1178_);
lean_ctor_set(v_reuseFailAlloc_1196_, 6, v_messages_1179_);
lean_ctor_set(v_reuseFailAlloc_1196_, 7, v_infoState_1180_);
lean_ctor_set(v_reuseFailAlloc_1196_, 8, v_snapshotTasks_1181_);
v___x_1193_ = v_reuseFailAlloc_1196_;
goto v_reusejp_1192_;
}
v_reusejp_1192_:
{
lean_object* v___x_1194_; lean_object* v___x_1195_; 
v___x_1194_ = lean_st_ref_set(v___y_1167_, v___x_1193_);
v___x_1195_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1195_, 0, v_traces_1171_);
return v___x_1195_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_ForwardRuleMatch_getProof_spec__11___redArg___boxed(lean_object* v___y_1201_, lean_object* v___y_1202_){
_start:
{
lean_object* v_res_1203_; 
v_res_1203_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_ForwardRuleMatch_getProof_spec__11___redArg(v___y_1201_);
lean_dec(v___y_1201_);
return v_res_1203_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_ForwardRuleMatch_getProof_spec__11(lean_object* v___y_1204_, lean_object* v___y_1205_, lean_object* v___y_1206_, lean_object* v___y_1207_){
_start:
{
lean_object* v___x_1209_; 
v___x_1209_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_ForwardRuleMatch_getProof_spec__11___redArg(v___y_1207_);
return v___x_1209_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_ForwardRuleMatch_getProof_spec__11___boxed(lean_object* v___y_1210_, lean_object* v___y_1211_, lean_object* v___y_1212_, lean_object* v___y_1213_, lean_object* v___y_1214_){
_start:
{
lean_object* v_res_1215_; 
v_res_1215_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_ForwardRuleMatch_getProof_spec__11(v___y_1210_, v___y_1211_, v___y_1212_, v___y_1213_);
lean_dec(v___y_1213_);
lean_dec_ref(v___y_1212_);
lean_dec(v___y_1211_);
lean_dec_ref(v___y_1210_);
return v_res_1215_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_Option_get___at___00Aesop_ForwardRuleMatch_getProof_spec__12(lean_object* v_opts_1216_, lean_object* v_opt_1217_){
_start:
{
lean_object* v_name_1218_; lean_object* v_defValue_1219_; lean_object* v_map_1220_; lean_object* v___x_1221_; 
v_name_1218_ = lean_ctor_get(v_opt_1217_, 0);
v_defValue_1219_ = lean_ctor_get(v_opt_1217_, 1);
v_map_1220_ = lean_ctor_get(v_opts_1216_, 0);
v___x_1221_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_1220_, v_name_1218_);
if (lean_obj_tag(v___x_1221_) == 0)
{
uint8_t v___x_1222_; 
v___x_1222_ = lean_unbox(v_defValue_1219_);
return v___x_1222_;
}
else
{
lean_object* v_val_1223_; 
v_val_1223_ = lean_ctor_get(v___x_1221_, 0);
lean_inc(v_val_1223_);
lean_dec_ref_known(v___x_1221_, 1);
if (lean_obj_tag(v_val_1223_) == 1)
{
uint8_t v_v_1224_; 
v_v_1224_ = lean_ctor_get_uint8(v_val_1223_, 0);
lean_dec_ref_known(v_val_1223_, 0);
return v_v_1224_;
}
else
{
uint8_t v___x_1225_; 
lean_dec(v_val_1223_);
v___x_1225_ = lean_unbox(v_defValue_1219_);
return v___x_1225_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Aesop_ForwardRuleMatch_getProof_spec__12___boxed(lean_object* v_opts_1226_, lean_object* v_opt_1227_){
_start:
{
uint8_t v_res_1228_; lean_object* v_r_1229_; 
v_res_1228_ = lp_aesop_Lean_Option_get___at___00Aesop_ForwardRuleMatch_getProof_spec__12(v_opts_1226_, v_opt_1227_);
lean_dec_ref(v_opt_1227_);
lean_dec_ref(v_opts_1226_);
v_r_1229_ = lean_box(v_res_1228_);
return v_r_1229_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__0(lean_object* v_____r_1232_, lean_object* v___y_1233_, lean_object* v___y_1234_, lean_object* v___y_1235_, lean_object* v___y_1236_){
_start:
{
lean_object* v___x_1238_; lean_object* v___x_1239_; 
v___x_1238_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__0___closed__0));
v___x_1239_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1239_, 0, v___x_1238_);
return v___x_1239_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__0___boxed(lean_object* v_____r_1240_, lean_object* v___y_1241_, lean_object* v___y_1242_, lean_object* v___y_1243_, lean_object* v___y_1244_, lean_object* v___y_1245_){
_start:
{
lean_object* v_res_1246_; 
v_res_1246_ = lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__0(v_____r_1240_, v___y_1241_, v___y_1242_, v___y_1243_, v___y_1244_);
lean_dec(v___y_1244_);
lean_dec_ref(v___y_1243_);
lean_dec(v___y_1242_);
lean_dec_ref(v___y_1241_);
return v_res_1246_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_ForwardRuleMatch_getProof_spec__7(size_t v_sz_1247_, size_t v_i_1248_, lean_object* v_bs_1249_){
_start:
{
uint8_t v___x_1250_; 
v___x_1250_ = lean_usize_dec_lt(v_i_1248_, v_sz_1247_);
if (v___x_1250_ == 0)
{
return v_bs_1249_;
}
else
{
lean_object* v_v_1251_; lean_object* v___x_1252_; lean_object* v_bs_x27_1253_; lean_object* v___y_1255_; 
v_v_1251_ = lean_array_uget(v_bs_1249_, v_i_1248_);
v___x_1252_ = lean_unsigned_to_nat(0u);
v_bs_x27_1253_ = lean_array_uset(v_bs_1249_, v_i_1248_, v___x_1252_);
if (lean_obj_tag(v_v_1251_) == 0)
{
lean_object* v___x_1260_; 
v___x_1260_ = lean_obj_once(&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CompleteMatch_toMessageData_spec__0___closed__1, &lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CompleteMatch_toMessageData_spec__0___closed__1_once, _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CompleteMatch_toMessageData_spec__0___closed__1);
v___y_1255_ = v___x_1260_;
goto v___jp_1254_;
}
else
{
lean_object* v_val_1261_; lean_object* v___x_1262_; 
v_val_1261_ = lean_ctor_get(v_v_1251_, 0);
lean_inc(v_val_1261_);
lean_dec_ref_known(v_v_1251_, 1);
v___x_1262_ = l_Lean_MessageData_ofLevel(v_val_1261_);
v___y_1255_ = v___x_1262_;
goto v___jp_1254_;
}
v___jp_1254_:
{
size_t v___x_1256_; size_t v___x_1257_; lean_object* v___x_1258_; 
v___x_1256_ = ((size_t)1ULL);
v___x_1257_ = lean_usize_add(v_i_1248_, v___x_1256_);
v___x_1258_ = lean_array_uset(v_bs_x27_1253_, v_i_1248_, v___y_1255_);
v_i_1248_ = v___x_1257_;
v_bs_1249_ = v___x_1258_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_ForwardRuleMatch_getProof_spec__7___boxed(lean_object* v_sz_1263_, lean_object* v_i_1264_, lean_object* v_bs_1265_){
_start:
{
size_t v_sz_boxed_1266_; size_t v_i_boxed_1267_; lean_object* v_res_1268_; 
v_sz_boxed_1266_ = lean_unbox_usize(v_sz_1263_);
lean_dec(v_sz_1263_);
v_i_boxed_1267_ = lean_unbox_usize(v_i_1264_);
lean_dec(v_i_1264_);
v_res_1268_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_ForwardRuleMatch_getProof_spec__7(v_sz_boxed_1266_, v_i_boxed_1267_, v_bs_1265_);
return v_res_1268_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_ForwardRuleMatch_getProof_spec__3___redArg(lean_object* v_opt_1269_, lean_object* v___y_1270_){
_start:
{
lean_object* v_options_1272_; lean_object* v_option_1273_; uint8_t v___x_1274_; lean_object* v___x_1275_; lean_object* v___x_1276_; 
v_options_1272_ = lean_ctor_get(v___y_1270_, 2);
v_option_1273_ = lean_ctor_get(v_opt_1269_, 1);
v___x_1274_ = lp_aesop_Lean_Option_get___at___00Aesop_ForwardRuleMatch_getProof_spec__12(v_options_1272_, v_option_1273_);
v___x_1275_ = lean_box(v___x_1274_);
v___x_1276_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1276_, 0, v___x_1275_);
return v___x_1276_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_ForwardRuleMatch_getProof_spec__3___redArg___boxed(lean_object* v_opt_1277_, lean_object* v___y_1278_, lean_object* v___y_1279_){
_start:
{
lean_object* v_res_1280_; 
v_res_1280_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_ForwardRuleMatch_getProof_spec__3___redArg(v_opt_1277_, v___y_1278_);
lean_dec_ref(v___y_1278_);
lean_dec_ref(v_opt_1277_);
return v_res_1280_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_addTrace___at___00Aesop_ForwardRuleMatch_getProof_spec__4_spec__6(lean_object* v_msgData_1281_, lean_object* v___y_1282_, lean_object* v___y_1283_, lean_object* v___y_1284_, lean_object* v___y_1285_){
_start:
{
lean_object* v___x_1287_; lean_object* v_env_1288_; lean_object* v___x_1289_; lean_object* v_mctx_1290_; lean_object* v_lctx_1291_; lean_object* v_options_1292_; lean_object* v___x_1293_; lean_object* v___x_1294_; lean_object* v___x_1295_; 
v___x_1287_ = lean_st_ref_get(v___y_1285_);
v_env_1288_ = lean_ctor_get(v___x_1287_, 0);
lean_inc_ref(v_env_1288_);
lean_dec(v___x_1287_);
v___x_1289_ = lean_st_ref_get(v___y_1283_);
v_mctx_1290_ = lean_ctor_get(v___x_1289_, 0);
lean_inc_ref(v_mctx_1290_);
lean_dec(v___x_1289_);
v_lctx_1291_ = lean_ctor_get(v___y_1282_, 2);
v_options_1292_ = lean_ctor_get(v___y_1284_, 2);
lean_inc_ref(v_options_1292_);
lean_inc_ref(v_lctx_1291_);
v___x_1293_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1293_, 0, v_env_1288_);
lean_ctor_set(v___x_1293_, 1, v_mctx_1290_);
lean_ctor_set(v___x_1293_, 2, v_lctx_1291_);
lean_ctor_set(v___x_1293_, 3, v_options_1292_);
v___x_1294_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_1294_, 0, v___x_1293_);
lean_ctor_set(v___x_1294_, 1, v_msgData_1281_);
v___x_1295_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1295_, 0, v___x_1294_);
return v___x_1295_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextFull___at___00Lean_addTrace___at___00Aesop_ForwardRuleMatch_getProof_spec__4_spec__6___boxed(lean_object* v_msgData_1296_, lean_object* v___y_1297_, lean_object* v___y_1298_, lean_object* v___y_1299_, lean_object* v___y_1300_, lean_object* v___y_1301_){
_start:
{
lean_object* v_res_1302_; 
v_res_1302_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_addTrace___at___00Aesop_ForwardRuleMatch_getProof_spec__4_spec__6(v_msgData_1296_, v___y_1297_, v___y_1298_, v___y_1299_, v___y_1300_);
lean_dec(v___y_1300_);
lean_dec_ref(v___y_1299_);
lean_dec(v___y_1298_);
lean_dec_ref(v___y_1297_);
return v_res_1302_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_ForwardRuleMatch_getProof_spec__8___redArg(lean_object* v_msg_1303_, lean_object* v___y_1304_, lean_object* v___y_1305_, lean_object* v___y_1306_, lean_object* v___y_1307_){
_start:
{
lean_object* v_ref_1309_; lean_object* v___x_1310_; lean_object* v_a_1311_; lean_object* v___x_1313_; uint8_t v_isShared_1314_; uint8_t v_isSharedCheck_1319_; 
v_ref_1309_ = lean_ctor_get(v___y_1306_, 5);
v___x_1310_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_addTrace___at___00Aesop_ForwardRuleMatch_getProof_spec__4_spec__6(v_msg_1303_, v___y_1304_, v___y_1305_, v___y_1306_, v___y_1307_);
v_a_1311_ = lean_ctor_get(v___x_1310_, 0);
v_isSharedCheck_1319_ = !lean_is_exclusive(v___x_1310_);
if (v_isSharedCheck_1319_ == 0)
{
v___x_1313_ = v___x_1310_;
v_isShared_1314_ = v_isSharedCheck_1319_;
goto v_resetjp_1312_;
}
else
{
lean_inc(v_a_1311_);
lean_dec(v___x_1310_);
v___x_1313_ = lean_box(0);
v_isShared_1314_ = v_isSharedCheck_1319_;
goto v_resetjp_1312_;
}
v_resetjp_1312_:
{
lean_object* v___x_1315_; lean_object* v___x_1317_; 
lean_inc(v_ref_1309_);
v___x_1315_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1315_, 0, v_ref_1309_);
lean_ctor_set(v___x_1315_, 1, v_a_1311_);
if (v_isShared_1314_ == 0)
{
lean_ctor_set_tag(v___x_1313_, 1);
lean_ctor_set(v___x_1313_, 0, v___x_1315_);
v___x_1317_ = v___x_1313_;
goto v_reusejp_1316_;
}
else
{
lean_object* v_reuseFailAlloc_1318_; 
v_reuseFailAlloc_1318_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1318_, 0, v___x_1315_);
v___x_1317_ = v_reuseFailAlloc_1318_;
goto v_reusejp_1316_;
}
v_reusejp_1316_:
{
return v___x_1317_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_ForwardRuleMatch_getProof_spec__8___redArg___boxed(lean_object* v_msg_1320_, lean_object* v___y_1321_, lean_object* v___y_1322_, lean_object* v___y_1323_, lean_object* v___y_1324_, lean_object* v___y_1325_){
_start:
{
lean_object* v_res_1326_; 
v_res_1326_ = lp_aesop_Lean_throwError___at___00Aesop_ForwardRuleMatch_getProof_spec__8___redArg(v_msg_1320_, v___y_1321_, v___y_1322_, v___y_1323_, v___y_1324_);
lean_dec(v___y_1324_);
lean_dec_ref(v___y_1323_);
lean_dec(v___y_1322_);
lean_dec_ref(v___y_1321_);
return v_res_1326_;
}
}
static double _init_lp_aesop_Lean_addTrace___at___00Aesop_ForwardRuleMatch_getProof_spec__4___closed__0(void){
_start:
{
lean_object* v___x_1327_; double v___x_1328_; 
v___x_1327_ = lean_unsigned_to_nat(0u);
v___x_1328_ = lean_float_of_nat(v___x_1327_);
return v___x_1328_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00Aesop_ForwardRuleMatch_getProof_spec__4(lean_object* v_cls_1332_, lean_object* v_msg_1333_, lean_object* v___y_1334_, lean_object* v___y_1335_, lean_object* v___y_1336_, lean_object* v___y_1337_){
_start:
{
lean_object* v_ref_1339_; lean_object* v___x_1340_; lean_object* v_a_1341_; lean_object* v___x_1343_; uint8_t v_isShared_1344_; uint8_t v_isSharedCheck_1385_; 
v_ref_1339_ = lean_ctor_get(v___y_1336_, 5);
v___x_1340_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_addTrace___at___00Aesop_ForwardRuleMatch_getProof_spec__4_spec__6(v_msg_1333_, v___y_1334_, v___y_1335_, v___y_1336_, v___y_1337_);
v_a_1341_ = lean_ctor_get(v___x_1340_, 0);
v_isSharedCheck_1385_ = !lean_is_exclusive(v___x_1340_);
if (v_isSharedCheck_1385_ == 0)
{
v___x_1343_ = v___x_1340_;
v_isShared_1344_ = v_isSharedCheck_1385_;
goto v_resetjp_1342_;
}
else
{
lean_inc(v_a_1341_);
lean_dec(v___x_1340_);
v___x_1343_ = lean_box(0);
v_isShared_1344_ = v_isSharedCheck_1385_;
goto v_resetjp_1342_;
}
v_resetjp_1342_:
{
lean_object* v___x_1345_; lean_object* v_traceState_1346_; lean_object* v_env_1347_; lean_object* v_nextMacroScope_1348_; lean_object* v_ngen_1349_; lean_object* v_auxDeclNGen_1350_; lean_object* v_cache_1351_; lean_object* v_messages_1352_; lean_object* v_infoState_1353_; lean_object* v_snapshotTasks_1354_; lean_object* v___x_1356_; uint8_t v_isShared_1357_; uint8_t v_isSharedCheck_1384_; 
v___x_1345_ = lean_st_ref_take(v___y_1337_);
v_traceState_1346_ = lean_ctor_get(v___x_1345_, 4);
v_env_1347_ = lean_ctor_get(v___x_1345_, 0);
v_nextMacroScope_1348_ = lean_ctor_get(v___x_1345_, 1);
v_ngen_1349_ = lean_ctor_get(v___x_1345_, 2);
v_auxDeclNGen_1350_ = lean_ctor_get(v___x_1345_, 3);
v_cache_1351_ = lean_ctor_get(v___x_1345_, 5);
v_messages_1352_ = lean_ctor_get(v___x_1345_, 6);
v_infoState_1353_ = lean_ctor_get(v___x_1345_, 7);
v_snapshotTasks_1354_ = lean_ctor_get(v___x_1345_, 8);
v_isSharedCheck_1384_ = !lean_is_exclusive(v___x_1345_);
if (v_isSharedCheck_1384_ == 0)
{
v___x_1356_ = v___x_1345_;
v_isShared_1357_ = v_isSharedCheck_1384_;
goto v_resetjp_1355_;
}
else
{
lean_inc(v_snapshotTasks_1354_);
lean_inc(v_infoState_1353_);
lean_inc(v_messages_1352_);
lean_inc(v_cache_1351_);
lean_inc(v_traceState_1346_);
lean_inc(v_auxDeclNGen_1350_);
lean_inc(v_ngen_1349_);
lean_inc(v_nextMacroScope_1348_);
lean_inc(v_env_1347_);
lean_dec(v___x_1345_);
v___x_1356_ = lean_box(0);
v_isShared_1357_ = v_isSharedCheck_1384_;
goto v_resetjp_1355_;
}
v_resetjp_1355_:
{
uint64_t v_tid_1358_; lean_object* v_traces_1359_; lean_object* v___x_1361_; uint8_t v_isShared_1362_; uint8_t v_isSharedCheck_1383_; 
v_tid_1358_ = lean_ctor_get_uint64(v_traceState_1346_, sizeof(void*)*1);
v_traces_1359_ = lean_ctor_get(v_traceState_1346_, 0);
v_isSharedCheck_1383_ = !lean_is_exclusive(v_traceState_1346_);
if (v_isSharedCheck_1383_ == 0)
{
v___x_1361_ = v_traceState_1346_;
v_isShared_1362_ = v_isSharedCheck_1383_;
goto v_resetjp_1360_;
}
else
{
lean_inc(v_traces_1359_);
lean_dec(v_traceState_1346_);
v___x_1361_ = lean_box(0);
v_isShared_1362_ = v_isSharedCheck_1383_;
goto v_resetjp_1360_;
}
v_resetjp_1360_:
{
lean_object* v___x_1363_; double v___x_1364_; uint8_t v___x_1365_; lean_object* v___x_1366_; lean_object* v___x_1367_; lean_object* v___x_1368_; lean_object* v___x_1369_; lean_object* v___x_1370_; lean_object* v___x_1371_; lean_object* v___x_1373_; 
v___x_1363_ = lean_box(0);
v___x_1364_ = lean_float_once(&lp_aesop_Lean_addTrace___at___00Aesop_ForwardRuleMatch_getProof_spec__4___closed__0, &lp_aesop_Lean_addTrace___at___00Aesop_ForwardRuleMatch_getProof_spec__4___closed__0_once, _init_lp_aesop_Lean_addTrace___at___00Aesop_ForwardRuleMatch_getProof_spec__4___closed__0);
v___x_1365_ = 0;
v___x_1366_ = ((lean_object*)(lp_aesop_Lean_addTrace___at___00Aesop_ForwardRuleMatch_getProof_spec__4___closed__1));
v___x_1367_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v___x_1367_, 0, v_cls_1332_);
lean_ctor_set(v___x_1367_, 1, v___x_1363_);
lean_ctor_set(v___x_1367_, 2, v___x_1366_);
lean_ctor_set_float(v___x_1367_, sizeof(void*)*3, v___x_1364_);
lean_ctor_set_float(v___x_1367_, sizeof(void*)*3 + 8, v___x_1364_);
lean_ctor_set_uint8(v___x_1367_, sizeof(void*)*3 + 16, v___x_1365_);
v___x_1368_ = ((lean_object*)(lp_aesop_Lean_addTrace___at___00Aesop_ForwardRuleMatch_getProof_spec__4___closed__2));
v___x_1369_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v___x_1369_, 0, v___x_1367_);
lean_ctor_set(v___x_1369_, 1, v_a_1341_);
lean_ctor_set(v___x_1369_, 2, v___x_1368_);
lean_inc(v_ref_1339_);
v___x_1370_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1370_, 0, v_ref_1339_);
lean_ctor_set(v___x_1370_, 1, v___x_1369_);
v___x_1371_ = l_Lean_PersistentArray_push___redArg(v_traces_1359_, v___x_1370_);
if (v_isShared_1362_ == 0)
{
lean_ctor_set(v___x_1361_, 0, v___x_1371_);
v___x_1373_ = v___x_1361_;
goto v_reusejp_1372_;
}
else
{
lean_object* v_reuseFailAlloc_1382_; 
v_reuseFailAlloc_1382_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_1382_, 0, v___x_1371_);
lean_ctor_set_uint64(v_reuseFailAlloc_1382_, sizeof(void*)*1, v_tid_1358_);
v___x_1373_ = v_reuseFailAlloc_1382_;
goto v_reusejp_1372_;
}
v_reusejp_1372_:
{
lean_object* v___x_1375_; 
if (v_isShared_1357_ == 0)
{
lean_ctor_set(v___x_1356_, 4, v___x_1373_);
v___x_1375_ = v___x_1356_;
goto v_reusejp_1374_;
}
else
{
lean_object* v_reuseFailAlloc_1381_; 
v_reuseFailAlloc_1381_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_1381_, 0, v_env_1347_);
lean_ctor_set(v_reuseFailAlloc_1381_, 1, v_nextMacroScope_1348_);
lean_ctor_set(v_reuseFailAlloc_1381_, 2, v_ngen_1349_);
lean_ctor_set(v_reuseFailAlloc_1381_, 3, v_auxDeclNGen_1350_);
lean_ctor_set(v_reuseFailAlloc_1381_, 4, v___x_1373_);
lean_ctor_set(v_reuseFailAlloc_1381_, 5, v_cache_1351_);
lean_ctor_set(v_reuseFailAlloc_1381_, 6, v_messages_1352_);
lean_ctor_set(v_reuseFailAlloc_1381_, 7, v_infoState_1353_);
lean_ctor_set(v_reuseFailAlloc_1381_, 8, v_snapshotTasks_1354_);
v___x_1375_ = v_reuseFailAlloc_1381_;
goto v_reusejp_1374_;
}
v_reusejp_1374_:
{
lean_object* v___x_1376_; lean_object* v___x_1377_; lean_object* v___x_1379_; 
v___x_1376_ = lean_st_ref_set(v___y_1337_, v___x_1375_);
v___x_1377_ = lean_box(0);
if (v_isShared_1344_ == 0)
{
lean_ctor_set(v___x_1343_, 0, v___x_1377_);
v___x_1379_ = v___x_1343_;
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
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00Aesop_ForwardRuleMatch_getProof_spec__4___boxed(lean_object* v_cls_1386_, lean_object* v_msg_1387_, lean_object* v___y_1388_, lean_object* v___y_1389_, lean_object* v___y_1390_, lean_object* v___y_1391_, lean_object* v___y_1392_){
_start:
{
lean_object* v_res_1393_; 
v_res_1393_ = lp_aesop_Lean_addTrace___at___00Aesop_ForwardRuleMatch_getProof_spec__4(v_cls_1386_, v_msg_1387_, v___y_1388_, v___y_1389_, v___y_1390_, v___y_1391_);
lean_dec(v___y_1391_);
lean_dec_ref(v___y_1390_);
lean_dec(v___y_1389_);
lean_dec_ref(v___y_1388_);
return v_res_1393_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_ForwardRuleMatch_getProof_spec__1_spec__2_spec__9_spec__23_spec__26___redArg(lean_object* v_x_1394_, lean_object* v_x_1395_, lean_object* v_x_1396_, lean_object* v_x_1397_){
_start:
{
lean_object* v_ks_1398_; lean_object* v_vs_1399_; lean_object* v___x_1401_; uint8_t v_isShared_1402_; uint8_t v_isSharedCheck_1423_; 
v_ks_1398_ = lean_ctor_get(v_x_1394_, 0);
v_vs_1399_ = lean_ctor_get(v_x_1394_, 1);
v_isSharedCheck_1423_ = !lean_is_exclusive(v_x_1394_);
if (v_isSharedCheck_1423_ == 0)
{
v___x_1401_ = v_x_1394_;
v_isShared_1402_ = v_isSharedCheck_1423_;
goto v_resetjp_1400_;
}
else
{
lean_inc(v_vs_1399_);
lean_inc(v_ks_1398_);
lean_dec(v_x_1394_);
v___x_1401_ = lean_box(0);
v_isShared_1402_ = v_isSharedCheck_1423_;
goto v_resetjp_1400_;
}
v_resetjp_1400_:
{
lean_object* v___x_1403_; uint8_t v___x_1404_; 
v___x_1403_ = lean_array_get_size(v_ks_1398_);
v___x_1404_ = lean_nat_dec_lt(v_x_1395_, v___x_1403_);
if (v___x_1404_ == 0)
{
lean_object* v___x_1405_; lean_object* v___x_1406_; lean_object* v___x_1408_; 
lean_dec(v_x_1395_);
v___x_1405_ = lean_array_push(v_ks_1398_, v_x_1396_);
v___x_1406_ = lean_array_push(v_vs_1399_, v_x_1397_);
if (v_isShared_1402_ == 0)
{
lean_ctor_set(v___x_1401_, 1, v___x_1406_);
lean_ctor_set(v___x_1401_, 0, v___x_1405_);
v___x_1408_ = v___x_1401_;
goto v_reusejp_1407_;
}
else
{
lean_object* v_reuseFailAlloc_1409_; 
v_reuseFailAlloc_1409_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1409_, 0, v___x_1405_);
lean_ctor_set(v_reuseFailAlloc_1409_, 1, v___x_1406_);
v___x_1408_ = v_reuseFailAlloc_1409_;
goto v_reusejp_1407_;
}
v_reusejp_1407_:
{
return v___x_1408_;
}
}
else
{
lean_object* v_k_x27_1410_; uint8_t v___x_1411_; 
v_k_x27_1410_ = lean_array_fget_borrowed(v_ks_1398_, v_x_1395_);
v___x_1411_ = l_Lean_instBEqLevelMVarId_beq(v_x_1396_, v_k_x27_1410_);
if (v___x_1411_ == 0)
{
lean_object* v___x_1413_; 
if (v_isShared_1402_ == 0)
{
v___x_1413_ = v___x_1401_;
goto v_reusejp_1412_;
}
else
{
lean_object* v_reuseFailAlloc_1417_; 
v_reuseFailAlloc_1417_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1417_, 0, v_ks_1398_);
lean_ctor_set(v_reuseFailAlloc_1417_, 1, v_vs_1399_);
v___x_1413_ = v_reuseFailAlloc_1417_;
goto v_reusejp_1412_;
}
v_reusejp_1412_:
{
lean_object* v___x_1414_; lean_object* v___x_1415_; 
v___x_1414_ = lean_unsigned_to_nat(1u);
v___x_1415_ = lean_nat_add(v_x_1395_, v___x_1414_);
lean_dec(v_x_1395_);
v_x_1394_ = v___x_1413_;
v_x_1395_ = v___x_1415_;
goto _start;
}
}
else
{
lean_object* v___x_1418_; lean_object* v___x_1419_; lean_object* v___x_1421_; 
v___x_1418_ = lean_array_fset(v_ks_1398_, v_x_1395_, v_x_1396_);
v___x_1419_ = lean_array_fset(v_vs_1399_, v_x_1395_, v_x_1397_);
lean_dec(v_x_1395_);
if (v_isShared_1402_ == 0)
{
lean_ctor_set(v___x_1401_, 1, v___x_1419_);
lean_ctor_set(v___x_1401_, 0, v___x_1418_);
v___x_1421_ = v___x_1401_;
goto v_reusejp_1420_;
}
else
{
lean_object* v_reuseFailAlloc_1422_; 
v_reuseFailAlloc_1422_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1422_, 0, v___x_1418_);
lean_ctor_set(v_reuseFailAlloc_1422_, 1, v___x_1419_);
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
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_ForwardRuleMatch_getProof_spec__1_spec__2_spec__9_spec__23___redArg(lean_object* v_n_1424_, lean_object* v_k_1425_, lean_object* v_v_1426_){
_start:
{
lean_object* v___x_1427_; lean_object* v___x_1428_; 
v___x_1427_ = lean_unsigned_to_nat(0u);
v___x_1428_ = lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_ForwardRuleMatch_getProof_spec__1_spec__2_spec__9_spec__23_spec__26___redArg(v_n_1424_, v___x_1427_, v_k_1425_, v_v_1426_);
return v___x_1428_;
}
}
static lean_object* _init_lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_ForwardRuleMatch_getProof_spec__1_spec__2_spec__9___redArg___closed__0(void){
_start:
{
lean_object* v___x_1429_; 
v___x_1429_ = l_Lean_PersistentHashMap_mkEmptyEntries(lean_box(0), lean_box(0));
return v___x_1429_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_ForwardRuleMatch_getProof_spec__1_spec__2_spec__9___redArg(lean_object* v_x_1430_, size_t v_x_1431_, size_t v_x_1432_, lean_object* v_x_1433_, lean_object* v_x_1434_){
_start:
{
if (lean_obj_tag(v_x_1430_) == 0)
{
lean_object* v_es_1435_; size_t v___x_1436_; size_t v___x_1437_; lean_object* v_j_1438_; lean_object* v___x_1439_; uint8_t v___x_1440_; 
v_es_1435_ = lean_ctor_get(v_x_1430_, 0);
v___x_1436_ = ((size_t)31ULL);
v___x_1437_ = lean_usize_land(v_x_1431_, v___x_1436_);
v_j_1438_ = lean_usize_to_nat(v___x_1437_);
v___x_1439_ = lean_array_get_size(v_es_1435_);
v___x_1440_ = lean_nat_dec_lt(v_j_1438_, v___x_1439_);
if (v___x_1440_ == 0)
{
lean_dec(v_j_1438_);
lean_dec(v_x_1434_);
lean_dec(v_x_1433_);
return v_x_1430_;
}
else
{
lean_object* v___x_1442_; uint8_t v_isShared_1443_; uint8_t v_isSharedCheck_1479_; 
lean_inc_ref(v_es_1435_);
v_isSharedCheck_1479_ = !lean_is_exclusive(v_x_1430_);
if (v_isSharedCheck_1479_ == 0)
{
lean_object* v_unused_1480_; 
v_unused_1480_ = lean_ctor_get(v_x_1430_, 0);
lean_dec(v_unused_1480_);
v___x_1442_ = v_x_1430_;
v_isShared_1443_ = v_isSharedCheck_1479_;
goto v_resetjp_1441_;
}
else
{
lean_dec(v_x_1430_);
v___x_1442_ = lean_box(0);
v_isShared_1443_ = v_isSharedCheck_1479_;
goto v_resetjp_1441_;
}
v_resetjp_1441_:
{
lean_object* v_v_1444_; lean_object* v___x_1445_; lean_object* v_xs_x27_1446_; lean_object* v___y_1448_; 
v_v_1444_ = lean_array_fget(v_es_1435_, v_j_1438_);
v___x_1445_ = lean_box(0);
v_xs_x27_1446_ = lean_array_fset(v_es_1435_, v_j_1438_, v___x_1445_);
switch(lean_obj_tag(v_v_1444_))
{
case 0:
{
lean_object* v_key_1453_; lean_object* v_val_1454_; lean_object* v___x_1456_; uint8_t v_isShared_1457_; uint8_t v_isSharedCheck_1464_; 
v_key_1453_ = lean_ctor_get(v_v_1444_, 0);
v_val_1454_ = lean_ctor_get(v_v_1444_, 1);
v_isSharedCheck_1464_ = !lean_is_exclusive(v_v_1444_);
if (v_isSharedCheck_1464_ == 0)
{
v___x_1456_ = v_v_1444_;
v_isShared_1457_ = v_isSharedCheck_1464_;
goto v_resetjp_1455_;
}
else
{
lean_inc(v_val_1454_);
lean_inc(v_key_1453_);
lean_dec(v_v_1444_);
v___x_1456_ = lean_box(0);
v_isShared_1457_ = v_isSharedCheck_1464_;
goto v_resetjp_1455_;
}
v_resetjp_1455_:
{
uint8_t v___x_1458_; 
v___x_1458_ = l_Lean_instBEqLevelMVarId_beq(v_x_1433_, v_key_1453_);
if (v___x_1458_ == 0)
{
lean_object* v___x_1459_; lean_object* v___x_1460_; 
lean_del_object(v___x_1456_);
v___x_1459_ = l_Lean_PersistentHashMap_mkCollisionNode___redArg(v_key_1453_, v_val_1454_, v_x_1433_, v_x_1434_);
v___x_1460_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1460_, 0, v___x_1459_);
v___y_1448_ = v___x_1460_;
goto v___jp_1447_;
}
else
{
lean_object* v___x_1462_; 
lean_dec(v_val_1454_);
lean_dec(v_key_1453_);
if (v_isShared_1457_ == 0)
{
lean_ctor_set(v___x_1456_, 1, v_x_1434_);
lean_ctor_set(v___x_1456_, 0, v_x_1433_);
v___x_1462_ = v___x_1456_;
goto v_reusejp_1461_;
}
else
{
lean_object* v_reuseFailAlloc_1463_; 
v_reuseFailAlloc_1463_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1463_, 0, v_x_1433_);
lean_ctor_set(v_reuseFailAlloc_1463_, 1, v_x_1434_);
v___x_1462_ = v_reuseFailAlloc_1463_;
goto v_reusejp_1461_;
}
v_reusejp_1461_:
{
v___y_1448_ = v___x_1462_;
goto v___jp_1447_;
}
}
}
}
case 1:
{
lean_object* v_node_1465_; lean_object* v___x_1467_; uint8_t v_isShared_1468_; uint8_t v_isSharedCheck_1477_; 
v_node_1465_ = lean_ctor_get(v_v_1444_, 0);
v_isSharedCheck_1477_ = !lean_is_exclusive(v_v_1444_);
if (v_isSharedCheck_1477_ == 0)
{
v___x_1467_ = v_v_1444_;
v_isShared_1468_ = v_isSharedCheck_1477_;
goto v_resetjp_1466_;
}
else
{
lean_inc(v_node_1465_);
lean_dec(v_v_1444_);
v___x_1467_ = lean_box(0);
v_isShared_1468_ = v_isSharedCheck_1477_;
goto v_resetjp_1466_;
}
v_resetjp_1466_:
{
size_t v___x_1469_; size_t v___x_1470_; size_t v___x_1471_; size_t v___x_1472_; lean_object* v___x_1473_; lean_object* v___x_1475_; 
v___x_1469_ = ((size_t)5ULL);
v___x_1470_ = lean_usize_shift_right(v_x_1431_, v___x_1469_);
v___x_1471_ = ((size_t)1ULL);
v___x_1472_ = lean_usize_add(v_x_1432_, v___x_1471_);
v___x_1473_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_ForwardRuleMatch_getProof_spec__1_spec__2_spec__9___redArg(v_node_1465_, v___x_1470_, v___x_1472_, v_x_1433_, v_x_1434_);
if (v_isShared_1468_ == 0)
{
lean_ctor_set(v___x_1467_, 0, v___x_1473_);
v___x_1475_ = v___x_1467_;
goto v_reusejp_1474_;
}
else
{
lean_object* v_reuseFailAlloc_1476_; 
v_reuseFailAlloc_1476_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1476_, 0, v___x_1473_);
v___x_1475_ = v_reuseFailAlloc_1476_;
goto v_reusejp_1474_;
}
v_reusejp_1474_:
{
v___y_1448_ = v___x_1475_;
goto v___jp_1447_;
}
}
}
default: 
{
lean_object* v___x_1478_; 
v___x_1478_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1478_, 0, v_x_1433_);
lean_ctor_set(v___x_1478_, 1, v_x_1434_);
v___y_1448_ = v___x_1478_;
goto v___jp_1447_;
}
}
v___jp_1447_:
{
lean_object* v___x_1449_; lean_object* v___x_1451_; 
v___x_1449_ = lean_array_fset(v_xs_x27_1446_, v_j_1438_, v___y_1448_);
lean_dec(v_j_1438_);
if (v_isShared_1443_ == 0)
{
lean_ctor_set(v___x_1442_, 0, v___x_1449_);
v___x_1451_ = v___x_1442_;
goto v_reusejp_1450_;
}
else
{
lean_object* v_reuseFailAlloc_1452_; 
v_reuseFailAlloc_1452_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1452_, 0, v___x_1449_);
v___x_1451_ = v_reuseFailAlloc_1452_;
goto v_reusejp_1450_;
}
v_reusejp_1450_:
{
return v___x_1451_;
}
}
}
}
}
else
{
lean_object* v_ks_1481_; lean_object* v_vs_1482_; lean_object* v___x_1484_; uint8_t v_isShared_1485_; uint8_t v_isSharedCheck_1502_; 
v_ks_1481_ = lean_ctor_get(v_x_1430_, 0);
v_vs_1482_ = lean_ctor_get(v_x_1430_, 1);
v_isSharedCheck_1502_ = !lean_is_exclusive(v_x_1430_);
if (v_isSharedCheck_1502_ == 0)
{
v___x_1484_ = v_x_1430_;
v_isShared_1485_ = v_isSharedCheck_1502_;
goto v_resetjp_1483_;
}
else
{
lean_inc(v_vs_1482_);
lean_inc(v_ks_1481_);
lean_dec(v_x_1430_);
v___x_1484_ = lean_box(0);
v_isShared_1485_ = v_isSharedCheck_1502_;
goto v_resetjp_1483_;
}
v_resetjp_1483_:
{
lean_object* v___x_1487_; 
if (v_isShared_1485_ == 0)
{
v___x_1487_ = v___x_1484_;
goto v_reusejp_1486_;
}
else
{
lean_object* v_reuseFailAlloc_1501_; 
v_reuseFailAlloc_1501_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1501_, 0, v_ks_1481_);
lean_ctor_set(v_reuseFailAlloc_1501_, 1, v_vs_1482_);
v___x_1487_ = v_reuseFailAlloc_1501_;
goto v_reusejp_1486_;
}
v_reusejp_1486_:
{
lean_object* v_newNode_1488_; uint8_t v___y_1490_; size_t v___x_1496_; uint8_t v___x_1497_; 
v_newNode_1488_ = lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_ForwardRuleMatch_getProof_spec__1_spec__2_spec__9_spec__23___redArg(v___x_1487_, v_x_1433_, v_x_1434_);
v___x_1496_ = ((size_t)7ULL);
v___x_1497_ = lean_usize_dec_le(v___x_1496_, v_x_1432_);
if (v___x_1497_ == 0)
{
lean_object* v___x_1498_; lean_object* v___x_1499_; uint8_t v___x_1500_; 
v___x_1498_ = l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(v_newNode_1488_);
v___x_1499_ = lean_unsigned_to_nat(4u);
v___x_1500_ = lean_nat_dec_lt(v___x_1498_, v___x_1499_);
lean_dec(v___x_1498_);
v___y_1490_ = v___x_1500_;
goto v___jp_1489_;
}
else
{
v___y_1490_ = v___x_1497_;
goto v___jp_1489_;
}
v___jp_1489_:
{
if (v___y_1490_ == 0)
{
lean_object* v_ks_1491_; lean_object* v_vs_1492_; lean_object* v___x_1493_; lean_object* v___x_1494_; lean_object* v___x_1495_; 
v_ks_1491_ = lean_ctor_get(v_newNode_1488_, 0);
lean_inc_ref(v_ks_1491_);
v_vs_1492_ = lean_ctor_get(v_newNode_1488_, 1);
lean_inc_ref(v_vs_1492_);
lean_dec_ref(v_newNode_1488_);
v___x_1493_ = lean_unsigned_to_nat(0u);
v___x_1494_ = lean_obj_once(&lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_ForwardRuleMatch_getProof_spec__1_spec__2_spec__9___redArg___closed__0, &lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_ForwardRuleMatch_getProof_spec__1_spec__2_spec__9___redArg___closed__0_once, _init_lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_ForwardRuleMatch_getProof_spec__1_spec__2_spec__9___redArg___closed__0);
v___x_1495_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_ForwardRuleMatch_getProof_spec__1_spec__2_spec__9_spec__24___redArg(v_x_1432_, v_ks_1491_, v_vs_1492_, v___x_1493_, v___x_1494_);
lean_dec_ref(v_vs_1492_);
lean_dec_ref(v_ks_1491_);
return v___x_1495_;
}
else
{
return v_newNode_1488_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_ForwardRuleMatch_getProof_spec__1_spec__2_spec__9_spec__24___redArg(size_t v_depth_1503_, lean_object* v_keys_1504_, lean_object* v_vals_1505_, lean_object* v_i_1506_, lean_object* v_entries_1507_){
_start:
{
lean_object* v___x_1508_; uint8_t v___x_1509_; 
v___x_1508_ = lean_array_get_size(v_keys_1504_);
v___x_1509_ = lean_nat_dec_lt(v_i_1506_, v___x_1508_);
if (v___x_1509_ == 0)
{
lean_dec(v_i_1506_);
return v_entries_1507_;
}
else
{
lean_object* v_k_1510_; lean_object* v_v_1511_; uint64_t v___x_1512_; size_t v_h_1513_; size_t v___x_1514_; lean_object* v___x_1515_; size_t v___x_1516_; size_t v___x_1517_; size_t v___x_1518_; size_t v_h_1519_; lean_object* v___x_1520_; lean_object* v___x_1521_; 
v_k_1510_ = lean_array_fget_borrowed(v_keys_1504_, v_i_1506_);
v_v_1511_ = lean_array_fget_borrowed(v_vals_1505_, v_i_1506_);
v___x_1512_ = l_Lean_instHashableLevelMVarId_hash(v_k_1510_);
v_h_1513_ = lean_uint64_to_usize(v___x_1512_);
v___x_1514_ = ((size_t)5ULL);
v___x_1515_ = lean_unsigned_to_nat(1u);
v___x_1516_ = ((size_t)1ULL);
v___x_1517_ = lean_usize_sub(v_depth_1503_, v___x_1516_);
v___x_1518_ = lean_usize_mul(v___x_1514_, v___x_1517_);
v_h_1519_ = lean_usize_shift_right(v_h_1513_, v___x_1518_);
v___x_1520_ = lean_nat_add(v_i_1506_, v___x_1515_);
lean_dec(v_i_1506_);
lean_inc(v_v_1511_);
lean_inc(v_k_1510_);
v___x_1521_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_ForwardRuleMatch_getProof_spec__1_spec__2_spec__9___redArg(v_entries_1507_, v_h_1519_, v_depth_1503_, v_k_1510_, v_v_1511_);
v_i_1506_ = v___x_1520_;
v_entries_1507_ = v___x_1521_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_ForwardRuleMatch_getProof_spec__1_spec__2_spec__9_spec__24___redArg___boxed(lean_object* v_depth_1523_, lean_object* v_keys_1524_, lean_object* v_vals_1525_, lean_object* v_i_1526_, lean_object* v_entries_1527_){
_start:
{
size_t v_depth_boxed_1528_; lean_object* v_res_1529_; 
v_depth_boxed_1528_ = lean_unbox_usize(v_depth_1523_);
lean_dec(v_depth_1523_);
v_res_1529_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_ForwardRuleMatch_getProof_spec__1_spec__2_spec__9_spec__24___redArg(v_depth_boxed_1528_, v_keys_1524_, v_vals_1525_, v_i_1526_, v_entries_1527_);
lean_dec_ref(v_vals_1525_);
lean_dec_ref(v_keys_1524_);
return v_res_1529_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_ForwardRuleMatch_getProof_spec__1_spec__2_spec__9___redArg___boxed(lean_object* v_x_1530_, lean_object* v_x_1531_, lean_object* v_x_1532_, lean_object* v_x_1533_, lean_object* v_x_1534_){
_start:
{
size_t v_x_28075__boxed_1535_; size_t v_x_28076__boxed_1536_; lean_object* v_res_1537_; 
v_x_28075__boxed_1535_ = lean_unbox_usize(v_x_1531_);
lean_dec(v_x_1531_);
v_x_28076__boxed_1536_ = lean_unbox_usize(v_x_1532_);
lean_dec(v_x_1532_);
v_res_1537_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_ForwardRuleMatch_getProof_spec__1_spec__2_spec__9___redArg(v_x_1530_, v_x_28075__boxed_1535_, v_x_28076__boxed_1536_, v_x_1533_, v_x_1534_);
return v_res_1537_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_ForwardRuleMatch_getProof_spec__1_spec__2___redArg(lean_object* v_x_1538_, lean_object* v_x_1539_, lean_object* v_x_1540_){
_start:
{
uint64_t v___x_1541_; size_t v___x_1542_; size_t v___x_1543_; lean_object* v___x_1544_; 
v___x_1541_ = l_Lean_instHashableLevelMVarId_hash(v_x_1539_);
v___x_1542_ = lean_uint64_to_usize(v___x_1541_);
v___x_1543_ = ((size_t)1ULL);
v___x_1544_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_ForwardRuleMatch_getProof_spec__1_spec__2_spec__9___redArg(v_x_1538_, v___x_1542_, v___x_1543_, v_x_1539_, v_x_1540_);
return v___x_1544_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_assignLevelMVar___at___00Aesop_ForwardRuleMatch_getProof_spec__1___redArg(lean_object* v_mvarId_1545_, lean_object* v_val_1546_, lean_object* v___y_1547_){
_start:
{
lean_object* v___x_1549_; lean_object* v_mctx_1550_; lean_object* v_cache_1551_; lean_object* v_zetaDeltaFVarIds_1552_; lean_object* v_postponed_1553_; lean_object* v_diag_1554_; lean_object* v___x_1556_; uint8_t v_isShared_1557_; uint8_t v_isSharedCheck_1582_; 
v___x_1549_ = lean_st_ref_take(v___y_1547_);
v_mctx_1550_ = lean_ctor_get(v___x_1549_, 0);
v_cache_1551_ = lean_ctor_get(v___x_1549_, 1);
v_zetaDeltaFVarIds_1552_ = lean_ctor_get(v___x_1549_, 2);
v_postponed_1553_ = lean_ctor_get(v___x_1549_, 3);
v_diag_1554_ = lean_ctor_get(v___x_1549_, 4);
v_isSharedCheck_1582_ = !lean_is_exclusive(v___x_1549_);
if (v_isSharedCheck_1582_ == 0)
{
v___x_1556_ = v___x_1549_;
v_isShared_1557_ = v_isSharedCheck_1582_;
goto v_resetjp_1555_;
}
else
{
lean_inc(v_diag_1554_);
lean_inc(v_postponed_1553_);
lean_inc(v_zetaDeltaFVarIds_1552_);
lean_inc(v_cache_1551_);
lean_inc(v_mctx_1550_);
lean_dec(v___x_1549_);
v___x_1556_ = lean_box(0);
v_isShared_1557_ = v_isSharedCheck_1582_;
goto v_resetjp_1555_;
}
v_resetjp_1555_:
{
lean_object* v_depth_1558_; lean_object* v_levelAssignDepth_1559_; lean_object* v_lmvarCounter_1560_; lean_object* v_mvarCounter_1561_; lean_object* v_lDecls_1562_; lean_object* v_decls_1563_; lean_object* v_userNames_1564_; lean_object* v_lAssignment_1565_; lean_object* v_eAssignment_1566_; lean_object* v_dAssignment_1567_; lean_object* v___x_1569_; uint8_t v_isShared_1570_; uint8_t v_isSharedCheck_1581_; 
v_depth_1558_ = lean_ctor_get(v_mctx_1550_, 0);
v_levelAssignDepth_1559_ = lean_ctor_get(v_mctx_1550_, 1);
v_lmvarCounter_1560_ = lean_ctor_get(v_mctx_1550_, 2);
v_mvarCounter_1561_ = lean_ctor_get(v_mctx_1550_, 3);
v_lDecls_1562_ = lean_ctor_get(v_mctx_1550_, 4);
v_decls_1563_ = lean_ctor_get(v_mctx_1550_, 5);
v_userNames_1564_ = lean_ctor_get(v_mctx_1550_, 6);
v_lAssignment_1565_ = lean_ctor_get(v_mctx_1550_, 7);
v_eAssignment_1566_ = lean_ctor_get(v_mctx_1550_, 8);
v_dAssignment_1567_ = lean_ctor_get(v_mctx_1550_, 9);
v_isSharedCheck_1581_ = !lean_is_exclusive(v_mctx_1550_);
if (v_isSharedCheck_1581_ == 0)
{
v___x_1569_ = v_mctx_1550_;
v_isShared_1570_ = v_isSharedCheck_1581_;
goto v_resetjp_1568_;
}
else
{
lean_inc(v_dAssignment_1567_);
lean_inc(v_eAssignment_1566_);
lean_inc(v_lAssignment_1565_);
lean_inc(v_userNames_1564_);
lean_inc(v_decls_1563_);
lean_inc(v_lDecls_1562_);
lean_inc(v_mvarCounter_1561_);
lean_inc(v_lmvarCounter_1560_);
lean_inc(v_levelAssignDepth_1559_);
lean_inc(v_depth_1558_);
lean_dec(v_mctx_1550_);
v___x_1569_ = lean_box(0);
v_isShared_1570_ = v_isSharedCheck_1581_;
goto v_resetjp_1568_;
}
v_resetjp_1568_:
{
lean_object* v___x_1571_; lean_object* v___x_1573_; 
v___x_1571_ = lp_aesop_Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_ForwardRuleMatch_getProof_spec__1_spec__2___redArg(v_lAssignment_1565_, v_mvarId_1545_, v_val_1546_);
if (v_isShared_1570_ == 0)
{
lean_ctor_set(v___x_1569_, 7, v___x_1571_);
v___x_1573_ = v___x_1569_;
goto v_reusejp_1572_;
}
else
{
lean_object* v_reuseFailAlloc_1580_; 
v_reuseFailAlloc_1580_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_1580_, 0, v_depth_1558_);
lean_ctor_set(v_reuseFailAlloc_1580_, 1, v_levelAssignDepth_1559_);
lean_ctor_set(v_reuseFailAlloc_1580_, 2, v_lmvarCounter_1560_);
lean_ctor_set(v_reuseFailAlloc_1580_, 3, v_mvarCounter_1561_);
lean_ctor_set(v_reuseFailAlloc_1580_, 4, v_lDecls_1562_);
lean_ctor_set(v_reuseFailAlloc_1580_, 5, v_decls_1563_);
lean_ctor_set(v_reuseFailAlloc_1580_, 6, v_userNames_1564_);
lean_ctor_set(v_reuseFailAlloc_1580_, 7, v___x_1571_);
lean_ctor_set(v_reuseFailAlloc_1580_, 8, v_eAssignment_1566_);
lean_ctor_set(v_reuseFailAlloc_1580_, 9, v_dAssignment_1567_);
v___x_1573_ = v_reuseFailAlloc_1580_;
goto v_reusejp_1572_;
}
v_reusejp_1572_:
{
lean_object* v___x_1575_; 
if (v_isShared_1557_ == 0)
{
lean_ctor_set(v___x_1556_, 0, v___x_1573_);
v___x_1575_ = v___x_1556_;
goto v_reusejp_1574_;
}
else
{
lean_object* v_reuseFailAlloc_1579_; 
v_reuseFailAlloc_1579_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1579_, 0, v___x_1573_);
lean_ctor_set(v_reuseFailAlloc_1579_, 1, v_cache_1551_);
lean_ctor_set(v_reuseFailAlloc_1579_, 2, v_zetaDeltaFVarIds_1552_);
lean_ctor_set(v_reuseFailAlloc_1579_, 3, v_postponed_1553_);
lean_ctor_set(v_reuseFailAlloc_1579_, 4, v_diag_1554_);
v___x_1575_ = v_reuseFailAlloc_1579_;
goto v_reusejp_1574_;
}
v_reusejp_1574_:
{
lean_object* v___x_1576_; lean_object* v___x_1577_; lean_object* v___x_1578_; 
v___x_1576_ = lean_st_ref_set(v___y_1547_, v___x_1575_);
v___x_1577_ = lean_box(0);
v___x_1578_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1578_, 0, v___x_1577_);
return v___x_1578_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_assignLevelMVar___at___00Aesop_ForwardRuleMatch_getProof_spec__1___redArg___boxed(lean_object* v_mvarId_1583_, lean_object* v_val_1584_, lean_object* v___y_1585_, lean_object* v___y_1586_){
_start:
{
lean_object* v_res_1587_; 
v_res_1587_ = lp_aesop_Lean_assignLevelMVar___at___00Aesop_ForwardRuleMatch_getProof_spec__1___redArg(v_mvarId_1583_, v_val_1584_, v___y_1585_);
lean_dec(v___y_1585_);
return v_res_1587_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_ForwardRuleMatch_getProof_spec__5(lean_object* v_as_1588_, size_t v_sz_1589_, size_t v_i_1590_, lean_object* v_b_1591_, lean_object* v___y_1592_, lean_object* v___y_1593_, lean_object* v___y_1594_, lean_object* v___y_1595_){
_start:
{
lean_object* v_a_1598_; uint8_t v___x_1602_; 
v___x_1602_ = lean_usize_dec_lt(v_i_1590_, v_sz_1589_);
if (v___x_1602_ == 0)
{
lean_object* v___x_1603_; 
v___x_1603_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1603_, 0, v_b_1591_);
return v___x_1603_;
}
else
{
lean_object* v_array_1604_; lean_object* v_start_1605_; lean_object* v_stop_1606_; uint8_t v___x_1607_; 
v_array_1604_ = lean_ctor_get(v_b_1591_, 0);
v_start_1605_ = lean_ctor_get(v_b_1591_, 1);
v_stop_1606_ = lean_ctor_get(v_b_1591_, 2);
v___x_1607_ = lean_nat_dec_lt(v_start_1605_, v_stop_1606_);
if (v___x_1607_ == 0)
{
lean_object* v___x_1608_; 
v___x_1608_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1608_, 0, v_b_1591_);
return v___x_1608_;
}
else
{
lean_object* v___x_1610_; uint8_t v_isShared_1611_; uint8_t v_isSharedCheck_1629_; 
lean_inc(v_stop_1606_);
lean_inc(v_start_1605_);
lean_inc_ref(v_array_1604_);
v_isSharedCheck_1629_ = !lean_is_exclusive(v_b_1591_);
if (v_isSharedCheck_1629_ == 0)
{
lean_object* v_unused_1630_; lean_object* v_unused_1631_; lean_object* v_unused_1632_; 
v_unused_1630_ = lean_ctor_get(v_b_1591_, 2);
lean_dec(v_unused_1630_);
v_unused_1631_ = lean_ctor_get(v_b_1591_, 1);
lean_dec(v_unused_1631_);
v_unused_1632_ = lean_ctor_get(v_b_1591_, 0);
lean_dec(v_unused_1632_);
v___x_1610_ = v_b_1591_;
v_isShared_1611_ = v_isSharedCheck_1629_;
goto v_resetjp_1609_;
}
else
{
lean_dec(v_b_1591_);
v___x_1610_ = lean_box(0);
v_isShared_1611_ = v_isSharedCheck_1629_;
goto v_resetjp_1609_;
}
v_resetjp_1609_:
{
lean_object* v_a_1612_; lean_object* v___x_1613_; lean_object* v___x_1614_; lean_object* v___x_1615_; lean_object* v___x_1617_; 
v_a_1612_ = lean_array_uget_borrowed(v_as_1588_, v_i_1590_);
v___x_1613_ = lean_array_fget(v_array_1604_, v_start_1605_);
v___x_1614_ = lean_unsigned_to_nat(1u);
v___x_1615_ = lean_nat_add(v_start_1605_, v___x_1614_);
lean_dec(v_start_1605_);
if (v_isShared_1611_ == 0)
{
lean_ctor_set(v___x_1610_, 1, v___x_1615_);
v___x_1617_ = v___x_1610_;
goto v_reusejp_1616_;
}
else
{
lean_object* v_reuseFailAlloc_1628_; 
v_reuseFailAlloc_1628_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1628_, 0, v_array_1604_);
lean_ctor_set(v_reuseFailAlloc_1628_, 1, v___x_1615_);
lean_ctor_set(v_reuseFailAlloc_1628_, 2, v_stop_1606_);
v___x_1617_ = v_reuseFailAlloc_1628_;
goto v_reusejp_1616_;
}
v_reusejp_1616_:
{
if (lean_obj_tag(v_a_1612_) == 1)
{
lean_object* v_val_1618_; lean_object* v___x_1619_; 
v_val_1618_ = lean_ctor_get(v_a_1612_, 0);
lean_inc(v_val_1618_);
v___x_1619_ = lp_aesop_Lean_assignLevelMVar___at___00Aesop_ForwardRuleMatch_getProof_spec__1___redArg(v___x_1613_, v_val_1618_, v___y_1593_);
if (lean_obj_tag(v___x_1619_) == 0)
{
lean_dec_ref_known(v___x_1619_, 1);
v_a_1598_ = v___x_1617_;
goto v___jp_1597_;
}
else
{
lean_object* v_a_1620_; lean_object* v___x_1622_; uint8_t v_isShared_1623_; uint8_t v_isSharedCheck_1627_; 
lean_dec_ref(v___x_1617_);
v_a_1620_ = lean_ctor_get(v___x_1619_, 0);
v_isSharedCheck_1627_ = !lean_is_exclusive(v___x_1619_);
if (v_isSharedCheck_1627_ == 0)
{
v___x_1622_ = v___x_1619_;
v_isShared_1623_ = v_isSharedCheck_1627_;
goto v_resetjp_1621_;
}
else
{
lean_inc(v_a_1620_);
lean_dec(v___x_1619_);
v___x_1622_ = lean_box(0);
v_isShared_1623_ = v_isSharedCheck_1627_;
goto v_resetjp_1621_;
}
v_resetjp_1621_:
{
lean_object* v___x_1625_; 
if (v_isShared_1623_ == 0)
{
v___x_1625_ = v___x_1622_;
goto v_reusejp_1624_;
}
else
{
lean_object* v_reuseFailAlloc_1626_; 
v_reuseFailAlloc_1626_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1626_, 0, v_a_1620_);
v___x_1625_ = v_reuseFailAlloc_1626_;
goto v_reusejp_1624_;
}
v_reusejp_1624_:
{
return v___x_1625_;
}
}
}
}
else
{
lean_dec(v___x_1613_);
v_a_1598_ = v___x_1617_;
goto v___jp_1597_;
}
}
}
}
}
v___jp_1597_:
{
size_t v___x_1599_; size_t v___x_1600_; 
v___x_1599_ = ((size_t)1ULL);
v___x_1600_ = lean_usize_add(v_i_1590_, v___x_1599_);
v_i_1590_ = v___x_1600_;
v_b_1591_ = v_a_1598_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_ForwardRuleMatch_getProof_spec__5___boxed(lean_object* v_as_1633_, lean_object* v_sz_1634_, lean_object* v_i_1635_, lean_object* v_b_1636_, lean_object* v___y_1637_, lean_object* v___y_1638_, lean_object* v___y_1639_, lean_object* v___y_1640_, lean_object* v___y_1641_){
_start:
{
size_t v_sz_boxed_1642_; size_t v_i_boxed_1643_; lean_object* v_res_1644_; 
v_sz_boxed_1642_ = lean_unbox_usize(v_sz_1634_);
lean_dec(v_sz_1634_);
v_i_boxed_1643_ = lean_unbox_usize(v_i_1635_);
lean_dec(v_i_1635_);
v_res_1644_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_ForwardRuleMatch_getProof_spec__5(v_as_1633_, v_sz_boxed_1642_, v_i_boxed_1643_, v_b_1636_, v___y_1637_, v___y_1638_, v___y_1639_, v___y_1640_);
lean_dec(v___y_1640_);
lean_dec_ref(v___y_1639_);
lean_dec(v___y_1638_);
lean_dec_ref(v___y_1637_);
lean_dec_ref(v_as_1633_);
return v_res_1644_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_ForwardRuleMatch_getProof_spec__0_spec__0_spec__6_spec__19_spec__22___redArg(lean_object* v_x_1645_, lean_object* v_x_1646_, lean_object* v_x_1647_, lean_object* v_x_1648_){
_start:
{
lean_object* v_ks_1649_; lean_object* v_vs_1650_; lean_object* v___x_1652_; uint8_t v_isShared_1653_; uint8_t v_isSharedCheck_1674_; 
v_ks_1649_ = lean_ctor_get(v_x_1645_, 0);
v_vs_1650_ = lean_ctor_get(v_x_1645_, 1);
v_isSharedCheck_1674_ = !lean_is_exclusive(v_x_1645_);
if (v_isSharedCheck_1674_ == 0)
{
v___x_1652_ = v_x_1645_;
v_isShared_1653_ = v_isSharedCheck_1674_;
goto v_resetjp_1651_;
}
else
{
lean_inc(v_vs_1650_);
lean_inc(v_ks_1649_);
lean_dec(v_x_1645_);
v___x_1652_ = lean_box(0);
v_isShared_1653_ = v_isSharedCheck_1674_;
goto v_resetjp_1651_;
}
v_resetjp_1651_:
{
lean_object* v___x_1654_; uint8_t v___x_1655_; 
v___x_1654_ = lean_array_get_size(v_ks_1649_);
v___x_1655_ = lean_nat_dec_lt(v_x_1646_, v___x_1654_);
if (v___x_1655_ == 0)
{
lean_object* v___x_1656_; lean_object* v___x_1657_; lean_object* v___x_1659_; 
lean_dec(v_x_1646_);
v___x_1656_ = lean_array_push(v_ks_1649_, v_x_1647_);
v___x_1657_ = lean_array_push(v_vs_1650_, v_x_1648_);
if (v_isShared_1653_ == 0)
{
lean_ctor_set(v___x_1652_, 1, v___x_1657_);
lean_ctor_set(v___x_1652_, 0, v___x_1656_);
v___x_1659_ = v___x_1652_;
goto v_reusejp_1658_;
}
else
{
lean_object* v_reuseFailAlloc_1660_; 
v_reuseFailAlloc_1660_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1660_, 0, v___x_1656_);
lean_ctor_set(v_reuseFailAlloc_1660_, 1, v___x_1657_);
v___x_1659_ = v_reuseFailAlloc_1660_;
goto v_reusejp_1658_;
}
v_reusejp_1658_:
{
return v___x_1659_;
}
}
else
{
lean_object* v_k_x27_1661_; uint8_t v___x_1662_; 
v_k_x27_1661_ = lean_array_fget_borrowed(v_ks_1649_, v_x_1646_);
v___x_1662_ = l_Lean_instBEqMVarId_beq(v_x_1647_, v_k_x27_1661_);
if (v___x_1662_ == 0)
{
lean_object* v___x_1664_; 
if (v_isShared_1653_ == 0)
{
v___x_1664_ = v___x_1652_;
goto v_reusejp_1663_;
}
else
{
lean_object* v_reuseFailAlloc_1668_; 
v_reuseFailAlloc_1668_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1668_, 0, v_ks_1649_);
lean_ctor_set(v_reuseFailAlloc_1668_, 1, v_vs_1650_);
v___x_1664_ = v_reuseFailAlloc_1668_;
goto v_reusejp_1663_;
}
v_reusejp_1663_:
{
lean_object* v___x_1665_; lean_object* v___x_1666_; 
v___x_1665_ = lean_unsigned_to_nat(1u);
v___x_1666_ = lean_nat_add(v_x_1646_, v___x_1665_);
lean_dec(v_x_1646_);
v_x_1645_ = v___x_1664_;
v_x_1646_ = v___x_1666_;
goto _start;
}
}
else
{
lean_object* v___x_1669_; lean_object* v___x_1670_; lean_object* v___x_1672_; 
v___x_1669_ = lean_array_fset(v_ks_1649_, v_x_1646_, v_x_1647_);
v___x_1670_ = lean_array_fset(v_vs_1650_, v_x_1646_, v_x_1648_);
lean_dec(v_x_1646_);
if (v_isShared_1653_ == 0)
{
lean_ctor_set(v___x_1652_, 1, v___x_1670_);
lean_ctor_set(v___x_1652_, 0, v___x_1669_);
v___x_1672_ = v___x_1652_;
goto v_reusejp_1671_;
}
else
{
lean_object* v_reuseFailAlloc_1673_; 
v_reuseFailAlloc_1673_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1673_, 0, v___x_1669_);
lean_ctor_set(v_reuseFailAlloc_1673_, 1, v___x_1670_);
v___x_1672_ = v_reuseFailAlloc_1673_;
goto v_reusejp_1671_;
}
v_reusejp_1671_:
{
return v___x_1672_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_ForwardRuleMatch_getProof_spec__0_spec__0_spec__6_spec__19___redArg(lean_object* v_n_1675_, lean_object* v_k_1676_, lean_object* v_v_1677_){
_start:
{
lean_object* v___x_1678_; lean_object* v___x_1679_; 
v___x_1678_ = lean_unsigned_to_nat(0u);
v___x_1679_ = lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_ForwardRuleMatch_getProof_spec__0_spec__0_spec__6_spec__19_spec__22___redArg(v_n_1675_, v___x_1678_, v_k_1676_, v_v_1677_);
return v___x_1679_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_ForwardRuleMatch_getProof_spec__0_spec__0_spec__6___redArg(lean_object* v_x_1680_, size_t v_x_1681_, size_t v_x_1682_, lean_object* v_x_1683_, lean_object* v_x_1684_){
_start:
{
if (lean_obj_tag(v_x_1680_) == 0)
{
lean_object* v_es_1685_; size_t v___x_1686_; size_t v___x_1687_; lean_object* v_j_1688_; lean_object* v___x_1689_; uint8_t v___x_1690_; 
v_es_1685_ = lean_ctor_get(v_x_1680_, 0);
v___x_1686_ = ((size_t)31ULL);
v___x_1687_ = lean_usize_land(v_x_1681_, v___x_1686_);
v_j_1688_ = lean_usize_to_nat(v___x_1687_);
v___x_1689_ = lean_array_get_size(v_es_1685_);
v___x_1690_ = lean_nat_dec_lt(v_j_1688_, v___x_1689_);
if (v___x_1690_ == 0)
{
lean_dec(v_j_1688_);
lean_dec(v_x_1684_);
lean_dec(v_x_1683_);
return v_x_1680_;
}
else
{
lean_object* v___x_1692_; uint8_t v_isShared_1693_; uint8_t v_isSharedCheck_1729_; 
lean_inc_ref(v_es_1685_);
v_isSharedCheck_1729_ = !lean_is_exclusive(v_x_1680_);
if (v_isSharedCheck_1729_ == 0)
{
lean_object* v_unused_1730_; 
v_unused_1730_ = lean_ctor_get(v_x_1680_, 0);
lean_dec(v_unused_1730_);
v___x_1692_ = v_x_1680_;
v_isShared_1693_ = v_isSharedCheck_1729_;
goto v_resetjp_1691_;
}
else
{
lean_dec(v_x_1680_);
v___x_1692_ = lean_box(0);
v_isShared_1693_ = v_isSharedCheck_1729_;
goto v_resetjp_1691_;
}
v_resetjp_1691_:
{
lean_object* v_v_1694_; lean_object* v___x_1695_; lean_object* v_xs_x27_1696_; lean_object* v___y_1698_; 
v_v_1694_ = lean_array_fget(v_es_1685_, v_j_1688_);
v___x_1695_ = lean_box(0);
v_xs_x27_1696_ = lean_array_fset(v_es_1685_, v_j_1688_, v___x_1695_);
switch(lean_obj_tag(v_v_1694_))
{
case 0:
{
lean_object* v_key_1703_; lean_object* v_val_1704_; lean_object* v___x_1706_; uint8_t v_isShared_1707_; uint8_t v_isSharedCheck_1714_; 
v_key_1703_ = lean_ctor_get(v_v_1694_, 0);
v_val_1704_ = lean_ctor_get(v_v_1694_, 1);
v_isSharedCheck_1714_ = !lean_is_exclusive(v_v_1694_);
if (v_isSharedCheck_1714_ == 0)
{
v___x_1706_ = v_v_1694_;
v_isShared_1707_ = v_isSharedCheck_1714_;
goto v_resetjp_1705_;
}
else
{
lean_inc(v_val_1704_);
lean_inc(v_key_1703_);
lean_dec(v_v_1694_);
v___x_1706_ = lean_box(0);
v_isShared_1707_ = v_isSharedCheck_1714_;
goto v_resetjp_1705_;
}
v_resetjp_1705_:
{
uint8_t v___x_1708_; 
v___x_1708_ = l_Lean_instBEqMVarId_beq(v_x_1683_, v_key_1703_);
if (v___x_1708_ == 0)
{
lean_object* v___x_1709_; lean_object* v___x_1710_; 
lean_del_object(v___x_1706_);
v___x_1709_ = l_Lean_PersistentHashMap_mkCollisionNode___redArg(v_key_1703_, v_val_1704_, v_x_1683_, v_x_1684_);
v___x_1710_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1710_, 0, v___x_1709_);
v___y_1698_ = v___x_1710_;
goto v___jp_1697_;
}
else
{
lean_object* v___x_1712_; 
lean_dec(v_val_1704_);
lean_dec(v_key_1703_);
if (v_isShared_1707_ == 0)
{
lean_ctor_set(v___x_1706_, 1, v_x_1684_);
lean_ctor_set(v___x_1706_, 0, v_x_1683_);
v___x_1712_ = v___x_1706_;
goto v_reusejp_1711_;
}
else
{
lean_object* v_reuseFailAlloc_1713_; 
v_reuseFailAlloc_1713_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1713_, 0, v_x_1683_);
lean_ctor_set(v_reuseFailAlloc_1713_, 1, v_x_1684_);
v___x_1712_ = v_reuseFailAlloc_1713_;
goto v_reusejp_1711_;
}
v_reusejp_1711_:
{
v___y_1698_ = v___x_1712_;
goto v___jp_1697_;
}
}
}
}
case 1:
{
lean_object* v_node_1715_; lean_object* v___x_1717_; uint8_t v_isShared_1718_; uint8_t v_isSharedCheck_1727_; 
v_node_1715_ = lean_ctor_get(v_v_1694_, 0);
v_isSharedCheck_1727_ = !lean_is_exclusive(v_v_1694_);
if (v_isSharedCheck_1727_ == 0)
{
v___x_1717_ = v_v_1694_;
v_isShared_1718_ = v_isSharedCheck_1727_;
goto v_resetjp_1716_;
}
else
{
lean_inc(v_node_1715_);
lean_dec(v_v_1694_);
v___x_1717_ = lean_box(0);
v_isShared_1718_ = v_isSharedCheck_1727_;
goto v_resetjp_1716_;
}
v_resetjp_1716_:
{
size_t v___x_1719_; size_t v___x_1720_; size_t v___x_1721_; size_t v___x_1722_; lean_object* v___x_1723_; lean_object* v___x_1725_; 
v___x_1719_ = ((size_t)5ULL);
v___x_1720_ = lean_usize_shift_right(v_x_1681_, v___x_1719_);
v___x_1721_ = ((size_t)1ULL);
v___x_1722_ = lean_usize_add(v_x_1682_, v___x_1721_);
v___x_1723_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_ForwardRuleMatch_getProof_spec__0_spec__0_spec__6___redArg(v_node_1715_, v___x_1720_, v___x_1722_, v_x_1683_, v_x_1684_);
if (v_isShared_1718_ == 0)
{
lean_ctor_set(v___x_1717_, 0, v___x_1723_);
v___x_1725_ = v___x_1717_;
goto v_reusejp_1724_;
}
else
{
lean_object* v_reuseFailAlloc_1726_; 
v_reuseFailAlloc_1726_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1726_, 0, v___x_1723_);
v___x_1725_ = v_reuseFailAlloc_1726_;
goto v_reusejp_1724_;
}
v_reusejp_1724_:
{
v___y_1698_ = v___x_1725_;
goto v___jp_1697_;
}
}
}
default: 
{
lean_object* v___x_1728_; 
v___x_1728_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1728_, 0, v_x_1683_);
lean_ctor_set(v___x_1728_, 1, v_x_1684_);
v___y_1698_ = v___x_1728_;
goto v___jp_1697_;
}
}
v___jp_1697_:
{
lean_object* v___x_1699_; lean_object* v___x_1701_; 
v___x_1699_ = lean_array_fset(v_xs_x27_1696_, v_j_1688_, v___y_1698_);
lean_dec(v_j_1688_);
if (v_isShared_1693_ == 0)
{
lean_ctor_set(v___x_1692_, 0, v___x_1699_);
v___x_1701_ = v___x_1692_;
goto v_reusejp_1700_;
}
else
{
lean_object* v_reuseFailAlloc_1702_; 
v_reuseFailAlloc_1702_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1702_, 0, v___x_1699_);
v___x_1701_ = v_reuseFailAlloc_1702_;
goto v_reusejp_1700_;
}
v_reusejp_1700_:
{
return v___x_1701_;
}
}
}
}
}
else
{
lean_object* v_ks_1731_; lean_object* v_vs_1732_; lean_object* v___x_1734_; uint8_t v_isShared_1735_; uint8_t v_isSharedCheck_1752_; 
v_ks_1731_ = lean_ctor_get(v_x_1680_, 0);
v_vs_1732_ = lean_ctor_get(v_x_1680_, 1);
v_isSharedCheck_1752_ = !lean_is_exclusive(v_x_1680_);
if (v_isSharedCheck_1752_ == 0)
{
v___x_1734_ = v_x_1680_;
v_isShared_1735_ = v_isSharedCheck_1752_;
goto v_resetjp_1733_;
}
else
{
lean_inc(v_vs_1732_);
lean_inc(v_ks_1731_);
lean_dec(v_x_1680_);
v___x_1734_ = lean_box(0);
v_isShared_1735_ = v_isSharedCheck_1752_;
goto v_resetjp_1733_;
}
v_resetjp_1733_:
{
lean_object* v___x_1737_; 
if (v_isShared_1735_ == 0)
{
v___x_1737_ = v___x_1734_;
goto v_reusejp_1736_;
}
else
{
lean_object* v_reuseFailAlloc_1751_; 
v_reuseFailAlloc_1751_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1751_, 0, v_ks_1731_);
lean_ctor_set(v_reuseFailAlloc_1751_, 1, v_vs_1732_);
v___x_1737_ = v_reuseFailAlloc_1751_;
goto v_reusejp_1736_;
}
v_reusejp_1736_:
{
lean_object* v_newNode_1738_; uint8_t v___y_1740_; size_t v___x_1746_; uint8_t v___x_1747_; 
v_newNode_1738_ = lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_ForwardRuleMatch_getProof_spec__0_spec__0_spec__6_spec__19___redArg(v___x_1737_, v_x_1683_, v_x_1684_);
v___x_1746_ = ((size_t)7ULL);
v___x_1747_ = lean_usize_dec_le(v___x_1746_, v_x_1682_);
if (v___x_1747_ == 0)
{
lean_object* v___x_1748_; lean_object* v___x_1749_; uint8_t v___x_1750_; 
v___x_1748_ = l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(v_newNode_1738_);
v___x_1749_ = lean_unsigned_to_nat(4u);
v___x_1750_ = lean_nat_dec_lt(v___x_1748_, v___x_1749_);
lean_dec(v___x_1748_);
v___y_1740_ = v___x_1750_;
goto v___jp_1739_;
}
else
{
v___y_1740_ = v___x_1747_;
goto v___jp_1739_;
}
v___jp_1739_:
{
if (v___y_1740_ == 0)
{
lean_object* v_ks_1741_; lean_object* v_vs_1742_; lean_object* v___x_1743_; lean_object* v___x_1744_; lean_object* v___x_1745_; 
v_ks_1741_ = lean_ctor_get(v_newNode_1738_, 0);
lean_inc_ref(v_ks_1741_);
v_vs_1742_ = lean_ctor_get(v_newNode_1738_, 1);
lean_inc_ref(v_vs_1742_);
lean_dec_ref(v_newNode_1738_);
v___x_1743_ = lean_unsigned_to_nat(0u);
v___x_1744_ = lean_obj_once(&lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_ForwardRuleMatch_getProof_spec__1_spec__2_spec__9___redArg___closed__0, &lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_ForwardRuleMatch_getProof_spec__1_spec__2_spec__9___redArg___closed__0_once, _init_lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_ForwardRuleMatch_getProof_spec__1_spec__2_spec__9___redArg___closed__0);
v___x_1745_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_ForwardRuleMatch_getProof_spec__0_spec__0_spec__6_spec__20___redArg(v_x_1682_, v_ks_1741_, v_vs_1742_, v___x_1743_, v___x_1744_);
lean_dec_ref(v_vs_1742_);
lean_dec_ref(v_ks_1741_);
return v___x_1745_;
}
else
{
return v_newNode_1738_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_ForwardRuleMatch_getProof_spec__0_spec__0_spec__6_spec__20___redArg(size_t v_depth_1753_, lean_object* v_keys_1754_, lean_object* v_vals_1755_, lean_object* v_i_1756_, lean_object* v_entries_1757_){
_start:
{
lean_object* v___x_1758_; uint8_t v___x_1759_; 
v___x_1758_ = lean_array_get_size(v_keys_1754_);
v___x_1759_ = lean_nat_dec_lt(v_i_1756_, v___x_1758_);
if (v___x_1759_ == 0)
{
lean_dec(v_i_1756_);
return v_entries_1757_;
}
else
{
lean_object* v_k_1760_; lean_object* v_v_1761_; uint64_t v___x_1762_; size_t v_h_1763_; size_t v___x_1764_; lean_object* v___x_1765_; size_t v___x_1766_; size_t v___x_1767_; size_t v___x_1768_; size_t v_h_1769_; lean_object* v___x_1770_; lean_object* v___x_1771_; 
v_k_1760_ = lean_array_fget_borrowed(v_keys_1754_, v_i_1756_);
v_v_1761_ = lean_array_fget_borrowed(v_vals_1755_, v_i_1756_);
v___x_1762_ = l_Lean_instHashableMVarId_hash(v_k_1760_);
v_h_1763_ = lean_uint64_to_usize(v___x_1762_);
v___x_1764_ = ((size_t)5ULL);
v___x_1765_ = lean_unsigned_to_nat(1u);
v___x_1766_ = ((size_t)1ULL);
v___x_1767_ = lean_usize_sub(v_depth_1753_, v___x_1766_);
v___x_1768_ = lean_usize_mul(v___x_1764_, v___x_1767_);
v_h_1769_ = lean_usize_shift_right(v_h_1763_, v___x_1768_);
v___x_1770_ = lean_nat_add(v_i_1756_, v___x_1765_);
lean_dec(v_i_1756_);
lean_inc(v_v_1761_);
lean_inc(v_k_1760_);
v___x_1771_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_ForwardRuleMatch_getProof_spec__0_spec__0_spec__6___redArg(v_entries_1757_, v_h_1769_, v_depth_1753_, v_k_1760_, v_v_1761_);
v_i_1756_ = v___x_1770_;
v_entries_1757_ = v___x_1771_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_ForwardRuleMatch_getProof_spec__0_spec__0_spec__6_spec__20___redArg___boxed(lean_object* v_depth_1773_, lean_object* v_keys_1774_, lean_object* v_vals_1775_, lean_object* v_i_1776_, lean_object* v_entries_1777_){
_start:
{
size_t v_depth_boxed_1778_; lean_object* v_res_1779_; 
v_depth_boxed_1778_ = lean_unbox_usize(v_depth_1773_);
lean_dec(v_depth_1773_);
v_res_1779_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_ForwardRuleMatch_getProof_spec__0_spec__0_spec__6_spec__20___redArg(v_depth_boxed_1778_, v_keys_1774_, v_vals_1775_, v_i_1776_, v_entries_1777_);
lean_dec_ref(v_vals_1775_);
lean_dec_ref(v_keys_1774_);
return v_res_1779_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_ForwardRuleMatch_getProof_spec__0_spec__0_spec__6___redArg___boxed(lean_object* v_x_1780_, lean_object* v_x_1781_, lean_object* v_x_1782_, lean_object* v_x_1783_, lean_object* v_x_1784_){
_start:
{
size_t v_x_28444__boxed_1785_; size_t v_x_28445__boxed_1786_; lean_object* v_res_1787_; 
v_x_28444__boxed_1785_ = lean_unbox_usize(v_x_1781_);
lean_dec(v_x_1781_);
v_x_28445__boxed_1786_ = lean_unbox_usize(v_x_1782_);
lean_dec(v_x_1782_);
v_res_1787_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_ForwardRuleMatch_getProof_spec__0_spec__0_spec__6___redArg(v_x_1780_, v_x_28444__boxed_1785_, v_x_28445__boxed_1786_, v_x_1783_, v_x_1784_);
return v_res_1787_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_ForwardRuleMatch_getProof_spec__0_spec__0___redArg(lean_object* v_x_1788_, lean_object* v_x_1789_, lean_object* v_x_1790_){
_start:
{
uint64_t v___x_1791_; size_t v___x_1792_; size_t v___x_1793_; lean_object* v___x_1794_; 
v___x_1791_ = l_Lean_instHashableMVarId_hash(v_x_1789_);
v___x_1792_ = lean_uint64_to_usize(v___x_1791_);
v___x_1793_ = ((size_t)1ULL);
v___x_1794_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_ForwardRuleMatch_getProof_spec__0_spec__0_spec__6___redArg(v_x_1788_, v___x_1792_, v___x_1793_, v_x_1789_, v_x_1790_);
return v___x_1794_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_assign___at___00Aesop_ForwardRuleMatch_getProof_spec__0___redArg(lean_object* v_mvarId_1795_, lean_object* v_val_1796_, lean_object* v___y_1797_){
_start:
{
lean_object* v___x_1799_; lean_object* v_mctx_1800_; lean_object* v_cache_1801_; lean_object* v_zetaDeltaFVarIds_1802_; lean_object* v_postponed_1803_; lean_object* v_diag_1804_; lean_object* v___x_1806_; uint8_t v_isShared_1807_; uint8_t v_isSharedCheck_1832_; 
v___x_1799_ = lean_st_ref_take(v___y_1797_);
v_mctx_1800_ = lean_ctor_get(v___x_1799_, 0);
v_cache_1801_ = lean_ctor_get(v___x_1799_, 1);
v_zetaDeltaFVarIds_1802_ = lean_ctor_get(v___x_1799_, 2);
v_postponed_1803_ = lean_ctor_get(v___x_1799_, 3);
v_diag_1804_ = lean_ctor_get(v___x_1799_, 4);
v_isSharedCheck_1832_ = !lean_is_exclusive(v___x_1799_);
if (v_isSharedCheck_1832_ == 0)
{
v___x_1806_ = v___x_1799_;
v_isShared_1807_ = v_isSharedCheck_1832_;
goto v_resetjp_1805_;
}
else
{
lean_inc(v_diag_1804_);
lean_inc(v_postponed_1803_);
lean_inc(v_zetaDeltaFVarIds_1802_);
lean_inc(v_cache_1801_);
lean_inc(v_mctx_1800_);
lean_dec(v___x_1799_);
v___x_1806_ = lean_box(0);
v_isShared_1807_ = v_isSharedCheck_1832_;
goto v_resetjp_1805_;
}
v_resetjp_1805_:
{
lean_object* v_depth_1808_; lean_object* v_levelAssignDepth_1809_; lean_object* v_lmvarCounter_1810_; lean_object* v_mvarCounter_1811_; lean_object* v_lDecls_1812_; lean_object* v_decls_1813_; lean_object* v_userNames_1814_; lean_object* v_lAssignment_1815_; lean_object* v_eAssignment_1816_; lean_object* v_dAssignment_1817_; lean_object* v___x_1819_; uint8_t v_isShared_1820_; uint8_t v_isSharedCheck_1831_; 
v_depth_1808_ = lean_ctor_get(v_mctx_1800_, 0);
v_levelAssignDepth_1809_ = lean_ctor_get(v_mctx_1800_, 1);
v_lmvarCounter_1810_ = lean_ctor_get(v_mctx_1800_, 2);
v_mvarCounter_1811_ = lean_ctor_get(v_mctx_1800_, 3);
v_lDecls_1812_ = lean_ctor_get(v_mctx_1800_, 4);
v_decls_1813_ = lean_ctor_get(v_mctx_1800_, 5);
v_userNames_1814_ = lean_ctor_get(v_mctx_1800_, 6);
v_lAssignment_1815_ = lean_ctor_get(v_mctx_1800_, 7);
v_eAssignment_1816_ = lean_ctor_get(v_mctx_1800_, 8);
v_dAssignment_1817_ = lean_ctor_get(v_mctx_1800_, 9);
v_isSharedCheck_1831_ = !lean_is_exclusive(v_mctx_1800_);
if (v_isSharedCheck_1831_ == 0)
{
v___x_1819_ = v_mctx_1800_;
v_isShared_1820_ = v_isSharedCheck_1831_;
goto v_resetjp_1818_;
}
else
{
lean_inc(v_dAssignment_1817_);
lean_inc(v_eAssignment_1816_);
lean_inc(v_lAssignment_1815_);
lean_inc(v_userNames_1814_);
lean_inc(v_decls_1813_);
lean_inc(v_lDecls_1812_);
lean_inc(v_mvarCounter_1811_);
lean_inc(v_lmvarCounter_1810_);
lean_inc(v_levelAssignDepth_1809_);
lean_inc(v_depth_1808_);
lean_dec(v_mctx_1800_);
v___x_1819_ = lean_box(0);
v_isShared_1820_ = v_isSharedCheck_1831_;
goto v_resetjp_1818_;
}
v_resetjp_1818_:
{
lean_object* v___x_1821_; lean_object* v___x_1823_; 
v___x_1821_ = lp_aesop_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_ForwardRuleMatch_getProof_spec__0_spec__0___redArg(v_eAssignment_1816_, v_mvarId_1795_, v_val_1796_);
if (v_isShared_1820_ == 0)
{
lean_ctor_set(v___x_1819_, 8, v___x_1821_);
v___x_1823_ = v___x_1819_;
goto v_reusejp_1822_;
}
else
{
lean_object* v_reuseFailAlloc_1830_; 
v_reuseFailAlloc_1830_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_1830_, 0, v_depth_1808_);
lean_ctor_set(v_reuseFailAlloc_1830_, 1, v_levelAssignDepth_1809_);
lean_ctor_set(v_reuseFailAlloc_1830_, 2, v_lmvarCounter_1810_);
lean_ctor_set(v_reuseFailAlloc_1830_, 3, v_mvarCounter_1811_);
lean_ctor_set(v_reuseFailAlloc_1830_, 4, v_lDecls_1812_);
lean_ctor_set(v_reuseFailAlloc_1830_, 5, v_decls_1813_);
lean_ctor_set(v_reuseFailAlloc_1830_, 6, v_userNames_1814_);
lean_ctor_set(v_reuseFailAlloc_1830_, 7, v_lAssignment_1815_);
lean_ctor_set(v_reuseFailAlloc_1830_, 8, v___x_1821_);
lean_ctor_set(v_reuseFailAlloc_1830_, 9, v_dAssignment_1817_);
v___x_1823_ = v_reuseFailAlloc_1830_;
goto v_reusejp_1822_;
}
v_reusejp_1822_:
{
lean_object* v___x_1825_; 
if (v_isShared_1807_ == 0)
{
lean_ctor_set(v___x_1806_, 0, v___x_1823_);
v___x_1825_ = v___x_1806_;
goto v_reusejp_1824_;
}
else
{
lean_object* v_reuseFailAlloc_1829_; 
v_reuseFailAlloc_1829_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1829_, 0, v___x_1823_);
lean_ctor_set(v_reuseFailAlloc_1829_, 1, v_cache_1801_);
lean_ctor_set(v_reuseFailAlloc_1829_, 2, v_zetaDeltaFVarIds_1802_);
lean_ctor_set(v_reuseFailAlloc_1829_, 3, v_postponed_1803_);
lean_ctor_set(v_reuseFailAlloc_1829_, 4, v_diag_1804_);
v___x_1825_ = v_reuseFailAlloc_1829_;
goto v_reusejp_1824_;
}
v_reusejp_1824_:
{
lean_object* v___x_1826_; lean_object* v___x_1827_; lean_object* v___x_1828_; 
v___x_1826_ = lean_st_ref_set(v___y_1797_, v___x_1825_);
v___x_1827_ = lean_box(0);
v___x_1828_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1828_, 0, v___x_1827_);
return v___x_1828_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_assign___at___00Aesop_ForwardRuleMatch_getProof_spec__0___redArg___boxed(lean_object* v_mvarId_1833_, lean_object* v_val_1834_, lean_object* v___y_1835_, lean_object* v___y_1836_){
_start:
{
lean_object* v_res_1837_; 
v_res_1837_ = lp_aesop_Lean_MVarId_assign___at___00Aesop_ForwardRuleMatch_getProof_spec__0___redArg(v_mvarId_1833_, v_val_1834_, v___y_1835_);
lean_dec(v___y_1835_);
return v_res_1837_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_ForwardRuleMatch_getProof_spec__6(lean_object* v_as_1838_, size_t v_sz_1839_, size_t v_i_1840_, lean_object* v_b_1841_, lean_object* v___y_1842_, lean_object* v___y_1843_, lean_object* v___y_1844_, lean_object* v___y_1845_){
_start:
{
lean_object* v_a_1848_; uint8_t v___x_1852_; 
v___x_1852_ = lean_usize_dec_lt(v_i_1840_, v_sz_1839_);
if (v___x_1852_ == 0)
{
lean_object* v___x_1853_; 
v___x_1853_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1853_, 0, v_b_1841_);
return v___x_1853_;
}
else
{
lean_object* v_array_1854_; lean_object* v_start_1855_; lean_object* v_stop_1856_; uint8_t v___x_1857_; 
v_array_1854_ = lean_ctor_get(v_b_1841_, 0);
v_start_1855_ = lean_ctor_get(v_b_1841_, 1);
v_stop_1856_ = lean_ctor_get(v_b_1841_, 2);
v___x_1857_ = lean_nat_dec_lt(v_start_1855_, v_stop_1856_);
if (v___x_1857_ == 0)
{
lean_object* v___x_1858_; 
v___x_1858_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1858_, 0, v_b_1841_);
return v___x_1858_;
}
else
{
lean_object* v___x_1860_; uint8_t v_isShared_1861_; uint8_t v_isSharedCheck_1880_; 
lean_inc(v_stop_1856_);
lean_inc(v_start_1855_);
lean_inc_ref(v_array_1854_);
v_isSharedCheck_1880_ = !lean_is_exclusive(v_b_1841_);
if (v_isSharedCheck_1880_ == 0)
{
lean_object* v_unused_1881_; lean_object* v_unused_1882_; lean_object* v_unused_1883_; 
v_unused_1881_ = lean_ctor_get(v_b_1841_, 2);
lean_dec(v_unused_1881_);
v_unused_1882_ = lean_ctor_get(v_b_1841_, 1);
lean_dec(v_unused_1882_);
v_unused_1883_ = lean_ctor_get(v_b_1841_, 0);
lean_dec(v_unused_1883_);
v___x_1860_ = v_b_1841_;
v_isShared_1861_ = v_isSharedCheck_1880_;
goto v_resetjp_1859_;
}
else
{
lean_dec(v_b_1841_);
v___x_1860_ = lean_box(0);
v_isShared_1861_ = v_isSharedCheck_1880_;
goto v_resetjp_1859_;
}
v_resetjp_1859_:
{
lean_object* v_a_1862_; lean_object* v___x_1863_; lean_object* v___x_1864_; lean_object* v___x_1865_; lean_object* v___x_1867_; 
v_a_1862_ = lean_array_uget_borrowed(v_as_1838_, v_i_1840_);
v___x_1863_ = lean_array_fget(v_array_1854_, v_start_1855_);
v___x_1864_ = lean_unsigned_to_nat(1u);
v___x_1865_ = lean_nat_add(v_start_1855_, v___x_1864_);
lean_dec(v_start_1855_);
if (v_isShared_1861_ == 0)
{
lean_ctor_set(v___x_1860_, 1, v___x_1865_);
v___x_1867_ = v___x_1860_;
goto v_reusejp_1866_;
}
else
{
lean_object* v_reuseFailAlloc_1879_; 
v_reuseFailAlloc_1879_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1879_, 0, v_array_1854_);
lean_ctor_set(v_reuseFailAlloc_1879_, 1, v___x_1865_);
lean_ctor_set(v_reuseFailAlloc_1879_, 2, v_stop_1856_);
v___x_1867_ = v_reuseFailAlloc_1879_;
goto v_reusejp_1866_;
}
v_reusejp_1866_:
{
if (lean_obj_tag(v_a_1862_) == 1)
{
lean_object* v_val_1868_; lean_object* v___x_1869_; lean_object* v___x_1870_; 
v_val_1868_ = lean_ctor_get(v_a_1862_, 0);
v___x_1869_ = l_Lean_Expr_mvarId_x21(v___x_1863_);
lean_dec(v___x_1863_);
lean_inc(v_val_1868_);
v___x_1870_ = lp_aesop_Lean_MVarId_assign___at___00Aesop_ForwardRuleMatch_getProof_spec__0___redArg(v___x_1869_, v_val_1868_, v___y_1843_);
if (lean_obj_tag(v___x_1870_) == 0)
{
lean_dec_ref_known(v___x_1870_, 1);
v_a_1848_ = v___x_1867_;
goto v___jp_1847_;
}
else
{
lean_object* v_a_1871_; lean_object* v___x_1873_; uint8_t v_isShared_1874_; uint8_t v_isSharedCheck_1878_; 
lean_dec_ref(v___x_1867_);
v_a_1871_ = lean_ctor_get(v___x_1870_, 0);
v_isSharedCheck_1878_ = !lean_is_exclusive(v___x_1870_);
if (v_isSharedCheck_1878_ == 0)
{
v___x_1873_ = v___x_1870_;
v_isShared_1874_ = v_isSharedCheck_1878_;
goto v_resetjp_1872_;
}
else
{
lean_inc(v_a_1871_);
lean_dec(v___x_1870_);
v___x_1873_ = lean_box(0);
v_isShared_1874_ = v_isSharedCheck_1878_;
goto v_resetjp_1872_;
}
v_resetjp_1872_:
{
lean_object* v___x_1876_; 
if (v_isShared_1874_ == 0)
{
v___x_1876_ = v___x_1873_;
goto v_reusejp_1875_;
}
else
{
lean_object* v_reuseFailAlloc_1877_; 
v_reuseFailAlloc_1877_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1877_, 0, v_a_1871_);
v___x_1876_ = v_reuseFailAlloc_1877_;
goto v_reusejp_1875_;
}
v_reusejp_1875_:
{
return v___x_1876_;
}
}
}
}
else
{
lean_dec(v___x_1863_);
v_a_1848_ = v___x_1867_;
goto v___jp_1847_;
}
}
}
}
}
v___jp_1847_:
{
size_t v___x_1849_; size_t v___x_1850_; 
v___x_1849_ = ((size_t)1ULL);
v___x_1850_ = lean_usize_add(v_i_1840_, v___x_1849_);
v_i_1840_ = v___x_1850_;
v_b_1841_ = v_a_1848_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_ForwardRuleMatch_getProof_spec__6___boxed(lean_object* v_as_1884_, lean_object* v_sz_1885_, lean_object* v_i_1886_, lean_object* v_b_1887_, lean_object* v___y_1888_, lean_object* v___y_1889_, lean_object* v___y_1890_, lean_object* v___y_1891_, lean_object* v___y_1892_){
_start:
{
size_t v_sz_boxed_1893_; size_t v_i_boxed_1894_; lean_object* v_res_1895_; 
v_sz_boxed_1893_ = lean_unbox_usize(v_sz_1885_);
lean_dec(v_sz_1885_);
v_i_boxed_1894_ = lean_unbox_usize(v_i_1886_);
lean_dec(v_i_1886_);
v_res_1895_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_ForwardRuleMatch_getProof_spec__6(v_as_1884_, v_sz_boxed_1893_, v_i_boxed_1894_, v_b_1887_, v___y_1888_, v___y_1889_, v___y_1890_, v___y_1891_);
lean_dec(v___y_1891_);
lean_dec_ref(v___y_1890_);
lean_dec(v___y_1889_);
lean_dec_ref(v___y_1888_);
lean_dec_ref(v_as_1884_);
return v_res_1895_;
}
}
static lean_object* _init_lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__1(void){
_start:
{
lean_object* v___x_1897_; lean_object* v___x_1898_; 
v___x_1897_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__0));
v___x_1898_ = l_Lean_stringToMessageData(v___x_1897_);
return v___x_1898_;
}
}
static lean_object* _init_lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__3(void){
_start:
{
lean_object* v___x_1900_; lean_object* v___x_1901_; 
v___x_1900_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__2));
v___x_1901_ = l_Lean_stringToMessageData(v___x_1900_);
return v___x_1901_;
}
}
static lean_object* _init_lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__7(void){
_start:
{
lean_object* v___x_1906_; lean_object* v___x_1907_; 
v___x_1906_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__6));
v___x_1907_ = l_Lean_stringToMessageData(v___x_1906_);
return v___x_1907_;
}
}
static lean_object* _init_lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__9(void){
_start:
{
lean_object* v___x_1909_; lean_object* v___x_1910_; 
v___x_1909_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__8));
v___x_1910_ = l_Lean_stringToMessageData(v___x_1909_);
return v___x_1910_;
}
}
static lean_object* _init_lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__11(void){
_start:
{
lean_object* v___x_1912_; lean_object* v___x_1913_; 
v___x_1912_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__10));
v___x_1913_ = l_Lean_stringToMessageData(v___x_1912_);
return v___x_1913_;
}
}
static lean_object* _init_lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__13(void){
_start:
{
lean_object* v___x_1915_; lean_object* v___x_1916_; 
v___x_1915_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__12));
v___x_1916_ = l_Lean_stringToMessageData(v___x_1915_);
return v___x_1916_;
}
}
static lean_object* _init_lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__15(void){
_start:
{
lean_object* v___x_1918_; lean_object* v___x_1919_; 
v___x_1918_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__14));
v___x_1919_ = l_Lean_stringToMessageData(v___x_1918_);
return v___x_1919_;
}
}
static lean_object* _init_lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__17(void){
_start:
{
lean_object* v___x_1921_; lean_object* v___x_1922_; 
v___x_1921_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__16));
v___x_1922_ = l_Lean_stringToMessageData(v___x_1921_);
return v___x_1922_;
}
}
static lean_object* _init_lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__19(void){
_start:
{
lean_object* v___x_1924_; lean_object* v___x_1925_; 
v___x_1924_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__18));
v___x_1925_ = l_Lean_stringToMessageData(v___x_1924_);
return v___x_1925_;
}
}
static lean_object* _init_lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__20(void){
_start:
{
lean_object* v___x_1926_; lean_object* v___x_1927_; lean_object* v___x_1928_; 
v___x_1926_ = lean_box(0);
v___x_1927_ = lean_unsigned_to_nat(16u);
v___x_1928_ = lean_mk_array(v___x_1927_, v___x_1926_);
return v___x_1928_;
}
}
static lean_object* _init_lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__21(void){
_start:
{
lean_object* v___x_1929_; lean_object* v___x_1930_; lean_object* v___x_1931_; 
v___x_1929_ = lean_obj_once(&lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__20, &lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__20_once, _init_lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__20);
v___x_1930_ = lean_unsigned_to_nat(0u);
v___x_1931_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1931_, 0, v___x_1930_);
lean_ctor_set(v___x_1931_, 1, v___x_1929_);
return v___x_1931_;
}
}
static lean_object* _init_lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__23(void){
_start:
{
lean_object* v___x_1934_; lean_object* v___x_1935_; lean_object* v___x_1936_; 
v___x_1934_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__22));
v___x_1935_ = lean_obj_once(&lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__21, &lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__21_once, _init_lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__21);
v___x_1936_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1936_, 0, v___x_1935_);
lean_ctor_set(v___x_1936_, 1, v___x_1935_);
lean_ctor_set(v___x_1936_, 2, v___x_1934_);
return v___x_1936_;
}
}
static lean_object* _init_lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__25(void){
_start:
{
lean_object* v___x_1938_; lean_object* v___x_1939_; 
v___x_1938_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__24));
v___x_1939_ = l_Lean_stringToMessageData(v___x_1938_);
return v___x_1939_;
}
}
static lean_object* _init_lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__27(void){
_start:
{
lean_object* v___x_1941_; lean_object* v___x_1942_; 
v___x_1941_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__26));
v___x_1942_ = l_Lean_stringToMessageData(v___x_1941_);
return v___x_1942_;
}
}
static lean_object* _init_lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__29(void){
_start:
{
lean_object* v___x_1944_; lean_object* v___x_1945_; 
v___x_1944_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__28));
v___x_1945_ = l_Lean_stringToMessageData(v___x_1944_);
return v___x_1945_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1(lean_object* v___x_1946_, lean_object* v___f_1947_, lean_object* v_goal_1948_, lean_object* v_m_1949_, lean_object* v___y_1950_, lean_object* v___y_1951_, lean_object* v___y_1952_, lean_object* v___y_1953_){
_start:
{
lean_object* v___y_1956_; lean_object* v___y_1960_; lean_object* v___y_1961_; lean_object* v___y_1962_; lean_object* v___y_1963_; lean_object* v___y_1964_; lean_object* v___y_1965_; lean_object* v___y_2004_; lean_object* v___y_2005_; lean_object* v___y_2006_; lean_object* v___y_2007_; lean_object* v___y_2008_; lean_object* v___y_2009_; lean_object* v___y_2010_; lean_object* v___y_2029_; lean_object* v___y_2030_; lean_object* v___y_2031_; lean_object* v___y_2032_; lean_object* v___y_2033_; lean_object* v___y_2034_; lean_object* v___y_2035_; uint8_t v___y_2036_; lean_object* v___y_2057_; lean_object* v___y_2058_; lean_object* v___y_2059_; lean_object* v___y_2060_; lean_object* v___y_2061_; lean_object* v___y_2062_; lean_object* v___y_2063_; lean_object* v___y_2064_; lean_object* v___y_2065_; lean_object* v___y_2066_; lean_object* v___y_2067_; lean_object* v___y_2100_; lean_object* v___y_2101_; lean_object* v___y_2102_; lean_object* v___y_2103_; lean_object* v___y_2104_; lean_object* v___y_2105_; lean_object* v___y_2106_; lean_object* v___y_2107_; lean_object* v___y_2108_; lean_object* v___y_2109_; lean_object* v___y_2110_; lean_object* v___y_2134_; lean_object* v___y_2135_; lean_object* v___y_2136_; lean_object* v___y_2137_; lean_object* v___y_2138_; lean_object* v___y_2139_; lean_object* v___y_2140_; lean_object* v___y_2141_; lean_object* v___y_2142_; lean_object* v___y_2143_; lean_object* v___y_2144_; lean_object* v___y_2177_; lean_object* v___y_2178_; lean_object* v___y_2179_; lean_object* v___y_2180_; lean_object* v___y_2181_; lean_object* v___y_2182_; lean_object* v___y_2183_; lean_object* v_numLevelParams_2184_; lean_object* v___y_2185_; lean_object* v___y_2186_; lean_object* v___y_2187_; lean_object* v___y_2188_; lean_object* v___y_2189_; lean_object* v___y_2221_; lean_object* v___y_2222_; lean_object* v___y_2223_; lean_object* v___y_2224_; lean_object* v___y_2225_; lean_object* v___y_2226_; lean_object* v___y_2227_; lean_object* v___y_2228_; lean_object* v___y_2229_; lean_object* v___y_2230_; lean_object* v___y_2231_; lean_object* v___y_2316_; lean_object* v___y_2317_; lean_object* v___y_2318_; lean_object* v___y_2319_; lean_object* v___x_2377_; lean_object* v_a_2378_; lean_object* v___x_2380_; uint8_t v_isShared_2381_; uint8_t v_isSharedCheck_2437_; 
v___x_2377_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_ForwardRuleMatch_getProof_spec__3___redArg(v___x_1946_, v___y_1952_);
v_a_2378_ = lean_ctor_get(v___x_2377_, 0);
v_isSharedCheck_2437_ = !lean_is_exclusive(v___x_2377_);
if (v_isSharedCheck_2437_ == 0)
{
v___x_2380_ = v___x_2377_;
v_isShared_2381_ = v_isSharedCheck_2437_;
goto v_resetjp_2379_;
}
else
{
lean_inc(v_a_2378_);
lean_dec(v___x_2377_);
v___x_2380_ = lean_box(0);
v_isShared_2381_ = v_isSharedCheck_2437_;
goto v_resetjp_2379_;
}
v___jp_1955_:
{
lean_object* v___x_1957_; lean_object* v___x_1958_; 
v___x_1957_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1957_, 0, v___y_1956_);
v___x_1958_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1958_, 0, v___x_1957_);
return v___x_1958_;
}
v___jp_1959_:
{
lean_object* v___x_1966_; uint8_t v___x_1967_; lean_object* v___x_1968_; 
v___x_1966_ = l_Lean_mkAppN(v___y_1965_, v___y_1961_);
lean_dec_ref(v___y_1961_);
v___x_1967_ = 1;
v___x_1968_ = l_Lean_Meta_abstractMVars(v___x_1966_, v___x_1967_, v___y_1962_, v___y_1960_, v___y_1964_, v___y_1963_);
if (lean_obj_tag(v___x_1968_) == 0)
{
lean_object* v_a_1969_; lean_object* v___x_1970_; lean_object* v_a_1971_; uint8_t v___x_1972_; 
v_a_1969_ = lean_ctor_get(v___x_1968_, 0);
lean_inc(v_a_1969_);
lean_dec_ref_known(v___x_1968_, 1);
v___x_1970_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_ForwardRuleMatch_getProof_spec__3___redArg(v___x_1946_, v___y_1964_);
v_a_1971_ = lean_ctor_get(v___x_1970_, 0);
lean_inc(v_a_1971_);
lean_dec_ref(v___x_1970_);
v___x_1972_ = lean_unbox(v_a_1971_);
lean_dec(v_a_1971_);
if (v___x_1972_ == 0)
{
lean_object* v_expr_1973_; 
lean_dec_ref(v___y_1964_);
lean_dec(v___y_1963_);
lean_dec_ref(v___y_1962_);
lean_dec(v___y_1960_);
lean_dec_ref(v___x_1946_);
v_expr_1973_ = lean_ctor_get(v_a_1969_, 2);
lean_inc_ref(v_expr_1973_);
lean_dec(v_a_1969_);
v___y_1956_ = v_expr_1973_;
goto v___jp_1955_;
}
else
{
lean_object* v_expr_1974_; lean_object* v_traceClass_1975_; lean_object* v___x_1977_; uint8_t v_isShared_1978_; uint8_t v_isSharedCheck_1993_; 
v_expr_1974_ = lean_ctor_get(v_a_1969_, 2);
lean_inc_ref(v_expr_1974_);
lean_dec(v_a_1969_);
v_traceClass_1975_ = lean_ctor_get(v___x_1946_, 0);
v_isSharedCheck_1993_ = !lean_is_exclusive(v___x_1946_);
if (v_isSharedCheck_1993_ == 0)
{
lean_object* v_unused_1994_; 
v_unused_1994_ = lean_ctor_get(v___x_1946_, 1);
lean_dec(v_unused_1994_);
v___x_1977_ = v___x_1946_;
v_isShared_1978_ = v_isSharedCheck_1993_;
goto v_resetjp_1976_;
}
else
{
lean_inc(v_traceClass_1975_);
lean_dec(v___x_1946_);
v___x_1977_ = lean_box(0);
v_isShared_1978_ = v_isSharedCheck_1993_;
goto v_resetjp_1976_;
}
v_resetjp_1976_:
{
lean_object* v___x_1979_; lean_object* v___x_1980_; lean_object* v___x_1982_; 
v___x_1979_ = lean_obj_once(&lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__1, &lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__1_once, _init_lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__1);
lean_inc_ref(v_expr_1974_);
v___x_1980_ = l_Lean_MessageData_ofExpr(v_expr_1974_);
if (v_isShared_1978_ == 0)
{
lean_ctor_set_tag(v___x_1977_, 7);
lean_ctor_set(v___x_1977_, 1, v___x_1980_);
lean_ctor_set(v___x_1977_, 0, v___x_1979_);
v___x_1982_ = v___x_1977_;
goto v_reusejp_1981_;
}
else
{
lean_object* v_reuseFailAlloc_1992_; 
v_reuseFailAlloc_1992_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1992_, 0, v___x_1979_);
lean_ctor_set(v_reuseFailAlloc_1992_, 1, v___x_1980_);
v___x_1982_ = v_reuseFailAlloc_1992_;
goto v_reusejp_1981_;
}
v_reusejp_1981_:
{
lean_object* v___x_1983_; 
v___x_1983_ = lp_aesop_Lean_addTrace___at___00Aesop_ForwardRuleMatch_getProof_spec__4(v_traceClass_1975_, v___x_1982_, v___y_1962_, v___y_1960_, v___y_1964_, v___y_1963_);
lean_dec(v___y_1963_);
lean_dec_ref(v___y_1964_);
lean_dec(v___y_1960_);
lean_dec_ref(v___y_1962_);
if (lean_obj_tag(v___x_1983_) == 0)
{
lean_dec_ref_known(v___x_1983_, 1);
v___y_1956_ = v_expr_1974_;
goto v___jp_1955_;
}
else
{
lean_object* v_a_1984_; lean_object* v___x_1986_; uint8_t v_isShared_1987_; uint8_t v_isSharedCheck_1991_; 
lean_dec_ref(v_expr_1974_);
v_a_1984_ = lean_ctor_get(v___x_1983_, 0);
v_isSharedCheck_1991_ = !lean_is_exclusive(v___x_1983_);
if (v_isSharedCheck_1991_ == 0)
{
v___x_1986_ = v___x_1983_;
v_isShared_1987_ = v_isSharedCheck_1991_;
goto v_resetjp_1985_;
}
else
{
lean_inc(v_a_1984_);
lean_dec(v___x_1983_);
v___x_1986_ = lean_box(0);
v_isShared_1987_ = v_isSharedCheck_1991_;
goto v_resetjp_1985_;
}
v_resetjp_1985_:
{
lean_object* v___x_1989_; 
if (v_isShared_1987_ == 0)
{
v___x_1989_ = v___x_1986_;
goto v_reusejp_1988_;
}
else
{
lean_object* v_reuseFailAlloc_1990_; 
v_reuseFailAlloc_1990_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1990_, 0, v_a_1984_);
v___x_1989_ = v_reuseFailAlloc_1990_;
goto v_reusejp_1988_;
}
v_reusejp_1988_:
{
return v___x_1989_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_1995_; lean_object* v___x_1997_; uint8_t v_isShared_1998_; uint8_t v_isSharedCheck_2002_; 
lean_dec_ref(v___y_1964_);
lean_dec(v___y_1963_);
lean_dec_ref(v___y_1962_);
lean_dec(v___y_1960_);
lean_dec_ref(v___x_1946_);
v_a_1995_ = lean_ctor_get(v___x_1968_, 0);
v_isSharedCheck_2002_ = !lean_is_exclusive(v___x_1968_);
if (v_isSharedCheck_2002_ == 0)
{
v___x_1997_ = v___x_1968_;
v_isShared_1998_ = v_isSharedCheck_2002_;
goto v_resetjp_1996_;
}
else
{
lean_inc(v_a_1995_);
lean_dec(v___x_1968_);
v___x_1997_ = lean_box(0);
v_isShared_1998_ = v_isSharedCheck_2002_;
goto v_resetjp_1996_;
}
v_resetjp_1996_:
{
lean_object* v___x_2000_; 
if (v_isShared_1998_ == 0)
{
v___x_2000_ = v___x_1997_;
goto v_reusejp_1999_;
}
else
{
lean_object* v_reuseFailAlloc_2001_; 
v_reuseFailAlloc_2001_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2001_, 0, v_a_1995_);
v___x_2000_ = v_reuseFailAlloc_2001_;
goto v_reusejp_1999_;
}
v_reusejp_1999_:
{
return v___x_2000_;
}
}
}
}
v___jp_2003_:
{
if (lean_obj_tag(v___y_2010_) == 0)
{
lean_object* v_a_2011_; lean_object* v___x_2013_; uint8_t v_isShared_2014_; uint8_t v_isSharedCheck_2019_; 
v_a_2011_ = lean_ctor_get(v___y_2010_, 0);
v_isSharedCheck_2019_ = !lean_is_exclusive(v___y_2010_);
if (v_isSharedCheck_2019_ == 0)
{
v___x_2013_ = v___y_2010_;
v_isShared_2014_ = v_isSharedCheck_2019_;
goto v_resetjp_2012_;
}
else
{
lean_inc(v_a_2011_);
lean_dec(v___y_2010_);
v___x_2013_ = lean_box(0);
v_isShared_2014_ = v_isSharedCheck_2019_;
goto v_resetjp_2012_;
}
v_resetjp_2012_:
{
if (lean_obj_tag(v_a_2011_) == 0)
{
lean_object* v_a_2015_; lean_object* v___x_2017_; 
lean_dec_ref(v___y_2009_);
lean_dec_ref(v___y_2008_);
lean_dec(v___y_2007_);
lean_dec_ref(v___y_2006_);
lean_dec_ref(v___y_2005_);
lean_dec(v___y_2004_);
lean_dec_ref(v___x_1946_);
v_a_2015_ = lean_ctor_get(v_a_2011_, 0);
lean_inc(v_a_2015_);
lean_dec_ref_known(v_a_2011_, 1);
if (v_isShared_2014_ == 0)
{
lean_ctor_set(v___x_2013_, 0, v_a_2015_);
v___x_2017_ = v___x_2013_;
goto v_reusejp_2016_;
}
else
{
lean_object* v_reuseFailAlloc_2018_; 
v_reuseFailAlloc_2018_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2018_, 0, v_a_2015_);
v___x_2017_ = v_reuseFailAlloc_2018_;
goto v_reusejp_2016_;
}
v_reusejp_2016_:
{
return v___x_2017_;
}
}
else
{
lean_dec_ref_known(v_a_2011_, 1);
lean_del_object(v___x_2013_);
v___y_1960_ = v___y_2004_;
v___y_1961_ = v___y_2005_;
v___y_1962_ = v___y_2006_;
v___y_1963_ = v___y_2007_;
v___y_1964_ = v___y_2008_;
v___y_1965_ = v___y_2009_;
goto v___jp_1959_;
}
}
}
else
{
lean_object* v_a_2020_; lean_object* v___x_2022_; uint8_t v_isShared_2023_; uint8_t v_isSharedCheck_2027_; 
lean_dec_ref(v___y_2009_);
lean_dec_ref(v___y_2008_);
lean_dec(v___y_2007_);
lean_dec_ref(v___y_2006_);
lean_dec_ref(v___y_2005_);
lean_dec(v___y_2004_);
lean_dec_ref(v___x_1946_);
v_a_2020_ = lean_ctor_get(v___y_2010_, 0);
v_isSharedCheck_2027_ = !lean_is_exclusive(v___y_2010_);
if (v_isSharedCheck_2027_ == 0)
{
v___x_2022_ = v___y_2010_;
v_isShared_2023_ = v_isSharedCheck_2027_;
goto v_resetjp_2021_;
}
else
{
lean_inc(v_a_2020_);
lean_dec(v___y_2010_);
v___x_2022_ = lean_box(0);
v_isShared_2023_ = v_isSharedCheck_2027_;
goto v_resetjp_2021_;
}
v_resetjp_2021_:
{
lean_object* v___x_2025_; 
if (v_isShared_2023_ == 0)
{
v___x_2025_ = v___x_2022_;
goto v_reusejp_2024_;
}
else
{
lean_object* v_reuseFailAlloc_2026_; 
v_reuseFailAlloc_2026_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2026_, 0, v_a_2020_);
v___x_2025_ = v_reuseFailAlloc_2026_;
goto v_reusejp_2024_;
}
v_reusejp_2024_:
{
return v___x_2025_;
}
}
}
}
v___jp_2028_:
{
if (v___y_2036_ == 0)
{
lean_object* v___x_2037_; lean_object* v_a_2038_; uint8_t v___x_2039_; 
lean_dec_ref(v___y_2032_);
v___x_2037_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_ForwardRuleMatch_getProof_spec__3___redArg(v___x_1946_, v___y_2034_);
v_a_2038_ = lean_ctor_get(v___x_2037_, 0);
lean_inc(v_a_2038_);
lean_dec_ref(v___x_2037_);
v___x_2039_ = lean_unbox(v_a_2038_);
lean_dec(v_a_2038_);
if (v___x_2039_ == 0)
{
lean_object* v___x_2040_; lean_object* v___x_2041_; 
v___x_2040_ = lean_box(0);
lean_inc(v___y_2033_);
lean_inc_ref(v___y_2034_);
lean_inc(v___y_2029_);
lean_inc_ref(v___y_2031_);
v___x_2041_ = lean_apply_6(v___f_1947_, v___x_2040_, v___y_2031_, v___y_2029_, v___y_2034_, v___y_2033_, lean_box(0));
v___y_2004_ = v___y_2029_;
v___y_2005_ = v___y_2030_;
v___y_2006_ = v___y_2031_;
v___y_2007_ = v___y_2033_;
v___y_2008_ = v___y_2034_;
v___y_2009_ = v___y_2035_;
v___y_2010_ = v___x_2041_;
goto v___jp_2003_;
}
else
{
lean_object* v_traceClass_2042_; lean_object* v___x_2043_; lean_object* v___x_2044_; 
v_traceClass_2042_ = lean_ctor_get(v___x_1946_, 0);
v___x_2043_ = lean_obj_once(&lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__3, &lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__3_once, _init_lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__3);
lean_inc(v_traceClass_2042_);
v___x_2044_ = lp_aesop_Lean_addTrace___at___00Aesop_ForwardRuleMatch_getProof_spec__4(v_traceClass_2042_, v___x_2043_, v___y_2031_, v___y_2029_, v___y_2034_, v___y_2033_);
if (lean_obj_tag(v___x_2044_) == 0)
{
lean_object* v_a_2045_; lean_object* v___x_2046_; 
v_a_2045_ = lean_ctor_get(v___x_2044_, 0);
lean_inc(v_a_2045_);
lean_dec_ref_known(v___x_2044_, 1);
lean_inc(v___y_2033_);
lean_inc_ref(v___y_2034_);
lean_inc(v___y_2029_);
lean_inc_ref(v___y_2031_);
v___x_2046_ = lean_apply_6(v___f_1947_, v_a_2045_, v___y_2031_, v___y_2029_, v___y_2034_, v___y_2033_, lean_box(0));
v___y_2004_ = v___y_2029_;
v___y_2005_ = v___y_2030_;
v___y_2006_ = v___y_2031_;
v___y_2007_ = v___y_2033_;
v___y_2008_ = v___y_2034_;
v___y_2009_ = v___y_2035_;
v___y_2010_ = v___x_2046_;
goto v___jp_2003_;
}
else
{
lean_object* v_a_2047_; lean_object* v___x_2049_; uint8_t v_isShared_2050_; uint8_t v_isSharedCheck_2054_; 
lean_dec_ref(v___y_2035_);
lean_dec_ref(v___y_2034_);
lean_dec(v___y_2033_);
lean_dec_ref(v___y_2031_);
lean_dec_ref(v___y_2030_);
lean_dec(v___y_2029_);
lean_dec_ref(v___f_1947_);
lean_dec_ref(v___x_1946_);
v_a_2047_ = lean_ctor_get(v___x_2044_, 0);
v_isSharedCheck_2054_ = !lean_is_exclusive(v___x_2044_);
if (v_isSharedCheck_2054_ == 0)
{
v___x_2049_ = v___x_2044_;
v_isShared_2050_ = v_isSharedCheck_2054_;
goto v_resetjp_2048_;
}
else
{
lean_inc(v_a_2047_);
lean_dec(v___x_2044_);
v___x_2049_ = lean_box(0);
v_isShared_2050_ = v_isSharedCheck_2054_;
goto v_resetjp_2048_;
}
v_resetjp_2048_:
{
lean_object* v___x_2052_; 
if (v_isShared_2050_ == 0)
{
v___x_2052_ = v___x_2049_;
goto v_reusejp_2051_;
}
else
{
lean_object* v_reuseFailAlloc_2053_; 
v_reuseFailAlloc_2053_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2053_, 0, v_a_2047_);
v___x_2052_ = v_reuseFailAlloc_2053_;
goto v_reusejp_2051_;
}
v_reusejp_2051_:
{
return v___x_2052_;
}
}
}
}
}
else
{
lean_object* v___x_2055_; 
lean_dec_ref(v___y_2035_);
lean_dec_ref(v___y_2034_);
lean_dec(v___y_2033_);
lean_dec_ref(v___y_2031_);
lean_dec_ref(v___y_2030_);
lean_dec(v___y_2029_);
lean_dec_ref(v___f_1947_);
lean_dec_ref(v___x_1946_);
v___x_2055_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2055_, 0, v___y_2032_);
return v___x_2055_;
}
}
v___jp_2056_:
{
lean_object* v___x_2068_; lean_object* v___x_2069_; size_t v_sz_2070_; size_t v___x_2071_; lean_object* v___x_2072_; 
v___x_2068_ = lean_array_get_size(v___y_2061_);
lean_inc(v___y_2062_);
v___x_2069_ = l_Array_toSubarray___redArg(v___y_2061_, v___y_2062_, v___x_2068_);
v_sz_2070_ = lean_array_size(v___y_2057_);
v___x_2071_ = ((size_t)0ULL);
v___x_2072_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_ForwardRuleMatch_getProof_spec__5(v___y_2057_, v_sz_2070_, v___x_2071_, v___x_2069_, v___y_2064_, v___y_2065_, v___y_2066_, v___y_2067_);
lean_dec_ref(v___y_2057_);
if (lean_obj_tag(v___x_2072_) == 0)
{
lean_object* v___x_2073_; lean_object* v___x_2074_; size_t v_sz_2075_; lean_object* v___x_2076_; 
lean_dec_ref_known(v___x_2072_, 1);
v___x_2073_ = lean_array_get_size(v___y_2059_);
lean_inc_ref(v___y_2059_);
v___x_2074_ = l_Array_toSubarray___redArg(v___y_2059_, v___y_2062_, v___x_2073_);
v_sz_2075_ = lean_array_size(v___y_2058_);
v___x_2076_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_ForwardRuleMatch_getProof_spec__6(v___y_2058_, v_sz_2075_, v___x_2071_, v___x_2074_, v___y_2064_, v___y_2065_, v___y_2066_, v___y_2067_);
lean_dec_ref(v___y_2058_);
if (lean_obj_tag(v___x_2076_) == 0)
{
lean_object* v___x_2077_; uint8_t v___x_2078_; lean_object* v___x_2079_; 
lean_dec_ref_known(v___x_2076_, 1);
v___x_2077_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__5));
v___x_2078_ = 0;
v___x_2079_ = l_Lean_Meta_synthAppInstances(v___x_2077_, v_goal_1948_, v___y_2059_, v___y_2060_, v___x_2078_, v___x_2078_, v___y_2064_, v___y_2065_, v___y_2066_, v___y_2067_);
if (lean_obj_tag(v___x_2079_) == 0)
{
lean_dec_ref_known(v___x_2079_, 1);
lean_dec_ref(v___f_1947_);
v___y_1960_ = v___y_2065_;
v___y_1961_ = v___y_2059_;
v___y_1962_ = v___y_2064_;
v___y_1963_ = v___y_2067_;
v___y_1964_ = v___y_2066_;
v___y_1965_ = v___y_2063_;
goto v___jp_1959_;
}
else
{
lean_object* v_a_2080_; uint8_t v___x_2081_; 
v_a_2080_ = lean_ctor_get(v___x_2079_, 0);
lean_inc(v_a_2080_);
lean_dec_ref_known(v___x_2079_, 1);
v___x_2081_ = l_Lean_Exception_isInterrupt(v_a_2080_);
if (v___x_2081_ == 0)
{
uint8_t v___x_2082_; 
lean_inc(v_a_2080_);
v___x_2082_ = l_Lean_Exception_isRuntime(v_a_2080_);
v___y_2029_ = v___y_2065_;
v___y_2030_ = v___y_2059_;
v___y_2031_ = v___y_2064_;
v___y_2032_ = v_a_2080_;
v___y_2033_ = v___y_2067_;
v___y_2034_ = v___y_2066_;
v___y_2035_ = v___y_2063_;
v___y_2036_ = v___x_2082_;
goto v___jp_2028_;
}
else
{
v___y_2029_ = v___y_2065_;
v___y_2030_ = v___y_2059_;
v___y_2031_ = v___y_2064_;
v___y_2032_ = v_a_2080_;
v___y_2033_ = v___y_2067_;
v___y_2034_ = v___y_2066_;
v___y_2035_ = v___y_2063_;
v___y_2036_ = v___x_2081_;
goto v___jp_2028_;
}
}
}
else
{
lean_object* v_a_2083_; lean_object* v___x_2085_; uint8_t v_isShared_2086_; uint8_t v_isSharedCheck_2090_; 
lean_dec(v___y_2067_);
lean_dec_ref(v___y_2066_);
lean_dec(v___y_2065_);
lean_dec_ref(v___y_2064_);
lean_dec_ref(v___y_2063_);
lean_dec_ref(v___y_2060_);
lean_dec_ref(v___y_2059_);
lean_dec(v_goal_1948_);
lean_dec_ref(v___f_1947_);
lean_dec_ref(v___x_1946_);
v_a_2083_ = lean_ctor_get(v___x_2076_, 0);
v_isSharedCheck_2090_ = !lean_is_exclusive(v___x_2076_);
if (v_isSharedCheck_2090_ == 0)
{
v___x_2085_ = v___x_2076_;
v_isShared_2086_ = v_isSharedCheck_2090_;
goto v_resetjp_2084_;
}
else
{
lean_inc(v_a_2083_);
lean_dec(v___x_2076_);
v___x_2085_ = lean_box(0);
v_isShared_2086_ = v_isSharedCheck_2090_;
goto v_resetjp_2084_;
}
v_resetjp_2084_:
{
lean_object* v___x_2088_; 
if (v_isShared_2086_ == 0)
{
v___x_2088_ = v___x_2085_;
goto v_reusejp_2087_;
}
else
{
lean_object* v_reuseFailAlloc_2089_; 
v_reuseFailAlloc_2089_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2089_, 0, v_a_2083_);
v___x_2088_ = v_reuseFailAlloc_2089_;
goto v_reusejp_2087_;
}
v_reusejp_2087_:
{
return v___x_2088_;
}
}
}
}
else
{
lean_object* v_a_2091_; lean_object* v___x_2093_; uint8_t v_isShared_2094_; uint8_t v_isSharedCheck_2098_; 
lean_dec(v___y_2067_);
lean_dec_ref(v___y_2066_);
lean_dec(v___y_2065_);
lean_dec_ref(v___y_2064_);
lean_dec_ref(v___y_2063_);
lean_dec(v___y_2062_);
lean_dec_ref(v___y_2060_);
lean_dec_ref(v___y_2059_);
lean_dec_ref(v___y_2058_);
lean_dec(v_goal_1948_);
lean_dec_ref(v___f_1947_);
lean_dec_ref(v___x_1946_);
v_a_2091_ = lean_ctor_get(v___x_2072_, 0);
v_isSharedCheck_2098_ = !lean_is_exclusive(v___x_2072_);
if (v_isSharedCheck_2098_ == 0)
{
v___x_2093_ = v___x_2072_;
v_isShared_2094_ = v_isSharedCheck_2098_;
goto v_resetjp_2092_;
}
else
{
lean_inc(v_a_2091_);
lean_dec(v___x_2072_);
v___x_2093_ = lean_box(0);
v_isShared_2094_ = v_isSharedCheck_2098_;
goto v_resetjp_2092_;
}
v_resetjp_2092_:
{
lean_object* v___x_2096_; 
if (v_isShared_2094_ == 0)
{
v___x_2096_ = v___x_2093_;
goto v_reusejp_2095_;
}
else
{
lean_object* v_reuseFailAlloc_2097_; 
v_reuseFailAlloc_2097_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2097_, 0, v_a_2091_);
v___x_2096_ = v_reuseFailAlloc_2097_;
goto v_reusejp_2095_;
}
v_reusejp_2095_:
{
return v___x_2096_;
}
}
}
}
v___jp_2099_:
{
lean_object* v___x_2111_; lean_object* v_a_2112_; uint8_t v___x_2113_; 
v___x_2111_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_ForwardRuleMatch_getProof_spec__3___redArg(v___x_1946_, v___y_2109_);
v_a_2112_ = lean_ctor_get(v___x_2111_, 0);
lean_inc(v_a_2112_);
lean_dec_ref(v___x_2111_);
v___x_2113_ = lean_unbox(v_a_2112_);
lean_dec(v_a_2112_);
if (v___x_2113_ == 0)
{
v___y_2057_ = v___y_2100_;
v___y_2058_ = v___y_2102_;
v___y_2059_ = v___y_2101_;
v___y_2060_ = v___y_2104_;
v___y_2061_ = v___y_2103_;
v___y_2062_ = v___y_2105_;
v___y_2063_ = v___y_2106_;
v___y_2064_ = v___y_2107_;
v___y_2065_ = v___y_2108_;
v___y_2066_ = v___y_2109_;
v___y_2067_ = v___y_2110_;
goto v___jp_2056_;
}
else
{
lean_object* v_traceClass_2114_; lean_object* v___x_2115_; size_t v_sz_2116_; size_t v___x_2117_; lean_object* v___x_2118_; lean_object* v___x_2119_; lean_object* v___x_2120_; lean_object* v___x_2121_; lean_object* v___x_2122_; lean_object* v___x_2123_; lean_object* v___x_2124_; 
v_traceClass_2114_ = lean_ctor_get(v___x_1946_, 0);
v___x_2115_ = lean_obj_once(&lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__7, &lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__7_once, _init_lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__7);
v_sz_2116_ = lean_array_size(v___y_2100_);
v___x_2117_ = ((size_t)0ULL);
lean_inc_ref(v___y_2100_);
v___x_2118_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_ForwardRuleMatch_getProof_spec__7(v_sz_2116_, v___x_2117_, v___y_2100_);
v___x_2119_ = lean_array_to_list(v___x_2118_);
v___x_2120_ = lean_box(0);
v___x_2121_ = lp_aesop_List_mapTR_loop___at___00Aesop_CompleteMatch_toMessageData_spec__1(v___x_2119_, v___x_2120_);
v___x_2122_ = l_Lean_MessageData_ofList(v___x_2121_);
v___x_2123_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2123_, 0, v___x_2115_);
lean_ctor_set(v___x_2123_, 1, v___x_2122_);
lean_inc(v_traceClass_2114_);
v___x_2124_ = lp_aesop_Lean_addTrace___at___00Aesop_ForwardRuleMatch_getProof_spec__4(v_traceClass_2114_, v___x_2123_, v___y_2107_, v___y_2108_, v___y_2109_, v___y_2110_);
if (lean_obj_tag(v___x_2124_) == 0)
{
lean_dec_ref_known(v___x_2124_, 1);
v___y_2057_ = v___y_2100_;
v___y_2058_ = v___y_2102_;
v___y_2059_ = v___y_2101_;
v___y_2060_ = v___y_2104_;
v___y_2061_ = v___y_2103_;
v___y_2062_ = v___y_2105_;
v___y_2063_ = v___y_2106_;
v___y_2064_ = v___y_2107_;
v___y_2065_ = v___y_2108_;
v___y_2066_ = v___y_2109_;
v___y_2067_ = v___y_2110_;
goto v___jp_2056_;
}
else
{
lean_object* v_a_2125_; lean_object* v___x_2127_; uint8_t v_isShared_2128_; uint8_t v_isSharedCheck_2132_; 
lean_dec(v___y_2110_);
lean_dec_ref(v___y_2109_);
lean_dec(v___y_2108_);
lean_dec_ref(v___y_2107_);
lean_dec_ref(v___y_2106_);
lean_dec(v___y_2105_);
lean_dec_ref(v___y_2104_);
lean_dec_ref(v___y_2103_);
lean_dec_ref(v___y_2102_);
lean_dec_ref(v___y_2101_);
lean_dec_ref(v___y_2100_);
lean_dec(v_goal_1948_);
lean_dec_ref(v___f_1947_);
lean_dec_ref(v___x_1946_);
v_a_2125_ = lean_ctor_get(v___x_2124_, 0);
v_isSharedCheck_2132_ = !lean_is_exclusive(v___x_2124_);
if (v_isSharedCheck_2132_ == 0)
{
v___x_2127_ = v___x_2124_;
v_isShared_2128_ = v_isSharedCheck_2132_;
goto v_resetjp_2126_;
}
else
{
lean_inc(v_a_2125_);
lean_dec(v___x_2124_);
v___x_2127_ = lean_box(0);
v_isShared_2128_ = v_isSharedCheck_2132_;
goto v_resetjp_2126_;
}
v_resetjp_2126_:
{
lean_object* v___x_2130_; 
if (v_isShared_2128_ == 0)
{
v___x_2130_ = v___x_2127_;
goto v_reusejp_2129_;
}
else
{
lean_object* v_reuseFailAlloc_2131_; 
v_reuseFailAlloc_2131_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2131_, 0, v_a_2125_);
v___x_2130_ = v_reuseFailAlloc_2131_;
goto v_reusejp_2129_;
}
v_reusejp_2129_:
{
return v___x_2130_;
}
}
}
}
}
v___jp_2133_:
{
lean_object* v___x_2145_; lean_object* v_fst_2146_; lean_object* v_snd_2147_; lean_object* v___x_2149_; uint8_t v_isShared_2150_; uint8_t v_isSharedCheck_2175_; 
v___x_2145_ = lp_aesop_Aesop_CompleteMatch_reconstructArgs(v___y_2140_, v___y_2137_);
lean_dec_ref(v___y_2137_);
v_fst_2146_ = lean_ctor_get(v___x_2145_, 0);
v_snd_2147_ = lean_ctor_get(v___x_2145_, 1);
v_isSharedCheck_2175_ = !lean_is_exclusive(v___x_2145_);
if (v_isSharedCheck_2175_ == 0)
{
v___x_2149_ = v___x_2145_;
v_isShared_2150_ = v_isSharedCheck_2175_;
goto v_resetjp_2148_;
}
else
{
lean_inc(v_snd_2147_);
lean_inc(v_fst_2146_);
lean_dec(v___x_2145_);
v___x_2149_ = lean_box(0);
v_isShared_2150_ = v_isSharedCheck_2175_;
goto v_resetjp_2148_;
}
v_resetjp_2148_:
{
lean_object* v___x_2151_; lean_object* v_a_2152_; uint8_t v___x_2153_; 
v___x_2151_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_ForwardRuleMatch_getProof_spec__3___redArg(v___x_1946_, v___y_2143_);
v_a_2152_ = lean_ctor_get(v___x_2151_, 0);
lean_inc(v_a_2152_);
lean_dec_ref(v___x_2151_);
v___x_2153_ = lean_unbox(v_a_2152_);
lean_dec(v_a_2152_);
if (v___x_2153_ == 0)
{
lean_del_object(v___x_2149_);
v___y_2100_ = v_snd_2147_;
v___y_2101_ = v___y_2134_;
v___y_2102_ = v_fst_2146_;
v___y_2103_ = v___y_2135_;
v___y_2104_ = v___y_2136_;
v___y_2105_ = v___y_2138_;
v___y_2106_ = v___y_2139_;
v___y_2107_ = v___y_2141_;
v___y_2108_ = v___y_2142_;
v___y_2109_ = v___y_2143_;
v___y_2110_ = v___y_2144_;
goto v___jp_2099_;
}
else
{
lean_object* v_traceClass_2154_; lean_object* v___x_2155_; size_t v_sz_2156_; size_t v___x_2157_; lean_object* v___x_2158_; lean_object* v___x_2159_; lean_object* v___x_2160_; lean_object* v___x_2161_; lean_object* v___x_2162_; lean_object* v___x_2164_; 
v_traceClass_2154_ = lean_ctor_get(v___x_1946_, 0);
v___x_2155_ = lean_obj_once(&lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__9, &lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__9_once, _init_lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__9);
v_sz_2156_ = lean_array_size(v_fst_2146_);
v___x_2157_ = ((size_t)0ULL);
lean_inc(v_fst_2146_);
v___x_2158_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_CompleteMatch_toMessageData_spec__0(v_sz_2156_, v___x_2157_, v_fst_2146_);
v___x_2159_ = lean_array_to_list(v___x_2158_);
v___x_2160_ = lean_box(0);
v___x_2161_ = lp_aesop_List_mapTR_loop___at___00Aesop_CompleteMatch_toMessageData_spec__1(v___x_2159_, v___x_2160_);
v___x_2162_ = l_Lean_MessageData_ofList(v___x_2161_);
if (v_isShared_2150_ == 0)
{
lean_ctor_set_tag(v___x_2149_, 7);
lean_ctor_set(v___x_2149_, 1, v___x_2162_);
lean_ctor_set(v___x_2149_, 0, v___x_2155_);
v___x_2164_ = v___x_2149_;
goto v_reusejp_2163_;
}
else
{
lean_object* v_reuseFailAlloc_2174_; 
v_reuseFailAlloc_2174_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2174_, 0, v___x_2155_);
lean_ctor_set(v_reuseFailAlloc_2174_, 1, v___x_2162_);
v___x_2164_ = v_reuseFailAlloc_2174_;
goto v_reusejp_2163_;
}
v_reusejp_2163_:
{
lean_object* v___x_2165_; 
lean_inc(v_traceClass_2154_);
v___x_2165_ = lp_aesop_Lean_addTrace___at___00Aesop_ForwardRuleMatch_getProof_spec__4(v_traceClass_2154_, v___x_2164_, v___y_2141_, v___y_2142_, v___y_2143_, v___y_2144_);
if (lean_obj_tag(v___x_2165_) == 0)
{
lean_dec_ref_known(v___x_2165_, 1);
v___y_2100_ = v_snd_2147_;
v___y_2101_ = v___y_2134_;
v___y_2102_ = v_fst_2146_;
v___y_2103_ = v___y_2135_;
v___y_2104_ = v___y_2136_;
v___y_2105_ = v___y_2138_;
v___y_2106_ = v___y_2139_;
v___y_2107_ = v___y_2141_;
v___y_2108_ = v___y_2142_;
v___y_2109_ = v___y_2143_;
v___y_2110_ = v___y_2144_;
goto v___jp_2099_;
}
else
{
lean_object* v_a_2166_; lean_object* v___x_2168_; uint8_t v_isShared_2169_; uint8_t v_isSharedCheck_2173_; 
lean_dec(v_snd_2147_);
lean_dec(v_fst_2146_);
lean_dec(v___y_2144_);
lean_dec_ref(v___y_2143_);
lean_dec(v___y_2142_);
lean_dec_ref(v___y_2141_);
lean_dec_ref(v___y_2139_);
lean_dec(v___y_2138_);
lean_dec_ref(v___y_2136_);
lean_dec_ref(v___y_2135_);
lean_dec_ref(v___y_2134_);
lean_dec(v_goal_1948_);
lean_dec_ref(v___f_1947_);
lean_dec_ref(v___x_1946_);
v_a_2166_ = lean_ctor_get(v___x_2165_, 0);
v_isSharedCheck_2173_ = !lean_is_exclusive(v___x_2165_);
if (v_isSharedCheck_2173_ == 0)
{
v___x_2168_ = v___x_2165_;
v_isShared_2169_ = v_isSharedCheck_2173_;
goto v_resetjp_2167_;
}
else
{
lean_inc(v_a_2166_);
lean_dec(v___x_2165_);
v___x_2168_ = lean_box(0);
v_isShared_2169_ = v_isSharedCheck_2173_;
goto v_resetjp_2167_;
}
v_resetjp_2167_:
{
lean_object* v___x_2171_; 
if (v_isShared_2169_ == 0)
{
v___x_2171_ = v___x_2168_;
goto v_reusejp_2170_;
}
else
{
lean_object* v_reuseFailAlloc_2172_; 
v_reuseFailAlloc_2172_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2172_, 0, v_a_2166_);
v___x_2171_ = v_reuseFailAlloc_2172_;
goto v_reusejp_2170_;
}
v_reusejp_2170_:
{
return v___x_2171_;
}
}
}
}
}
}
}
v___jp_2176_:
{
lean_object* v___x_2190_; uint8_t v___x_2191_; 
v___x_2190_ = lean_array_get_size(v___y_2180_);
v___x_2191_ = lean_nat_dec_eq(v___x_2190_, v_numLevelParams_2184_);
if (v___x_2191_ == 0)
{
lean_object* v___x_2192_; lean_object* v___x_2193_; lean_object* v___x_2194_; lean_object* v___x_2195_; lean_object* v___x_2196_; lean_object* v___x_2197_; lean_object* v___x_2198_; lean_object* v___x_2199_; lean_object* v___x_2200_; lean_object* v___x_2201_; lean_object* v___x_2202_; lean_object* v___x_2203_; lean_object* v___x_2204_; lean_object* v___x_2205_; lean_object* v___x_2206_; lean_object* v___x_2207_; lean_object* v___x_2208_; lean_object* v___x_2209_; lean_object* v___x_2210_; lean_object* v___x_2211_; lean_object* v_a_2212_; lean_object* v___x_2214_; uint8_t v_isShared_2215_; uint8_t v_isSharedCheck_2219_; 
lean_dec_ref(v___y_2185_);
lean_dec(v___y_2182_);
lean_dec_ref(v___y_2181_);
lean_dec_ref(v___y_2180_);
lean_dec_ref(v___y_2178_);
lean_dec_ref(v___y_2177_);
lean_dec(v_goal_1948_);
lean_dec_ref(v___f_1947_);
lean_dec_ref(v___x_1946_);
v___x_2192_ = lean_obj_once(&lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__11, &lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__11_once, _init_lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__11);
v___x_2193_ = l_Lean_indentExpr(v___y_2183_);
v___x_2194_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2194_, 0, v___x_2192_);
lean_ctor_set(v___x_2194_, 1, v___x_2193_);
v___x_2195_ = lean_obj_once(&lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__13, &lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__13_once, _init_lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__13);
v___x_2196_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2196_, 0, v___x_2194_);
lean_ctor_set(v___x_2196_, 1, v___x_2195_);
v___x_2197_ = l_Lean_indentExpr(v___y_2179_);
v___x_2198_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2198_, 0, v___x_2196_);
lean_ctor_set(v___x_2198_, 1, v___x_2197_);
v___x_2199_ = lean_obj_once(&lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__15, &lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__15_once, _init_lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__15);
v___x_2200_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2200_, 0, v___x_2198_);
lean_ctor_set(v___x_2200_, 1, v___x_2199_);
v___x_2201_ = l_Nat_reprFast(v_numLevelParams_2184_);
v___x_2202_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2202_, 0, v___x_2201_);
v___x_2203_ = l_Lean_MessageData_ofFormat(v___x_2202_);
v___x_2204_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2204_, 0, v___x_2200_);
lean_ctor_set(v___x_2204_, 1, v___x_2203_);
v___x_2205_ = lean_obj_once(&lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__17, &lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__17_once, _init_lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__17);
v___x_2206_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2206_, 0, v___x_2204_);
lean_ctor_set(v___x_2206_, 1, v___x_2205_);
v___x_2207_ = l_Nat_reprFast(v___x_2190_);
v___x_2208_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2208_, 0, v___x_2207_);
v___x_2209_ = l_Lean_MessageData_ofFormat(v___x_2208_);
v___x_2210_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2210_, 0, v___x_2206_);
lean_ctor_set(v___x_2210_, 1, v___x_2209_);
v___x_2211_ = lp_aesop_Lean_throwError___at___00Aesop_ForwardRuleMatch_getProof_spec__8___redArg(v___x_2210_, v___y_2186_, v___y_2187_, v___y_2188_, v___y_2189_);
lean_dec(v___y_2189_);
lean_dec_ref(v___y_2188_);
lean_dec(v___y_2187_);
lean_dec_ref(v___y_2186_);
v_a_2212_ = lean_ctor_get(v___x_2211_, 0);
v_isSharedCheck_2219_ = !lean_is_exclusive(v___x_2211_);
if (v_isSharedCheck_2219_ == 0)
{
v___x_2214_ = v___x_2211_;
v_isShared_2215_ = v_isSharedCheck_2219_;
goto v_resetjp_2213_;
}
else
{
lean_inc(v_a_2212_);
lean_dec(v___x_2211_);
v___x_2214_ = lean_box(0);
v_isShared_2215_ = v_isSharedCheck_2219_;
goto v_resetjp_2213_;
}
v_resetjp_2213_:
{
lean_object* v___x_2217_; 
if (v_isShared_2215_ == 0)
{
v___x_2217_ = v___x_2214_;
goto v_reusejp_2216_;
}
else
{
lean_object* v_reuseFailAlloc_2218_; 
v_reuseFailAlloc_2218_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2218_, 0, v_a_2212_);
v___x_2217_ = v_reuseFailAlloc_2218_;
goto v_reusejp_2216_;
}
v_reusejp_2216_:
{
return v___x_2217_;
}
}
}
else
{
lean_dec(v_numLevelParams_2184_);
lean_dec_ref(v___y_2179_);
v___y_2134_ = v___y_2177_;
v___y_2135_ = v___y_2180_;
v___y_2136_ = v___y_2178_;
v___y_2137_ = v___y_2181_;
v___y_2138_ = v___y_2182_;
v___y_2139_ = v___y_2183_;
v___y_2140_ = v___y_2185_;
v___y_2141_ = v___y_2186_;
v___y_2142_ = v___y_2187_;
v___y_2143_ = v___y_2188_;
v___y_2144_ = v___y_2189_;
goto v___jp_2133_;
}
}
v___jp_2220_:
{
lean_object* v_keyedConfig_2232_; uint8_t v_trackZetaDelta_2233_; lean_object* v_zetaDeltaSet_2234_; lean_object* v_lctx_2235_; lean_object* v_localInstances_2236_; lean_object* v_defEqCtx_x3f_2237_; lean_object* v_synthPendingDepth_2238_; lean_object* v_customCanUnfoldPredicate_x3f_2239_; uint8_t v_univApprox_2240_; uint8_t v_inTypeClassResolution_2241_; uint8_t v_cacheInferType_2242_; uint8_t v___x_2243_; lean_object* v___x_2244_; lean_object* v___x_2245_; lean_object* v___x_2246_; 
v_keyedConfig_2232_ = lean_ctor_get(v___y_2228_, 0);
v_trackZetaDelta_2233_ = lean_ctor_get_uint8(v___y_2228_, sizeof(void*)*7);
v_zetaDeltaSet_2234_ = lean_ctor_get(v___y_2228_, 1);
v_lctx_2235_ = lean_ctor_get(v___y_2228_, 2);
v_localInstances_2236_ = lean_ctor_get(v___y_2228_, 3);
v_defEqCtx_x3f_2237_ = lean_ctor_get(v___y_2228_, 4);
v_synthPendingDepth_2238_ = lean_ctor_get(v___y_2228_, 5);
v_customCanUnfoldPredicate_x3f_2239_ = lean_ctor_get(v___y_2228_, 6);
v_univApprox_2240_ = lean_ctor_get_uint8(v___y_2228_, sizeof(void*)*7 + 1);
v_inTypeClassResolution_2241_ = lean_ctor_get_uint8(v___y_2228_, sizeof(void*)*7 + 2);
v_cacheInferType_2242_ = lean_ctor_get_uint8(v___y_2228_, sizeof(void*)*7 + 3);
v___x_2243_ = 2;
lean_inc_ref(v_keyedConfig_2232_);
v___x_2244_ = l_Lean_Meta_ConfigWithKey_setTransparency(v___x_2243_, v_keyedConfig_2232_);
lean_inc(v_customCanUnfoldPredicate_x3f_2239_);
lean_inc(v_synthPendingDepth_2238_);
lean_inc(v_defEqCtx_x3f_2237_);
lean_inc_ref(v_localInstances_2236_);
lean_inc_ref(v_lctx_2235_);
lean_inc(v_zetaDeltaSet_2234_);
v___x_2245_ = lean_alloc_ctor(0, 7, 4);
lean_ctor_set(v___x_2245_, 0, v___x_2244_);
lean_ctor_set(v___x_2245_, 1, v_zetaDeltaSet_2234_);
lean_ctor_set(v___x_2245_, 2, v_lctx_2235_);
lean_ctor_set(v___x_2245_, 3, v_localInstances_2236_);
lean_ctor_set(v___x_2245_, 4, v_defEqCtx_x3f_2237_);
lean_ctor_set(v___x_2245_, 5, v_synthPendingDepth_2238_);
lean_ctor_set(v___x_2245_, 6, v_customCanUnfoldPredicate_x3f_2239_);
lean_ctor_set_uint8(v___x_2245_, sizeof(void*)*7, v_trackZetaDelta_2233_);
lean_ctor_set_uint8(v___x_2245_, sizeof(void*)*7 + 1, v_univApprox_2240_);
lean_ctor_set_uint8(v___x_2245_, sizeof(void*)*7 + 2, v_inTypeClassResolution_2241_);
lean_ctor_set_uint8(v___x_2245_, sizeof(void*)*7 + 3, v_cacheInferType_2242_);
lean_inc(v___y_2231_);
lean_inc_ref(v___y_2230_);
lean_inc(v___y_2229_);
lean_inc_ref(v___x_2245_);
lean_inc_ref(v___y_2225_);
v___x_2246_ = lean_infer_type(v___y_2225_, v___x_2245_, v___y_2229_, v___y_2230_, v___y_2231_);
if (lean_obj_tag(v___x_2246_) == 0)
{
lean_object* v_a_2247_; uint8_t v___x_2248_; lean_object* v___x_2249_; 
v_a_2247_ = lean_ctor_get(v___x_2246_, 0);
lean_inc(v_a_2247_);
lean_dec_ref_known(v___x_2246_, 1);
v___x_2248_ = 0;
v___x_2249_ = l_Lean_Meta_forallMetaTelescope(v_a_2247_, v___x_2248_, v___x_2245_, v___y_2229_, v___y_2230_, v___y_2231_);
lean_dec_ref_known(v___x_2245_, 7);
if (lean_obj_tag(v___x_2249_) == 0)
{
lean_object* v_a_2250_; lean_object* v_snd_2251_; lean_object* v_fst_2252_; lean_object* v___x_2254_; uint8_t v_isShared_2255_; uint8_t v_isSharedCheck_2298_; 
v_a_2250_ = lean_ctor_get(v___x_2249_, 0);
lean_inc(v_a_2250_);
lean_dec_ref_known(v___x_2249_, 1);
v_snd_2251_ = lean_ctor_get(v_a_2250_, 1);
v_fst_2252_ = lean_ctor_get(v_a_2250_, 0);
v_isSharedCheck_2298_ = !lean_is_exclusive(v_a_2250_);
if (v_isSharedCheck_2298_ == 0)
{
v___x_2254_ = v_a_2250_;
v_isShared_2255_ = v_isSharedCheck_2298_;
goto v_resetjp_2253_;
}
else
{
lean_inc(v_snd_2251_);
lean_inc(v_fst_2252_);
lean_dec(v_a_2250_);
v___x_2254_ = lean_box(0);
v_isShared_2255_ = v_isSharedCheck_2298_;
goto v_resetjp_2253_;
}
v_resetjp_2253_:
{
lean_object* v_fst_2256_; lean_object* v___x_2258_; uint8_t v_isShared_2259_; uint8_t v_isSharedCheck_2296_; 
v_fst_2256_ = lean_ctor_get(v_snd_2251_, 0);
v_isSharedCheck_2296_ = !lean_is_exclusive(v_snd_2251_);
if (v_isSharedCheck_2296_ == 0)
{
lean_object* v_unused_2297_; 
v_unused_2297_ = lean_ctor_get(v_snd_2251_, 1);
lean_dec(v_unused_2297_);
v___x_2258_ = v_snd_2251_;
v_isShared_2259_ = v_isSharedCheck_2296_;
goto v_resetjp_2257_;
}
else
{
lean_inc(v_fst_2256_);
lean_dec(v_snd_2251_);
v___x_2258_ = lean_box(0);
v_isShared_2259_ = v_isSharedCheck_2296_;
goto v_resetjp_2257_;
}
v_resetjp_2257_:
{
lean_object* v_numPremises_2260_; lean_object* v_numLevelParams_2261_; lean_object* v___x_2262_; uint8_t v___x_2263_; 
v_numPremises_2260_ = lean_ctor_get(v___y_2226_, 0);
lean_inc(v_numPremises_2260_);
v_numLevelParams_2261_ = lean_ctor_get(v___y_2226_, 1);
lean_inc(v_numLevelParams_2261_);
lean_dec_ref(v___y_2226_);
v___x_2262_ = lean_array_get_size(v_fst_2252_);
v___x_2263_ = lean_nat_dec_eq(v___x_2262_, v_numPremises_2260_);
if (v___x_2263_ == 0)
{
lean_object* v___x_2264_; lean_object* v___x_2265_; lean_object* v___x_2267_; 
lean_dec(v_numLevelParams_2261_);
lean_dec(v_fst_2256_);
lean_dec(v_fst_2252_);
lean_dec_ref(v___y_2227_);
lean_dec(v___y_2224_);
lean_dec_ref(v___y_2223_);
lean_dec_ref(v___y_2222_);
lean_dec(v_goal_1948_);
lean_dec_ref(v___f_1947_);
lean_dec_ref(v___x_1946_);
v___x_2264_ = lean_obj_once(&lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__11, &lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__11_once, _init_lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__11);
v___x_2265_ = l_Lean_indentExpr(v___y_2225_);
if (v_isShared_2259_ == 0)
{
lean_ctor_set_tag(v___x_2258_, 7);
lean_ctor_set(v___x_2258_, 1, v___x_2265_);
lean_ctor_set(v___x_2258_, 0, v___x_2264_);
v___x_2267_ = v___x_2258_;
goto v_reusejp_2266_;
}
else
{
lean_object* v_reuseFailAlloc_2295_; 
v_reuseFailAlloc_2295_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2295_, 0, v___x_2264_);
lean_ctor_set(v_reuseFailAlloc_2295_, 1, v___x_2265_);
v___x_2267_ = v_reuseFailAlloc_2295_;
goto v_reusejp_2266_;
}
v_reusejp_2266_:
{
lean_object* v___x_2268_; lean_object* v___x_2270_; 
v___x_2268_ = lean_obj_once(&lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__13, &lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__13_once, _init_lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__13);
if (v_isShared_2255_ == 0)
{
lean_ctor_set_tag(v___x_2254_, 7);
lean_ctor_set(v___x_2254_, 1, v___x_2268_);
lean_ctor_set(v___x_2254_, 0, v___x_2267_);
v___x_2270_ = v___x_2254_;
goto v_reusejp_2269_;
}
else
{
lean_object* v_reuseFailAlloc_2294_; 
v_reuseFailAlloc_2294_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2294_, 0, v___x_2267_);
lean_ctor_set(v_reuseFailAlloc_2294_, 1, v___x_2268_);
v___x_2270_ = v_reuseFailAlloc_2294_;
goto v_reusejp_2269_;
}
v_reusejp_2269_:
{
lean_object* v___x_2271_; lean_object* v___x_2272_; lean_object* v___x_2273_; lean_object* v___x_2274_; lean_object* v___x_2275_; lean_object* v___x_2276_; lean_object* v___x_2277_; lean_object* v___x_2278_; lean_object* v___x_2279_; lean_object* v___x_2280_; lean_object* v___x_2281_; lean_object* v___x_2282_; lean_object* v___x_2283_; lean_object* v___x_2284_; lean_object* v___x_2285_; lean_object* v_a_2286_; lean_object* v___x_2288_; uint8_t v_isShared_2289_; uint8_t v_isSharedCheck_2293_; 
v___x_2271_ = l_Lean_indentExpr(v___y_2221_);
v___x_2272_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2272_, 0, v___x_2270_);
lean_ctor_set(v___x_2272_, 1, v___x_2271_);
v___x_2273_ = lean_obj_once(&lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__15, &lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__15_once, _init_lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__15);
v___x_2274_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2274_, 0, v___x_2272_);
lean_ctor_set(v___x_2274_, 1, v___x_2273_);
v___x_2275_ = l_Nat_reprFast(v_numPremises_2260_);
v___x_2276_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2276_, 0, v___x_2275_);
v___x_2277_ = l_Lean_MessageData_ofFormat(v___x_2276_);
v___x_2278_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2278_, 0, v___x_2274_);
lean_ctor_set(v___x_2278_, 1, v___x_2277_);
v___x_2279_ = lean_obj_once(&lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__19, &lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__19_once, _init_lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__19);
v___x_2280_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2280_, 0, v___x_2278_);
lean_ctor_set(v___x_2280_, 1, v___x_2279_);
v___x_2281_ = l_Nat_reprFast(v___x_2262_);
v___x_2282_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2282_, 0, v___x_2281_);
v___x_2283_ = l_Lean_MessageData_ofFormat(v___x_2282_);
v___x_2284_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2284_, 0, v___x_2280_);
lean_ctor_set(v___x_2284_, 1, v___x_2283_);
v___x_2285_ = lp_aesop_Lean_throwError___at___00Aesop_ForwardRuleMatch_getProof_spec__8___redArg(v___x_2284_, v___y_2228_, v___y_2229_, v___y_2230_, v___y_2231_);
lean_dec(v___y_2231_);
lean_dec_ref(v___y_2230_);
lean_dec(v___y_2229_);
lean_dec_ref(v___y_2228_);
v_a_2286_ = lean_ctor_get(v___x_2285_, 0);
v_isSharedCheck_2293_ = !lean_is_exclusive(v___x_2285_);
if (v_isSharedCheck_2293_ == 0)
{
v___x_2288_ = v___x_2285_;
v_isShared_2289_ = v_isSharedCheck_2293_;
goto v_resetjp_2287_;
}
else
{
lean_inc(v_a_2286_);
lean_dec(v___x_2285_);
v___x_2288_ = lean_box(0);
v_isShared_2289_ = v_isSharedCheck_2293_;
goto v_resetjp_2287_;
}
v_resetjp_2287_:
{
lean_object* v___x_2291_; 
if (v_isShared_2289_ == 0)
{
v___x_2291_ = v___x_2288_;
goto v_reusejp_2290_;
}
else
{
lean_object* v_reuseFailAlloc_2292_; 
v_reuseFailAlloc_2292_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2292_, 0, v_a_2286_);
v___x_2291_ = v_reuseFailAlloc_2292_;
goto v_reusejp_2290_;
}
v_reusejp_2290_:
{
return v___x_2291_;
}
}
}
}
}
else
{
lean_dec(v_numPremises_2260_);
lean_del_object(v___x_2258_);
lean_del_object(v___x_2254_);
v___y_2177_ = v_fst_2252_;
v___y_2178_ = v_fst_2256_;
v___y_2179_ = v___y_2221_;
v___y_2180_ = v___y_2222_;
v___y_2181_ = v___y_2223_;
v___y_2182_ = v___y_2224_;
v___y_2183_ = v___y_2225_;
v_numLevelParams_2184_ = v_numLevelParams_2261_;
v___y_2185_ = v___y_2227_;
v___y_2186_ = v___y_2228_;
v___y_2187_ = v___y_2229_;
v___y_2188_ = v___y_2230_;
v___y_2189_ = v___y_2231_;
goto v___jp_2176_;
}
}
}
}
else
{
lean_object* v_a_2299_; lean_object* v___x_2301_; uint8_t v_isShared_2302_; uint8_t v_isSharedCheck_2306_; 
lean_dec(v___y_2231_);
lean_dec_ref(v___y_2230_);
lean_dec(v___y_2229_);
lean_dec_ref(v___y_2228_);
lean_dec_ref(v___y_2227_);
lean_dec_ref(v___y_2226_);
lean_dec_ref(v___y_2225_);
lean_dec(v___y_2224_);
lean_dec_ref(v___y_2223_);
lean_dec_ref(v___y_2222_);
lean_dec_ref(v___y_2221_);
lean_dec(v_goal_1948_);
lean_dec_ref(v___f_1947_);
lean_dec_ref(v___x_1946_);
v_a_2299_ = lean_ctor_get(v___x_2249_, 0);
v_isSharedCheck_2306_ = !lean_is_exclusive(v___x_2249_);
if (v_isSharedCheck_2306_ == 0)
{
v___x_2301_ = v___x_2249_;
v_isShared_2302_ = v_isSharedCheck_2306_;
goto v_resetjp_2300_;
}
else
{
lean_inc(v_a_2299_);
lean_dec(v___x_2249_);
v___x_2301_ = lean_box(0);
v_isShared_2302_ = v_isSharedCheck_2306_;
goto v_resetjp_2300_;
}
v_resetjp_2300_:
{
lean_object* v___x_2304_; 
if (v_isShared_2302_ == 0)
{
v___x_2304_ = v___x_2301_;
goto v_reusejp_2303_;
}
else
{
lean_object* v_reuseFailAlloc_2305_; 
v_reuseFailAlloc_2305_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2305_, 0, v_a_2299_);
v___x_2304_ = v_reuseFailAlloc_2305_;
goto v_reusejp_2303_;
}
v_reusejp_2303_:
{
return v___x_2304_;
}
}
}
}
else
{
lean_object* v_a_2307_; lean_object* v___x_2309_; uint8_t v_isShared_2310_; uint8_t v_isSharedCheck_2314_; 
lean_dec_ref_known(v___x_2245_, 7);
lean_dec(v___y_2231_);
lean_dec_ref(v___y_2230_);
lean_dec(v___y_2229_);
lean_dec_ref(v___y_2228_);
lean_dec_ref(v___y_2227_);
lean_dec_ref(v___y_2226_);
lean_dec_ref(v___y_2225_);
lean_dec(v___y_2224_);
lean_dec_ref(v___y_2223_);
lean_dec_ref(v___y_2222_);
lean_dec_ref(v___y_2221_);
lean_dec(v_goal_1948_);
lean_dec_ref(v___f_1947_);
lean_dec_ref(v___x_1946_);
v_a_2307_ = lean_ctor_get(v___x_2246_, 0);
v_isSharedCheck_2314_ = !lean_is_exclusive(v___x_2246_);
if (v_isSharedCheck_2314_ == 0)
{
v___x_2309_ = v___x_2246_;
v_isShared_2310_ = v_isSharedCheck_2314_;
goto v_resetjp_2308_;
}
else
{
lean_inc(v_a_2307_);
lean_dec(v___x_2246_);
v___x_2309_ = lean_box(0);
v_isShared_2310_ = v_isSharedCheck_2314_;
goto v_resetjp_2308_;
}
v_resetjp_2308_:
{
lean_object* v___x_2312_; 
if (v_isShared_2310_ == 0)
{
v___x_2312_ = v___x_2309_;
goto v_reusejp_2311_;
}
else
{
lean_object* v_reuseFailAlloc_2313_; 
v_reuseFailAlloc_2313_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2313_, 0, v_a_2307_);
v___x_2312_ = v_reuseFailAlloc_2313_;
goto v_reusejp_2311_;
}
v_reusejp_2311_:
{
return v___x_2312_;
}
}
}
}
v___jp_2315_:
{
lean_object* v_rule_2320_; lean_object* v_match_2321_; lean_object* v___x_2323_; uint8_t v_isShared_2324_; uint8_t v_isSharedCheck_2376_; 
v_rule_2320_ = lean_ctor_get(v_m_1949_, 0);
v_match_2321_ = lean_ctor_get(v_m_1949_, 1);
v_isSharedCheck_2376_ = !lean_is_exclusive(v_m_1949_);
if (v_isSharedCheck_2376_ == 0)
{
v___x_2323_ = v_m_1949_;
v_isShared_2324_ = v_isSharedCheck_2376_;
goto v_resetjp_2322_;
}
else
{
lean_inc(v_match_2321_);
lean_inc(v_rule_2320_);
lean_dec(v_m_1949_);
v___x_2323_ = lean_box(0);
v_isShared_2324_ = v_isSharedCheck_2376_;
goto v_resetjp_2322_;
}
v_resetjp_2322_:
{
lean_object* v_toForwardRuleInfo_2325_; lean_object* v_term_2326_; lean_object* v___x_2327_; 
v_toForwardRuleInfo_2325_ = lean_ctor_get(v_rule_2320_, 0);
lean_inc_ref(v_toForwardRuleInfo_2325_);
v_term_2326_ = lean_ctor_get(v_rule_2320_, 2);
lean_inc_ref(v_term_2326_);
lean_inc(v_goal_1948_);
v___x_2327_ = lp_aesop_Aesop_elabForwardRuleTerm(v_goal_1948_, v_term_2326_, v___y_2316_, v___y_2317_, v___y_2318_, v___y_2319_);
if (lean_obj_tag(v___x_2327_) == 0)
{
lean_object* v_a_2328_; lean_object* v___x_2329_; lean_object* v___x_2330_; lean_object* v___x_2331_; lean_object* v___x_2332_; 
v_a_2328_ = lean_ctor_get(v___x_2327_, 0);
lean_inc_n(v_a_2328_, 3);
lean_dec_ref_known(v___x_2327_, 1);
v___x_2329_ = lean_unsigned_to_nat(0u);
v___x_2330_ = lean_obj_once(&lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__23, &lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__23_once, _init_lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__23);
v___x_2331_ = l_Lean_collectLevelMVars(v___x_2330_, v_a_2328_);
lean_inc(v___y_2319_);
lean_inc_ref(v___y_2318_);
lean_inc(v___y_2317_);
lean_inc_ref(v___y_2316_);
v___x_2332_ = lean_infer_type(v_a_2328_, v___y_2316_, v___y_2317_, v___y_2318_, v___y_2319_);
if (lean_obj_tag(v___x_2332_) == 0)
{
lean_object* v_a_2333_; lean_object* v___x_2334_; lean_object* v_a_2335_; lean_object* v___x_2336_; lean_object* v_a_2337_; uint8_t v___x_2338_; 
v_a_2333_ = lean_ctor_get(v___x_2332_, 0);
lean_inc(v_a_2333_);
lean_dec_ref_known(v___x_2332_, 1);
v___x_2334_ = lp_aesop_Lean_instantiateMVars___at___00Aesop_ForwardRuleMatch_getProof_spec__2___redArg(v_a_2333_, v___y_2317_);
v_a_2335_ = lean_ctor_get(v___x_2334_, 0);
lean_inc(v_a_2335_);
lean_dec_ref(v___x_2334_);
v___x_2336_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_ForwardRuleMatch_getProof_spec__3___redArg(v___x_1946_, v___y_2318_);
v_a_2337_ = lean_ctor_get(v___x_2336_, 0);
lean_inc(v_a_2337_);
lean_dec_ref(v___x_2336_);
v___x_2338_ = lean_unbox(v_a_2337_);
lean_dec(v_a_2337_);
if (v___x_2338_ == 0)
{
lean_object* v_result_2339_; 
lean_del_object(v___x_2323_);
v_result_2339_ = lean_ctor_get(v___x_2331_, 2);
lean_inc_ref(v_result_2339_);
lean_dec_ref(v___x_2331_);
v___y_2221_ = v_a_2335_;
v___y_2222_ = v_result_2339_;
v___y_2223_ = v_match_2321_;
v___y_2224_ = v___x_2329_;
v___y_2225_ = v_a_2328_;
v___y_2226_ = v_toForwardRuleInfo_2325_;
v___y_2227_ = v_rule_2320_;
v___y_2228_ = v___y_2316_;
v___y_2229_ = v___y_2317_;
v___y_2230_ = v___y_2318_;
v___y_2231_ = v___y_2319_;
goto v___jp_2220_;
}
else
{
lean_object* v_result_2340_; lean_object* v_traceClass_2341_; lean_object* v___x_2342_; lean_object* v___x_2343_; lean_object* v___x_2345_; 
v_result_2340_ = lean_ctor_get(v___x_2331_, 2);
lean_inc_ref(v_result_2340_);
lean_dec_ref(v___x_2331_);
v_traceClass_2341_ = lean_ctor_get(v___x_1946_, 0);
v___x_2342_ = lean_obj_once(&lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__25, &lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__25_once, _init_lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__25);
lean_inc(v_a_2328_);
v___x_2343_ = l_Lean_MessageData_ofExpr(v_a_2328_);
if (v_isShared_2324_ == 0)
{
lean_ctor_set_tag(v___x_2323_, 7);
lean_ctor_set(v___x_2323_, 1, v___x_2343_);
lean_ctor_set(v___x_2323_, 0, v___x_2342_);
v___x_2345_ = v___x_2323_;
goto v_reusejp_2344_;
}
else
{
lean_object* v_reuseFailAlloc_2359_; 
v_reuseFailAlloc_2359_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2359_, 0, v___x_2342_);
lean_ctor_set(v_reuseFailAlloc_2359_, 1, v___x_2343_);
v___x_2345_ = v_reuseFailAlloc_2359_;
goto v_reusejp_2344_;
}
v_reusejp_2344_:
{
lean_object* v___x_2346_; lean_object* v___x_2347_; lean_object* v___x_2348_; lean_object* v___x_2349_; lean_object* v___x_2350_; 
v___x_2346_ = lean_obj_once(&lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__27, &lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__27_once, _init_lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__27);
v___x_2347_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2347_, 0, v___x_2345_);
lean_ctor_set(v___x_2347_, 1, v___x_2346_);
lean_inc(v_a_2335_);
v___x_2348_ = l_Lean_MessageData_ofExpr(v_a_2335_);
v___x_2349_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2349_, 0, v___x_2347_);
lean_ctor_set(v___x_2349_, 1, v___x_2348_);
lean_inc(v_traceClass_2341_);
v___x_2350_ = lp_aesop_Lean_addTrace___at___00Aesop_ForwardRuleMatch_getProof_spec__4(v_traceClass_2341_, v___x_2349_, v___y_2316_, v___y_2317_, v___y_2318_, v___y_2319_);
if (lean_obj_tag(v___x_2350_) == 0)
{
lean_dec_ref_known(v___x_2350_, 1);
v___y_2221_ = v_a_2335_;
v___y_2222_ = v_result_2340_;
v___y_2223_ = v_match_2321_;
v___y_2224_ = v___x_2329_;
v___y_2225_ = v_a_2328_;
v___y_2226_ = v_toForwardRuleInfo_2325_;
v___y_2227_ = v_rule_2320_;
v___y_2228_ = v___y_2316_;
v___y_2229_ = v___y_2317_;
v___y_2230_ = v___y_2318_;
v___y_2231_ = v___y_2319_;
goto v___jp_2220_;
}
else
{
lean_object* v_a_2351_; lean_object* v___x_2353_; uint8_t v_isShared_2354_; uint8_t v_isSharedCheck_2358_; 
lean_dec_ref(v_result_2340_);
lean_dec(v_a_2335_);
lean_dec(v_a_2328_);
lean_dec_ref(v_toForwardRuleInfo_2325_);
lean_dec_ref(v_match_2321_);
lean_dec_ref(v_rule_2320_);
lean_dec(v___y_2319_);
lean_dec_ref(v___y_2318_);
lean_dec(v___y_2317_);
lean_dec_ref(v___y_2316_);
lean_dec(v_goal_1948_);
lean_dec_ref(v___f_1947_);
lean_dec_ref(v___x_1946_);
v_a_2351_ = lean_ctor_get(v___x_2350_, 0);
v_isSharedCheck_2358_ = !lean_is_exclusive(v___x_2350_);
if (v_isSharedCheck_2358_ == 0)
{
v___x_2353_ = v___x_2350_;
v_isShared_2354_ = v_isSharedCheck_2358_;
goto v_resetjp_2352_;
}
else
{
lean_inc(v_a_2351_);
lean_dec(v___x_2350_);
v___x_2353_ = lean_box(0);
v_isShared_2354_ = v_isSharedCheck_2358_;
goto v_resetjp_2352_;
}
v_resetjp_2352_:
{
lean_object* v___x_2356_; 
if (v_isShared_2354_ == 0)
{
v___x_2356_ = v___x_2353_;
goto v_reusejp_2355_;
}
else
{
lean_object* v_reuseFailAlloc_2357_; 
v_reuseFailAlloc_2357_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2357_, 0, v_a_2351_);
v___x_2356_ = v_reuseFailAlloc_2357_;
goto v_reusejp_2355_;
}
v_reusejp_2355_:
{
return v___x_2356_;
}
}
}
}
}
}
else
{
lean_object* v_a_2360_; lean_object* v___x_2362_; uint8_t v_isShared_2363_; uint8_t v_isSharedCheck_2367_; 
lean_dec_ref(v___x_2331_);
lean_dec(v_a_2328_);
lean_dec_ref(v_toForwardRuleInfo_2325_);
lean_del_object(v___x_2323_);
lean_dec_ref(v_match_2321_);
lean_dec_ref(v_rule_2320_);
lean_dec(v___y_2319_);
lean_dec_ref(v___y_2318_);
lean_dec(v___y_2317_);
lean_dec_ref(v___y_2316_);
lean_dec(v_goal_1948_);
lean_dec_ref(v___f_1947_);
lean_dec_ref(v___x_1946_);
v_a_2360_ = lean_ctor_get(v___x_2332_, 0);
v_isSharedCheck_2367_ = !lean_is_exclusive(v___x_2332_);
if (v_isSharedCheck_2367_ == 0)
{
v___x_2362_ = v___x_2332_;
v_isShared_2363_ = v_isSharedCheck_2367_;
goto v_resetjp_2361_;
}
else
{
lean_inc(v_a_2360_);
lean_dec(v___x_2332_);
v___x_2362_ = lean_box(0);
v_isShared_2363_ = v_isSharedCheck_2367_;
goto v_resetjp_2361_;
}
v_resetjp_2361_:
{
lean_object* v___x_2365_; 
if (v_isShared_2363_ == 0)
{
v___x_2365_ = v___x_2362_;
goto v_reusejp_2364_;
}
else
{
lean_object* v_reuseFailAlloc_2366_; 
v_reuseFailAlloc_2366_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2366_, 0, v_a_2360_);
v___x_2365_ = v_reuseFailAlloc_2366_;
goto v_reusejp_2364_;
}
v_reusejp_2364_:
{
return v___x_2365_;
}
}
}
}
else
{
lean_object* v_a_2368_; lean_object* v___x_2370_; uint8_t v_isShared_2371_; uint8_t v_isSharedCheck_2375_; 
lean_dec_ref(v_toForwardRuleInfo_2325_);
lean_del_object(v___x_2323_);
lean_dec_ref(v_match_2321_);
lean_dec_ref(v_rule_2320_);
lean_dec(v___y_2319_);
lean_dec_ref(v___y_2318_);
lean_dec(v___y_2317_);
lean_dec_ref(v___y_2316_);
lean_dec(v_goal_1948_);
lean_dec_ref(v___f_1947_);
lean_dec_ref(v___x_1946_);
v_a_2368_ = lean_ctor_get(v___x_2327_, 0);
v_isSharedCheck_2375_ = !lean_is_exclusive(v___x_2327_);
if (v_isSharedCheck_2375_ == 0)
{
v___x_2370_ = v___x_2327_;
v_isShared_2371_ = v_isSharedCheck_2375_;
goto v_resetjp_2369_;
}
else
{
lean_inc(v_a_2368_);
lean_dec(v___x_2327_);
v___x_2370_ = lean_box(0);
v_isShared_2371_ = v_isSharedCheck_2375_;
goto v_resetjp_2369_;
}
v_resetjp_2369_:
{
lean_object* v___x_2373_; 
if (v_isShared_2371_ == 0)
{
v___x_2373_ = v___x_2370_;
goto v_reusejp_2372_;
}
else
{
lean_object* v_reuseFailAlloc_2374_; 
v_reuseFailAlloc_2374_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2374_, 0, v_a_2368_);
v___x_2373_ = v_reuseFailAlloc_2374_;
goto v_reusejp_2372_;
}
v_reusejp_2372_:
{
return v___x_2373_;
}
}
}
}
}
v_resetjp_2379_:
{
uint8_t v___x_2382_; 
v___x_2382_ = lean_unbox(v_a_2378_);
if (v___x_2382_ == 0)
{
lean_del_object(v___x_2380_);
lean_dec(v_a_2378_);
v___y_2316_ = v___y_1950_;
v___y_2317_ = v___y_1951_;
v___y_2318_ = v___y_1952_;
v___y_2319_ = v___y_1953_;
goto v___jp_2315_;
}
else
{
lean_object* v_rule_2383_; lean_object* v_name_2384_; lean_object* v_traceClass_2385_; lean_object* v_name_2386_; uint8_t v_builder_2387_; uint8_t v_phase_2388_; uint8_t v_scope_2389_; lean_object* v___x_2390_; lean_object* v___y_2392_; lean_object* v___y_2393_; lean_object* v___y_2394_; lean_object* v___y_2415_; lean_object* v___y_2416_; lean_object* v___y_2417_; lean_object* v___y_2423_; 
v_rule_2383_ = lean_ctor_get(v_m_1949_, 0);
v_name_2384_ = lean_ctor_get(v_rule_2383_, 1);
v_traceClass_2385_ = lean_ctor_get(v___x_1946_, 0);
v_name_2386_ = lean_ctor_get(v_name_2384_, 0);
v_builder_2387_ = lean_ctor_get_uint8(v_name_2384_, sizeof(void*)*1 + 8);
v_phase_2388_ = lean_ctor_get_uint8(v_name_2384_, sizeof(void*)*1 + 9);
v_scope_2389_ = lean_ctor_get_uint8(v_name_2384_, sizeof(void*)*1 + 10);
v___x_2390_ = lean_obj_once(&lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__29, &lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__29_once, _init_lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___closed__29);
switch(v_phase_2388_)
{
case 0:
{
lean_object* v___x_2434_; 
v___x_2434_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__13));
v___y_2423_ = v___x_2434_;
goto v___jp_2422_;
}
case 1:
{
lean_object* v___x_2435_; 
v___x_2435_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__14));
v___y_2423_ = v___x_2435_;
goto v___jp_2422_;
}
default: 
{
lean_object* v___x_2436_; 
v___x_2436_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__15));
v___y_2423_ = v___x_2436_;
goto v___jp_2422_;
}
}
v___jp_2391_:
{
lean_object* v___x_2395_; lean_object* v___x_2396_; uint8_t v___x_2397_; lean_object* v___x_2398_; lean_object* v___x_2399_; lean_object* v___x_2401_; 
v___x_2395_ = lean_string_append(v___y_2393_, v___y_2394_);
v___x_2396_ = lean_string_append(v___x_2395_, v___y_2392_);
v___x_2397_ = lean_unbox(v_a_2378_);
lean_dec(v_a_2378_);
lean_inc(v_name_2386_);
v___x_2398_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_name_2386_, v___x_2397_);
v___x_2399_ = lean_string_append(v___x_2396_, v___x_2398_);
lean_dec_ref(v___x_2398_);
if (v_isShared_2381_ == 0)
{
lean_ctor_set_tag(v___x_2380_, 3);
lean_ctor_set(v___x_2380_, 0, v___x_2399_);
v___x_2401_ = v___x_2380_;
goto v_reusejp_2400_;
}
else
{
lean_object* v_reuseFailAlloc_2413_; 
v_reuseFailAlloc_2413_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2413_, 0, v___x_2399_);
v___x_2401_ = v_reuseFailAlloc_2413_;
goto v_reusejp_2400_;
}
v_reusejp_2400_:
{
lean_object* v___x_2402_; lean_object* v___x_2403_; lean_object* v___x_2404_; 
v___x_2402_ = l_Lean_MessageData_ofFormat(v___x_2401_);
v___x_2403_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2403_, 0, v___x_2390_);
lean_ctor_set(v___x_2403_, 1, v___x_2402_);
lean_inc(v_traceClass_2385_);
v___x_2404_ = lp_aesop_Lean_addTrace___at___00Aesop_ForwardRuleMatch_getProof_spec__4(v_traceClass_2385_, v___x_2403_, v___y_1950_, v___y_1951_, v___y_1952_, v___y_1953_);
if (lean_obj_tag(v___x_2404_) == 0)
{
lean_dec_ref_known(v___x_2404_, 1);
v___y_2316_ = v___y_1950_;
v___y_2317_ = v___y_1951_;
v___y_2318_ = v___y_1952_;
v___y_2319_ = v___y_1953_;
goto v___jp_2315_;
}
else
{
lean_object* v_a_2405_; lean_object* v___x_2407_; uint8_t v_isShared_2408_; uint8_t v_isSharedCheck_2412_; 
lean_dec(v___y_1953_);
lean_dec_ref(v___y_1952_);
lean_dec(v___y_1951_);
lean_dec_ref(v___y_1950_);
lean_dec_ref(v_m_1949_);
lean_dec(v_goal_1948_);
lean_dec_ref(v___f_1947_);
lean_dec_ref(v___x_1946_);
v_a_2405_ = lean_ctor_get(v___x_2404_, 0);
v_isSharedCheck_2412_ = !lean_is_exclusive(v___x_2404_);
if (v_isSharedCheck_2412_ == 0)
{
v___x_2407_ = v___x_2404_;
v_isShared_2408_ = v_isSharedCheck_2412_;
goto v_resetjp_2406_;
}
else
{
lean_inc(v_a_2405_);
lean_dec(v___x_2404_);
v___x_2407_ = lean_box(0);
v_isShared_2408_ = v_isSharedCheck_2412_;
goto v_resetjp_2406_;
}
v_resetjp_2406_:
{
lean_object* v___x_2410_; 
if (v_isShared_2408_ == 0)
{
v___x_2410_ = v___x_2407_;
goto v_reusejp_2409_;
}
else
{
lean_object* v_reuseFailAlloc_2411_; 
v_reuseFailAlloc_2411_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2411_, 0, v_a_2405_);
v___x_2410_ = v_reuseFailAlloc_2411_;
goto v_reusejp_2409_;
}
v_reusejp_2409_:
{
return v___x_2410_;
}
}
}
}
}
v___jp_2414_:
{
lean_object* v___x_2418_; lean_object* v___x_2419_; 
v___x_2418_ = lean_string_append(v___y_2415_, v___y_2417_);
v___x_2419_ = lean_string_append(v___x_2418_, v___y_2416_);
if (v_scope_2389_ == 0)
{
lean_object* v___x_2420_; 
v___x_2420_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__2));
v___y_2392_ = v___y_2416_;
v___y_2393_ = v___x_2419_;
v___y_2394_ = v___x_2420_;
goto v___jp_2391_;
}
else
{
lean_object* v___x_2421_; 
v___x_2421_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__3));
v___y_2392_ = v___y_2416_;
v___y_2393_ = v___x_2419_;
v___y_2394_ = v___x_2421_;
goto v___jp_2391_;
}
}
v___jp_2422_:
{
lean_object* v___x_2424_; lean_object* v___x_2425_; 
v___x_2424_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__4));
lean_inc_ref(v___y_2423_);
v___x_2425_ = lean_string_append(v___y_2423_, v___x_2424_);
switch(v_builder_2387_)
{
case 0:
{
lean_object* v___x_2426_; 
v___x_2426_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__5));
v___y_2415_ = v___x_2425_;
v___y_2416_ = v___x_2424_;
v___y_2417_ = v___x_2426_;
goto v___jp_2414_;
}
case 1:
{
lean_object* v___x_2427_; 
v___x_2427_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__6));
v___y_2415_ = v___x_2425_;
v___y_2416_ = v___x_2424_;
v___y_2417_ = v___x_2427_;
goto v___jp_2414_;
}
case 2:
{
lean_object* v___x_2428_; 
v___x_2428_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__7));
v___y_2415_ = v___x_2425_;
v___y_2416_ = v___x_2424_;
v___y_2417_ = v___x_2428_;
goto v___jp_2414_;
}
case 3:
{
lean_object* v___x_2429_; 
v___x_2429_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__8));
v___y_2415_ = v___x_2425_;
v___y_2416_ = v___x_2424_;
v___y_2417_ = v___x_2429_;
goto v___jp_2414_;
}
case 4:
{
lean_object* v___x_2430_; 
v___x_2430_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__9));
v___y_2415_ = v___x_2425_;
v___y_2416_ = v___x_2424_;
v___y_2417_ = v___x_2430_;
goto v___jp_2414_;
}
case 5:
{
lean_object* v___x_2431_; 
v___x_2431_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__10));
v___y_2415_ = v___x_2425_;
v___y_2416_ = v___x_2424_;
v___y_2417_ = v___x_2431_;
goto v___jp_2414_;
}
case 6:
{
lean_object* v___x_2432_; 
v___x_2432_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__11));
v___y_2415_ = v___x_2425_;
v___y_2416_ = v___x_2424_;
v___y_2417_ = v___x_2432_;
goto v___jp_2414_;
}
default: 
{
lean_object* v___x_2433_; 
v___x_2433_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__12));
v___y_2415_ = v___x_2425_;
v___y_2416_ = v___x_2424_;
v___y_2417_ = v___x_2433_;
goto v___jp_2414_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___boxed(lean_object* v___x_2438_, lean_object* v___f_2439_, lean_object* v_goal_2440_, lean_object* v_m_2441_, lean_object* v___y_2442_, lean_object* v___y_2443_, lean_object* v___y_2444_, lean_object* v___y_2445_, lean_object* v___y_2446_){
_start:
{
lean_object* v_res_2447_; 
v_res_2447_ = lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1(v___x_2438_, v___f_2439_, v_goal_2440_, v_m_2441_, v___y_2442_, v___y_2443_, v___y_2444_, v___y_2445_);
return v_res_2447_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__2(lean_object* v___x_2448_, lean_object* v_x_2449_, lean_object* v___y_2450_, lean_object* v___y_2451_, lean_object* v___y_2452_, lean_object* v___y_2453_){
_start:
{
lean_object* v___x_2455_; 
v___x_2455_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2455_, 0, v___x_2448_);
return v___x_2455_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__2___boxed(lean_object* v___x_2456_, lean_object* v_x_2457_, lean_object* v___y_2458_, lean_object* v___y_2459_, lean_object* v___y_2460_, lean_object* v___y_2461_, lean_object* v___y_2462_){
_start:
{
lean_object* v_res_2463_; 
v_res_2463_ = lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__2(v___x_2456_, v_x_2457_, v___y_2458_, v___y_2459_, v___y_2460_, v___y_2461_);
lean_dec(v___y_2461_);
lean_dec_ref(v___y_2460_);
lean_dec(v___y_2459_);
lean_dec_ref(v___y_2458_);
lean_dec_ref(v_x_2457_);
return v_res_2463_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13_spec__16_spec__19(size_t v_sz_2464_, size_t v_i_2465_, lean_object* v_bs_2466_){
_start:
{
uint8_t v___x_2467_; 
v___x_2467_ = lean_usize_dec_lt(v_i_2465_, v_sz_2464_);
if (v___x_2467_ == 0)
{
return v_bs_2466_;
}
else
{
lean_object* v_v_2468_; lean_object* v_msg_2469_; lean_object* v___x_2470_; lean_object* v_bs_x27_2471_; size_t v___x_2472_; size_t v___x_2473_; lean_object* v___x_2474_; 
v_v_2468_ = lean_array_uget_borrowed(v_bs_2466_, v_i_2465_);
v_msg_2469_ = lean_ctor_get(v_v_2468_, 1);
lean_inc_ref(v_msg_2469_);
v___x_2470_ = lean_unsigned_to_nat(0u);
v_bs_x27_2471_ = lean_array_uset(v_bs_2466_, v_i_2465_, v___x_2470_);
v___x_2472_ = ((size_t)1ULL);
v___x_2473_ = lean_usize_add(v_i_2465_, v___x_2472_);
v___x_2474_ = lean_array_uset(v_bs_x27_2471_, v_i_2465_, v_msg_2469_);
v_i_2465_ = v___x_2473_;
v_bs_2466_ = v___x_2474_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13_spec__16_spec__19___boxed(lean_object* v_sz_2476_, lean_object* v_i_2477_, lean_object* v_bs_2478_){
_start:
{
size_t v_sz_boxed_2479_; size_t v_i_boxed_2480_; lean_object* v_res_2481_; 
v_sz_boxed_2479_ = lean_unbox_usize(v_sz_2476_);
lean_dec(v_sz_2476_);
v_i_boxed_2480_ = lean_unbox_usize(v_i_2477_);
lean_dec(v_i_2477_);
v_res_2481_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13_spec__16_spec__19(v_sz_boxed_2479_, v_i_boxed_2480_, v_bs_2478_);
return v_res_2481_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13_spec__16(lean_object* v_oldTraces_2482_, lean_object* v_data_2483_, lean_object* v_ref_2484_, lean_object* v_msg_2485_, lean_object* v___y_2486_, lean_object* v___y_2487_, lean_object* v___y_2488_, lean_object* v___y_2489_){
_start:
{
lean_object* v_fileName_2491_; lean_object* v_fileMap_2492_; lean_object* v_options_2493_; lean_object* v_currRecDepth_2494_; lean_object* v_maxRecDepth_2495_; lean_object* v_ref_2496_; lean_object* v_currNamespace_2497_; lean_object* v_openDecls_2498_; lean_object* v_initHeartbeats_2499_; lean_object* v_maxHeartbeats_2500_; lean_object* v_quotContext_2501_; lean_object* v_currMacroScope_2502_; uint8_t v_diag_2503_; lean_object* v_cancelTk_x3f_2504_; uint8_t v_suppressElabErrors_2505_; lean_object* v_inheritedTraceOptions_2506_; lean_object* v___x_2507_; lean_object* v_traceState_2508_; lean_object* v_traces_2509_; lean_object* v_ref_2510_; lean_object* v___x_2511_; lean_object* v___x_2512_; size_t v_sz_2513_; size_t v___x_2514_; lean_object* v___x_2515_; lean_object* v_msg_2516_; lean_object* v___x_2517_; lean_object* v_a_2518_; lean_object* v___x_2520_; uint8_t v_isShared_2521_; uint8_t v_isSharedCheck_2555_; 
v_fileName_2491_ = lean_ctor_get(v___y_2488_, 0);
v_fileMap_2492_ = lean_ctor_get(v___y_2488_, 1);
v_options_2493_ = lean_ctor_get(v___y_2488_, 2);
v_currRecDepth_2494_ = lean_ctor_get(v___y_2488_, 3);
v_maxRecDepth_2495_ = lean_ctor_get(v___y_2488_, 4);
v_ref_2496_ = lean_ctor_get(v___y_2488_, 5);
v_currNamespace_2497_ = lean_ctor_get(v___y_2488_, 6);
v_openDecls_2498_ = lean_ctor_get(v___y_2488_, 7);
v_initHeartbeats_2499_ = lean_ctor_get(v___y_2488_, 8);
v_maxHeartbeats_2500_ = lean_ctor_get(v___y_2488_, 9);
v_quotContext_2501_ = lean_ctor_get(v___y_2488_, 10);
v_currMacroScope_2502_ = lean_ctor_get(v___y_2488_, 11);
v_diag_2503_ = lean_ctor_get_uint8(v___y_2488_, sizeof(void*)*14);
v_cancelTk_x3f_2504_ = lean_ctor_get(v___y_2488_, 12);
v_suppressElabErrors_2505_ = lean_ctor_get_uint8(v___y_2488_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_2506_ = lean_ctor_get(v___y_2488_, 13);
v___x_2507_ = lean_st_ref_get(v___y_2489_);
v_traceState_2508_ = lean_ctor_get(v___x_2507_, 4);
lean_inc_ref(v_traceState_2508_);
lean_dec(v___x_2507_);
v_traces_2509_ = lean_ctor_get(v_traceState_2508_, 0);
lean_inc_ref(v_traces_2509_);
lean_dec_ref(v_traceState_2508_);
v_ref_2510_ = l_Lean_replaceRef(v_ref_2484_, v_ref_2496_);
lean_inc_ref(v_inheritedTraceOptions_2506_);
lean_inc(v_cancelTk_x3f_2504_);
lean_inc(v_currMacroScope_2502_);
lean_inc(v_quotContext_2501_);
lean_inc(v_maxHeartbeats_2500_);
lean_inc(v_initHeartbeats_2499_);
lean_inc(v_openDecls_2498_);
lean_inc(v_currNamespace_2497_);
lean_inc(v_maxRecDepth_2495_);
lean_inc(v_currRecDepth_2494_);
lean_inc_ref(v_options_2493_);
lean_inc_ref(v_fileMap_2492_);
lean_inc_ref(v_fileName_2491_);
v___x_2511_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_2511_, 0, v_fileName_2491_);
lean_ctor_set(v___x_2511_, 1, v_fileMap_2492_);
lean_ctor_set(v___x_2511_, 2, v_options_2493_);
lean_ctor_set(v___x_2511_, 3, v_currRecDepth_2494_);
lean_ctor_set(v___x_2511_, 4, v_maxRecDepth_2495_);
lean_ctor_set(v___x_2511_, 5, v_ref_2510_);
lean_ctor_set(v___x_2511_, 6, v_currNamespace_2497_);
lean_ctor_set(v___x_2511_, 7, v_openDecls_2498_);
lean_ctor_set(v___x_2511_, 8, v_initHeartbeats_2499_);
lean_ctor_set(v___x_2511_, 9, v_maxHeartbeats_2500_);
lean_ctor_set(v___x_2511_, 10, v_quotContext_2501_);
lean_ctor_set(v___x_2511_, 11, v_currMacroScope_2502_);
lean_ctor_set(v___x_2511_, 12, v_cancelTk_x3f_2504_);
lean_ctor_set(v___x_2511_, 13, v_inheritedTraceOptions_2506_);
lean_ctor_set_uint8(v___x_2511_, sizeof(void*)*14, v_diag_2503_);
lean_ctor_set_uint8(v___x_2511_, sizeof(void*)*14 + 1, v_suppressElabErrors_2505_);
v___x_2512_ = l_Lean_PersistentArray_toArray___redArg(v_traces_2509_);
lean_dec_ref(v_traces_2509_);
v_sz_2513_ = lean_array_size(v___x_2512_);
v___x_2514_ = ((size_t)0ULL);
v___x_2515_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13_spec__16_spec__19(v_sz_2513_, v___x_2514_, v___x_2512_);
v_msg_2516_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v_msg_2516_, 0, v_data_2483_);
lean_ctor_set(v_msg_2516_, 1, v_msg_2485_);
lean_ctor_set(v_msg_2516_, 2, v___x_2515_);
v___x_2517_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_addTrace___at___00Aesop_ForwardRuleMatch_getProof_spec__4_spec__6(v_msg_2516_, v___y_2486_, v___y_2487_, v___x_2511_, v___y_2489_);
lean_dec_ref_known(v___x_2511_, 14);
v_a_2518_ = lean_ctor_get(v___x_2517_, 0);
v_isSharedCheck_2555_ = !lean_is_exclusive(v___x_2517_);
if (v_isSharedCheck_2555_ == 0)
{
v___x_2520_ = v___x_2517_;
v_isShared_2521_ = v_isSharedCheck_2555_;
goto v_resetjp_2519_;
}
else
{
lean_inc(v_a_2518_);
lean_dec(v___x_2517_);
v___x_2520_ = lean_box(0);
v_isShared_2521_ = v_isSharedCheck_2555_;
goto v_resetjp_2519_;
}
v_resetjp_2519_:
{
lean_object* v___x_2522_; lean_object* v_traceState_2523_; lean_object* v_env_2524_; lean_object* v_nextMacroScope_2525_; lean_object* v_ngen_2526_; lean_object* v_auxDeclNGen_2527_; lean_object* v_cache_2528_; lean_object* v_messages_2529_; lean_object* v_infoState_2530_; lean_object* v_snapshotTasks_2531_; lean_object* v___x_2533_; uint8_t v_isShared_2534_; uint8_t v_isSharedCheck_2554_; 
v___x_2522_ = lean_st_ref_take(v___y_2489_);
v_traceState_2523_ = lean_ctor_get(v___x_2522_, 4);
v_env_2524_ = lean_ctor_get(v___x_2522_, 0);
v_nextMacroScope_2525_ = lean_ctor_get(v___x_2522_, 1);
v_ngen_2526_ = lean_ctor_get(v___x_2522_, 2);
v_auxDeclNGen_2527_ = lean_ctor_get(v___x_2522_, 3);
v_cache_2528_ = lean_ctor_get(v___x_2522_, 5);
v_messages_2529_ = lean_ctor_get(v___x_2522_, 6);
v_infoState_2530_ = lean_ctor_get(v___x_2522_, 7);
v_snapshotTasks_2531_ = lean_ctor_get(v___x_2522_, 8);
v_isSharedCheck_2554_ = !lean_is_exclusive(v___x_2522_);
if (v_isSharedCheck_2554_ == 0)
{
v___x_2533_ = v___x_2522_;
v_isShared_2534_ = v_isSharedCheck_2554_;
goto v_resetjp_2532_;
}
else
{
lean_inc(v_snapshotTasks_2531_);
lean_inc(v_infoState_2530_);
lean_inc(v_messages_2529_);
lean_inc(v_cache_2528_);
lean_inc(v_traceState_2523_);
lean_inc(v_auxDeclNGen_2527_);
lean_inc(v_ngen_2526_);
lean_inc(v_nextMacroScope_2525_);
lean_inc(v_env_2524_);
lean_dec(v___x_2522_);
v___x_2533_ = lean_box(0);
v_isShared_2534_ = v_isSharedCheck_2554_;
goto v_resetjp_2532_;
}
v_resetjp_2532_:
{
uint64_t v_tid_2535_; lean_object* v___x_2537_; uint8_t v_isShared_2538_; uint8_t v_isSharedCheck_2552_; 
v_tid_2535_ = lean_ctor_get_uint64(v_traceState_2523_, sizeof(void*)*1);
v_isSharedCheck_2552_ = !lean_is_exclusive(v_traceState_2523_);
if (v_isSharedCheck_2552_ == 0)
{
lean_object* v_unused_2553_; 
v_unused_2553_ = lean_ctor_get(v_traceState_2523_, 0);
lean_dec(v_unused_2553_);
v___x_2537_ = v_traceState_2523_;
v_isShared_2538_ = v_isSharedCheck_2552_;
goto v_resetjp_2536_;
}
else
{
lean_dec(v_traceState_2523_);
v___x_2537_ = lean_box(0);
v_isShared_2538_ = v_isSharedCheck_2552_;
goto v_resetjp_2536_;
}
v_resetjp_2536_:
{
lean_object* v___x_2539_; lean_object* v___x_2540_; lean_object* v___x_2542_; 
v___x_2539_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2539_, 0, v_ref_2484_);
lean_ctor_set(v___x_2539_, 1, v_a_2518_);
v___x_2540_ = l_Lean_PersistentArray_push___redArg(v_oldTraces_2482_, v___x_2539_);
if (v_isShared_2538_ == 0)
{
lean_ctor_set(v___x_2537_, 0, v___x_2540_);
v___x_2542_ = v___x_2537_;
goto v_reusejp_2541_;
}
else
{
lean_object* v_reuseFailAlloc_2551_; 
v_reuseFailAlloc_2551_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_2551_, 0, v___x_2540_);
lean_ctor_set_uint64(v_reuseFailAlloc_2551_, sizeof(void*)*1, v_tid_2535_);
v___x_2542_ = v_reuseFailAlloc_2551_;
goto v_reusejp_2541_;
}
v_reusejp_2541_:
{
lean_object* v___x_2544_; 
if (v_isShared_2534_ == 0)
{
lean_ctor_set(v___x_2533_, 4, v___x_2542_);
v___x_2544_ = v___x_2533_;
goto v_reusejp_2543_;
}
else
{
lean_object* v_reuseFailAlloc_2550_; 
v_reuseFailAlloc_2550_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_2550_, 0, v_env_2524_);
lean_ctor_set(v_reuseFailAlloc_2550_, 1, v_nextMacroScope_2525_);
lean_ctor_set(v_reuseFailAlloc_2550_, 2, v_ngen_2526_);
lean_ctor_set(v_reuseFailAlloc_2550_, 3, v_auxDeclNGen_2527_);
lean_ctor_set(v_reuseFailAlloc_2550_, 4, v___x_2542_);
lean_ctor_set(v_reuseFailAlloc_2550_, 5, v_cache_2528_);
lean_ctor_set(v_reuseFailAlloc_2550_, 6, v_messages_2529_);
lean_ctor_set(v_reuseFailAlloc_2550_, 7, v_infoState_2530_);
lean_ctor_set(v_reuseFailAlloc_2550_, 8, v_snapshotTasks_2531_);
v___x_2544_ = v_reuseFailAlloc_2550_;
goto v_reusejp_2543_;
}
v_reusejp_2543_:
{
lean_object* v___x_2545_; lean_object* v___x_2546_; lean_object* v___x_2548_; 
v___x_2545_ = lean_st_ref_set(v___y_2489_, v___x_2544_);
v___x_2546_ = lean_box(0);
if (v_isShared_2521_ == 0)
{
lean_ctor_set(v___x_2520_, 0, v___x_2546_);
v___x_2548_ = v___x_2520_;
goto v_reusejp_2547_;
}
else
{
lean_object* v_reuseFailAlloc_2549_; 
v_reuseFailAlloc_2549_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2549_, 0, v___x_2546_);
v___x_2548_ = v_reuseFailAlloc_2549_;
goto v_reusejp_2547_;
}
v_reusejp_2547_:
{
return v___x_2548_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13_spec__16___boxed(lean_object* v_oldTraces_2556_, lean_object* v_data_2557_, lean_object* v_ref_2558_, lean_object* v_msg_2559_, lean_object* v___y_2560_, lean_object* v___y_2561_, lean_object* v___y_2562_, lean_object* v___y_2563_, lean_object* v___y_2564_){
_start:
{
lean_object* v_res_2565_; 
v_res_2565_ = lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13_spec__16(v_oldTraces_2556_, v_data_2557_, v_ref_2558_, v_msg_2559_, v___y_2560_, v___y_2561_, v___y_2562_, v___y_2563_);
lean_dec(v___y_2563_);
lean_dec_ref(v___y_2562_);
lean_dec(v___y_2561_);
lean_dec_ref(v___y_2560_);
return v_res_2565_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13_spec__18(lean_object* v_e_2566_){
_start:
{
if (lean_obj_tag(v_e_2566_) == 0)
{
uint8_t v___x_2567_; 
v___x_2567_ = 2;
return v___x_2567_;
}
else
{
lean_object* v_a_2568_; 
v_a_2568_ = lean_ctor_get(v_e_2566_, 0);
if (lean_obj_tag(v_a_2568_) == 0)
{
uint8_t v___x_2569_; 
v___x_2569_ = 1;
return v___x_2569_;
}
else
{
uint8_t v___x_2570_; 
v___x_2570_ = 0;
return v___x_2570_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13_spec__18___boxed(lean_object* v_e_2571_){
_start:
{
uint8_t v_res_2572_; lean_object* v_r_2573_; 
v_res_2572_ = lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13_spec__18(v_e_2571_);
lean_dec_ref(v_e_2571_);
v_r_2573_ = lean_box(v_res_2572_);
return v_r_2573_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13_spec__19(lean_object* v_opts_2574_, lean_object* v_opt_2575_){
_start:
{
lean_object* v_name_2576_; lean_object* v_defValue_2577_; lean_object* v_map_2578_; lean_object* v___x_2579_; 
v_name_2576_ = lean_ctor_get(v_opt_2575_, 0);
v_defValue_2577_ = lean_ctor_get(v_opt_2575_, 1);
v_map_2578_ = lean_ctor_get(v_opts_2574_, 0);
v___x_2579_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_2578_, v_name_2576_);
if (lean_obj_tag(v___x_2579_) == 0)
{
lean_inc(v_defValue_2577_);
return v_defValue_2577_;
}
else
{
lean_object* v_val_2580_; 
v_val_2580_ = lean_ctor_get(v___x_2579_, 0);
lean_inc(v_val_2580_);
lean_dec_ref_known(v___x_2579_, 1);
if (lean_obj_tag(v_val_2580_) == 3)
{
lean_object* v_v_2581_; 
v_v_2581_ = lean_ctor_get(v_val_2580_, 0);
lean_inc(v_v_2581_);
lean_dec_ref_known(v_val_2580_, 1);
return v_v_2581_;
}
else
{
lean_dec(v_val_2580_);
lean_inc(v_defValue_2577_);
return v_defValue_2577_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13_spec__19___boxed(lean_object* v_opts_2582_, lean_object* v_opt_2583_){
_start:
{
lean_object* v_res_2584_; 
v_res_2584_ = lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13_spec__19(v_opts_2582_, v_opt_2583_);
lean_dec_ref(v_opt_2583_);
lean_dec_ref(v_opts_2582_);
return v_res_2584_;
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13_spec__17___redArg(lean_object* v_x_2585_){
_start:
{
if (lean_obj_tag(v_x_2585_) == 0)
{
lean_object* v_a_2587_; lean_object* v___x_2589_; uint8_t v_isShared_2590_; uint8_t v_isSharedCheck_2594_; 
v_a_2587_ = lean_ctor_get(v_x_2585_, 0);
v_isSharedCheck_2594_ = !lean_is_exclusive(v_x_2585_);
if (v_isSharedCheck_2594_ == 0)
{
v___x_2589_ = v_x_2585_;
v_isShared_2590_ = v_isSharedCheck_2594_;
goto v_resetjp_2588_;
}
else
{
lean_inc(v_a_2587_);
lean_dec(v_x_2585_);
v___x_2589_ = lean_box(0);
v_isShared_2590_ = v_isSharedCheck_2594_;
goto v_resetjp_2588_;
}
v_resetjp_2588_:
{
lean_object* v___x_2592_; 
if (v_isShared_2590_ == 0)
{
lean_ctor_set_tag(v___x_2589_, 1);
v___x_2592_ = v___x_2589_;
goto v_reusejp_2591_;
}
else
{
lean_object* v_reuseFailAlloc_2593_; 
v_reuseFailAlloc_2593_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2593_, 0, v_a_2587_);
v___x_2592_ = v_reuseFailAlloc_2593_;
goto v_reusejp_2591_;
}
v_reusejp_2591_:
{
return v___x_2592_;
}
}
}
else
{
lean_object* v_a_2595_; lean_object* v___x_2597_; uint8_t v_isShared_2598_; uint8_t v_isSharedCheck_2602_; 
v_a_2595_ = lean_ctor_get(v_x_2585_, 0);
v_isSharedCheck_2602_ = !lean_is_exclusive(v_x_2585_);
if (v_isSharedCheck_2602_ == 0)
{
v___x_2597_ = v_x_2585_;
v_isShared_2598_ = v_isSharedCheck_2602_;
goto v_resetjp_2596_;
}
else
{
lean_inc(v_a_2595_);
lean_dec(v_x_2585_);
v___x_2597_ = lean_box(0);
v_isShared_2598_ = v_isSharedCheck_2602_;
goto v_resetjp_2596_;
}
v_resetjp_2596_:
{
lean_object* v___x_2600_; 
if (v_isShared_2598_ == 0)
{
lean_ctor_set_tag(v___x_2597_, 0);
v___x_2600_ = v___x_2597_;
goto v_reusejp_2599_;
}
else
{
lean_object* v_reuseFailAlloc_2601_; 
v_reuseFailAlloc_2601_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2601_, 0, v_a_2595_);
v___x_2600_ = v_reuseFailAlloc_2601_;
goto v_reusejp_2599_;
}
v_reusejp_2599_:
{
return v___x_2600_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13_spec__17___redArg___boxed(lean_object* v_x_2603_, lean_object* v___y_2604_){
_start:
{
lean_object* v_res_2605_; 
v_res_2605_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13_spec__17___redArg(v_x_2603_);
return v_res_2605_;
}
}
static lean_object* _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13___closed__1(void){
_start:
{
lean_object* v___x_2607_; lean_object* v___x_2608_; 
v___x_2607_ = ((lean_object*)(lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13___closed__0));
v___x_2608_ = l_Lean_stringToMessageData(v___x_2607_);
return v___x_2608_;
}
}
static double _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13___closed__2(void){
_start:
{
lean_object* v___x_2609_; double v___x_2610_; 
v___x_2609_ = lean_unsigned_to_nat(1000u);
v___x_2610_ = lean_float_of_nat(v___x_2609_);
return v___x_2610_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13(lean_object* v_cls_2611_, uint8_t v_collapsed_2612_, lean_object* v_tag_2613_, lean_object* v_opts_2614_, uint8_t v_clsEnabled_2615_, lean_object* v_oldTraces_2616_, lean_object* v_msg_2617_, lean_object* v_resStartStop_2618_, lean_object* v___y_2619_, lean_object* v___y_2620_, lean_object* v___y_2621_, lean_object* v___y_2622_){
_start:
{
lean_object* v_fst_2624_; lean_object* v_snd_2625_; lean_object* v___y_2627_; lean_object* v___y_2628_; lean_object* v_data_2629_; lean_object* v_fst_2640_; lean_object* v_snd_2641_; lean_object* v___x_2642_; uint8_t v___x_2643_; lean_object* v___y_2645_; lean_object* v_a_2646_; uint8_t v___y_2661_; double v___y_2692_; 
v_fst_2624_ = lean_ctor_get(v_resStartStop_2618_, 0);
lean_inc(v_fst_2624_);
v_snd_2625_ = lean_ctor_get(v_resStartStop_2618_, 1);
lean_inc(v_snd_2625_);
lean_dec_ref(v_resStartStop_2618_);
v_fst_2640_ = lean_ctor_get(v_snd_2625_, 0);
lean_inc(v_fst_2640_);
v_snd_2641_ = lean_ctor_get(v_snd_2625_, 1);
lean_inc(v_snd_2641_);
lean_dec(v_snd_2625_);
v___x_2642_ = l_Lean_trace_profiler;
v___x_2643_ = lp_aesop_Lean_Option_get___at___00Aesop_ForwardRuleMatch_getProof_spec__12(v_opts_2614_, v___x_2642_);
if (v___x_2643_ == 0)
{
v___y_2661_ = v___x_2643_;
goto v___jp_2660_;
}
else
{
lean_object* v___x_2697_; uint8_t v___x_2698_; 
v___x_2697_ = l_Lean_trace_profiler_useHeartbeats;
v___x_2698_ = lp_aesop_Lean_Option_get___at___00Aesop_ForwardRuleMatch_getProof_spec__12(v_opts_2614_, v___x_2697_);
if (v___x_2698_ == 0)
{
lean_object* v___x_2699_; lean_object* v___x_2700_; double v___x_2701_; double v___x_2702_; double v___x_2703_; 
v___x_2699_ = l_Lean_trace_profiler_threshold;
v___x_2700_ = lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13_spec__19(v_opts_2614_, v___x_2699_);
v___x_2701_ = lean_float_of_nat(v___x_2700_);
v___x_2702_ = lean_float_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13___closed__2, &lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13___closed__2_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13___closed__2);
v___x_2703_ = lean_float_div(v___x_2701_, v___x_2702_);
v___y_2692_ = v___x_2703_;
goto v___jp_2691_;
}
else
{
lean_object* v___x_2704_; lean_object* v___x_2705_; double v___x_2706_; 
v___x_2704_ = l_Lean_trace_profiler_threshold;
v___x_2705_ = lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13_spec__19(v_opts_2614_, v___x_2704_);
v___x_2706_ = lean_float_of_nat(v___x_2705_);
v___y_2692_ = v___x_2706_;
goto v___jp_2691_;
}
}
v___jp_2626_:
{
lean_object* v___x_2630_; 
lean_inc(v___y_2628_);
v___x_2630_ = lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13_spec__16(v_oldTraces_2616_, v_data_2629_, v___y_2628_, v___y_2627_, v___y_2619_, v___y_2620_, v___y_2621_, v___y_2622_);
if (lean_obj_tag(v___x_2630_) == 0)
{
lean_object* v___x_2631_; 
lean_dec_ref_known(v___x_2630_, 1);
v___x_2631_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13_spec__17___redArg(v_fst_2624_);
return v___x_2631_;
}
else
{
lean_object* v_a_2632_; lean_object* v___x_2634_; uint8_t v_isShared_2635_; uint8_t v_isSharedCheck_2639_; 
lean_dec(v_fst_2624_);
v_a_2632_ = lean_ctor_get(v___x_2630_, 0);
v_isSharedCheck_2639_ = !lean_is_exclusive(v___x_2630_);
if (v_isSharedCheck_2639_ == 0)
{
v___x_2634_ = v___x_2630_;
v_isShared_2635_ = v_isSharedCheck_2639_;
goto v_resetjp_2633_;
}
else
{
lean_inc(v_a_2632_);
lean_dec(v___x_2630_);
v___x_2634_ = lean_box(0);
v_isShared_2635_ = v_isSharedCheck_2639_;
goto v_resetjp_2633_;
}
v_resetjp_2633_:
{
lean_object* v___x_2637_; 
if (v_isShared_2635_ == 0)
{
v___x_2637_ = v___x_2634_;
goto v_reusejp_2636_;
}
else
{
lean_object* v_reuseFailAlloc_2638_; 
v_reuseFailAlloc_2638_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2638_, 0, v_a_2632_);
v___x_2637_ = v_reuseFailAlloc_2638_;
goto v_reusejp_2636_;
}
v_reusejp_2636_:
{
return v___x_2637_;
}
}
}
}
v___jp_2644_:
{
uint8_t v_result_2647_; lean_object* v___x_2648_; lean_object* v___x_2649_; double v___x_2650_; lean_object* v_data_2651_; 
v_result_2647_ = lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13_spec__18(v_fst_2624_);
v___x_2648_ = lean_box(v_result_2647_);
v___x_2649_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2649_, 0, v___x_2648_);
v___x_2650_ = lean_float_once(&lp_aesop_Lean_addTrace___at___00Aesop_ForwardRuleMatch_getProof_spec__4___closed__0, &lp_aesop_Lean_addTrace___at___00Aesop_ForwardRuleMatch_getProof_spec__4___closed__0_once, _init_lp_aesop_Lean_addTrace___at___00Aesop_ForwardRuleMatch_getProof_spec__4___closed__0);
lean_inc_ref(v_tag_2613_);
lean_inc_ref(v___x_2649_);
lean_inc(v_cls_2611_);
v_data_2651_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v_data_2651_, 0, v_cls_2611_);
lean_ctor_set(v_data_2651_, 1, v___x_2649_);
lean_ctor_set(v_data_2651_, 2, v_tag_2613_);
lean_ctor_set_float(v_data_2651_, sizeof(void*)*3, v___x_2650_);
lean_ctor_set_float(v_data_2651_, sizeof(void*)*3 + 8, v___x_2650_);
lean_ctor_set_uint8(v_data_2651_, sizeof(void*)*3 + 16, v_collapsed_2612_);
if (v___x_2643_ == 0)
{
lean_dec_ref_known(v___x_2649_, 1);
lean_dec(v_snd_2641_);
lean_dec(v_fst_2640_);
lean_dec_ref(v_tag_2613_);
lean_dec(v_cls_2611_);
v___y_2627_ = v_a_2646_;
v___y_2628_ = v___y_2645_;
v_data_2629_ = v_data_2651_;
goto v___jp_2626_;
}
else
{
lean_object* v_data_2652_; double v___x_2653_; double v___x_2654_; 
lean_dec_ref_known(v_data_2651_, 3);
v_data_2652_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v_data_2652_, 0, v_cls_2611_);
lean_ctor_set(v_data_2652_, 1, v___x_2649_);
lean_ctor_set(v_data_2652_, 2, v_tag_2613_);
v___x_2653_ = lean_unbox_float(v_fst_2640_);
lean_dec(v_fst_2640_);
lean_ctor_set_float(v_data_2652_, sizeof(void*)*3, v___x_2653_);
v___x_2654_ = lean_unbox_float(v_snd_2641_);
lean_dec(v_snd_2641_);
lean_ctor_set_float(v_data_2652_, sizeof(void*)*3 + 8, v___x_2654_);
lean_ctor_set_uint8(v_data_2652_, sizeof(void*)*3 + 16, v_collapsed_2612_);
v___y_2627_ = v_a_2646_;
v___y_2628_ = v___y_2645_;
v_data_2629_ = v_data_2652_;
goto v___jp_2626_;
}
}
v___jp_2655_:
{
lean_object* v_ref_2656_; lean_object* v___x_2657_; 
v_ref_2656_ = lean_ctor_get(v___y_2621_, 5);
lean_inc(v___y_2622_);
lean_inc_ref(v___y_2621_);
lean_inc(v___y_2620_);
lean_inc_ref(v___y_2619_);
lean_inc(v_fst_2624_);
v___x_2657_ = lean_apply_6(v_msg_2617_, v_fst_2624_, v___y_2619_, v___y_2620_, v___y_2621_, v___y_2622_, lean_box(0));
if (lean_obj_tag(v___x_2657_) == 0)
{
lean_object* v_a_2658_; 
v_a_2658_ = lean_ctor_get(v___x_2657_, 0);
lean_inc(v_a_2658_);
lean_dec_ref_known(v___x_2657_, 1);
v___y_2645_ = v_ref_2656_;
v_a_2646_ = v_a_2658_;
goto v___jp_2644_;
}
else
{
lean_object* v___x_2659_; 
lean_dec_ref_known(v___x_2657_, 1);
v___x_2659_ = lean_obj_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13___closed__1, &lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13___closed__1_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13___closed__1);
v___y_2645_ = v_ref_2656_;
v_a_2646_ = v___x_2659_;
goto v___jp_2644_;
}
}
v___jp_2660_:
{
if (v_clsEnabled_2615_ == 0)
{
if (v___y_2661_ == 0)
{
lean_object* v___x_2662_; lean_object* v_traceState_2663_; lean_object* v_env_2664_; lean_object* v_nextMacroScope_2665_; lean_object* v_ngen_2666_; lean_object* v_auxDeclNGen_2667_; lean_object* v_cache_2668_; lean_object* v_messages_2669_; lean_object* v_infoState_2670_; lean_object* v_snapshotTasks_2671_; lean_object* v___x_2673_; uint8_t v_isShared_2674_; uint8_t v_isSharedCheck_2690_; 
lean_dec(v_snd_2641_);
lean_dec(v_fst_2640_);
lean_dec_ref(v_msg_2617_);
lean_dec_ref(v_tag_2613_);
lean_dec(v_cls_2611_);
v___x_2662_ = lean_st_ref_take(v___y_2622_);
v_traceState_2663_ = lean_ctor_get(v___x_2662_, 4);
v_env_2664_ = lean_ctor_get(v___x_2662_, 0);
v_nextMacroScope_2665_ = lean_ctor_get(v___x_2662_, 1);
v_ngen_2666_ = lean_ctor_get(v___x_2662_, 2);
v_auxDeclNGen_2667_ = lean_ctor_get(v___x_2662_, 3);
v_cache_2668_ = lean_ctor_get(v___x_2662_, 5);
v_messages_2669_ = lean_ctor_get(v___x_2662_, 6);
v_infoState_2670_ = lean_ctor_get(v___x_2662_, 7);
v_snapshotTasks_2671_ = lean_ctor_get(v___x_2662_, 8);
v_isSharedCheck_2690_ = !lean_is_exclusive(v___x_2662_);
if (v_isSharedCheck_2690_ == 0)
{
v___x_2673_ = v___x_2662_;
v_isShared_2674_ = v_isSharedCheck_2690_;
goto v_resetjp_2672_;
}
else
{
lean_inc(v_snapshotTasks_2671_);
lean_inc(v_infoState_2670_);
lean_inc(v_messages_2669_);
lean_inc(v_cache_2668_);
lean_inc(v_traceState_2663_);
lean_inc(v_auxDeclNGen_2667_);
lean_inc(v_ngen_2666_);
lean_inc(v_nextMacroScope_2665_);
lean_inc(v_env_2664_);
lean_dec(v___x_2662_);
v___x_2673_ = lean_box(0);
v_isShared_2674_ = v_isSharedCheck_2690_;
goto v_resetjp_2672_;
}
v_resetjp_2672_:
{
uint64_t v_tid_2675_; lean_object* v_traces_2676_; lean_object* v___x_2678_; uint8_t v_isShared_2679_; uint8_t v_isSharedCheck_2689_; 
v_tid_2675_ = lean_ctor_get_uint64(v_traceState_2663_, sizeof(void*)*1);
v_traces_2676_ = lean_ctor_get(v_traceState_2663_, 0);
v_isSharedCheck_2689_ = !lean_is_exclusive(v_traceState_2663_);
if (v_isSharedCheck_2689_ == 0)
{
v___x_2678_ = v_traceState_2663_;
v_isShared_2679_ = v_isSharedCheck_2689_;
goto v_resetjp_2677_;
}
else
{
lean_inc(v_traces_2676_);
lean_dec(v_traceState_2663_);
v___x_2678_ = lean_box(0);
v_isShared_2679_ = v_isSharedCheck_2689_;
goto v_resetjp_2677_;
}
v_resetjp_2677_:
{
lean_object* v___x_2680_; lean_object* v___x_2682_; 
v___x_2680_ = l_Lean_PersistentArray_append___redArg(v_oldTraces_2616_, v_traces_2676_);
lean_dec_ref(v_traces_2676_);
if (v_isShared_2679_ == 0)
{
lean_ctor_set(v___x_2678_, 0, v___x_2680_);
v___x_2682_ = v___x_2678_;
goto v_reusejp_2681_;
}
else
{
lean_object* v_reuseFailAlloc_2688_; 
v_reuseFailAlloc_2688_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_2688_, 0, v___x_2680_);
lean_ctor_set_uint64(v_reuseFailAlloc_2688_, sizeof(void*)*1, v_tid_2675_);
v___x_2682_ = v_reuseFailAlloc_2688_;
goto v_reusejp_2681_;
}
v_reusejp_2681_:
{
lean_object* v___x_2684_; 
if (v_isShared_2674_ == 0)
{
lean_ctor_set(v___x_2673_, 4, v___x_2682_);
v___x_2684_ = v___x_2673_;
goto v_reusejp_2683_;
}
else
{
lean_object* v_reuseFailAlloc_2687_; 
v_reuseFailAlloc_2687_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_2687_, 0, v_env_2664_);
lean_ctor_set(v_reuseFailAlloc_2687_, 1, v_nextMacroScope_2665_);
lean_ctor_set(v_reuseFailAlloc_2687_, 2, v_ngen_2666_);
lean_ctor_set(v_reuseFailAlloc_2687_, 3, v_auxDeclNGen_2667_);
lean_ctor_set(v_reuseFailAlloc_2687_, 4, v___x_2682_);
lean_ctor_set(v_reuseFailAlloc_2687_, 5, v_cache_2668_);
lean_ctor_set(v_reuseFailAlloc_2687_, 6, v_messages_2669_);
lean_ctor_set(v_reuseFailAlloc_2687_, 7, v_infoState_2670_);
lean_ctor_set(v_reuseFailAlloc_2687_, 8, v_snapshotTasks_2671_);
v___x_2684_ = v_reuseFailAlloc_2687_;
goto v_reusejp_2683_;
}
v_reusejp_2683_:
{
lean_object* v___x_2685_; lean_object* v___x_2686_; 
v___x_2685_ = lean_st_ref_set(v___y_2622_, v___x_2684_);
v___x_2686_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13_spec__17___redArg(v_fst_2624_);
return v___x_2686_;
}
}
}
}
}
else
{
goto v___jp_2655_;
}
}
else
{
goto v___jp_2655_;
}
}
v___jp_2691_:
{
double v___x_2693_; double v___x_2694_; double v___x_2695_; uint8_t v___x_2696_; 
v___x_2693_ = lean_unbox_float(v_snd_2641_);
v___x_2694_ = lean_unbox_float(v_fst_2640_);
v___x_2695_ = lean_float_sub(v___x_2693_, v___x_2694_);
v___x_2696_ = lean_float_decLt(v___y_2692_, v___x_2695_);
v___y_2661_ = v___x_2696_;
goto v___jp_2660_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13___boxed(lean_object* v_cls_2707_, lean_object* v_collapsed_2708_, lean_object* v_tag_2709_, lean_object* v_opts_2710_, lean_object* v_clsEnabled_2711_, lean_object* v_oldTraces_2712_, lean_object* v_msg_2713_, lean_object* v_resStartStop_2714_, lean_object* v___y_2715_, lean_object* v___y_2716_, lean_object* v___y_2717_, lean_object* v___y_2718_, lean_object* v___y_2719_){
_start:
{
uint8_t v_collapsed_boxed_2720_; uint8_t v_clsEnabled_boxed_2721_; lean_object* v_res_2722_; 
v_collapsed_boxed_2720_ = lean_unbox(v_collapsed_2708_);
v_clsEnabled_boxed_2721_ = lean_unbox(v_clsEnabled_2711_);
v_res_2722_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13(v_cls_2707_, v_collapsed_boxed_2720_, v_tag_2709_, v_opts_2710_, v_clsEnabled_boxed_2721_, v_oldTraces_2712_, v_msg_2713_, v_resStartStop_2714_, v___y_2715_, v___y_2716_, v___y_2717_, v___y_2718_);
lean_dec(v___y_2718_);
lean_dec_ref(v___y_2717_);
lean_dec(v___y_2716_);
lean_dec_ref(v___y_2715_);
lean_dec_ref(v_opts_2710_);
return v_res_2722_;
}
}
static lean_object* _init_lp_aesop_Aesop_ForwardRuleMatch_getProof___closed__4(void){
_start:
{
lean_object* v___x_2727_; lean_object* v___x_2728_; 
v___x_2727_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_getProof___closed__3));
v___x_2728_ = l_Lean_stringToMessageData(v___x_2727_);
return v___x_2728_;
}
}
static lean_object* _init_lp_aesop_Aesop_ForwardRuleMatch_getProof___closed__5(void){
_start:
{
lean_object* v___x_2729_; lean_object* v___f_2730_; 
v___x_2729_ = lean_obj_once(&lp_aesop_Aesop_ForwardRuleMatch_getProof___closed__4, &lp_aesop_Aesop_ForwardRuleMatch_getProof___closed__4_once, _init_lp_aesop_Aesop_ForwardRuleMatch_getProof___closed__4);
v___f_2730_ = lean_alloc_closure((void*)(lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__2___boxed), 7, 1);
lean_closure_set(v___f_2730_, 0, v___x_2729_);
return v___f_2730_;
}
}
static double _init_lp_aesop_Aesop_ForwardRuleMatch_getProof___closed__8(void){
_start:
{
lean_object* v___x_2734_; double v___x_2735_; 
v___x_2734_ = lean_unsigned_to_nat(1000000000u);
v___x_2735_ = lean_float_of_nat(v___x_2734_);
return v___x_2735_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_getProof(lean_object* v_goal_2736_, lean_object* v_m_2737_, lean_object* v_a_2738_, lean_object* v_a_2739_, lean_object* v_a_2740_, lean_object* v_a_2741_){
_start:
{
lean_object* v___y_2744_; lean_object* v___y_2745_; lean_object* v___y_2746_; lean_object* v___y_2747_; lean_object* v___y_2748_; lean_object* v_name_2749_; lean_object* v___y_2750_; lean_object* v___y_2765_; lean_object* v___y_2766_; lean_object* v___y_2767_; lean_object* v___y_2768_; lean_object* v_name_2769_; uint8_t v_scope_2770_; lean_object* v___y_2771_; lean_object* v___y_2772_; lean_object* v___y_2778_; lean_object* v___y_2779_; lean_object* v___y_2780_; lean_object* v_name_2781_; uint8_t v_builder_2782_; uint8_t v_scope_2783_; lean_object* v___y_2784_; lean_object* v___y_2796_; lean_object* v___y_2797_; uint8_t v___y_2798_; lean_object* v___y_2812_; lean_object* v_a_2813_; lean_object* v___y_2817_; lean_object* v_options_2819_; lean_object* v_inheritedTraceOptions_2820_; uint8_t v_hasTrace_2821_; lean_object* v___f_2822_; lean_object* v___x_2823_; lean_object* v___f_2824_; uint8_t v___x_2825_; lean_object* v___x_2826_; lean_object* v___x_2827_; 
v_options_2819_ = lean_ctor_get(v_a_2740_, 2);
v_inheritedTraceOptions_2820_ = lean_ctor_get(v_a_2740_, 13);
v_hasTrace_2821_ = lean_ctor_get_uint8(v_options_2819_, sizeof(void*)*1);
v___f_2822_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_getProof___closed__2));
v___x_2823_ = lp_aesop_Aesop_TraceOption_forward;
lean_inc_ref(v_m_2737_);
lean_inc(v_goal_2736_);
v___f_2824_ = lean_alloc_closure((void*)(lp_aesop_Aesop_ForwardRuleMatch_getProof___lam__1___boxed), 9, 4);
lean_closure_set(v___f_2824_, 0, v___x_2823_);
lean_closure_set(v___f_2824_, 1, v___f_2822_);
lean_closure_set(v___f_2824_, 2, v_goal_2736_);
lean_closure_set(v___f_2824_, 3, v_m_2737_);
v___x_2825_ = 0;
v___x_2826_ = lean_box(v___x_2825_);
v___x_2827_ = lean_alloc_closure((void*)(lp_aesop_Lean_Meta_withNewMCtxDepth___at___00Aesop_ForwardRuleMatch_getProof_spec__9___boxed), 8, 3);
lean_closure_set(v___x_2827_, 0, lean_box(0));
lean_closure_set(v___x_2827_, 1, v___f_2824_);
lean_closure_set(v___x_2827_, 2, v___x_2826_);
if (v_hasTrace_2821_ == 0)
{
lean_object* v___x_2828_; 
v___x_2828_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_ForwardRuleMatch_getProof_spec__10___redArg(v_goal_2736_, v___x_2827_, v_a_2738_, v_a_2739_, v_a_2740_, v_a_2741_);
v___y_2817_ = v___x_2828_;
goto v___jp_2816_;
}
else
{
lean_object* v_traceClass_2829_; lean_object* v___f_2830_; lean_object* v___x_2831_; lean_object* v___x_2832_; lean_object* v___x_2833_; uint8_t v___x_2834_; lean_object* v___y_2836_; lean_object* v___y_2837_; lean_object* v_a_2838_; lean_object* v___y_2851_; lean_object* v___y_2852_; lean_object* v_a_2853_; 
v_traceClass_2829_ = lean_ctor_get(v___x_2823_, 0);
v___f_2830_ = lean_obj_once(&lp_aesop_Aesop_ForwardRuleMatch_getProof___closed__5, &lp_aesop_Aesop_ForwardRuleMatch_getProof___closed__5_once, _init_lp_aesop_Aesop_ForwardRuleMatch_getProof___closed__5);
v___x_2831_ = ((lean_object*)(lp_aesop_Lean_addTrace___at___00Aesop_ForwardRuleMatch_getProof_spec__4___closed__1));
v___x_2832_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_getProof___closed__7));
lean_inc(v_traceClass_2829_);
v___x_2833_ = l_Lean_Name_append(v___x_2832_, v_traceClass_2829_);
v___x_2834_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_2820_, v_options_2819_, v___x_2833_);
lean_dec(v___x_2833_);
if (v___x_2834_ == 0)
{
lean_object* v___x_2903_; uint8_t v___x_2904_; 
v___x_2903_ = l_Lean_trace_profiler;
v___x_2904_ = lp_aesop_Lean_Option_get___at___00Aesop_ForwardRuleMatch_getProof_spec__12(v_options_2819_, v___x_2903_);
if (v___x_2904_ == 0)
{
lean_object* v___x_2905_; 
v___x_2905_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_ForwardRuleMatch_getProof_spec__10___redArg(v_goal_2736_, v___x_2827_, v_a_2738_, v_a_2739_, v_a_2740_, v_a_2741_);
v___y_2817_ = v___x_2905_;
goto v___jp_2816_;
}
else
{
goto v___jp_2862_;
}
}
else
{
goto v___jp_2862_;
}
v___jp_2835_:
{
lean_object* v___x_2839_; double v___x_2840_; double v___x_2841_; double v___x_2842_; double v___x_2843_; double v___x_2844_; lean_object* v___x_2845_; lean_object* v___x_2846_; lean_object* v___x_2847_; lean_object* v___x_2848_; lean_object* v___x_2849_; 
v___x_2839_ = lean_io_mono_nanos_now();
v___x_2840_ = lean_float_of_nat(v___y_2837_);
v___x_2841_ = lean_float_once(&lp_aesop_Aesop_ForwardRuleMatch_getProof___closed__8, &lp_aesop_Aesop_ForwardRuleMatch_getProof___closed__8_once, _init_lp_aesop_Aesop_ForwardRuleMatch_getProof___closed__8);
v___x_2842_ = lean_float_div(v___x_2840_, v___x_2841_);
v___x_2843_ = lean_float_of_nat(v___x_2839_);
v___x_2844_ = lean_float_div(v___x_2843_, v___x_2841_);
v___x_2845_ = lean_box_float(v___x_2842_);
v___x_2846_ = lean_box_float(v___x_2844_);
v___x_2847_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2847_, 0, v___x_2845_);
lean_ctor_set(v___x_2847_, 1, v___x_2846_);
v___x_2848_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2848_, 0, v_a_2838_);
lean_ctor_set(v___x_2848_, 1, v___x_2847_);
lean_inc(v_traceClass_2829_);
v___x_2849_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13(v_traceClass_2829_, v_hasTrace_2821_, v___x_2831_, v_options_2819_, v___x_2834_, v___y_2836_, v___f_2830_, v___x_2848_, v_a_2738_, v_a_2739_, v_a_2740_, v_a_2741_);
v___y_2817_ = v___x_2849_;
goto v___jp_2816_;
}
v___jp_2850_:
{
lean_object* v___x_2854_; double v___x_2855_; double v___x_2856_; lean_object* v___x_2857_; lean_object* v___x_2858_; lean_object* v___x_2859_; lean_object* v___x_2860_; lean_object* v___x_2861_; 
v___x_2854_ = lean_io_get_num_heartbeats();
v___x_2855_ = lean_float_of_nat(v___y_2851_);
v___x_2856_ = lean_float_of_nat(v___x_2854_);
v___x_2857_ = lean_box_float(v___x_2855_);
v___x_2858_ = lean_box_float(v___x_2856_);
v___x_2859_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2859_, 0, v___x_2857_);
lean_ctor_set(v___x_2859_, 1, v___x_2858_);
v___x_2860_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2860_, 0, v_a_2853_);
lean_ctor_set(v___x_2860_, 1, v___x_2859_);
lean_inc(v_traceClass_2829_);
v___x_2861_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13(v_traceClass_2829_, v_hasTrace_2821_, v___x_2831_, v_options_2819_, v___x_2834_, v___y_2852_, v___f_2830_, v___x_2860_, v_a_2738_, v_a_2739_, v_a_2740_, v_a_2741_);
v___y_2817_ = v___x_2861_;
goto v___jp_2816_;
}
v___jp_2862_:
{
lean_object* v___x_2863_; lean_object* v_a_2864_; lean_object* v___x_2865_; uint8_t v___x_2866_; 
v___x_2863_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_ForwardRuleMatch_getProof_spec__11___redArg(v_a_2741_);
v_a_2864_ = lean_ctor_get(v___x_2863_, 0);
lean_inc(v_a_2864_);
lean_dec_ref(v___x_2863_);
v___x_2865_ = l_Lean_trace_profiler_useHeartbeats;
v___x_2866_ = lp_aesop_Lean_Option_get___at___00Aesop_ForwardRuleMatch_getProof_spec__12(v_options_2819_, v___x_2865_);
if (v___x_2866_ == 0)
{
lean_object* v___x_2867_; lean_object* v___x_2868_; 
v___x_2867_ = lean_io_mono_nanos_now();
v___x_2868_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_ForwardRuleMatch_getProof_spec__10___redArg(v_goal_2736_, v___x_2827_, v_a_2738_, v_a_2739_, v_a_2740_, v_a_2741_);
if (lean_obj_tag(v___x_2868_) == 0)
{
lean_object* v_a_2869_; lean_object* v___x_2871_; uint8_t v_isShared_2872_; uint8_t v_isSharedCheck_2876_; 
v_a_2869_ = lean_ctor_get(v___x_2868_, 0);
v_isSharedCheck_2876_ = !lean_is_exclusive(v___x_2868_);
if (v_isSharedCheck_2876_ == 0)
{
v___x_2871_ = v___x_2868_;
v_isShared_2872_ = v_isSharedCheck_2876_;
goto v_resetjp_2870_;
}
else
{
lean_inc(v_a_2869_);
lean_dec(v___x_2868_);
v___x_2871_ = lean_box(0);
v_isShared_2872_ = v_isSharedCheck_2876_;
goto v_resetjp_2870_;
}
v_resetjp_2870_:
{
lean_object* v___x_2874_; 
if (v_isShared_2872_ == 0)
{
lean_ctor_set_tag(v___x_2871_, 1);
v___x_2874_ = v___x_2871_;
goto v_reusejp_2873_;
}
else
{
lean_object* v_reuseFailAlloc_2875_; 
v_reuseFailAlloc_2875_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2875_, 0, v_a_2869_);
v___x_2874_ = v_reuseFailAlloc_2875_;
goto v_reusejp_2873_;
}
v_reusejp_2873_:
{
v___y_2836_ = v_a_2864_;
v___y_2837_ = v___x_2867_;
v_a_2838_ = v___x_2874_;
goto v___jp_2835_;
}
}
}
else
{
lean_object* v_a_2877_; lean_object* v___x_2879_; uint8_t v_isShared_2880_; uint8_t v_isSharedCheck_2884_; 
v_a_2877_ = lean_ctor_get(v___x_2868_, 0);
v_isSharedCheck_2884_ = !lean_is_exclusive(v___x_2868_);
if (v_isSharedCheck_2884_ == 0)
{
v___x_2879_ = v___x_2868_;
v_isShared_2880_ = v_isSharedCheck_2884_;
goto v_resetjp_2878_;
}
else
{
lean_inc(v_a_2877_);
lean_dec(v___x_2868_);
v___x_2879_ = lean_box(0);
v_isShared_2880_ = v_isSharedCheck_2884_;
goto v_resetjp_2878_;
}
v_resetjp_2878_:
{
lean_object* v___x_2882_; 
if (v_isShared_2880_ == 0)
{
lean_ctor_set_tag(v___x_2879_, 0);
v___x_2882_ = v___x_2879_;
goto v_reusejp_2881_;
}
else
{
lean_object* v_reuseFailAlloc_2883_; 
v_reuseFailAlloc_2883_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2883_, 0, v_a_2877_);
v___x_2882_ = v_reuseFailAlloc_2883_;
goto v_reusejp_2881_;
}
v_reusejp_2881_:
{
v___y_2836_ = v_a_2864_;
v___y_2837_ = v___x_2867_;
v_a_2838_ = v___x_2882_;
goto v___jp_2835_;
}
}
}
}
else
{
lean_object* v___x_2885_; lean_object* v___x_2886_; 
v___x_2885_ = lean_io_get_num_heartbeats();
v___x_2886_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_ForwardRuleMatch_getProof_spec__10___redArg(v_goal_2736_, v___x_2827_, v_a_2738_, v_a_2739_, v_a_2740_, v_a_2741_);
if (lean_obj_tag(v___x_2886_) == 0)
{
lean_object* v_a_2887_; lean_object* v___x_2889_; uint8_t v_isShared_2890_; uint8_t v_isSharedCheck_2894_; 
v_a_2887_ = lean_ctor_get(v___x_2886_, 0);
v_isSharedCheck_2894_ = !lean_is_exclusive(v___x_2886_);
if (v_isSharedCheck_2894_ == 0)
{
v___x_2889_ = v___x_2886_;
v_isShared_2890_ = v_isSharedCheck_2894_;
goto v_resetjp_2888_;
}
else
{
lean_inc(v_a_2887_);
lean_dec(v___x_2886_);
v___x_2889_ = lean_box(0);
v_isShared_2890_ = v_isSharedCheck_2894_;
goto v_resetjp_2888_;
}
v_resetjp_2888_:
{
lean_object* v___x_2892_; 
if (v_isShared_2890_ == 0)
{
lean_ctor_set_tag(v___x_2889_, 1);
v___x_2892_ = v___x_2889_;
goto v_reusejp_2891_;
}
else
{
lean_object* v_reuseFailAlloc_2893_; 
v_reuseFailAlloc_2893_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2893_, 0, v_a_2887_);
v___x_2892_ = v_reuseFailAlloc_2893_;
goto v_reusejp_2891_;
}
v_reusejp_2891_:
{
v___y_2851_ = v___x_2885_;
v___y_2852_ = v_a_2864_;
v_a_2853_ = v___x_2892_;
goto v___jp_2850_;
}
}
}
else
{
lean_object* v_a_2895_; lean_object* v___x_2897_; uint8_t v_isShared_2898_; uint8_t v_isSharedCheck_2902_; 
v_a_2895_ = lean_ctor_get(v___x_2886_, 0);
v_isSharedCheck_2902_ = !lean_is_exclusive(v___x_2886_);
if (v_isSharedCheck_2902_ == 0)
{
v___x_2897_ = v___x_2886_;
v_isShared_2898_ = v_isSharedCheck_2902_;
goto v_resetjp_2896_;
}
else
{
lean_inc(v_a_2895_);
lean_dec(v___x_2886_);
v___x_2897_ = lean_box(0);
v_isShared_2898_ = v_isSharedCheck_2902_;
goto v_resetjp_2896_;
}
v_resetjp_2896_:
{
lean_object* v___x_2900_; 
if (v_isShared_2898_ == 0)
{
lean_ctor_set_tag(v___x_2897_, 0);
v___x_2900_ = v___x_2897_;
goto v_reusejp_2899_;
}
else
{
lean_object* v_reuseFailAlloc_2901_; 
v_reuseFailAlloc_2901_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2901_, 0, v_a_2895_);
v___x_2900_ = v_reuseFailAlloc_2901_;
goto v_reusejp_2899_;
}
v_reusejp_2899_:
{
v___y_2851_ = v___x_2885_;
v___y_2852_ = v_a_2864_;
v_a_2853_ = v___x_2900_;
goto v___jp_2850_;
}
}
}
}
}
}
v___jp_2743_:
{
lean_object* v___x_2751_; lean_object* v___x_2752_; uint8_t v___x_2753_; lean_object* v___x_2754_; lean_object* v___x_2755_; lean_object* v___x_2756_; lean_object* v___x_2757_; lean_object* v___x_2758_; lean_object* v___x_2759_; lean_object* v___x_2760_; lean_object* v___x_2761_; lean_object* v___x_2762_; lean_object* v___x_2763_; 
v___x_2751_ = lean_string_append(v___y_2746_, v___y_2750_);
v___x_2752_ = lean_string_append(v___x_2751_, v___y_2745_);
v___x_2753_ = 1;
v___x_2754_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_name_2749_, v___x_2753_);
v___x_2755_ = lean_string_append(v___x_2752_, v___x_2754_);
lean_dec_ref(v___x_2754_);
lean_inc_ref(v___y_2744_);
v___x_2756_ = lean_string_append(v___y_2744_, v___x_2755_);
lean_dec_ref(v___x_2755_);
v___x_2757_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_getProof___closed__0));
v___x_2758_ = lean_string_append(v___x_2756_, v___x_2757_);
v___x_2759_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_2759_, 0, v___x_2758_);
v___x_2760_ = l_Lean_MessageData_ofFormat(v___x_2759_);
v___x_2761_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2761_, 0, v___x_2760_);
lean_ctor_set(v___x_2761_, 1, v___y_2748_);
v___x_2762_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2762_, 0, v___y_2747_);
lean_ctor_set(v___x_2762_, 1, v___x_2761_);
v___x_2763_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2763_, 0, v___x_2762_);
return v___x_2763_;
}
v___jp_2764_:
{
lean_object* v___x_2773_; lean_object* v___x_2774_; 
v___x_2773_ = lean_string_append(v___y_2765_, v___y_2772_);
v___x_2774_ = lean_string_append(v___x_2773_, v___y_2767_);
if (v_scope_2770_ == 0)
{
lean_object* v___x_2775_; 
v___x_2775_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__2));
v___y_2744_ = v___y_2766_;
v___y_2745_ = v___y_2767_;
v___y_2746_ = v___x_2774_;
v___y_2747_ = v___y_2768_;
v___y_2748_ = v___y_2771_;
v_name_2749_ = v_name_2769_;
v___y_2750_ = v___x_2775_;
goto v___jp_2743_;
}
else
{
lean_object* v___x_2776_; 
v___x_2776_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__3));
v___y_2744_ = v___y_2766_;
v___y_2745_ = v___y_2767_;
v___y_2746_ = v___x_2774_;
v___y_2747_ = v___y_2768_;
v___y_2748_ = v___y_2771_;
v_name_2749_ = v_name_2769_;
v___y_2750_ = v___x_2776_;
goto v___jp_2743_;
}
}
v___jp_2777_:
{
lean_object* v___x_2785_; lean_object* v___x_2786_; 
v___x_2785_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__4));
lean_inc_ref(v___y_2784_);
v___x_2786_ = lean_string_append(v___y_2784_, v___x_2785_);
switch(v_builder_2782_)
{
case 0:
{
lean_object* v___x_2787_; 
v___x_2787_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__5));
v___y_2765_ = v___x_2786_;
v___y_2766_ = v___y_2778_;
v___y_2767_ = v___x_2785_;
v___y_2768_ = v___y_2779_;
v_name_2769_ = v_name_2781_;
v_scope_2770_ = v_scope_2783_;
v___y_2771_ = v___y_2780_;
v___y_2772_ = v___x_2787_;
goto v___jp_2764_;
}
case 1:
{
lean_object* v___x_2788_; 
v___x_2788_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__6));
v___y_2765_ = v___x_2786_;
v___y_2766_ = v___y_2778_;
v___y_2767_ = v___x_2785_;
v___y_2768_ = v___y_2779_;
v_name_2769_ = v_name_2781_;
v_scope_2770_ = v_scope_2783_;
v___y_2771_ = v___y_2780_;
v___y_2772_ = v___x_2788_;
goto v___jp_2764_;
}
case 2:
{
lean_object* v___x_2789_; 
v___x_2789_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__7));
v___y_2765_ = v___x_2786_;
v___y_2766_ = v___y_2778_;
v___y_2767_ = v___x_2785_;
v___y_2768_ = v___y_2779_;
v_name_2769_ = v_name_2781_;
v_scope_2770_ = v_scope_2783_;
v___y_2771_ = v___y_2780_;
v___y_2772_ = v___x_2789_;
goto v___jp_2764_;
}
case 3:
{
lean_object* v___x_2790_; 
v___x_2790_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__8));
v___y_2765_ = v___x_2786_;
v___y_2766_ = v___y_2778_;
v___y_2767_ = v___x_2785_;
v___y_2768_ = v___y_2779_;
v_name_2769_ = v_name_2781_;
v_scope_2770_ = v_scope_2783_;
v___y_2771_ = v___y_2780_;
v___y_2772_ = v___x_2790_;
goto v___jp_2764_;
}
case 4:
{
lean_object* v___x_2791_; 
v___x_2791_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__9));
v___y_2765_ = v___x_2786_;
v___y_2766_ = v___y_2778_;
v___y_2767_ = v___x_2785_;
v___y_2768_ = v___y_2779_;
v_name_2769_ = v_name_2781_;
v_scope_2770_ = v_scope_2783_;
v___y_2771_ = v___y_2780_;
v___y_2772_ = v___x_2791_;
goto v___jp_2764_;
}
case 5:
{
lean_object* v___x_2792_; 
v___x_2792_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__10));
v___y_2765_ = v___x_2786_;
v___y_2766_ = v___y_2778_;
v___y_2767_ = v___x_2785_;
v___y_2768_ = v___y_2779_;
v_name_2769_ = v_name_2781_;
v_scope_2770_ = v_scope_2783_;
v___y_2771_ = v___y_2780_;
v___y_2772_ = v___x_2792_;
goto v___jp_2764_;
}
case 6:
{
lean_object* v___x_2793_; 
v___x_2793_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__11));
v___y_2765_ = v___x_2786_;
v___y_2766_ = v___y_2778_;
v___y_2767_ = v___x_2785_;
v___y_2768_ = v___y_2779_;
v_name_2769_ = v_name_2781_;
v_scope_2770_ = v_scope_2783_;
v___y_2771_ = v___y_2780_;
v___y_2772_ = v___x_2793_;
goto v___jp_2764_;
}
default: 
{
lean_object* v___x_2794_; 
v___x_2794_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__12));
v___y_2765_ = v___x_2786_;
v___y_2766_ = v___y_2778_;
v___y_2767_ = v___x_2785_;
v___y_2768_ = v___y_2779_;
v_name_2769_ = v_name_2781_;
v_scope_2770_ = v_scope_2783_;
v___y_2771_ = v___y_2780_;
v___y_2772_ = v___x_2794_;
goto v___jp_2764_;
}
}
}
v___jp_2795_:
{
if (v___y_2798_ == 0)
{
if (lean_obj_tag(v___y_2796_) == 0)
{
lean_object* v_rule_2799_; lean_object* v_name_2800_; lean_object* v_ref_2801_; lean_object* v_msg_2802_; lean_object* v_name_2803_; uint8_t v_builder_2804_; uint8_t v_phase_2805_; uint8_t v_scope_2806_; lean_object* v___x_2807_; 
lean_dec_ref(v___y_2797_);
v_rule_2799_ = lean_ctor_get(v_m_2737_, 0);
lean_inc_ref(v_rule_2799_);
lean_dec_ref(v_m_2737_);
v_name_2800_ = lean_ctor_get(v_rule_2799_, 1);
lean_inc_ref(v_name_2800_);
lean_dec_ref(v_rule_2799_);
v_ref_2801_ = lean_ctor_get(v___y_2796_, 0);
lean_inc(v_ref_2801_);
v_msg_2802_ = lean_ctor_get(v___y_2796_, 1);
lean_inc_ref(v_msg_2802_);
lean_dec_ref_known(v___y_2796_, 2);
v_name_2803_ = lean_ctor_get(v_name_2800_, 0);
lean_inc(v_name_2803_);
v_builder_2804_ = lean_ctor_get_uint8(v_name_2800_, sizeof(void*)*1 + 8);
v_phase_2805_ = lean_ctor_get_uint8(v_name_2800_, sizeof(void*)*1 + 9);
v_scope_2806_ = lean_ctor_get_uint8(v_name_2800_, sizeof(void*)*1 + 10);
lean_dec_ref(v_name_2800_);
v___x_2807_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_getProof___closed__1));
switch(v_phase_2805_)
{
case 0:
{
lean_object* v___x_2808_; 
v___x_2808_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__13));
v___y_2778_ = v___x_2807_;
v___y_2779_ = v_ref_2801_;
v___y_2780_ = v_msg_2802_;
v_name_2781_ = v_name_2803_;
v_builder_2782_ = v_builder_2804_;
v_scope_2783_ = v_scope_2806_;
v___y_2784_ = v___x_2808_;
goto v___jp_2777_;
}
case 1:
{
lean_object* v___x_2809_; 
v___x_2809_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__14));
v___y_2778_ = v___x_2807_;
v___y_2779_ = v_ref_2801_;
v___y_2780_ = v_msg_2802_;
v_name_2781_ = v_name_2803_;
v_builder_2782_ = v_builder_2804_;
v_scope_2783_ = v_scope_2806_;
v___y_2784_ = v___x_2809_;
goto v___jp_2777_;
}
default: 
{
lean_object* v___x_2810_; 
v___x_2810_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_instToMessageData___lam__0___closed__15));
v___y_2778_ = v___x_2807_;
v___y_2779_ = v_ref_2801_;
v___y_2780_ = v_msg_2802_;
v_name_2781_ = v_name_2803_;
v_builder_2782_ = v_builder_2804_;
v_scope_2783_ = v_scope_2806_;
v___y_2784_ = v___x_2810_;
goto v___jp_2777_;
}
}
}
else
{
lean_dec_ref_known(v___y_2796_, 2);
lean_dec_ref(v_m_2737_);
return v___y_2797_;
}
}
else
{
lean_dec_ref(v___y_2796_);
lean_dec_ref(v_m_2737_);
return v___y_2797_;
}
}
v___jp_2811_:
{
uint8_t v___x_2814_; 
v___x_2814_ = l_Lean_Exception_isInterrupt(v_a_2813_);
if (v___x_2814_ == 0)
{
uint8_t v___x_2815_; 
lean_inc_ref(v_a_2813_);
v___x_2815_ = l_Lean_Exception_isRuntime(v_a_2813_);
v___y_2796_ = v_a_2813_;
v___y_2797_ = v___y_2812_;
v___y_2798_ = v___x_2815_;
goto v___jp_2795_;
}
else
{
v___y_2796_ = v_a_2813_;
v___y_2797_ = v___y_2812_;
v___y_2798_ = v___x_2814_;
goto v___jp_2795_;
}
}
v___jp_2816_:
{
if (lean_obj_tag(v___y_2817_) == 0)
{
lean_dec_ref(v_m_2737_);
return v___y_2817_;
}
else
{
lean_object* v_a_2818_; 
v_a_2818_ = lean_ctor_get(v___y_2817_, 0);
lean_inc(v_a_2818_);
v___y_2812_ = v___y_2817_;
v_a_2813_ = v_a_2818_;
goto v___jp_2811_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_getProof___boxed(lean_object* v_goal_2906_, lean_object* v_m_2907_, lean_object* v_a_2908_, lean_object* v_a_2909_, lean_object* v_a_2910_, lean_object* v_a_2911_, lean_object* v_a_2912_){
_start:
{
lean_object* v_res_2913_; 
v_res_2913_ = lp_aesop_Aesop_ForwardRuleMatch_getProof(v_goal_2906_, v_m_2907_, v_a_2908_, v_a_2909_, v_a_2910_, v_a_2911_);
lean_dec(v_a_2911_);
lean_dec_ref(v_a_2910_);
lean_dec(v_a_2909_);
lean_dec_ref(v_a_2908_);
return v_res_2913_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_assign___at___00Aesop_ForwardRuleMatch_getProof_spec__0(lean_object* v_mvarId_2914_, lean_object* v_val_2915_, lean_object* v___y_2916_, lean_object* v___y_2917_, lean_object* v___y_2918_, lean_object* v___y_2919_){
_start:
{
lean_object* v___x_2921_; 
v___x_2921_ = lp_aesop_Lean_MVarId_assign___at___00Aesop_ForwardRuleMatch_getProof_spec__0___redArg(v_mvarId_2914_, v_val_2915_, v___y_2917_);
return v___x_2921_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_assign___at___00Aesop_ForwardRuleMatch_getProof_spec__0___boxed(lean_object* v_mvarId_2922_, lean_object* v_val_2923_, lean_object* v___y_2924_, lean_object* v___y_2925_, lean_object* v___y_2926_, lean_object* v___y_2927_, lean_object* v___y_2928_){
_start:
{
lean_object* v_res_2929_; 
v_res_2929_ = lp_aesop_Lean_MVarId_assign___at___00Aesop_ForwardRuleMatch_getProof_spec__0(v_mvarId_2922_, v_val_2923_, v___y_2924_, v___y_2925_, v___y_2926_, v___y_2927_);
lean_dec(v___y_2927_);
lean_dec_ref(v___y_2926_);
lean_dec(v___y_2925_);
lean_dec_ref(v___y_2924_);
return v_res_2929_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_assignLevelMVar___at___00Aesop_ForwardRuleMatch_getProof_spec__1(lean_object* v_mvarId_2930_, lean_object* v_val_2931_, lean_object* v___y_2932_, lean_object* v___y_2933_, lean_object* v___y_2934_, lean_object* v___y_2935_){
_start:
{
lean_object* v___x_2937_; 
v___x_2937_ = lp_aesop_Lean_assignLevelMVar___at___00Aesop_ForwardRuleMatch_getProof_spec__1___redArg(v_mvarId_2930_, v_val_2931_, v___y_2933_);
return v___x_2937_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_assignLevelMVar___at___00Aesop_ForwardRuleMatch_getProof_spec__1___boxed(lean_object* v_mvarId_2938_, lean_object* v_val_2939_, lean_object* v___y_2940_, lean_object* v___y_2941_, lean_object* v___y_2942_, lean_object* v___y_2943_, lean_object* v___y_2944_){
_start:
{
lean_object* v_res_2945_; 
v_res_2945_ = lp_aesop_Lean_assignLevelMVar___at___00Aesop_ForwardRuleMatch_getProof_spec__1(v_mvarId_2938_, v_val_2939_, v___y_2940_, v___y_2941_, v___y_2942_, v___y_2943_);
lean_dec(v___y_2943_);
lean_dec_ref(v___y_2942_);
lean_dec(v___y_2941_);
lean_dec_ref(v___y_2940_);
return v_res_2945_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_ForwardRuleMatch_getProof_spec__3(lean_object* v_opt_2946_, lean_object* v___y_2947_, lean_object* v___y_2948_, lean_object* v___y_2949_, lean_object* v___y_2950_){
_start:
{
lean_object* v___x_2952_; 
v___x_2952_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_ForwardRuleMatch_getProof_spec__3___redArg(v_opt_2946_, v___y_2949_);
return v___x_2952_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_ForwardRuleMatch_getProof_spec__3___boxed(lean_object* v_opt_2953_, lean_object* v___y_2954_, lean_object* v___y_2955_, lean_object* v___y_2956_, lean_object* v___y_2957_, lean_object* v___y_2958_){
_start:
{
lean_object* v_res_2959_; 
v_res_2959_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_ForwardRuleMatch_getProof_spec__3(v_opt_2953_, v___y_2954_, v___y_2955_, v___y_2956_, v___y_2957_);
lean_dec(v___y_2957_);
lean_dec_ref(v___y_2956_);
lean_dec(v___y_2955_);
lean_dec_ref(v___y_2954_);
lean_dec_ref(v_opt_2953_);
return v_res_2959_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_ForwardRuleMatch_getProof_spec__8(lean_object* v_00_u03b1_2960_, lean_object* v_msg_2961_, lean_object* v___y_2962_, lean_object* v___y_2963_, lean_object* v___y_2964_, lean_object* v___y_2965_){
_start:
{
lean_object* v___x_2967_; 
v___x_2967_ = lp_aesop_Lean_throwError___at___00Aesop_ForwardRuleMatch_getProof_spec__8___redArg(v_msg_2961_, v___y_2962_, v___y_2963_, v___y_2964_, v___y_2965_);
return v___x_2967_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_throwError___at___00Aesop_ForwardRuleMatch_getProof_spec__8___boxed(lean_object* v_00_u03b1_2968_, lean_object* v_msg_2969_, lean_object* v___y_2970_, lean_object* v___y_2971_, lean_object* v___y_2972_, lean_object* v___y_2973_, lean_object* v___y_2974_){
_start:
{
lean_object* v_res_2975_; 
v_res_2975_ = lp_aesop_Lean_throwError___at___00Aesop_ForwardRuleMatch_getProof_spec__8(v_00_u03b1_2968_, v_msg_2969_, v___y_2970_, v___y_2971_, v___y_2972_, v___y_2973_);
lean_dec(v___y_2973_);
lean_dec_ref(v___y_2972_);
lean_dec(v___y_2971_);
lean_dec_ref(v___y_2970_);
return v_res_2975_;
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13_spec__17(lean_object* v_00_u03b1_2976_, lean_object* v_x_2977_, lean_object* v___y_2978_, lean_object* v___y_2979_, lean_object* v___y_2980_, lean_object* v___y_2981_){
_start:
{
lean_object* v___x_2983_; 
v___x_2983_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13_spec__17___redArg(v_x_2977_);
return v___x_2983_;
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13_spec__17___boxed(lean_object* v_00_u03b1_2984_, lean_object* v_x_2985_, lean_object* v___y_2986_, lean_object* v___y_2987_, lean_object* v___y_2988_, lean_object* v___y_2989_, lean_object* v___y_2990_){
_start:
{
lean_object* v_res_2991_; 
v_res_2991_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13_spec__17(v_00_u03b1_2984_, v_x_2985_, v___y_2986_, v___y_2987_, v___y_2988_, v___y_2989_);
lean_dec(v___y_2989_);
lean_dec_ref(v___y_2988_);
lean_dec(v___y_2987_);
lean_dec_ref(v___y_2986_);
return v_res_2991_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_ForwardRuleMatch_getProof_spec__0_spec__0(lean_object* v_00_u03b2_2992_, lean_object* v_x_2993_, lean_object* v_x_2994_, lean_object* v_x_2995_){
_start:
{
lean_object* v___x_2996_; 
v___x_2996_ = lp_aesop_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_ForwardRuleMatch_getProof_spec__0_spec__0___redArg(v_x_2993_, v_x_2994_, v_x_2995_);
return v___x_2996_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_ForwardRuleMatch_getProof_spec__1_spec__2(lean_object* v_00_u03b2_2997_, lean_object* v_x_2998_, lean_object* v_x_2999_, lean_object* v_x_3000_){
_start:
{
lean_object* v___x_3001_; 
v___x_3001_ = lp_aesop_Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_ForwardRuleMatch_getProof_spec__1_spec__2___redArg(v_x_2998_, v_x_2999_, v_x_3000_);
return v___x_3001_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_ForwardRuleMatch_getProof_spec__0_spec__0_spec__6(lean_object* v_00_u03b2_3002_, lean_object* v_x_3003_, size_t v_x_3004_, size_t v_x_3005_, lean_object* v_x_3006_, lean_object* v_x_3007_){
_start:
{
lean_object* v___x_3008_; 
v___x_3008_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_ForwardRuleMatch_getProof_spec__0_spec__0_spec__6___redArg(v_x_3003_, v_x_3004_, v_x_3005_, v_x_3006_, v_x_3007_);
return v___x_3008_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_ForwardRuleMatch_getProof_spec__0_spec__0_spec__6___boxed(lean_object* v_00_u03b2_3009_, lean_object* v_x_3010_, lean_object* v_x_3011_, lean_object* v_x_3012_, lean_object* v_x_3013_, lean_object* v_x_3014_){
_start:
{
size_t v_x_30643__boxed_3015_; size_t v_x_30644__boxed_3016_; lean_object* v_res_3017_; 
v_x_30643__boxed_3015_ = lean_unbox_usize(v_x_3011_);
lean_dec(v_x_3011_);
v_x_30644__boxed_3016_ = lean_unbox_usize(v_x_3012_);
lean_dec(v_x_3012_);
v_res_3017_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_ForwardRuleMatch_getProof_spec__0_spec__0_spec__6(v_00_u03b2_3009_, v_x_3010_, v_x_30643__boxed_3015_, v_x_30644__boxed_3016_, v_x_3013_, v_x_3014_);
return v_res_3017_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_ForwardRuleMatch_getProof_spec__1_spec__2_spec__9(lean_object* v_00_u03b2_3018_, lean_object* v_x_3019_, size_t v_x_3020_, size_t v_x_3021_, lean_object* v_x_3022_, lean_object* v_x_3023_){
_start:
{
lean_object* v___x_3024_; 
v___x_3024_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_ForwardRuleMatch_getProof_spec__1_spec__2_spec__9___redArg(v_x_3019_, v_x_3020_, v_x_3021_, v_x_3022_, v_x_3023_);
return v___x_3024_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_ForwardRuleMatch_getProof_spec__1_spec__2_spec__9___boxed(lean_object* v_00_u03b2_3025_, lean_object* v_x_3026_, lean_object* v_x_3027_, lean_object* v_x_3028_, lean_object* v_x_3029_, lean_object* v_x_3030_){
_start:
{
size_t v_x_30660__boxed_3031_; size_t v_x_30661__boxed_3032_; lean_object* v_res_3033_; 
v_x_30660__boxed_3031_ = lean_unbox_usize(v_x_3027_);
lean_dec(v_x_3027_);
v_x_30661__boxed_3032_ = lean_unbox_usize(v_x_3028_);
lean_dec(v_x_3028_);
v_res_3033_ = lp_aesop_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_ForwardRuleMatch_getProof_spec__1_spec__2_spec__9(v_00_u03b2_3025_, v_x_3026_, v_x_30660__boxed_3031_, v_x_30661__boxed_3032_, v_x_3029_, v_x_3030_);
return v_res_3033_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_ForwardRuleMatch_getProof_spec__0_spec__0_spec__6_spec__19(lean_object* v_00_u03b2_3034_, lean_object* v_n_3035_, lean_object* v_k_3036_, lean_object* v_v_3037_){
_start:
{
lean_object* v___x_3038_; 
v___x_3038_ = lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_ForwardRuleMatch_getProof_spec__0_spec__0_spec__6_spec__19___redArg(v_n_3035_, v_k_3036_, v_v_3037_);
return v___x_3038_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_ForwardRuleMatch_getProof_spec__0_spec__0_spec__6_spec__20(lean_object* v_00_u03b2_3039_, size_t v_depth_3040_, lean_object* v_keys_3041_, lean_object* v_vals_3042_, lean_object* v_heq_3043_, lean_object* v_i_3044_, lean_object* v_entries_3045_){
_start:
{
lean_object* v___x_3046_; 
v___x_3046_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_ForwardRuleMatch_getProof_spec__0_spec__0_spec__6_spec__20___redArg(v_depth_3040_, v_keys_3041_, v_vals_3042_, v_i_3044_, v_entries_3045_);
return v___x_3046_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_ForwardRuleMatch_getProof_spec__0_spec__0_spec__6_spec__20___boxed(lean_object* v_00_u03b2_3047_, lean_object* v_depth_3048_, lean_object* v_keys_3049_, lean_object* v_vals_3050_, lean_object* v_heq_3051_, lean_object* v_i_3052_, lean_object* v_entries_3053_){
_start:
{
size_t v_depth_boxed_3054_; lean_object* v_res_3055_; 
v_depth_boxed_3054_ = lean_unbox_usize(v_depth_3048_);
lean_dec(v_depth_3048_);
v_res_3055_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_ForwardRuleMatch_getProof_spec__0_spec__0_spec__6_spec__20(v_00_u03b2_3047_, v_depth_boxed_3054_, v_keys_3049_, v_vals_3050_, v_heq_3051_, v_i_3052_, v_entries_3053_);
lean_dec_ref(v_vals_3050_);
lean_dec_ref(v_keys_3049_);
return v_res_3055_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_ForwardRuleMatch_getProof_spec__1_spec__2_spec__9_spec__23(lean_object* v_00_u03b2_3056_, lean_object* v_n_3057_, lean_object* v_k_3058_, lean_object* v_v_3059_){
_start:
{
lean_object* v___x_3060_; 
v___x_3060_ = lp_aesop_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_ForwardRuleMatch_getProof_spec__1_spec__2_spec__9_spec__23___redArg(v_n_3057_, v_k_3058_, v_v_3059_);
return v___x_3060_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_ForwardRuleMatch_getProof_spec__1_spec__2_spec__9_spec__24(lean_object* v_00_u03b2_3061_, size_t v_depth_3062_, lean_object* v_keys_3063_, lean_object* v_vals_3064_, lean_object* v_heq_3065_, lean_object* v_i_3066_, lean_object* v_entries_3067_){
_start:
{
lean_object* v___x_3068_; 
v___x_3068_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_ForwardRuleMatch_getProof_spec__1_spec__2_spec__9_spec__24___redArg(v_depth_3062_, v_keys_3063_, v_vals_3064_, v_i_3066_, v_entries_3067_);
return v___x_3068_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_ForwardRuleMatch_getProof_spec__1_spec__2_spec__9_spec__24___boxed(lean_object* v_00_u03b2_3069_, lean_object* v_depth_3070_, lean_object* v_keys_3071_, lean_object* v_vals_3072_, lean_object* v_heq_3073_, lean_object* v_i_3074_, lean_object* v_entries_3075_){
_start:
{
size_t v_depth_boxed_3076_; lean_object* v_res_3077_; 
v_depth_boxed_3076_ = lean_unbox_usize(v_depth_3070_);
lean_dec(v_depth_3070_);
v_res_3077_ = lp_aesop___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_ForwardRuleMatch_getProof_spec__1_spec__2_spec__9_spec__24(v_00_u03b2_3069_, v_depth_boxed_3076_, v_keys_3071_, v_vals_3072_, v_heq_3073_, v_i_3074_, v_entries_3075_);
lean_dec_ref(v_vals_3072_);
lean_dec_ref(v_keys_3071_);
return v_res_3077_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_ForwardRuleMatch_getProof_spec__0_spec__0_spec__6_spec__19_spec__22(lean_object* v_00_u03b2_3078_, lean_object* v_x_3079_, lean_object* v_x_3080_, lean_object* v_x_3081_, lean_object* v_x_3082_){
_start:
{
lean_object* v___x_3083_; 
v___x_3083_ = lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Aesop_ForwardRuleMatch_getProof_spec__0_spec__0_spec__6_spec__19_spec__22___redArg(v_x_3079_, v_x_3080_, v_x_3081_, v_x_3082_);
return v___x_3083_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_ForwardRuleMatch_getProof_spec__1_spec__2_spec__9_spec__23_spec__26(lean_object* v_00_u03b2_3084_, lean_object* v_x_3085_, lean_object* v_x_3086_, lean_object* v_x_3087_, lean_object* v_x_3088_){
_start:
{
lean_object* v___x_3089_; 
v___x_3089_ = lp_aesop_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_assignLevelMVar___at___00Aesop_ForwardRuleMatch_getProof_spec__1_spec__2_spec__9_spec__23_spec__26___redArg(v_x_3085_, v_x_3086_, v_x_3087_, v_x_3088_);
return v___x_3089_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_getUnusedUserName___at___00Aesop_ForwardRuleMatch_apply_spec__0___redArg(lean_object* v_suggestion_3090_, lean_object* v___y_3091_){
_start:
{
lean_object* v_lctx_3093_; lean_object* v___x_3094_; lean_object* v___x_3095_; 
v_lctx_3093_ = lean_ctor_get(v___y_3091_, 2);
v___x_3094_ = lp_batteries_Lean_LocalContext_getUnusedUserName(v_lctx_3093_, v_suggestion_3090_);
v___x_3095_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3095_, 0, v___x_3094_);
return v___x_3095_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_getUnusedUserName___at___00Aesop_ForwardRuleMatch_apply_spec__0___redArg___boxed(lean_object* v_suggestion_3096_, lean_object* v___y_3097_, lean_object* v___y_3098_){
_start:
{
lean_object* v_res_3099_; 
v_res_3099_ = lp_aesop_Lean_Meta_getUnusedUserName___at___00Aesop_ForwardRuleMatch_apply_spec__0___redArg(v_suggestion_3096_, v___y_3097_);
lean_dec_ref(v___y_3097_);
lean_dec(v_suggestion_3096_);
return v_res_3099_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_getUnusedUserName___at___00Aesop_ForwardRuleMatch_apply_spec__0(lean_object* v_suggestion_3100_, lean_object* v___y_3101_, lean_object* v___y_3102_, lean_object* v___y_3103_, lean_object* v___y_3104_, lean_object* v___y_3105_, lean_object* v___y_3106_){
_start:
{
lean_object* v___x_3108_; 
v___x_3108_ = lp_aesop_Lean_Meta_getUnusedUserName___at___00Aesop_ForwardRuleMatch_apply_spec__0___redArg(v_suggestion_3100_, v___y_3103_);
return v___x_3108_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Meta_getUnusedUserName___at___00Aesop_ForwardRuleMatch_apply_spec__0___boxed(lean_object* v_suggestion_3109_, lean_object* v___y_3110_, lean_object* v___y_3111_, lean_object* v___y_3112_, lean_object* v___y_3113_, lean_object* v___y_3114_, lean_object* v___y_3115_, lean_object* v___y_3116_){
_start:
{
lean_object* v_res_3117_; 
v_res_3117_ = lp_aesop_Lean_Meta_getUnusedUserName___at___00Aesop_ForwardRuleMatch_apply_spec__0(v_suggestion_3109_, v___y_3110_, v___y_3111_, v___y_3112_, v___y_3113_, v___y_3114_, v___y_3115_);
lean_dec(v___y_3115_);
lean_dec_ref(v___y_3114_);
lean_dec(v___y_3113_);
lean_dec_ref(v___y_3112_);
lean_dec(v___y_3111_);
lean_dec(v___y_3110_);
lean_dec(v_suggestion_3109_);
return v_res_3117_;
}
}
static lean_object* _init_lp_aesop_panic___at___00Aesop_ForwardRuleMatch_apply_spec__1___closed__0(void){
_start:
{
lean_object* v___x_3118_; 
v___x_3118_ = l_instMonadEIO(lean_box(0));
return v___x_3118_;
}
}
LEAN_EXPORT lean_object* lp_aesop_panic___at___00Aesop_ForwardRuleMatch_apply_spec__1(lean_object* v_msg_3123_, lean_object* v___y_3124_, lean_object* v___y_3125_, lean_object* v___y_3126_, lean_object* v___y_3127_, lean_object* v___y_3128_, lean_object* v___y_3129_){
_start:
{
lean_object* v___x_3131_; lean_object* v___x_3132_; lean_object* v_toApplicative_3133_; lean_object* v___x_3135_; uint8_t v_isShared_3136_; uint8_t v_isSharedCheck_3196_; 
v___x_3131_ = lean_obj_once(&lp_aesop_panic___at___00Aesop_ForwardRuleMatch_apply_spec__1___closed__0, &lp_aesop_panic___at___00Aesop_ForwardRuleMatch_apply_spec__1___closed__0_once, _init_lp_aesop_panic___at___00Aesop_ForwardRuleMatch_apply_spec__1___closed__0);
v___x_3132_ = l_StateRefT_x27_instMonad___redArg(v___x_3131_);
v_toApplicative_3133_ = lean_ctor_get(v___x_3132_, 0);
v_isSharedCheck_3196_ = !lean_is_exclusive(v___x_3132_);
if (v_isSharedCheck_3196_ == 0)
{
lean_object* v_unused_3197_; 
v_unused_3197_ = lean_ctor_get(v___x_3132_, 1);
lean_dec(v_unused_3197_);
v___x_3135_ = v___x_3132_;
v_isShared_3136_ = v_isSharedCheck_3196_;
goto v_resetjp_3134_;
}
else
{
lean_inc(v_toApplicative_3133_);
lean_dec(v___x_3132_);
v___x_3135_ = lean_box(0);
v_isShared_3136_ = v_isSharedCheck_3196_;
goto v_resetjp_3134_;
}
v_resetjp_3134_:
{
lean_object* v_toFunctor_3137_; lean_object* v_toSeq_3138_; lean_object* v_toSeqLeft_3139_; lean_object* v_toSeqRight_3140_; lean_object* v___x_3142_; uint8_t v_isShared_3143_; uint8_t v_isSharedCheck_3194_; 
v_toFunctor_3137_ = lean_ctor_get(v_toApplicative_3133_, 0);
v_toSeq_3138_ = lean_ctor_get(v_toApplicative_3133_, 2);
v_toSeqLeft_3139_ = lean_ctor_get(v_toApplicative_3133_, 3);
v_toSeqRight_3140_ = lean_ctor_get(v_toApplicative_3133_, 4);
v_isSharedCheck_3194_ = !lean_is_exclusive(v_toApplicative_3133_);
if (v_isSharedCheck_3194_ == 0)
{
lean_object* v_unused_3195_; 
v_unused_3195_ = lean_ctor_get(v_toApplicative_3133_, 1);
lean_dec(v_unused_3195_);
v___x_3142_ = v_toApplicative_3133_;
v_isShared_3143_ = v_isSharedCheck_3194_;
goto v_resetjp_3141_;
}
else
{
lean_inc(v_toSeqRight_3140_);
lean_inc(v_toSeqLeft_3139_);
lean_inc(v_toSeq_3138_);
lean_inc(v_toFunctor_3137_);
lean_dec(v_toApplicative_3133_);
v___x_3142_ = lean_box(0);
v_isShared_3143_ = v_isSharedCheck_3194_;
goto v_resetjp_3141_;
}
v_resetjp_3141_:
{
lean_object* v___f_3144_; lean_object* v___f_3145_; lean_object* v___f_3146_; lean_object* v___f_3147_; lean_object* v___x_3148_; lean_object* v___f_3149_; lean_object* v___f_3150_; lean_object* v___f_3151_; lean_object* v___x_3153_; 
v___f_3144_ = ((lean_object*)(lp_aesop_panic___at___00Aesop_ForwardRuleMatch_apply_spec__1___closed__1));
v___f_3145_ = ((lean_object*)(lp_aesop_panic___at___00Aesop_ForwardRuleMatch_apply_spec__1___closed__2));
lean_inc_ref(v_toFunctor_3137_);
v___f_3146_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_3146_, 0, v_toFunctor_3137_);
v___f_3147_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_3147_, 0, v_toFunctor_3137_);
v___x_3148_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3148_, 0, v___f_3146_);
lean_ctor_set(v___x_3148_, 1, v___f_3147_);
v___f_3149_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_3149_, 0, v_toSeqRight_3140_);
v___f_3150_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_3150_, 0, v_toSeqLeft_3139_);
v___f_3151_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_3151_, 0, v_toSeq_3138_);
if (v_isShared_3143_ == 0)
{
lean_ctor_set(v___x_3142_, 4, v___f_3149_);
lean_ctor_set(v___x_3142_, 3, v___f_3150_);
lean_ctor_set(v___x_3142_, 2, v___f_3151_);
lean_ctor_set(v___x_3142_, 1, v___f_3144_);
lean_ctor_set(v___x_3142_, 0, v___x_3148_);
v___x_3153_ = v___x_3142_;
goto v_reusejp_3152_;
}
else
{
lean_object* v_reuseFailAlloc_3193_; 
v_reuseFailAlloc_3193_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3193_, 0, v___x_3148_);
lean_ctor_set(v_reuseFailAlloc_3193_, 1, v___f_3144_);
lean_ctor_set(v_reuseFailAlloc_3193_, 2, v___f_3151_);
lean_ctor_set(v_reuseFailAlloc_3193_, 3, v___f_3150_);
lean_ctor_set(v_reuseFailAlloc_3193_, 4, v___f_3149_);
v___x_3153_ = v_reuseFailAlloc_3193_;
goto v_reusejp_3152_;
}
v_reusejp_3152_:
{
lean_object* v___x_3155_; 
if (v_isShared_3136_ == 0)
{
lean_ctor_set(v___x_3135_, 1, v___f_3145_);
lean_ctor_set(v___x_3135_, 0, v___x_3153_);
v___x_3155_ = v___x_3135_;
goto v_reusejp_3154_;
}
else
{
lean_object* v_reuseFailAlloc_3192_; 
v_reuseFailAlloc_3192_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3192_, 0, v___x_3153_);
lean_ctor_set(v_reuseFailAlloc_3192_, 1, v___f_3145_);
v___x_3155_ = v_reuseFailAlloc_3192_;
goto v_reusejp_3154_;
}
v_reusejp_3154_:
{
lean_object* v___x_3156_; lean_object* v_toApplicative_3157_; lean_object* v___x_3159_; uint8_t v_isShared_3160_; uint8_t v_isSharedCheck_3190_; 
v___x_3156_ = l_StateRefT_x27_instMonad___redArg(v___x_3155_);
v_toApplicative_3157_ = lean_ctor_get(v___x_3156_, 0);
v_isSharedCheck_3190_ = !lean_is_exclusive(v___x_3156_);
if (v_isSharedCheck_3190_ == 0)
{
lean_object* v_unused_3191_; 
v_unused_3191_ = lean_ctor_get(v___x_3156_, 1);
lean_dec(v_unused_3191_);
v___x_3159_ = v___x_3156_;
v_isShared_3160_ = v_isSharedCheck_3190_;
goto v_resetjp_3158_;
}
else
{
lean_inc(v_toApplicative_3157_);
lean_dec(v___x_3156_);
v___x_3159_ = lean_box(0);
v_isShared_3160_ = v_isSharedCheck_3190_;
goto v_resetjp_3158_;
}
v_resetjp_3158_:
{
lean_object* v_toFunctor_3161_; lean_object* v_toSeq_3162_; lean_object* v_toSeqLeft_3163_; lean_object* v_toSeqRight_3164_; lean_object* v___x_3166_; uint8_t v_isShared_3167_; uint8_t v_isSharedCheck_3188_; 
v_toFunctor_3161_ = lean_ctor_get(v_toApplicative_3157_, 0);
v_toSeq_3162_ = lean_ctor_get(v_toApplicative_3157_, 2);
v_toSeqLeft_3163_ = lean_ctor_get(v_toApplicative_3157_, 3);
v_toSeqRight_3164_ = lean_ctor_get(v_toApplicative_3157_, 4);
v_isSharedCheck_3188_ = !lean_is_exclusive(v_toApplicative_3157_);
if (v_isSharedCheck_3188_ == 0)
{
lean_object* v_unused_3189_; 
v_unused_3189_ = lean_ctor_get(v_toApplicative_3157_, 1);
lean_dec(v_unused_3189_);
v___x_3166_ = v_toApplicative_3157_;
v_isShared_3167_ = v_isSharedCheck_3188_;
goto v_resetjp_3165_;
}
else
{
lean_inc(v_toSeqRight_3164_);
lean_inc(v_toSeqLeft_3163_);
lean_inc(v_toSeq_3162_);
lean_inc(v_toFunctor_3161_);
lean_dec(v_toApplicative_3157_);
v___x_3166_ = lean_box(0);
v_isShared_3167_ = v_isSharedCheck_3188_;
goto v_resetjp_3165_;
}
v_resetjp_3165_:
{
lean_object* v___f_3168_; lean_object* v___f_3169_; lean_object* v___f_3170_; lean_object* v___f_3171_; lean_object* v___x_3172_; lean_object* v___f_3173_; lean_object* v___f_3174_; lean_object* v___f_3175_; lean_object* v___x_3177_; 
v___f_3168_ = ((lean_object*)(lp_aesop_panic___at___00Aesop_ForwardRuleMatch_apply_spec__1___closed__3));
v___f_3169_ = ((lean_object*)(lp_aesop_panic___at___00Aesop_ForwardRuleMatch_apply_spec__1___closed__4));
lean_inc_ref(v_toFunctor_3161_);
v___f_3170_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_3170_, 0, v_toFunctor_3161_);
v___f_3171_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_3171_, 0, v_toFunctor_3161_);
v___x_3172_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3172_, 0, v___f_3170_);
lean_ctor_set(v___x_3172_, 1, v___f_3171_);
v___f_3173_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_3173_, 0, v_toSeqRight_3164_);
v___f_3174_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_3174_, 0, v_toSeqLeft_3163_);
v___f_3175_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_3175_, 0, v_toSeq_3162_);
if (v_isShared_3167_ == 0)
{
lean_ctor_set(v___x_3166_, 4, v___f_3173_);
lean_ctor_set(v___x_3166_, 3, v___f_3174_);
lean_ctor_set(v___x_3166_, 2, v___f_3175_);
lean_ctor_set(v___x_3166_, 1, v___f_3168_);
lean_ctor_set(v___x_3166_, 0, v___x_3172_);
v___x_3177_ = v___x_3166_;
goto v_reusejp_3176_;
}
else
{
lean_object* v_reuseFailAlloc_3187_; 
v_reuseFailAlloc_3187_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_3187_, 0, v___x_3172_);
lean_ctor_set(v_reuseFailAlloc_3187_, 1, v___f_3168_);
lean_ctor_set(v_reuseFailAlloc_3187_, 2, v___f_3175_);
lean_ctor_set(v_reuseFailAlloc_3187_, 3, v___f_3174_);
lean_ctor_set(v_reuseFailAlloc_3187_, 4, v___f_3173_);
v___x_3177_ = v_reuseFailAlloc_3187_;
goto v_reusejp_3176_;
}
v_reusejp_3176_:
{
lean_object* v___x_3179_; 
if (v_isShared_3160_ == 0)
{
lean_ctor_set(v___x_3159_, 1, v___f_3169_);
lean_ctor_set(v___x_3159_, 0, v___x_3177_);
v___x_3179_ = v___x_3159_;
goto v_reusejp_3178_;
}
else
{
lean_object* v_reuseFailAlloc_3186_; 
v_reuseFailAlloc_3186_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3186_, 0, v___x_3177_);
lean_ctor_set(v_reuseFailAlloc_3186_, 1, v___f_3169_);
v___x_3179_ = v_reuseFailAlloc_3186_;
goto v_reusejp_3178_;
}
v_reusejp_3178_:
{
lean_object* v___x_3180_; lean_object* v___x_3181_; lean_object* v___x_3182_; lean_object* v___x_3183_; lean_object* v___x_35313__overap_3184_; lean_object* v___x_3185_; 
v___x_3180_ = l_StateRefT_x27_instMonad___redArg(v___x_3179_);
v___x_3181_ = l_StateRefT_x27_instMonad___redArg(v___x_3180_);
v___x_3182_ = lean_box(0);
v___x_3183_ = l_instInhabitedOfMonad___redArg(v___x_3181_, v___x_3182_);
v___x_35313__overap_3184_ = lean_panic_fn_borrowed(v___x_3183_, v_msg_3123_);
lean_dec(v___x_3183_);
lean_inc(v___y_3129_);
lean_inc_ref(v___y_3128_);
lean_inc(v___y_3127_);
lean_inc_ref(v___y_3126_);
lean_inc(v___y_3125_);
lean_inc(v___y_3124_);
v___x_3185_ = lean_apply_7(v___x_35313__overap_3184_, v___y_3124_, v___y_3125_, v___y_3126_, v___y_3127_, v___y_3128_, v___y_3129_, lean_box(0));
return v___x_3185_;
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
LEAN_EXPORT lean_object* lp_aesop_panic___at___00Aesop_ForwardRuleMatch_apply_spec__1___boxed(lean_object* v_msg_3198_, lean_object* v___y_3199_, lean_object* v___y_3200_, lean_object* v___y_3201_, lean_object* v___y_3202_, lean_object* v___y_3203_, lean_object* v___y_3204_, lean_object* v___y_3205_){
_start:
{
lean_object* v_res_3206_; 
v_res_3206_ = lp_aesop_panic___at___00Aesop_ForwardRuleMatch_apply_spec__1(v_msg_3198_, v___y_3199_, v___y_3200_, v___y_3201_, v___y_3202_, v___y_3203_, v___y_3204_);
lean_dec(v___y_3204_);
lean_dec_ref(v___y_3203_);
lean_dec(v___y_3202_);
lean_dec_ref(v___y_3201_);
lean_dec(v___y_3200_);
lean_dec(v___y_3199_);
return v_res_3206_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_ForwardRuleMatch_apply_spec__2___redArg___lam__0(lean_object* v_x_3207_, lean_object* v___y_3208_, lean_object* v___y_3209_, lean_object* v___y_3210_, lean_object* v___y_3211_, lean_object* v___y_3212_, lean_object* v___y_3213_){
_start:
{
lean_object* v___x_3215_; 
lean_inc(v___y_3209_);
lean_inc(v___y_3208_);
v___x_3215_ = lean_apply_7(v_x_3207_, v___y_3208_, v___y_3209_, v___y_3210_, v___y_3211_, v___y_3212_, v___y_3213_, lean_box(0));
return v___x_3215_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_ForwardRuleMatch_apply_spec__2___redArg___lam__0___boxed(lean_object* v_x_3216_, lean_object* v___y_3217_, lean_object* v___y_3218_, lean_object* v___y_3219_, lean_object* v___y_3220_, lean_object* v___y_3221_, lean_object* v___y_3222_, lean_object* v___y_3223_){
_start:
{
lean_object* v_res_3224_; 
v_res_3224_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_ForwardRuleMatch_apply_spec__2___redArg___lam__0(v_x_3216_, v___y_3217_, v___y_3218_, v___y_3219_, v___y_3220_, v___y_3221_, v___y_3222_);
lean_dec(v___y_3218_);
lean_dec(v___y_3217_);
return v_res_3224_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_ForwardRuleMatch_apply_spec__2___redArg(lean_object* v_mvarId_3225_, lean_object* v_x_3226_, lean_object* v___y_3227_, lean_object* v___y_3228_, lean_object* v___y_3229_, lean_object* v___y_3230_, lean_object* v___y_3231_, lean_object* v___y_3232_){
_start:
{
lean_object* v___f_3234_; lean_object* v___x_3235_; 
lean_inc(v___y_3228_);
lean_inc(v___y_3227_);
v___f_3234_ = lean_alloc_closure((void*)(lp_aesop_Lean_MVarId_withContext___at___00Aesop_ForwardRuleMatch_apply_spec__2___redArg___lam__0___boxed), 8, 3);
lean_closure_set(v___f_3234_, 0, v_x_3226_);
lean_closure_set(v___f_3234_, 1, v___y_3227_);
lean_closure_set(v___f_3234_, 2, v___y_3228_);
v___x_3235_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_box(0), v_mvarId_3225_, v___f_3234_, v___y_3229_, v___y_3230_, v___y_3231_, v___y_3232_);
if (lean_obj_tag(v___x_3235_) == 0)
{
return v___x_3235_;
}
else
{
lean_object* v_a_3236_; lean_object* v___x_3238_; uint8_t v_isShared_3239_; uint8_t v_isSharedCheck_3243_; 
v_a_3236_ = lean_ctor_get(v___x_3235_, 0);
v_isSharedCheck_3243_ = !lean_is_exclusive(v___x_3235_);
if (v_isSharedCheck_3243_ == 0)
{
v___x_3238_ = v___x_3235_;
v_isShared_3239_ = v_isSharedCheck_3243_;
goto v_resetjp_3237_;
}
else
{
lean_inc(v_a_3236_);
lean_dec(v___x_3235_);
v___x_3238_ = lean_box(0);
v_isShared_3239_ = v_isSharedCheck_3243_;
goto v_resetjp_3237_;
}
v_resetjp_3237_:
{
lean_object* v___x_3241_; 
if (v_isShared_3239_ == 0)
{
v___x_3241_ = v___x_3238_;
goto v_reusejp_3240_;
}
else
{
lean_object* v_reuseFailAlloc_3242_; 
v_reuseFailAlloc_3242_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3242_, 0, v_a_3236_);
v___x_3241_ = v_reuseFailAlloc_3242_;
goto v_reusejp_3240_;
}
v_reusejp_3240_:
{
return v___x_3241_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_ForwardRuleMatch_apply_spec__2___redArg___boxed(lean_object* v_mvarId_3244_, lean_object* v_x_3245_, lean_object* v___y_3246_, lean_object* v___y_3247_, lean_object* v___y_3248_, lean_object* v___y_3249_, lean_object* v___y_3250_, lean_object* v___y_3251_, lean_object* v___y_3252_){
_start:
{
lean_object* v_res_3253_; 
v_res_3253_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_ForwardRuleMatch_apply_spec__2___redArg(v_mvarId_3244_, v_x_3245_, v___y_3246_, v___y_3247_, v___y_3248_, v___y_3249_, v___y_3250_, v___y_3251_);
lean_dec(v___y_3251_);
lean_dec_ref(v___y_3250_);
lean_dec(v___y_3249_);
lean_dec_ref(v___y_3248_);
lean_dec(v___y_3247_);
lean_dec(v___y_3246_);
return v_res_3253_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_ForwardRuleMatch_apply_spec__2(lean_object* v_00_u03b1_3254_, lean_object* v_mvarId_3255_, lean_object* v_x_3256_, lean_object* v___y_3257_, lean_object* v___y_3258_, lean_object* v___y_3259_, lean_object* v___y_3260_, lean_object* v___y_3261_, lean_object* v___y_3262_){
_start:
{
lean_object* v___x_3264_; 
v___x_3264_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_ForwardRuleMatch_apply_spec__2___redArg(v_mvarId_3255_, v_x_3256_, v___y_3257_, v___y_3258_, v___y_3259_, v___y_3260_, v___y_3261_, v___y_3262_);
return v___x_3264_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_MVarId_withContext___at___00Aesop_ForwardRuleMatch_apply_spec__2___boxed(lean_object* v_00_u03b1_3265_, lean_object* v_mvarId_3266_, lean_object* v_x_3267_, lean_object* v___y_3268_, lean_object* v___y_3269_, lean_object* v___y_3270_, lean_object* v___y_3271_, lean_object* v___y_3272_, lean_object* v___y_3273_, lean_object* v___y_3274_){
_start:
{
lean_object* v_res_3275_; 
v_res_3275_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_ForwardRuleMatch_apply_spec__2(v_00_u03b1_3265_, v_mvarId_3266_, v_x_3267_, v___y_3268_, v___y_3269_, v___y_3270_, v___y_3271_, v___y_3272_, v___y_3273_);
lean_dec(v___y_3273_);
lean_dec_ref(v___y_3272_);
lean_dec(v___y_3271_);
lean_dec_ref(v___y_3270_);
lean_dec(v___y_3269_);
lean_dec(v___y_3268_);
return v_res_3275_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_ForwardRuleMatch_apply_spec__5___redArg(lean_object* v___y_3276_){
_start:
{
lean_object* v___x_3278_; lean_object* v_traceState_3279_; lean_object* v_traces_3280_; lean_object* v___x_3281_; lean_object* v_traceState_3282_; lean_object* v_env_3283_; lean_object* v_nextMacroScope_3284_; lean_object* v_ngen_3285_; lean_object* v_auxDeclNGen_3286_; lean_object* v_cache_3287_; lean_object* v_messages_3288_; lean_object* v_infoState_3289_; lean_object* v_snapshotTasks_3290_; lean_object* v___x_3292_; uint8_t v_isShared_3293_; uint8_t v_isSharedCheck_3311_; 
v___x_3278_ = lean_st_ref_get(v___y_3276_);
v_traceState_3279_ = lean_ctor_get(v___x_3278_, 4);
lean_inc_ref(v_traceState_3279_);
lean_dec(v___x_3278_);
v_traces_3280_ = lean_ctor_get(v_traceState_3279_, 0);
lean_inc_ref(v_traces_3280_);
lean_dec_ref(v_traceState_3279_);
v___x_3281_ = lean_st_ref_take(v___y_3276_);
v_traceState_3282_ = lean_ctor_get(v___x_3281_, 4);
v_env_3283_ = lean_ctor_get(v___x_3281_, 0);
v_nextMacroScope_3284_ = lean_ctor_get(v___x_3281_, 1);
v_ngen_3285_ = lean_ctor_get(v___x_3281_, 2);
v_auxDeclNGen_3286_ = lean_ctor_get(v___x_3281_, 3);
v_cache_3287_ = lean_ctor_get(v___x_3281_, 5);
v_messages_3288_ = lean_ctor_get(v___x_3281_, 6);
v_infoState_3289_ = lean_ctor_get(v___x_3281_, 7);
v_snapshotTasks_3290_ = lean_ctor_get(v___x_3281_, 8);
v_isSharedCheck_3311_ = !lean_is_exclusive(v___x_3281_);
if (v_isSharedCheck_3311_ == 0)
{
v___x_3292_ = v___x_3281_;
v_isShared_3293_ = v_isSharedCheck_3311_;
goto v_resetjp_3291_;
}
else
{
lean_inc(v_snapshotTasks_3290_);
lean_inc(v_infoState_3289_);
lean_inc(v_messages_3288_);
lean_inc(v_cache_3287_);
lean_inc(v_traceState_3282_);
lean_inc(v_auxDeclNGen_3286_);
lean_inc(v_ngen_3285_);
lean_inc(v_nextMacroScope_3284_);
lean_inc(v_env_3283_);
lean_dec(v___x_3281_);
v___x_3292_ = lean_box(0);
v_isShared_3293_ = v_isSharedCheck_3311_;
goto v_resetjp_3291_;
}
v_resetjp_3291_:
{
uint64_t v_tid_3294_; lean_object* v___x_3296_; uint8_t v_isShared_3297_; uint8_t v_isSharedCheck_3309_; 
v_tid_3294_ = lean_ctor_get_uint64(v_traceState_3282_, sizeof(void*)*1);
v_isSharedCheck_3309_ = !lean_is_exclusive(v_traceState_3282_);
if (v_isSharedCheck_3309_ == 0)
{
lean_object* v_unused_3310_; 
v_unused_3310_ = lean_ctor_get(v_traceState_3282_, 0);
lean_dec(v_unused_3310_);
v___x_3296_ = v_traceState_3282_;
v_isShared_3297_ = v_isSharedCheck_3309_;
goto v_resetjp_3295_;
}
else
{
lean_dec(v_traceState_3282_);
v___x_3296_ = lean_box(0);
v_isShared_3297_ = v_isSharedCheck_3309_;
goto v_resetjp_3295_;
}
v_resetjp_3295_:
{
lean_object* v___x_3298_; lean_object* v___x_3299_; lean_object* v___x_3300_; lean_object* v___x_3302_; 
v___x_3298_ = lean_unsigned_to_nat(32u);
v___x_3299_ = lean_mk_empty_array_with_capacity(v___x_3298_);
lean_dec_ref(v___x_3299_);
v___x_3300_ = lean_obj_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_ForwardRuleMatch_getProof_spec__11___redArg___closed__1, &lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_ForwardRuleMatch_getProof_spec__11___redArg___closed__1_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_ForwardRuleMatch_getProof_spec__11___redArg___closed__1);
if (v_isShared_3297_ == 0)
{
lean_ctor_set(v___x_3296_, 0, v___x_3300_);
v___x_3302_ = v___x_3296_;
goto v_reusejp_3301_;
}
else
{
lean_object* v_reuseFailAlloc_3308_; 
v_reuseFailAlloc_3308_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_3308_, 0, v___x_3300_);
lean_ctor_set_uint64(v_reuseFailAlloc_3308_, sizeof(void*)*1, v_tid_3294_);
v___x_3302_ = v_reuseFailAlloc_3308_;
goto v_reusejp_3301_;
}
v_reusejp_3301_:
{
lean_object* v___x_3304_; 
if (v_isShared_3293_ == 0)
{
lean_ctor_set(v___x_3292_, 4, v___x_3302_);
v___x_3304_ = v___x_3292_;
goto v_reusejp_3303_;
}
else
{
lean_object* v_reuseFailAlloc_3307_; 
v_reuseFailAlloc_3307_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_3307_, 0, v_env_3283_);
lean_ctor_set(v_reuseFailAlloc_3307_, 1, v_nextMacroScope_3284_);
lean_ctor_set(v_reuseFailAlloc_3307_, 2, v_ngen_3285_);
lean_ctor_set(v_reuseFailAlloc_3307_, 3, v_auxDeclNGen_3286_);
lean_ctor_set(v_reuseFailAlloc_3307_, 4, v___x_3302_);
lean_ctor_set(v_reuseFailAlloc_3307_, 5, v_cache_3287_);
lean_ctor_set(v_reuseFailAlloc_3307_, 6, v_messages_3288_);
lean_ctor_set(v_reuseFailAlloc_3307_, 7, v_infoState_3289_);
lean_ctor_set(v_reuseFailAlloc_3307_, 8, v_snapshotTasks_3290_);
v___x_3304_ = v_reuseFailAlloc_3307_;
goto v_reusejp_3303_;
}
v_reusejp_3303_:
{
lean_object* v___x_3305_; lean_object* v___x_3306_; 
v___x_3305_ = lean_st_ref_set(v___y_3276_, v___x_3304_);
v___x_3306_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3306_, 0, v_traces_3280_);
return v___x_3306_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_ForwardRuleMatch_apply_spec__5___redArg___boxed(lean_object* v___y_3312_, lean_object* v___y_3313_){
_start:
{
lean_object* v_res_3314_; 
v_res_3314_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_ForwardRuleMatch_apply_spec__5___redArg(v___y_3312_);
lean_dec(v___y_3312_);
return v_res_3314_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_ForwardRuleMatch_apply_spec__5(lean_object* v___y_3315_, lean_object* v___y_3316_, lean_object* v___y_3317_, lean_object* v___y_3318_, lean_object* v___y_3319_, lean_object* v___y_3320_){
_start:
{
lean_object* v___x_3322_; 
v___x_3322_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_ForwardRuleMatch_apply_spec__5___redArg(v___y_3320_);
return v___x_3322_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_ForwardRuleMatch_apply_spec__5___boxed(lean_object* v___y_3323_, lean_object* v___y_3324_, lean_object* v___y_3325_, lean_object* v___y_3326_, lean_object* v___y_3327_, lean_object* v___y_3328_, lean_object* v___y_3329_){
_start:
{
lean_object* v_res_3330_; 
v_res_3330_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_ForwardRuleMatch_apply_spec__5(v___y_3323_, v___y_3324_, v___y_3325_, v___y_3326_, v___y_3327_, v___y_3328_);
lean_dec(v___y_3328_);
lean_dec_ref(v___y_3327_);
lean_dec(v___y_3326_);
lean_dec_ref(v___y_3325_);
lean_dec(v___y_3324_);
lean_dec(v___y_3323_);
return v_res_3330_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_apply___lam__0(lean_object* v_m_3331_, lean_object* v___y_3332_, lean_object* v___y_3333_, lean_object* v___y_3334_, lean_object* v___y_3335_, lean_object* v___y_3336_, lean_object* v___y_3337_){
_start:
{
lean_object* v___x_3339_; 
v___x_3339_ = lp_aesop_Aesop_ForwardRuleMatch_getPropHyps(v_m_3331_, v___y_3334_, v___y_3335_, v___y_3336_, v___y_3337_);
return v___x_3339_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_apply___lam__0___boxed(lean_object* v_m_3340_, lean_object* v___y_3341_, lean_object* v___y_3342_, lean_object* v___y_3343_, lean_object* v___y_3344_, lean_object* v___y_3345_, lean_object* v___y_3346_, lean_object* v___y_3347_){
_start:
{
lean_object* v_res_3348_; 
v_res_3348_ = lp_aesop_Aesop_ForwardRuleMatch_apply___lam__0(v_m_3340_, v___y_3341_, v___y_3342_, v___y_3343_, v___y_3344_, v___y_3345_, v___y_3346_);
lean_dec(v___y_3346_);
lean_dec_ref(v___y_3345_);
lean_dec(v___y_3344_);
lean_dec_ref(v___y_3343_);
lean_dec(v___y_3342_);
lean_dec(v___y_3341_);
lean_dec_ref(v_m_3340_);
return v_res_3348_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00Aesop_ForwardRuleMatch_apply_spec__4___redArg(lean_object* v_cls_3349_, lean_object* v_msg_3350_, lean_object* v___y_3351_, lean_object* v___y_3352_, lean_object* v___y_3353_, lean_object* v___y_3354_){
_start:
{
lean_object* v_ref_3356_; lean_object* v___x_3357_; lean_object* v_a_3358_; lean_object* v___x_3360_; uint8_t v_isShared_3361_; uint8_t v_isSharedCheck_3402_; 
v_ref_3356_ = lean_ctor_get(v___y_3353_, 5);
v___x_3357_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_addTrace___at___00Aesop_ForwardRuleMatch_getProof_spec__4_spec__6(v_msg_3350_, v___y_3351_, v___y_3352_, v___y_3353_, v___y_3354_);
v_a_3358_ = lean_ctor_get(v___x_3357_, 0);
v_isSharedCheck_3402_ = !lean_is_exclusive(v___x_3357_);
if (v_isSharedCheck_3402_ == 0)
{
v___x_3360_ = v___x_3357_;
v_isShared_3361_ = v_isSharedCheck_3402_;
goto v_resetjp_3359_;
}
else
{
lean_inc(v_a_3358_);
lean_dec(v___x_3357_);
v___x_3360_ = lean_box(0);
v_isShared_3361_ = v_isSharedCheck_3402_;
goto v_resetjp_3359_;
}
v_resetjp_3359_:
{
lean_object* v___x_3362_; lean_object* v_traceState_3363_; lean_object* v_env_3364_; lean_object* v_nextMacroScope_3365_; lean_object* v_ngen_3366_; lean_object* v_auxDeclNGen_3367_; lean_object* v_cache_3368_; lean_object* v_messages_3369_; lean_object* v_infoState_3370_; lean_object* v_snapshotTasks_3371_; lean_object* v___x_3373_; uint8_t v_isShared_3374_; uint8_t v_isSharedCheck_3401_; 
v___x_3362_ = lean_st_ref_take(v___y_3354_);
v_traceState_3363_ = lean_ctor_get(v___x_3362_, 4);
v_env_3364_ = lean_ctor_get(v___x_3362_, 0);
v_nextMacroScope_3365_ = lean_ctor_get(v___x_3362_, 1);
v_ngen_3366_ = lean_ctor_get(v___x_3362_, 2);
v_auxDeclNGen_3367_ = lean_ctor_get(v___x_3362_, 3);
v_cache_3368_ = lean_ctor_get(v___x_3362_, 5);
v_messages_3369_ = lean_ctor_get(v___x_3362_, 6);
v_infoState_3370_ = lean_ctor_get(v___x_3362_, 7);
v_snapshotTasks_3371_ = lean_ctor_get(v___x_3362_, 8);
v_isSharedCheck_3401_ = !lean_is_exclusive(v___x_3362_);
if (v_isSharedCheck_3401_ == 0)
{
v___x_3373_ = v___x_3362_;
v_isShared_3374_ = v_isSharedCheck_3401_;
goto v_resetjp_3372_;
}
else
{
lean_inc(v_snapshotTasks_3371_);
lean_inc(v_infoState_3370_);
lean_inc(v_messages_3369_);
lean_inc(v_cache_3368_);
lean_inc(v_traceState_3363_);
lean_inc(v_auxDeclNGen_3367_);
lean_inc(v_ngen_3366_);
lean_inc(v_nextMacroScope_3365_);
lean_inc(v_env_3364_);
lean_dec(v___x_3362_);
v___x_3373_ = lean_box(0);
v_isShared_3374_ = v_isSharedCheck_3401_;
goto v_resetjp_3372_;
}
v_resetjp_3372_:
{
uint64_t v_tid_3375_; lean_object* v_traces_3376_; lean_object* v___x_3378_; uint8_t v_isShared_3379_; uint8_t v_isSharedCheck_3400_; 
v_tid_3375_ = lean_ctor_get_uint64(v_traceState_3363_, sizeof(void*)*1);
v_traces_3376_ = lean_ctor_get(v_traceState_3363_, 0);
v_isSharedCheck_3400_ = !lean_is_exclusive(v_traceState_3363_);
if (v_isSharedCheck_3400_ == 0)
{
v___x_3378_ = v_traceState_3363_;
v_isShared_3379_ = v_isSharedCheck_3400_;
goto v_resetjp_3377_;
}
else
{
lean_inc(v_traces_3376_);
lean_dec(v_traceState_3363_);
v___x_3378_ = lean_box(0);
v_isShared_3379_ = v_isSharedCheck_3400_;
goto v_resetjp_3377_;
}
v_resetjp_3377_:
{
lean_object* v___x_3380_; double v___x_3381_; uint8_t v___x_3382_; lean_object* v___x_3383_; lean_object* v___x_3384_; lean_object* v___x_3385_; lean_object* v___x_3386_; lean_object* v___x_3387_; lean_object* v___x_3388_; lean_object* v___x_3390_; 
v___x_3380_ = lean_box(0);
v___x_3381_ = lean_float_once(&lp_aesop_Lean_addTrace___at___00Aesop_ForwardRuleMatch_getProof_spec__4___closed__0, &lp_aesop_Lean_addTrace___at___00Aesop_ForwardRuleMatch_getProof_spec__4___closed__0_once, _init_lp_aesop_Lean_addTrace___at___00Aesop_ForwardRuleMatch_getProof_spec__4___closed__0);
v___x_3382_ = 0;
v___x_3383_ = ((lean_object*)(lp_aesop_Lean_addTrace___at___00Aesop_ForwardRuleMatch_getProof_spec__4___closed__1));
v___x_3384_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v___x_3384_, 0, v_cls_3349_);
lean_ctor_set(v___x_3384_, 1, v___x_3380_);
lean_ctor_set(v___x_3384_, 2, v___x_3383_);
lean_ctor_set_float(v___x_3384_, sizeof(void*)*3, v___x_3381_);
lean_ctor_set_float(v___x_3384_, sizeof(void*)*3 + 8, v___x_3381_);
lean_ctor_set_uint8(v___x_3384_, sizeof(void*)*3 + 16, v___x_3382_);
v___x_3385_ = ((lean_object*)(lp_aesop_Lean_addTrace___at___00Aesop_ForwardRuleMatch_getProof_spec__4___closed__2));
v___x_3386_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v___x_3386_, 0, v___x_3384_);
lean_ctor_set(v___x_3386_, 1, v_a_3358_);
lean_ctor_set(v___x_3386_, 2, v___x_3385_);
lean_inc(v_ref_3356_);
v___x_3387_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3387_, 0, v_ref_3356_);
lean_ctor_set(v___x_3387_, 1, v___x_3386_);
v___x_3388_ = l_Lean_PersistentArray_push___redArg(v_traces_3376_, v___x_3387_);
if (v_isShared_3379_ == 0)
{
lean_ctor_set(v___x_3378_, 0, v___x_3388_);
v___x_3390_ = v___x_3378_;
goto v_reusejp_3389_;
}
else
{
lean_object* v_reuseFailAlloc_3399_; 
v_reuseFailAlloc_3399_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_3399_, 0, v___x_3388_);
lean_ctor_set_uint64(v_reuseFailAlloc_3399_, sizeof(void*)*1, v_tid_3375_);
v___x_3390_ = v_reuseFailAlloc_3399_;
goto v_reusejp_3389_;
}
v_reusejp_3389_:
{
lean_object* v___x_3392_; 
if (v_isShared_3374_ == 0)
{
lean_ctor_set(v___x_3373_, 4, v___x_3390_);
v___x_3392_ = v___x_3373_;
goto v_reusejp_3391_;
}
else
{
lean_object* v_reuseFailAlloc_3398_; 
v_reuseFailAlloc_3398_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_3398_, 0, v_env_3364_);
lean_ctor_set(v_reuseFailAlloc_3398_, 1, v_nextMacroScope_3365_);
lean_ctor_set(v_reuseFailAlloc_3398_, 2, v_ngen_3366_);
lean_ctor_set(v_reuseFailAlloc_3398_, 3, v_auxDeclNGen_3367_);
lean_ctor_set(v_reuseFailAlloc_3398_, 4, v___x_3390_);
lean_ctor_set(v_reuseFailAlloc_3398_, 5, v_cache_3368_);
lean_ctor_set(v_reuseFailAlloc_3398_, 6, v_messages_3369_);
lean_ctor_set(v_reuseFailAlloc_3398_, 7, v_infoState_3370_);
lean_ctor_set(v_reuseFailAlloc_3398_, 8, v_snapshotTasks_3371_);
v___x_3392_ = v_reuseFailAlloc_3398_;
goto v_reusejp_3391_;
}
v_reusejp_3391_:
{
lean_object* v___x_3393_; lean_object* v___x_3394_; lean_object* v___x_3396_; 
v___x_3393_ = lean_st_ref_set(v___y_3354_, v___x_3392_);
v___x_3394_ = lean_box(0);
if (v_isShared_3361_ == 0)
{
lean_ctor_set(v___x_3360_, 0, v___x_3394_);
v___x_3396_ = v___x_3360_;
goto v_reusejp_3395_;
}
else
{
lean_object* v_reuseFailAlloc_3397_; 
v_reuseFailAlloc_3397_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3397_, 0, v___x_3394_);
v___x_3396_ = v_reuseFailAlloc_3397_;
goto v_reusejp_3395_;
}
v_reusejp_3395_:
{
return v___x_3396_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00Aesop_ForwardRuleMatch_apply_spec__4___redArg___boxed(lean_object* v_cls_3403_, lean_object* v_msg_3404_, lean_object* v___y_3405_, lean_object* v___y_3406_, lean_object* v___y_3407_, lean_object* v___y_3408_, lean_object* v___y_3409_){
_start:
{
lean_object* v_res_3410_; 
v_res_3410_ = lp_aesop_Lean_addTrace___at___00Aesop_ForwardRuleMatch_apply_spec__4___redArg(v_cls_3403_, v_msg_3404_, v___y_3405_, v___y_3406_, v___y_3407_, v___y_3408_);
lean_dec(v___y_3408_);
lean_dec_ref(v___y_3407_);
lean_dec(v___y_3406_);
lean_dec_ref(v___y_3405_);
return v_res_3410_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_ForwardRuleMatch_apply_spec__3___redArg(lean_object* v_opt_3411_, lean_object* v___y_3412_){
_start:
{
lean_object* v_options_3414_; lean_object* v_option_3415_; uint8_t v___x_3416_; lean_object* v___x_3417_; lean_object* v___x_3418_; 
v_options_3414_ = lean_ctor_get(v___y_3412_, 2);
v_option_3415_ = lean_ctor_get(v_opt_3411_, 1);
v___x_3416_ = lp_aesop_Lean_Option_get___at___00Aesop_ForwardRuleMatch_getProof_spec__12(v_options_3414_, v_option_3415_);
v___x_3417_ = lean_box(v___x_3416_);
v___x_3418_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3418_, 0, v___x_3417_);
return v___x_3418_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_ForwardRuleMatch_apply_spec__3___redArg___boxed(lean_object* v_opt_3419_, lean_object* v___y_3420_, lean_object* v___y_3421_){
_start:
{
lean_object* v_res_3422_; 
v_res_3422_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_ForwardRuleMatch_apply_spec__3___redArg(v_opt_3419_, v___y_3420_);
lean_dec_ref(v___y_3420_);
lean_dec_ref(v_opt_3419_);
return v_res_3422_;
}
}
static lean_object* _init_lp_aesop_Aesop_ForwardRuleMatch_apply___lam__1___closed__1(void){
_start:
{
lean_object* v___x_3424_; lean_object* v___x_3425_; 
v___x_3424_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_apply___lam__1___closed__0));
v___x_3425_ = l_Lean_stringToMessageData(v___x_3424_);
return v___x_3425_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_apply___lam__1(lean_object* v___x_3428_, uint8_t v_result_3429_, lean_object* v___y_3430_, lean_object* v___y_3431_, lean_object* v___y_3432_, lean_object* v___y_3433_, lean_object* v___y_3434_, lean_object* v___y_3435_){
_start:
{
lean_object* v___x_3437_; lean_object* v_a_3438_; lean_object* v___x_3440_; uint8_t v_isShared_3441_; uint8_t v_isSharedCheck_3481_; 
v___x_3437_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_ForwardRuleMatch_apply_spec__3___redArg(v___x_3428_, v___y_3434_);
v_a_3438_ = lean_ctor_get(v___x_3437_, 0);
v_isSharedCheck_3481_ = !lean_is_exclusive(v___x_3437_);
if (v_isSharedCheck_3481_ == 0)
{
v___x_3440_ = v___x_3437_;
v_isShared_3441_ = v_isSharedCheck_3481_;
goto v_resetjp_3439_;
}
else
{
lean_inc(v_a_3438_);
lean_dec(v___x_3437_);
v___x_3440_ = lean_box(0);
v_isShared_3441_ = v_isSharedCheck_3481_;
goto v_resetjp_3439_;
}
v_resetjp_3439_:
{
uint8_t v___x_3442_; 
v___x_3442_ = lean_unbox(v_a_3438_);
lean_dec(v_a_3438_);
if (v___x_3442_ == 0)
{
lean_object* v___x_3443_; lean_object* v___x_3445_; 
lean_dec_ref(v___x_3428_);
v___x_3443_ = lean_box(v_result_3429_);
if (v_isShared_3441_ == 0)
{
lean_ctor_set(v___x_3440_, 0, v___x_3443_);
v___x_3445_ = v___x_3440_;
goto v_reusejp_3444_;
}
else
{
lean_object* v_reuseFailAlloc_3446_; 
v_reuseFailAlloc_3446_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3446_, 0, v___x_3443_);
v___x_3445_ = v_reuseFailAlloc_3446_;
goto v_reusejp_3444_;
}
v_reusejp_3444_:
{
return v___x_3445_;
}
}
else
{
lean_object* v_traceClass_3447_; lean_object* v___x_3449_; uint8_t v_isShared_3450_; uint8_t v_isSharedCheck_3479_; 
lean_del_object(v___x_3440_);
v_traceClass_3447_ = lean_ctor_get(v___x_3428_, 0);
v_isSharedCheck_3479_ = !lean_is_exclusive(v___x_3428_);
if (v_isSharedCheck_3479_ == 0)
{
lean_object* v_unused_3480_; 
v_unused_3480_ = lean_ctor_get(v___x_3428_, 1);
lean_dec(v_unused_3480_);
v___x_3449_ = v___x_3428_;
v_isShared_3450_ = v_isSharedCheck_3479_;
goto v_resetjp_3448_;
}
else
{
lean_inc(v_traceClass_3447_);
lean_dec(v___x_3428_);
v___x_3449_ = lean_box(0);
v_isShared_3450_ = v_isSharedCheck_3479_;
goto v_resetjp_3448_;
}
v_resetjp_3448_:
{
lean_object* v___x_3451_; lean_object* v___y_3453_; 
v___x_3451_ = lean_obj_once(&lp_aesop_Aesop_ForwardRuleMatch_apply___lam__1___closed__1, &lp_aesop_Aesop_ForwardRuleMatch_apply___lam__1___closed__1_once, _init_lp_aesop_Aesop_ForwardRuleMatch_apply___lam__1___closed__1);
if (v_result_3429_ == 0)
{
lean_object* v___x_3477_; 
v___x_3477_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_apply___lam__1___closed__2));
v___y_3453_ = v___x_3477_;
goto v___jp_3452_;
}
else
{
lean_object* v___x_3478_; 
v___x_3478_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_apply___lam__1___closed__3));
v___y_3453_ = v___x_3478_;
goto v___jp_3452_;
}
v___jp_3452_:
{
lean_object* v___x_3454_; lean_object* v___x_3455_; lean_object* v___x_3457_; 
lean_inc_ref(v___y_3453_);
v___x_3454_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_3454_, 0, v___y_3453_);
v___x_3455_ = l_Lean_MessageData_ofFormat(v___x_3454_);
if (v_isShared_3450_ == 0)
{
lean_ctor_set_tag(v___x_3449_, 7);
lean_ctor_set(v___x_3449_, 1, v___x_3455_);
lean_ctor_set(v___x_3449_, 0, v___x_3451_);
v___x_3457_ = v___x_3449_;
goto v_reusejp_3456_;
}
else
{
lean_object* v_reuseFailAlloc_3476_; 
v_reuseFailAlloc_3476_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3476_, 0, v___x_3451_);
lean_ctor_set(v_reuseFailAlloc_3476_, 1, v___x_3455_);
v___x_3457_ = v_reuseFailAlloc_3476_;
goto v_reusejp_3456_;
}
v_reusejp_3456_:
{
lean_object* v___x_3458_; 
v___x_3458_ = lp_aesop_Lean_addTrace___at___00Aesop_ForwardRuleMatch_apply_spec__4___redArg(v_traceClass_3447_, v___x_3457_, v___y_3432_, v___y_3433_, v___y_3434_, v___y_3435_);
if (lean_obj_tag(v___x_3458_) == 0)
{
lean_object* v___x_3460_; uint8_t v_isShared_3461_; uint8_t v_isSharedCheck_3466_; 
v_isSharedCheck_3466_ = !lean_is_exclusive(v___x_3458_);
if (v_isSharedCheck_3466_ == 0)
{
lean_object* v_unused_3467_; 
v_unused_3467_ = lean_ctor_get(v___x_3458_, 0);
lean_dec(v_unused_3467_);
v___x_3460_ = v___x_3458_;
v_isShared_3461_ = v_isSharedCheck_3466_;
goto v_resetjp_3459_;
}
else
{
lean_dec(v___x_3458_);
v___x_3460_ = lean_box(0);
v_isShared_3461_ = v_isSharedCheck_3466_;
goto v_resetjp_3459_;
}
v_resetjp_3459_:
{
lean_object* v___x_3462_; lean_object* v___x_3464_; 
v___x_3462_ = lean_box(v_result_3429_);
if (v_isShared_3461_ == 0)
{
lean_ctor_set(v___x_3460_, 0, v___x_3462_);
v___x_3464_ = v___x_3460_;
goto v_reusejp_3463_;
}
else
{
lean_object* v_reuseFailAlloc_3465_; 
v_reuseFailAlloc_3465_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3465_, 0, v___x_3462_);
v___x_3464_ = v_reuseFailAlloc_3465_;
goto v_reusejp_3463_;
}
v_reusejp_3463_:
{
return v___x_3464_;
}
}
}
else
{
lean_object* v_a_3468_; lean_object* v___x_3470_; uint8_t v_isShared_3471_; uint8_t v_isSharedCheck_3475_; 
v_a_3468_ = lean_ctor_get(v___x_3458_, 0);
v_isSharedCheck_3475_ = !lean_is_exclusive(v___x_3458_);
if (v_isSharedCheck_3475_ == 0)
{
v___x_3470_ = v___x_3458_;
v_isShared_3471_ = v_isSharedCheck_3475_;
goto v_resetjp_3469_;
}
else
{
lean_inc(v_a_3468_);
lean_dec(v___x_3458_);
v___x_3470_ = lean_box(0);
v_isShared_3471_ = v_isSharedCheck_3475_;
goto v_resetjp_3469_;
}
v_resetjp_3469_:
{
lean_object* v___x_3473_; 
if (v_isShared_3471_ == 0)
{
v___x_3473_ = v___x_3470_;
goto v_reusejp_3472_;
}
else
{
lean_object* v_reuseFailAlloc_3474_; 
v_reuseFailAlloc_3474_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3474_, 0, v_a_3468_);
v___x_3473_ = v_reuseFailAlloc_3474_;
goto v_reusejp_3472_;
}
v_reusejp_3472_:
{
return v___x_3473_;
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
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_apply___lam__1___boxed(lean_object* v___x_3482_, lean_object* v_result_3483_, lean_object* v___y_3484_, lean_object* v___y_3485_, lean_object* v___y_3486_, lean_object* v___y_3487_, lean_object* v___y_3488_, lean_object* v___y_3489_, lean_object* v___y_3490_){
_start:
{
uint8_t v_result_boxed_3491_; lean_object* v_res_3492_; 
v_result_boxed_3491_ = lean_unbox(v_result_3483_);
v_res_3492_ = lp_aesop_Aesop_ForwardRuleMatch_apply___lam__1(v___x_3482_, v_result_boxed_3491_, v___y_3484_, v___y_3485_, v___y_3486_, v___y_3487_, v___y_3488_, v___y_3489_);
lean_dec(v___y_3489_);
lean_dec_ref(v___y_3488_);
lean_dec(v___y_3487_);
lean_dec_ref(v___y_3486_);
lean_dec(v___y_3485_);
lean_dec(v___y_3484_);
return v_res_3492_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_apply___lam__2(lean_object* v___x_3493_, lean_object* v_x_3494_, lean_object* v___y_3495_, lean_object* v___y_3496_, lean_object* v___y_3497_, lean_object* v___y_3498_, lean_object* v___y_3499_, lean_object* v___y_3500_){
_start:
{
lean_object* v___x_3502_; 
v___x_3502_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3502_, 0, v___x_3493_);
return v___x_3502_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_apply___lam__2___boxed(lean_object* v___x_3503_, lean_object* v_x_3504_, lean_object* v___y_3505_, lean_object* v___y_3506_, lean_object* v___y_3507_, lean_object* v___y_3508_, lean_object* v___y_3509_, lean_object* v___y_3510_, lean_object* v___y_3511_){
_start:
{
lean_object* v_res_3512_; 
v_res_3512_ = lp_aesop_Aesop_ForwardRuleMatch_apply___lam__2(v___x_3503_, v_x_3504_, v___y_3505_, v___y_3506_, v___y_3507_, v___y_3508_, v___y_3509_, v___y_3510_);
lean_dec(v___y_3510_);
lean_dec_ref(v___y_3509_);
lean_dec(v___y_3508_);
lean_dec_ref(v___y_3507_);
lean_dec(v___y_3506_);
lean_dec(v___y_3505_);
lean_dec_ref(v_x_3504_);
return v_res_3512_;
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_apply_spec__6_spec__7___redArg(lean_object* v_x_3513_){
_start:
{
if (lean_obj_tag(v_x_3513_) == 0)
{
lean_object* v_a_3515_; lean_object* v___x_3517_; uint8_t v_isShared_3518_; uint8_t v_isSharedCheck_3522_; 
v_a_3515_ = lean_ctor_get(v_x_3513_, 0);
v_isSharedCheck_3522_ = !lean_is_exclusive(v_x_3513_);
if (v_isSharedCheck_3522_ == 0)
{
v___x_3517_ = v_x_3513_;
v_isShared_3518_ = v_isSharedCheck_3522_;
goto v_resetjp_3516_;
}
else
{
lean_inc(v_a_3515_);
lean_dec(v_x_3513_);
v___x_3517_ = lean_box(0);
v_isShared_3518_ = v_isSharedCheck_3522_;
goto v_resetjp_3516_;
}
v_resetjp_3516_:
{
lean_object* v___x_3520_; 
if (v_isShared_3518_ == 0)
{
lean_ctor_set_tag(v___x_3517_, 1);
v___x_3520_ = v___x_3517_;
goto v_reusejp_3519_;
}
else
{
lean_object* v_reuseFailAlloc_3521_; 
v_reuseFailAlloc_3521_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3521_, 0, v_a_3515_);
v___x_3520_ = v_reuseFailAlloc_3521_;
goto v_reusejp_3519_;
}
v_reusejp_3519_:
{
return v___x_3520_;
}
}
}
else
{
lean_object* v_a_3523_; lean_object* v___x_3525_; uint8_t v_isShared_3526_; uint8_t v_isSharedCheck_3530_; 
v_a_3523_ = lean_ctor_get(v_x_3513_, 0);
v_isSharedCheck_3530_ = !lean_is_exclusive(v_x_3513_);
if (v_isSharedCheck_3530_ == 0)
{
v___x_3525_ = v_x_3513_;
v_isShared_3526_ = v_isSharedCheck_3530_;
goto v_resetjp_3524_;
}
else
{
lean_inc(v_a_3523_);
lean_dec(v_x_3513_);
v___x_3525_ = lean_box(0);
v_isShared_3526_ = v_isSharedCheck_3530_;
goto v_resetjp_3524_;
}
v_resetjp_3524_:
{
lean_object* v___x_3528_; 
if (v_isShared_3526_ == 0)
{
lean_ctor_set_tag(v___x_3525_, 0);
v___x_3528_ = v___x_3525_;
goto v_reusejp_3527_;
}
else
{
lean_object* v_reuseFailAlloc_3529_; 
v_reuseFailAlloc_3529_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3529_, 0, v_a_3523_);
v___x_3528_ = v_reuseFailAlloc_3529_;
goto v_reusejp_3527_;
}
v_reusejp_3527_:
{
return v___x_3528_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_apply_spec__6_spec__7___redArg___boxed(lean_object* v_x_3531_, lean_object* v___y_3532_){
_start:
{
lean_object* v_res_3533_; 
v_res_3533_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_apply_spec__6_spec__7___redArg(v_x_3531_);
return v_res_3533_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_apply_spec__6_spec__8(lean_object* v_e_3534_){
_start:
{
if (lean_obj_tag(v_e_3534_) == 0)
{
uint8_t v___x_3535_; 
v___x_3535_ = 2;
return v___x_3535_;
}
else
{
lean_object* v_a_3536_; uint8_t v___x_3537_; 
v_a_3536_ = lean_ctor_get(v_e_3534_, 0);
v___x_3537_ = lean_unbox(v_a_3536_);
if (v___x_3537_ == 0)
{
uint8_t v___x_3538_; 
v___x_3538_ = 1;
return v___x_3538_;
}
else
{
uint8_t v___x_3539_; 
v___x_3539_ = 0;
return v___x_3539_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_apply_spec__6_spec__8___boxed(lean_object* v_e_3540_){
_start:
{
uint8_t v_res_3541_; lean_object* v_r_3542_; 
v_res_3541_ = lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_apply_spec__6_spec__8(v_e_3540_);
lean_dec_ref(v_e_3540_);
v_r_3542_ = lean_box(v_res_3541_);
return v_r_3542_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_apply_spec__6_spec__6___redArg(lean_object* v_oldTraces_3543_, lean_object* v_data_3544_, lean_object* v_ref_3545_, lean_object* v_msg_3546_, lean_object* v___y_3547_, lean_object* v___y_3548_, lean_object* v___y_3549_, lean_object* v___y_3550_){
_start:
{
lean_object* v_fileName_3552_; lean_object* v_fileMap_3553_; lean_object* v_options_3554_; lean_object* v_currRecDepth_3555_; lean_object* v_maxRecDepth_3556_; lean_object* v_ref_3557_; lean_object* v_currNamespace_3558_; lean_object* v_openDecls_3559_; lean_object* v_initHeartbeats_3560_; lean_object* v_maxHeartbeats_3561_; lean_object* v_quotContext_3562_; lean_object* v_currMacroScope_3563_; uint8_t v_diag_3564_; lean_object* v_cancelTk_x3f_3565_; uint8_t v_suppressElabErrors_3566_; lean_object* v_inheritedTraceOptions_3567_; lean_object* v___x_3568_; lean_object* v_traceState_3569_; lean_object* v_traces_3570_; lean_object* v_ref_3571_; lean_object* v___x_3572_; lean_object* v___x_3573_; size_t v_sz_3574_; size_t v___x_3575_; lean_object* v___x_3576_; lean_object* v_msg_3577_; lean_object* v___x_3578_; lean_object* v_a_3579_; lean_object* v___x_3581_; uint8_t v_isShared_3582_; uint8_t v_isSharedCheck_3616_; 
v_fileName_3552_ = lean_ctor_get(v___y_3549_, 0);
v_fileMap_3553_ = lean_ctor_get(v___y_3549_, 1);
v_options_3554_ = lean_ctor_get(v___y_3549_, 2);
v_currRecDepth_3555_ = lean_ctor_get(v___y_3549_, 3);
v_maxRecDepth_3556_ = lean_ctor_get(v___y_3549_, 4);
v_ref_3557_ = lean_ctor_get(v___y_3549_, 5);
v_currNamespace_3558_ = lean_ctor_get(v___y_3549_, 6);
v_openDecls_3559_ = lean_ctor_get(v___y_3549_, 7);
v_initHeartbeats_3560_ = lean_ctor_get(v___y_3549_, 8);
v_maxHeartbeats_3561_ = lean_ctor_get(v___y_3549_, 9);
v_quotContext_3562_ = lean_ctor_get(v___y_3549_, 10);
v_currMacroScope_3563_ = lean_ctor_get(v___y_3549_, 11);
v_diag_3564_ = lean_ctor_get_uint8(v___y_3549_, sizeof(void*)*14);
v_cancelTk_x3f_3565_ = lean_ctor_get(v___y_3549_, 12);
v_suppressElabErrors_3566_ = lean_ctor_get_uint8(v___y_3549_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_3567_ = lean_ctor_get(v___y_3549_, 13);
v___x_3568_ = lean_st_ref_get(v___y_3550_);
v_traceState_3569_ = lean_ctor_get(v___x_3568_, 4);
lean_inc_ref(v_traceState_3569_);
lean_dec(v___x_3568_);
v_traces_3570_ = lean_ctor_get(v_traceState_3569_, 0);
lean_inc_ref(v_traces_3570_);
lean_dec_ref(v_traceState_3569_);
v_ref_3571_ = l_Lean_replaceRef(v_ref_3545_, v_ref_3557_);
lean_inc_ref(v_inheritedTraceOptions_3567_);
lean_inc(v_cancelTk_x3f_3565_);
lean_inc(v_currMacroScope_3563_);
lean_inc(v_quotContext_3562_);
lean_inc(v_maxHeartbeats_3561_);
lean_inc(v_initHeartbeats_3560_);
lean_inc(v_openDecls_3559_);
lean_inc(v_currNamespace_3558_);
lean_inc(v_maxRecDepth_3556_);
lean_inc(v_currRecDepth_3555_);
lean_inc_ref(v_options_3554_);
lean_inc_ref(v_fileMap_3553_);
lean_inc_ref(v_fileName_3552_);
v___x_3572_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_3572_, 0, v_fileName_3552_);
lean_ctor_set(v___x_3572_, 1, v_fileMap_3553_);
lean_ctor_set(v___x_3572_, 2, v_options_3554_);
lean_ctor_set(v___x_3572_, 3, v_currRecDepth_3555_);
lean_ctor_set(v___x_3572_, 4, v_maxRecDepth_3556_);
lean_ctor_set(v___x_3572_, 5, v_ref_3571_);
lean_ctor_set(v___x_3572_, 6, v_currNamespace_3558_);
lean_ctor_set(v___x_3572_, 7, v_openDecls_3559_);
lean_ctor_set(v___x_3572_, 8, v_initHeartbeats_3560_);
lean_ctor_set(v___x_3572_, 9, v_maxHeartbeats_3561_);
lean_ctor_set(v___x_3572_, 10, v_quotContext_3562_);
lean_ctor_set(v___x_3572_, 11, v_currMacroScope_3563_);
lean_ctor_set(v___x_3572_, 12, v_cancelTk_x3f_3565_);
lean_ctor_set(v___x_3572_, 13, v_inheritedTraceOptions_3567_);
lean_ctor_set_uint8(v___x_3572_, sizeof(void*)*14, v_diag_3564_);
lean_ctor_set_uint8(v___x_3572_, sizeof(void*)*14 + 1, v_suppressElabErrors_3566_);
v___x_3573_ = l_Lean_PersistentArray_toArray___redArg(v_traces_3570_);
lean_dec_ref(v_traces_3570_);
v_sz_3574_ = lean_array_size(v___x_3573_);
v___x_3575_ = ((size_t)0ULL);
v___x_3576_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13_spec__16_spec__19(v_sz_3574_, v___x_3575_, v___x_3573_);
v_msg_3577_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v_msg_3577_, 0, v_data_3544_);
lean_ctor_set(v_msg_3577_, 1, v_msg_3546_);
lean_ctor_set(v_msg_3577_, 2, v___x_3576_);
v___x_3578_ = lp_aesop_Lean_addMessageContextFull___at___00Lean_addTrace___at___00Aesop_ForwardRuleMatch_getProof_spec__4_spec__6(v_msg_3577_, v___y_3547_, v___y_3548_, v___x_3572_, v___y_3550_);
lean_dec_ref_known(v___x_3572_, 14);
v_a_3579_ = lean_ctor_get(v___x_3578_, 0);
v_isSharedCheck_3616_ = !lean_is_exclusive(v___x_3578_);
if (v_isSharedCheck_3616_ == 0)
{
v___x_3581_ = v___x_3578_;
v_isShared_3582_ = v_isSharedCheck_3616_;
goto v_resetjp_3580_;
}
else
{
lean_inc(v_a_3579_);
lean_dec(v___x_3578_);
v___x_3581_ = lean_box(0);
v_isShared_3582_ = v_isSharedCheck_3616_;
goto v_resetjp_3580_;
}
v_resetjp_3580_:
{
lean_object* v___x_3583_; lean_object* v_traceState_3584_; lean_object* v_env_3585_; lean_object* v_nextMacroScope_3586_; lean_object* v_ngen_3587_; lean_object* v_auxDeclNGen_3588_; lean_object* v_cache_3589_; lean_object* v_messages_3590_; lean_object* v_infoState_3591_; lean_object* v_snapshotTasks_3592_; lean_object* v___x_3594_; uint8_t v_isShared_3595_; uint8_t v_isSharedCheck_3615_; 
v___x_3583_ = lean_st_ref_take(v___y_3550_);
v_traceState_3584_ = lean_ctor_get(v___x_3583_, 4);
v_env_3585_ = lean_ctor_get(v___x_3583_, 0);
v_nextMacroScope_3586_ = lean_ctor_get(v___x_3583_, 1);
v_ngen_3587_ = lean_ctor_get(v___x_3583_, 2);
v_auxDeclNGen_3588_ = lean_ctor_get(v___x_3583_, 3);
v_cache_3589_ = lean_ctor_get(v___x_3583_, 5);
v_messages_3590_ = lean_ctor_get(v___x_3583_, 6);
v_infoState_3591_ = lean_ctor_get(v___x_3583_, 7);
v_snapshotTasks_3592_ = lean_ctor_get(v___x_3583_, 8);
v_isSharedCheck_3615_ = !lean_is_exclusive(v___x_3583_);
if (v_isSharedCheck_3615_ == 0)
{
v___x_3594_ = v___x_3583_;
v_isShared_3595_ = v_isSharedCheck_3615_;
goto v_resetjp_3593_;
}
else
{
lean_inc(v_snapshotTasks_3592_);
lean_inc(v_infoState_3591_);
lean_inc(v_messages_3590_);
lean_inc(v_cache_3589_);
lean_inc(v_traceState_3584_);
lean_inc(v_auxDeclNGen_3588_);
lean_inc(v_ngen_3587_);
lean_inc(v_nextMacroScope_3586_);
lean_inc(v_env_3585_);
lean_dec(v___x_3583_);
v___x_3594_ = lean_box(0);
v_isShared_3595_ = v_isSharedCheck_3615_;
goto v_resetjp_3593_;
}
v_resetjp_3593_:
{
uint64_t v_tid_3596_; lean_object* v___x_3598_; uint8_t v_isShared_3599_; uint8_t v_isSharedCheck_3613_; 
v_tid_3596_ = lean_ctor_get_uint64(v_traceState_3584_, sizeof(void*)*1);
v_isSharedCheck_3613_ = !lean_is_exclusive(v_traceState_3584_);
if (v_isSharedCheck_3613_ == 0)
{
lean_object* v_unused_3614_; 
v_unused_3614_ = lean_ctor_get(v_traceState_3584_, 0);
lean_dec(v_unused_3614_);
v___x_3598_ = v_traceState_3584_;
v_isShared_3599_ = v_isSharedCheck_3613_;
goto v_resetjp_3597_;
}
else
{
lean_dec(v_traceState_3584_);
v___x_3598_ = lean_box(0);
v_isShared_3599_ = v_isSharedCheck_3613_;
goto v_resetjp_3597_;
}
v_resetjp_3597_:
{
lean_object* v___x_3600_; lean_object* v___x_3601_; lean_object* v___x_3603_; 
v___x_3600_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3600_, 0, v_ref_3545_);
lean_ctor_set(v___x_3600_, 1, v_a_3579_);
v___x_3601_ = l_Lean_PersistentArray_push___redArg(v_oldTraces_3543_, v___x_3600_);
if (v_isShared_3599_ == 0)
{
lean_ctor_set(v___x_3598_, 0, v___x_3601_);
v___x_3603_ = v___x_3598_;
goto v_reusejp_3602_;
}
else
{
lean_object* v_reuseFailAlloc_3612_; 
v_reuseFailAlloc_3612_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_3612_, 0, v___x_3601_);
lean_ctor_set_uint64(v_reuseFailAlloc_3612_, sizeof(void*)*1, v_tid_3596_);
v___x_3603_ = v_reuseFailAlloc_3612_;
goto v_reusejp_3602_;
}
v_reusejp_3602_:
{
lean_object* v___x_3605_; 
if (v_isShared_3595_ == 0)
{
lean_ctor_set(v___x_3594_, 4, v___x_3603_);
v___x_3605_ = v___x_3594_;
goto v_reusejp_3604_;
}
else
{
lean_object* v_reuseFailAlloc_3611_; 
v_reuseFailAlloc_3611_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_3611_, 0, v_env_3585_);
lean_ctor_set(v_reuseFailAlloc_3611_, 1, v_nextMacroScope_3586_);
lean_ctor_set(v_reuseFailAlloc_3611_, 2, v_ngen_3587_);
lean_ctor_set(v_reuseFailAlloc_3611_, 3, v_auxDeclNGen_3588_);
lean_ctor_set(v_reuseFailAlloc_3611_, 4, v___x_3603_);
lean_ctor_set(v_reuseFailAlloc_3611_, 5, v_cache_3589_);
lean_ctor_set(v_reuseFailAlloc_3611_, 6, v_messages_3590_);
lean_ctor_set(v_reuseFailAlloc_3611_, 7, v_infoState_3591_);
lean_ctor_set(v_reuseFailAlloc_3611_, 8, v_snapshotTasks_3592_);
v___x_3605_ = v_reuseFailAlloc_3611_;
goto v_reusejp_3604_;
}
v_reusejp_3604_:
{
lean_object* v___x_3606_; lean_object* v___x_3607_; lean_object* v___x_3609_; 
v___x_3606_ = lean_st_ref_set(v___y_3550_, v___x_3605_);
v___x_3607_ = lean_box(0);
if (v_isShared_3582_ == 0)
{
lean_ctor_set(v___x_3581_, 0, v___x_3607_);
v___x_3609_ = v___x_3581_;
goto v_reusejp_3608_;
}
else
{
lean_object* v_reuseFailAlloc_3610_; 
v_reuseFailAlloc_3610_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3610_, 0, v___x_3607_);
v___x_3609_ = v_reuseFailAlloc_3610_;
goto v_reusejp_3608_;
}
v_reusejp_3608_:
{
return v___x_3609_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_apply_spec__6_spec__6___redArg___boxed(lean_object* v_oldTraces_3617_, lean_object* v_data_3618_, lean_object* v_ref_3619_, lean_object* v_msg_3620_, lean_object* v___y_3621_, lean_object* v___y_3622_, lean_object* v___y_3623_, lean_object* v___y_3624_, lean_object* v___y_3625_){
_start:
{
lean_object* v_res_3626_; 
v_res_3626_ = lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_apply_spec__6_spec__6___redArg(v_oldTraces_3617_, v_data_3618_, v_ref_3619_, v_msg_3620_, v___y_3621_, v___y_3622_, v___y_3623_, v___y_3624_);
lean_dec(v___y_3624_);
lean_dec_ref(v___y_3623_);
lean_dec(v___y_3622_);
lean_dec_ref(v___y_3621_);
return v_res_3626_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_apply_spec__6(lean_object* v_cls_3627_, uint8_t v_collapsed_3628_, lean_object* v_tag_3629_, lean_object* v_opts_3630_, uint8_t v_clsEnabled_3631_, lean_object* v_oldTraces_3632_, lean_object* v_msg_3633_, lean_object* v_resStartStop_3634_, lean_object* v___y_3635_, lean_object* v___y_3636_, lean_object* v___y_3637_, lean_object* v___y_3638_, lean_object* v___y_3639_, lean_object* v___y_3640_){
_start:
{
lean_object* v_fst_3642_; lean_object* v_snd_3643_; lean_object* v___y_3645_; lean_object* v___y_3646_; lean_object* v_data_3647_; lean_object* v_fst_3658_; lean_object* v_snd_3659_; lean_object* v___x_3660_; uint8_t v___x_3661_; lean_object* v___y_3663_; lean_object* v_a_3664_; uint8_t v___y_3679_; double v___y_3710_; 
v_fst_3642_ = lean_ctor_get(v_resStartStop_3634_, 0);
lean_inc(v_fst_3642_);
v_snd_3643_ = lean_ctor_get(v_resStartStop_3634_, 1);
lean_inc(v_snd_3643_);
lean_dec_ref(v_resStartStop_3634_);
v_fst_3658_ = lean_ctor_get(v_snd_3643_, 0);
lean_inc(v_fst_3658_);
v_snd_3659_ = lean_ctor_get(v_snd_3643_, 1);
lean_inc(v_snd_3659_);
lean_dec(v_snd_3643_);
v___x_3660_ = l_Lean_trace_profiler;
v___x_3661_ = lp_aesop_Lean_Option_get___at___00Aesop_ForwardRuleMatch_getProof_spec__12(v_opts_3630_, v___x_3660_);
if (v___x_3661_ == 0)
{
v___y_3679_ = v___x_3661_;
goto v___jp_3678_;
}
else
{
lean_object* v___x_3715_; uint8_t v___x_3716_; 
v___x_3715_ = l_Lean_trace_profiler_useHeartbeats;
v___x_3716_ = lp_aesop_Lean_Option_get___at___00Aesop_ForwardRuleMatch_getProof_spec__12(v_opts_3630_, v___x_3715_);
if (v___x_3716_ == 0)
{
lean_object* v___x_3717_; lean_object* v___x_3718_; double v___x_3719_; double v___x_3720_; double v___x_3721_; 
v___x_3717_ = l_Lean_trace_profiler_threshold;
v___x_3718_ = lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13_spec__19(v_opts_3630_, v___x_3717_);
v___x_3719_ = lean_float_of_nat(v___x_3718_);
v___x_3720_ = lean_float_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13___closed__2, &lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13___closed__2_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13___closed__2);
v___x_3721_ = lean_float_div(v___x_3719_, v___x_3720_);
v___y_3710_ = v___x_3721_;
goto v___jp_3709_;
}
else
{
lean_object* v___x_3722_; lean_object* v___x_3723_; double v___x_3724_; 
v___x_3722_ = l_Lean_trace_profiler_threshold;
v___x_3723_ = lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13_spec__19(v_opts_3630_, v___x_3722_);
v___x_3724_ = lean_float_of_nat(v___x_3723_);
v___y_3710_ = v___x_3724_;
goto v___jp_3709_;
}
}
v___jp_3644_:
{
lean_object* v___x_3648_; 
lean_inc(v___y_3646_);
v___x_3648_ = lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_apply_spec__6_spec__6___redArg(v_oldTraces_3632_, v_data_3647_, v___y_3646_, v___y_3645_, v___y_3637_, v___y_3638_, v___y_3639_, v___y_3640_);
if (lean_obj_tag(v___x_3648_) == 0)
{
lean_object* v___x_3649_; 
lean_dec_ref_known(v___x_3648_, 1);
v___x_3649_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_apply_spec__6_spec__7___redArg(v_fst_3642_);
return v___x_3649_;
}
else
{
lean_object* v_a_3650_; lean_object* v___x_3652_; uint8_t v_isShared_3653_; uint8_t v_isSharedCheck_3657_; 
lean_dec(v_fst_3642_);
v_a_3650_ = lean_ctor_get(v___x_3648_, 0);
v_isSharedCheck_3657_ = !lean_is_exclusive(v___x_3648_);
if (v_isSharedCheck_3657_ == 0)
{
v___x_3652_ = v___x_3648_;
v_isShared_3653_ = v_isSharedCheck_3657_;
goto v_resetjp_3651_;
}
else
{
lean_inc(v_a_3650_);
lean_dec(v___x_3648_);
v___x_3652_ = lean_box(0);
v_isShared_3653_ = v_isSharedCheck_3657_;
goto v_resetjp_3651_;
}
v_resetjp_3651_:
{
lean_object* v___x_3655_; 
if (v_isShared_3653_ == 0)
{
v___x_3655_ = v___x_3652_;
goto v_reusejp_3654_;
}
else
{
lean_object* v_reuseFailAlloc_3656_; 
v_reuseFailAlloc_3656_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3656_, 0, v_a_3650_);
v___x_3655_ = v_reuseFailAlloc_3656_;
goto v_reusejp_3654_;
}
v_reusejp_3654_:
{
return v___x_3655_;
}
}
}
}
v___jp_3662_:
{
uint8_t v_result_3665_; lean_object* v___x_3666_; lean_object* v___x_3667_; double v___x_3668_; lean_object* v_data_3669_; 
v_result_3665_ = lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_apply_spec__6_spec__8(v_fst_3642_);
v___x_3666_ = lean_box(v_result_3665_);
v___x_3667_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3667_, 0, v___x_3666_);
v___x_3668_ = lean_float_once(&lp_aesop_Lean_addTrace___at___00Aesop_ForwardRuleMatch_getProof_spec__4___closed__0, &lp_aesop_Lean_addTrace___at___00Aesop_ForwardRuleMatch_getProof_spec__4___closed__0_once, _init_lp_aesop_Lean_addTrace___at___00Aesop_ForwardRuleMatch_getProof_spec__4___closed__0);
lean_inc_ref(v_tag_3629_);
lean_inc_ref(v___x_3667_);
lean_inc(v_cls_3627_);
v_data_3669_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v_data_3669_, 0, v_cls_3627_);
lean_ctor_set(v_data_3669_, 1, v___x_3667_);
lean_ctor_set(v_data_3669_, 2, v_tag_3629_);
lean_ctor_set_float(v_data_3669_, sizeof(void*)*3, v___x_3668_);
lean_ctor_set_float(v_data_3669_, sizeof(void*)*3 + 8, v___x_3668_);
lean_ctor_set_uint8(v_data_3669_, sizeof(void*)*3 + 16, v_collapsed_3628_);
if (v___x_3661_ == 0)
{
lean_dec_ref_known(v___x_3667_, 1);
lean_dec(v_snd_3659_);
lean_dec(v_fst_3658_);
lean_dec_ref(v_tag_3629_);
lean_dec(v_cls_3627_);
v___y_3645_ = v_a_3664_;
v___y_3646_ = v___y_3663_;
v_data_3647_ = v_data_3669_;
goto v___jp_3644_;
}
else
{
lean_object* v_data_3670_; double v___x_3671_; double v___x_3672_; 
lean_dec_ref_known(v_data_3669_, 3);
v_data_3670_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v_data_3670_, 0, v_cls_3627_);
lean_ctor_set(v_data_3670_, 1, v___x_3667_);
lean_ctor_set(v_data_3670_, 2, v_tag_3629_);
v___x_3671_ = lean_unbox_float(v_fst_3658_);
lean_dec(v_fst_3658_);
lean_ctor_set_float(v_data_3670_, sizeof(void*)*3, v___x_3671_);
v___x_3672_ = lean_unbox_float(v_snd_3659_);
lean_dec(v_snd_3659_);
lean_ctor_set_float(v_data_3670_, sizeof(void*)*3 + 8, v___x_3672_);
lean_ctor_set_uint8(v_data_3670_, sizeof(void*)*3 + 16, v_collapsed_3628_);
v___y_3645_ = v_a_3664_;
v___y_3646_ = v___y_3663_;
v_data_3647_ = v_data_3670_;
goto v___jp_3644_;
}
}
v___jp_3673_:
{
lean_object* v_ref_3674_; lean_object* v___x_3675_; 
v_ref_3674_ = lean_ctor_get(v___y_3639_, 5);
lean_inc(v___y_3640_);
lean_inc_ref(v___y_3639_);
lean_inc(v___y_3638_);
lean_inc_ref(v___y_3637_);
lean_inc(v___y_3636_);
lean_inc(v___y_3635_);
lean_inc(v_fst_3642_);
v___x_3675_ = lean_apply_8(v_msg_3633_, v_fst_3642_, v___y_3635_, v___y_3636_, v___y_3637_, v___y_3638_, v___y_3639_, v___y_3640_, lean_box(0));
if (lean_obj_tag(v___x_3675_) == 0)
{
lean_object* v_a_3676_; 
v_a_3676_ = lean_ctor_get(v___x_3675_, 0);
lean_inc(v_a_3676_);
lean_dec_ref_known(v___x_3675_, 1);
v___y_3663_ = v_ref_3674_;
v_a_3664_ = v_a_3676_;
goto v___jp_3662_;
}
else
{
lean_object* v___x_3677_; 
lean_dec_ref_known(v___x_3675_, 1);
v___x_3677_ = lean_obj_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13___closed__1, &lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13___closed__1_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13___closed__1);
v___y_3663_ = v_ref_3674_;
v_a_3664_ = v___x_3677_;
goto v___jp_3662_;
}
}
v___jp_3678_:
{
if (v_clsEnabled_3631_ == 0)
{
if (v___y_3679_ == 0)
{
lean_object* v___x_3680_; lean_object* v_traceState_3681_; lean_object* v_env_3682_; lean_object* v_nextMacroScope_3683_; lean_object* v_ngen_3684_; lean_object* v_auxDeclNGen_3685_; lean_object* v_cache_3686_; lean_object* v_messages_3687_; lean_object* v_infoState_3688_; lean_object* v_snapshotTasks_3689_; lean_object* v___x_3691_; uint8_t v_isShared_3692_; uint8_t v_isSharedCheck_3708_; 
lean_dec(v_snd_3659_);
lean_dec(v_fst_3658_);
lean_dec_ref(v_msg_3633_);
lean_dec_ref(v_tag_3629_);
lean_dec(v_cls_3627_);
v___x_3680_ = lean_st_ref_take(v___y_3640_);
v_traceState_3681_ = lean_ctor_get(v___x_3680_, 4);
v_env_3682_ = lean_ctor_get(v___x_3680_, 0);
v_nextMacroScope_3683_ = lean_ctor_get(v___x_3680_, 1);
v_ngen_3684_ = lean_ctor_get(v___x_3680_, 2);
v_auxDeclNGen_3685_ = lean_ctor_get(v___x_3680_, 3);
v_cache_3686_ = lean_ctor_get(v___x_3680_, 5);
v_messages_3687_ = lean_ctor_get(v___x_3680_, 6);
v_infoState_3688_ = lean_ctor_get(v___x_3680_, 7);
v_snapshotTasks_3689_ = lean_ctor_get(v___x_3680_, 8);
v_isSharedCheck_3708_ = !lean_is_exclusive(v___x_3680_);
if (v_isSharedCheck_3708_ == 0)
{
v___x_3691_ = v___x_3680_;
v_isShared_3692_ = v_isSharedCheck_3708_;
goto v_resetjp_3690_;
}
else
{
lean_inc(v_snapshotTasks_3689_);
lean_inc(v_infoState_3688_);
lean_inc(v_messages_3687_);
lean_inc(v_cache_3686_);
lean_inc(v_traceState_3681_);
lean_inc(v_auxDeclNGen_3685_);
lean_inc(v_ngen_3684_);
lean_inc(v_nextMacroScope_3683_);
lean_inc(v_env_3682_);
lean_dec(v___x_3680_);
v___x_3691_ = lean_box(0);
v_isShared_3692_ = v_isSharedCheck_3708_;
goto v_resetjp_3690_;
}
v_resetjp_3690_:
{
uint64_t v_tid_3693_; lean_object* v_traces_3694_; lean_object* v___x_3696_; uint8_t v_isShared_3697_; uint8_t v_isSharedCheck_3707_; 
v_tid_3693_ = lean_ctor_get_uint64(v_traceState_3681_, sizeof(void*)*1);
v_traces_3694_ = lean_ctor_get(v_traceState_3681_, 0);
v_isSharedCheck_3707_ = !lean_is_exclusive(v_traceState_3681_);
if (v_isSharedCheck_3707_ == 0)
{
v___x_3696_ = v_traceState_3681_;
v_isShared_3697_ = v_isSharedCheck_3707_;
goto v_resetjp_3695_;
}
else
{
lean_inc(v_traces_3694_);
lean_dec(v_traceState_3681_);
v___x_3696_ = lean_box(0);
v_isShared_3697_ = v_isSharedCheck_3707_;
goto v_resetjp_3695_;
}
v_resetjp_3695_:
{
lean_object* v___x_3698_; lean_object* v___x_3700_; 
v___x_3698_ = l_Lean_PersistentArray_append___redArg(v_oldTraces_3632_, v_traces_3694_);
lean_dec_ref(v_traces_3694_);
if (v_isShared_3697_ == 0)
{
lean_ctor_set(v___x_3696_, 0, v___x_3698_);
v___x_3700_ = v___x_3696_;
goto v_reusejp_3699_;
}
else
{
lean_object* v_reuseFailAlloc_3706_; 
v_reuseFailAlloc_3706_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_3706_, 0, v___x_3698_);
lean_ctor_set_uint64(v_reuseFailAlloc_3706_, sizeof(void*)*1, v_tid_3693_);
v___x_3700_ = v_reuseFailAlloc_3706_;
goto v_reusejp_3699_;
}
v_reusejp_3699_:
{
lean_object* v___x_3702_; 
if (v_isShared_3692_ == 0)
{
lean_ctor_set(v___x_3691_, 4, v___x_3700_);
v___x_3702_ = v___x_3691_;
goto v_reusejp_3701_;
}
else
{
lean_object* v_reuseFailAlloc_3705_; 
v_reuseFailAlloc_3705_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_3705_, 0, v_env_3682_);
lean_ctor_set(v_reuseFailAlloc_3705_, 1, v_nextMacroScope_3683_);
lean_ctor_set(v_reuseFailAlloc_3705_, 2, v_ngen_3684_);
lean_ctor_set(v_reuseFailAlloc_3705_, 3, v_auxDeclNGen_3685_);
lean_ctor_set(v_reuseFailAlloc_3705_, 4, v___x_3700_);
lean_ctor_set(v_reuseFailAlloc_3705_, 5, v_cache_3686_);
lean_ctor_set(v_reuseFailAlloc_3705_, 6, v_messages_3687_);
lean_ctor_set(v_reuseFailAlloc_3705_, 7, v_infoState_3688_);
lean_ctor_set(v_reuseFailAlloc_3705_, 8, v_snapshotTasks_3689_);
v___x_3702_ = v_reuseFailAlloc_3705_;
goto v_reusejp_3701_;
}
v_reusejp_3701_:
{
lean_object* v___x_3703_; lean_object* v___x_3704_; 
v___x_3703_ = lean_st_ref_set(v___y_3640_, v___x_3702_);
v___x_3704_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_apply_spec__6_spec__7___redArg(v_fst_3642_);
return v___x_3704_;
}
}
}
}
}
else
{
goto v___jp_3673_;
}
}
else
{
goto v___jp_3673_;
}
}
v___jp_3709_:
{
double v___x_3711_; double v___x_3712_; double v___x_3713_; uint8_t v___x_3714_; 
v___x_3711_ = lean_unbox_float(v_snd_3659_);
v___x_3712_ = lean_unbox_float(v_fst_3658_);
v___x_3713_ = lean_float_sub(v___x_3711_, v___x_3712_);
v___x_3714_ = lean_float_decLt(v___y_3710_, v___x_3713_);
v___y_3679_ = v___x_3714_;
goto v___jp_3678_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_apply_spec__6___boxed(lean_object* v_cls_3725_, lean_object* v_collapsed_3726_, lean_object* v_tag_3727_, lean_object* v_opts_3728_, lean_object* v_clsEnabled_3729_, lean_object* v_oldTraces_3730_, lean_object* v_msg_3731_, lean_object* v_resStartStop_3732_, lean_object* v___y_3733_, lean_object* v___y_3734_, lean_object* v___y_3735_, lean_object* v___y_3736_, lean_object* v___y_3737_, lean_object* v___y_3738_, lean_object* v___y_3739_){
_start:
{
uint8_t v_collapsed_boxed_3740_; uint8_t v_clsEnabled_boxed_3741_; lean_object* v_res_3742_; 
v_collapsed_boxed_3740_ = lean_unbox(v_collapsed_3726_);
v_clsEnabled_boxed_3741_ = lean_unbox(v_clsEnabled_3729_);
v_res_3742_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_apply_spec__6(v_cls_3725_, v_collapsed_boxed_3740_, v_tag_3727_, v_opts_3728_, v_clsEnabled_boxed_3741_, v_oldTraces_3730_, v_msg_3731_, v_resStartStop_3732_, v___y_3733_, v___y_3734_, v___y_3735_, v___y_3736_, v___y_3737_, v___y_3738_);
lean_dec(v___y_3738_);
lean_dec_ref(v___y_3737_);
lean_dec(v___y_3736_);
lean_dec_ref(v___y_3735_);
lean_dec(v___y_3734_);
lean_dec(v___y_3733_);
lean_dec_ref(v_opts_3728_);
return v_res_3742_;
}
}
static lean_object* _init_lp_aesop_Aesop_ForwardRuleMatch_apply___lam__3___closed__2(void){
_start:
{
lean_object* v___x_3745_; lean_object* v___x_3746_; lean_object* v___x_3747_; lean_object* v___x_3748_; lean_object* v___x_3749_; lean_object* v___x_3750_; 
v___x_3745_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_apply___lam__3___closed__1));
v___x_3746_ = lean_unsigned_to_nat(8u);
v___x_3747_ = lean_unsigned_to_nat(185u);
v___x_3748_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_apply___lam__3___closed__0));
v___x_3749_ = ((lean_object*)(lp_aesop_Aesop_CompleteMatch_reconstructArgs___closed__0));
v___x_3750_ = l_mkPanicMessageWithDecl(v___x_3749_, v___x_3748_, v___x_3747_, v___x_3746_, v___x_3745_);
return v___x_3750_;
}
}
static lean_object* _init_lp_aesop_Aesop_ForwardRuleMatch_apply___lam__3___closed__5(void){
_start:
{
lean_object* v___x_3754_; lean_object* v___x_3755_; 
v___x_3754_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_apply___lam__3___closed__4));
v___x_3755_ = l_Lean_stringToMessageData(v___x_3754_);
return v___x_3755_;
}
}
static lean_object* _init_lp_aesop_Aesop_ForwardRuleMatch_apply___lam__3___closed__6(void){
_start:
{
lean_object* v___x_3756_; lean_object* v___f_3757_; 
v___x_3756_ = lean_obj_once(&lp_aesop_Aesop_ForwardRuleMatch_apply___lam__3___closed__5, &lp_aesop_Aesop_ForwardRuleMatch_apply___lam__3___closed__5_once, _init_lp_aesop_Aesop_ForwardRuleMatch_apply___lam__3___closed__5);
v___f_3757_ = lean_alloc_closure((void*)(lp_aesop_Aesop_ForwardRuleMatch_apply___lam__2___boxed), 9, 1);
lean_closure_set(v___f_3757_, 0, v___x_3756_);
return v___f_3757_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_apply___lam__3(lean_object* v___x_3758_, lean_object* v_goal_3759_, lean_object* v_m_3760_, lean_object* v___f_3761_, uint8_t v_skipExistingProps_3762_, lean_object* v___y_3763_, lean_object* v___y_3764_, lean_object* v___y_3765_, lean_object* v___y_3766_, lean_object* v___y_3767_, lean_object* v___y_3768_){
_start:
{
lean_object* v___x_3770_; lean_object* v_a_3771_; lean_object* v___x_3773_; uint8_t v_isShared_3774_; uint8_t v_isSharedCheck_4066_; 
v___x_3770_ = lp_aesop_Lean_Meta_getUnusedUserName___at___00Aesop_ForwardRuleMatch_apply_spec__0___redArg(v___x_3758_, v___y_3765_);
v_a_3771_ = lean_ctor_get(v___x_3770_, 0);
v_isSharedCheck_4066_ = !lean_is_exclusive(v___x_3770_);
if (v_isSharedCheck_4066_ == 0)
{
v___x_3773_ = v___x_3770_;
v_isShared_3774_ = v_isSharedCheck_4066_;
goto v_resetjp_3772_;
}
else
{
lean_inc(v_a_3771_);
lean_dec(v___x_3770_);
v___x_3773_ = lean_box(0);
v_isShared_3774_ = v_isSharedCheck_4066_;
goto v_resetjp_3772_;
}
v_resetjp_3772_:
{
lean_object* v___x_3775_; 
lean_inc_ref(v_m_3760_);
lean_inc(v_goal_3759_);
v___x_3775_ = lp_aesop_Aesop_ForwardRuleMatch_getProof(v_goal_3759_, v_m_3760_, v___y_3765_, v___y_3766_, v___y_3767_, v___y_3768_);
if (lean_obj_tag(v___x_3775_) == 0)
{
lean_object* v_a_3776_; lean_object* v___x_3778_; uint8_t v_isShared_3779_; uint8_t v_isSharedCheck_4057_; 
v_a_3776_ = lean_ctor_get(v___x_3775_, 0);
v_isSharedCheck_4057_ = !lean_is_exclusive(v___x_3775_);
if (v_isSharedCheck_4057_ == 0)
{
v___x_3778_ = v___x_3775_;
v_isShared_3779_ = v_isSharedCheck_4057_;
goto v_resetjp_3777_;
}
else
{
lean_inc(v_a_3776_);
lean_dec(v___x_3775_);
v___x_3778_ = lean_box(0);
v_isShared_3779_ = v_isSharedCheck_4057_;
goto v_resetjp_3777_;
}
v_resetjp_3777_:
{
if (lean_obj_tag(v_a_3776_) == 1)
{
lean_object* v_val_3780_; lean_object* v___x_3782_; uint8_t v_isShared_3783_; uint8_t v_isSharedCheck_4052_; 
lean_del_object(v___x_3778_);
v_val_3780_ = lean_ctor_get(v_a_3776_, 0);
v_isSharedCheck_4052_ = !lean_is_exclusive(v_a_3776_);
if (v_isSharedCheck_4052_ == 0)
{
v___x_3782_ = v_a_3776_;
v_isShared_3783_ = v_isSharedCheck_4052_;
goto v_resetjp_3781_;
}
else
{
lean_inc(v_val_3780_);
lean_dec(v_a_3776_);
v___x_3782_ = lean_box(0);
v_isShared_3783_ = v_isSharedCheck_4052_;
goto v_resetjp_3781_;
}
v_resetjp_3781_:
{
lean_object* v___x_3784_; 
lean_inc(v___y_3768_);
lean_inc_ref(v___y_3767_);
lean_inc(v___y_3766_);
lean_inc_ref(v___y_3765_);
lean_inc(v_val_3780_);
v___x_3784_ = lean_infer_type(v_val_3780_, v___y_3765_, v___y_3766_, v___y_3767_, v___y_3768_);
if (lean_obj_tag(v___x_3784_) == 0)
{
lean_object* v_a_3785_; lean_object* v___y_3787_; lean_object* v___y_3788_; lean_object* v___y_3789_; lean_object* v___y_3790_; lean_object* v___y_3791_; lean_object* v___y_3792_; lean_object* v___y_3887_; 
v_a_3785_ = lean_ctor_get(v___x_3784_, 0);
lean_inc(v_a_3785_);
lean_dec_ref_known(v___x_3784_, 1);
if (v_skipExistingProps_3762_ == 0)
{
lean_del_object(v___x_3773_);
v___y_3787_ = v___y_3763_;
v___y_3788_ = v___y_3764_;
v___y_3789_ = v___y_3765_;
v___y_3790_ = v___y_3766_;
v___y_3791_ = v___y_3767_;
v___y_3792_ = v___y_3768_;
goto v___jp_3786_;
}
else
{
lean_object* v_options_3906_; lean_object* v_inheritedTraceOptions_3907_; uint8_t v_hasTrace_3908_; lean_object* v___x_3909_; 
v_options_3906_ = lean_ctor_get(v___y_3767_, 2);
v_inheritedTraceOptions_3907_ = lean_ctor_get(v___y_3767_, 13);
v_hasTrace_3908_ = lean_ctor_get_uint8(v_options_3906_, sizeof(void*)*1);
v___x_3909_ = lp_aesop_Aesop_TraceOption_forwardDebug;
if (v_hasTrace_3908_ == 0)
{
lean_object* v___x_3910_; 
lean_del_object(v___x_3773_);
lean_inc(v_a_3785_);
v___x_3910_ = lp_aesop_Aesop_isHypRedundantReducibleRigid(v_a_3785_, v___y_3765_, v___y_3766_, v___y_3767_, v___y_3768_);
if (lean_obj_tag(v___x_3910_) == 0)
{
lean_object* v_a_3911_; uint8_t v___x_3912_; lean_object* v___x_3913_; 
v_a_3911_ = lean_ctor_get(v___x_3910_, 0);
lean_inc(v_a_3911_);
lean_dec_ref_known(v___x_3910_, 1);
v___x_3912_ = lean_unbox(v_a_3911_);
lean_dec(v_a_3911_);
v___x_3913_ = lp_aesop_Aesop_ForwardRuleMatch_apply___lam__1(v___x_3909_, v___x_3912_, v___y_3763_, v___y_3764_, v___y_3765_, v___y_3766_, v___y_3767_, v___y_3768_);
v___y_3887_ = v___x_3913_;
goto v___jp_3886_;
}
else
{
v___y_3887_ = v___x_3910_;
goto v___jp_3886_;
}
}
else
{
lean_object* v_traceClass_3914_; lean_object* v___f_3915_; lean_object* v___x_3916_; lean_object* v___x_3917_; lean_object* v___x_3918_; uint8_t v___x_3919_; lean_object* v___y_3921_; lean_object* v___y_3922_; lean_object* v_a_3923_; lean_object* v___y_3936_; lean_object* v___y_3937_; uint8_t v_a_3938_; lean_object* v___y_3944_; lean_object* v___y_3945_; lean_object* v_a_3946_; lean_object* v___y_3949_; lean_object* v___y_3950_; lean_object* v___y_3951_; lean_object* v___y_3956_; uint8_t v___y_3957_; lean_object* v___y_3958_; lean_object* v___y_3959_; lean_object* v___y_3960_; lean_object* v___y_3967_; lean_object* v___y_3968_; lean_object* v_a_3969_; lean_object* v___y_3979_; lean_object* v___y_3980_; lean_object* v_a_3981_; lean_object* v___y_3984_; lean_object* v___y_3985_; uint8_t v_a_3986_; lean_object* v___y_3990_; lean_object* v___y_3991_; uint8_t v___y_3992_; lean_object* v___y_3993_; lean_object* v___y_3994_; lean_object* v___y_4001_; lean_object* v___y_4002_; lean_object* v___y_4003_; 
v_traceClass_3914_ = lean_ctor_get(v___x_3909_, 0);
v___f_3915_ = lean_obj_once(&lp_aesop_Aesop_ForwardRuleMatch_apply___lam__3___closed__6, &lp_aesop_Aesop_ForwardRuleMatch_apply___lam__3___closed__6_once, _init_lp_aesop_Aesop_ForwardRuleMatch_apply___lam__3___closed__6);
v___x_3916_ = ((lean_object*)(lp_aesop_Lean_addTrace___at___00Aesop_ForwardRuleMatch_getProof_spec__4___closed__1));
v___x_3917_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_getProof___closed__7));
lean_inc(v_traceClass_3914_);
v___x_3918_ = l_Lean_Name_append(v___x_3917_, v_traceClass_3914_);
v___x_3919_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_3907_, v_options_3906_, v___x_3918_);
lean_dec(v___x_3918_);
if (v___x_3919_ == 0)
{
lean_object* v___x_4038_; uint8_t v___x_4039_; 
v___x_4038_ = l_Lean_trace_profiler;
v___x_4039_ = lp_aesop_Lean_Option_get___at___00Aesop_ForwardRuleMatch_getProof_spec__12(v_options_3906_, v___x_4038_);
if (v___x_4039_ == 0)
{
lean_object* v___x_4040_; 
lean_del_object(v___x_3773_);
lean_inc(v_a_3785_);
v___x_4040_ = lp_aesop_Aesop_isHypRedundantReducibleRigid(v_a_3785_, v___y_3765_, v___y_3766_, v___y_3767_, v___y_3768_);
if (lean_obj_tag(v___x_4040_) == 0)
{
lean_object* v_a_4041_; uint8_t v___x_4042_; lean_object* v___x_4043_; 
v_a_4041_ = lean_ctor_get(v___x_4040_, 0);
lean_inc(v_a_4041_);
lean_dec_ref_known(v___x_4040_, 1);
v___x_4042_ = lean_unbox(v_a_4041_);
lean_dec(v_a_4041_);
v___x_4043_ = lp_aesop_Aesop_ForwardRuleMatch_apply___lam__1(v___x_3909_, v___x_4042_, v___y_3763_, v___y_3764_, v___y_3765_, v___y_3766_, v___y_3767_, v___y_3768_);
v___y_3887_ = v___x_4043_;
goto v___jp_3886_;
}
else
{
v___y_3887_ = v___x_4040_;
goto v___jp_3886_;
}
}
else
{
goto v___jp_4007_;
}
}
else
{
goto v___jp_4007_;
}
v___jp_3920_:
{
lean_object* v___x_3924_; double v___x_3925_; double v___x_3926_; double v___x_3927_; double v___x_3928_; double v___x_3929_; lean_object* v___x_3930_; lean_object* v___x_3931_; lean_object* v___x_3932_; lean_object* v___x_3933_; lean_object* v___x_3934_; 
v___x_3924_ = lean_io_mono_nanos_now();
v___x_3925_ = lean_float_of_nat(v___y_3921_);
v___x_3926_ = lean_float_once(&lp_aesop_Aesop_ForwardRuleMatch_getProof___closed__8, &lp_aesop_Aesop_ForwardRuleMatch_getProof___closed__8_once, _init_lp_aesop_Aesop_ForwardRuleMatch_getProof___closed__8);
v___x_3927_ = lean_float_div(v___x_3925_, v___x_3926_);
v___x_3928_ = lean_float_of_nat(v___x_3924_);
v___x_3929_ = lean_float_div(v___x_3928_, v___x_3926_);
v___x_3930_ = lean_box_float(v___x_3927_);
v___x_3931_ = lean_box_float(v___x_3929_);
v___x_3932_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3932_, 0, v___x_3930_);
lean_ctor_set(v___x_3932_, 1, v___x_3931_);
v___x_3933_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3933_, 0, v_a_3923_);
lean_ctor_set(v___x_3933_, 1, v___x_3932_);
lean_inc(v_traceClass_3914_);
v___x_3934_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_apply_spec__6(v_traceClass_3914_, v_hasTrace_3908_, v___x_3916_, v_options_3906_, v___x_3919_, v___y_3922_, v___f_3915_, v___x_3933_, v___y_3763_, v___y_3764_, v___y_3765_, v___y_3766_, v___y_3767_, v___y_3768_);
v___y_3887_ = v___x_3934_;
goto v___jp_3886_;
}
v___jp_3935_:
{
lean_object* v___x_3939_; lean_object* v___x_3941_; 
v___x_3939_ = lean_box(v_a_3938_);
if (v_isShared_3774_ == 0)
{
lean_ctor_set_tag(v___x_3773_, 1);
lean_ctor_set(v___x_3773_, 0, v___x_3939_);
v___x_3941_ = v___x_3773_;
goto v_reusejp_3940_;
}
else
{
lean_object* v_reuseFailAlloc_3942_; 
v_reuseFailAlloc_3942_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3942_, 0, v___x_3939_);
v___x_3941_ = v_reuseFailAlloc_3942_;
goto v_reusejp_3940_;
}
v_reusejp_3940_:
{
v___y_3921_ = v___y_3936_;
v___y_3922_ = v___y_3937_;
v_a_3923_ = v___x_3941_;
goto v___jp_3920_;
}
}
v___jp_3943_:
{
lean_object* v___x_3947_; 
v___x_3947_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3947_, 0, v_a_3946_);
v___y_3921_ = v___y_3944_;
v___y_3922_ = v___y_3945_;
v_a_3923_ = v___x_3947_;
goto v___jp_3920_;
}
v___jp_3948_:
{
if (lean_obj_tag(v___y_3951_) == 0)
{
lean_object* v_a_3952_; uint8_t v___x_3953_; 
v_a_3952_ = lean_ctor_get(v___y_3951_, 0);
lean_inc(v_a_3952_);
lean_dec_ref_known(v___y_3951_, 1);
v___x_3953_ = lean_unbox(v_a_3952_);
lean_dec(v_a_3952_);
v___y_3936_ = v___y_3949_;
v___y_3937_ = v___y_3950_;
v_a_3938_ = v___x_3953_;
goto v___jp_3935_;
}
else
{
lean_object* v_a_3954_; 
lean_del_object(v___x_3773_);
v_a_3954_ = lean_ctor_get(v___y_3951_, 0);
lean_inc(v_a_3954_);
lean_dec_ref_known(v___y_3951_, 1);
v___y_3944_ = v___y_3949_;
v___y_3945_ = v___y_3950_;
v_a_3946_ = v_a_3954_;
goto v___jp_3943_;
}
}
v___jp_3955_:
{
lean_object* v___x_3961_; lean_object* v___x_3962_; lean_object* v___x_3963_; lean_object* v___x_3964_; 
lean_inc_ref(v___y_3960_);
v___x_3961_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_3961_, 0, v___y_3960_);
v___x_3962_ = l_Lean_MessageData_ofFormat(v___x_3961_);
lean_inc_ref(v___y_3958_);
v___x_3963_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3963_, 0, v___y_3958_);
lean_ctor_set(v___x_3963_, 1, v___x_3962_);
lean_inc(v_traceClass_3914_);
v___x_3964_ = lp_aesop_Lean_addTrace___at___00Aesop_ForwardRuleMatch_apply_spec__4___redArg(v_traceClass_3914_, v___x_3963_, v___y_3765_, v___y_3766_, v___y_3767_, v___y_3768_);
if (lean_obj_tag(v___x_3964_) == 0)
{
lean_dec_ref_known(v___x_3964_, 1);
v___y_3936_ = v___y_3956_;
v___y_3937_ = v___y_3959_;
v_a_3938_ = v___y_3957_;
goto v___jp_3935_;
}
else
{
lean_object* v_a_3965_; 
lean_del_object(v___x_3773_);
v_a_3965_ = lean_ctor_get(v___x_3964_, 0);
lean_inc(v_a_3965_);
lean_dec_ref_known(v___x_3964_, 1);
v___y_3944_ = v___y_3956_;
v___y_3945_ = v___y_3959_;
v_a_3946_ = v_a_3965_;
goto v___jp_3943_;
}
}
v___jp_3966_:
{
lean_object* v___x_3970_; double v___x_3971_; double v___x_3972_; lean_object* v___x_3973_; lean_object* v___x_3974_; lean_object* v___x_3975_; lean_object* v___x_3976_; lean_object* v___x_3977_; 
v___x_3970_ = lean_io_get_num_heartbeats();
v___x_3971_ = lean_float_of_nat(v___y_3967_);
v___x_3972_ = lean_float_of_nat(v___x_3970_);
v___x_3973_ = lean_box_float(v___x_3971_);
v___x_3974_ = lean_box_float(v___x_3972_);
v___x_3975_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3975_, 0, v___x_3973_);
lean_ctor_set(v___x_3975_, 1, v___x_3974_);
v___x_3976_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3976_, 0, v_a_3969_);
lean_ctor_set(v___x_3976_, 1, v___x_3975_);
lean_inc(v_traceClass_3914_);
v___x_3977_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_apply_spec__6(v_traceClass_3914_, v_hasTrace_3908_, v___x_3916_, v_options_3906_, v___x_3919_, v___y_3968_, v___f_3915_, v___x_3976_, v___y_3763_, v___y_3764_, v___y_3765_, v___y_3766_, v___y_3767_, v___y_3768_);
v___y_3887_ = v___x_3977_;
goto v___jp_3886_;
}
v___jp_3978_:
{
lean_object* v___x_3982_; 
v___x_3982_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3982_, 0, v_a_3981_);
v___y_3967_ = v___y_3979_;
v___y_3968_ = v___y_3980_;
v_a_3969_ = v___x_3982_;
goto v___jp_3966_;
}
v___jp_3983_:
{
lean_object* v___x_3987_; lean_object* v___x_3988_; 
v___x_3987_ = lean_box(v_a_3986_);
v___x_3988_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3988_, 0, v___x_3987_);
v___y_3967_ = v___y_3984_;
v___y_3968_ = v___y_3985_;
v_a_3969_ = v___x_3988_;
goto v___jp_3966_;
}
v___jp_3989_:
{
lean_object* v___x_3995_; lean_object* v___x_3996_; lean_object* v___x_3997_; lean_object* v___x_3998_; 
lean_inc_ref(v___y_3994_);
v___x_3995_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_3995_, 0, v___y_3994_);
v___x_3996_ = l_Lean_MessageData_ofFormat(v___x_3995_);
lean_inc_ref(v___y_3990_);
v___x_3997_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_3997_, 0, v___y_3990_);
lean_ctor_set(v___x_3997_, 1, v___x_3996_);
lean_inc(v_traceClass_3914_);
v___x_3998_ = lp_aesop_Lean_addTrace___at___00Aesop_ForwardRuleMatch_apply_spec__4___redArg(v_traceClass_3914_, v___x_3997_, v___y_3765_, v___y_3766_, v___y_3767_, v___y_3768_);
if (lean_obj_tag(v___x_3998_) == 0)
{
lean_dec_ref_known(v___x_3998_, 1);
v___y_3984_ = v___y_3991_;
v___y_3985_ = v___y_3993_;
v_a_3986_ = v___y_3992_;
goto v___jp_3983_;
}
else
{
lean_object* v_a_3999_; 
v_a_3999_ = lean_ctor_get(v___x_3998_, 0);
lean_inc(v_a_3999_);
lean_dec_ref_known(v___x_3998_, 1);
v___y_3979_ = v___y_3991_;
v___y_3980_ = v___y_3993_;
v_a_3981_ = v_a_3999_;
goto v___jp_3978_;
}
}
v___jp_4000_:
{
if (lean_obj_tag(v___y_4003_) == 0)
{
lean_object* v_a_4004_; uint8_t v___x_4005_; 
v_a_4004_ = lean_ctor_get(v___y_4003_, 0);
lean_inc(v_a_4004_);
lean_dec_ref_known(v___y_4003_, 1);
v___x_4005_ = lean_unbox(v_a_4004_);
lean_dec(v_a_4004_);
v___y_3984_ = v___y_4001_;
v___y_3985_ = v___y_4002_;
v_a_3986_ = v___x_4005_;
goto v___jp_3983_;
}
else
{
lean_object* v_a_4006_; 
v_a_4006_ = lean_ctor_get(v___y_4003_, 0);
lean_inc(v_a_4006_);
lean_dec_ref_known(v___y_4003_, 1);
v___y_3979_ = v___y_4001_;
v___y_3980_ = v___y_4002_;
v_a_3981_ = v_a_4006_;
goto v___jp_3978_;
}
}
v___jp_4007_:
{
lean_object* v___x_4008_; lean_object* v_a_4009_; lean_object* v___x_4010_; uint8_t v___x_4011_; 
v___x_4008_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_ForwardRuleMatch_apply_spec__5___redArg(v___y_3768_);
v_a_4009_ = lean_ctor_get(v___x_4008_, 0);
lean_inc(v_a_4009_);
lean_dec_ref(v___x_4008_);
v___x_4010_ = l_Lean_trace_profiler_useHeartbeats;
v___x_4011_ = lp_aesop_Lean_Option_get___at___00Aesop_ForwardRuleMatch_getProof_spec__12(v_options_3906_, v___x_4010_);
if (v___x_4011_ == 0)
{
lean_object* v___x_4012_; lean_object* v___x_4013_; 
v___x_4012_ = lean_io_mono_nanos_now();
lean_inc(v_a_3785_);
v___x_4013_ = lp_aesop_Aesop_isHypRedundantReducibleRigid(v_a_3785_, v___y_3765_, v___y_3766_, v___y_3767_, v___y_3768_);
if (lean_obj_tag(v___x_4013_) == 0)
{
lean_object* v_a_4014_; lean_object* v___x_4015_; lean_object* v_a_4016_; uint8_t v___x_4017_; 
v_a_4014_ = lean_ctor_get(v___x_4013_, 0);
lean_inc(v_a_4014_);
lean_dec_ref_known(v___x_4013_, 1);
v___x_4015_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_ForwardRuleMatch_apply_spec__3___redArg(v___x_3909_, v___y_3767_);
v_a_4016_ = lean_ctor_get(v___x_4015_, 0);
lean_inc(v_a_4016_);
lean_dec_ref(v___x_4015_);
v___x_4017_ = lean_unbox(v_a_4016_);
lean_dec(v_a_4016_);
if (v___x_4017_ == 0)
{
uint8_t v___x_4018_; 
v___x_4018_ = lean_unbox(v_a_4014_);
lean_dec(v_a_4014_);
v___y_3936_ = v___x_4012_;
v___y_3937_ = v_a_4009_;
v_a_3938_ = v___x_4018_;
goto v___jp_3935_;
}
else
{
lean_object* v___x_4019_; uint8_t v___x_4020_; 
v___x_4019_ = lean_obj_once(&lp_aesop_Aesop_ForwardRuleMatch_apply___lam__1___closed__1, &lp_aesop_Aesop_ForwardRuleMatch_apply___lam__1___closed__1_once, _init_lp_aesop_Aesop_ForwardRuleMatch_apply___lam__1___closed__1);
v___x_4020_ = lean_unbox(v_a_4014_);
if (v___x_4020_ == 0)
{
lean_object* v___x_4021_; uint8_t v___x_4022_; 
v___x_4021_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_apply___lam__1___closed__2));
v___x_4022_ = lean_unbox(v_a_4014_);
lean_dec(v_a_4014_);
v___y_3956_ = v___x_4012_;
v___y_3957_ = v___x_4022_;
v___y_3958_ = v___x_4019_;
v___y_3959_ = v_a_4009_;
v___y_3960_ = v___x_4021_;
goto v___jp_3955_;
}
else
{
lean_object* v___x_4023_; uint8_t v___x_4024_; 
v___x_4023_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_apply___lam__1___closed__3));
v___x_4024_ = lean_unbox(v_a_4014_);
lean_dec(v_a_4014_);
v___y_3956_ = v___x_4012_;
v___y_3957_ = v___x_4024_;
v___y_3958_ = v___x_4019_;
v___y_3959_ = v_a_4009_;
v___y_3960_ = v___x_4023_;
goto v___jp_3955_;
}
}
}
else
{
v___y_3949_ = v___x_4012_;
v___y_3950_ = v_a_4009_;
v___y_3951_ = v___x_4013_;
goto v___jp_3948_;
}
}
else
{
lean_object* v___x_4025_; lean_object* v___x_4026_; 
lean_del_object(v___x_3773_);
v___x_4025_ = lean_io_get_num_heartbeats();
lean_inc(v_a_3785_);
v___x_4026_ = lp_aesop_Aesop_isHypRedundantReducibleRigid(v_a_3785_, v___y_3765_, v___y_3766_, v___y_3767_, v___y_3768_);
if (lean_obj_tag(v___x_4026_) == 0)
{
lean_object* v_a_4027_; lean_object* v___x_4028_; lean_object* v_a_4029_; uint8_t v___x_4030_; 
v_a_4027_ = lean_ctor_get(v___x_4026_, 0);
lean_inc(v_a_4027_);
lean_dec_ref_known(v___x_4026_, 1);
v___x_4028_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_ForwardRuleMatch_apply_spec__3___redArg(v___x_3909_, v___y_3767_);
v_a_4029_ = lean_ctor_get(v___x_4028_, 0);
lean_inc(v_a_4029_);
lean_dec_ref(v___x_4028_);
v___x_4030_ = lean_unbox(v_a_4029_);
lean_dec(v_a_4029_);
if (v___x_4030_ == 0)
{
uint8_t v___x_4031_; 
v___x_4031_ = lean_unbox(v_a_4027_);
lean_dec(v_a_4027_);
v___y_3984_ = v___x_4025_;
v___y_3985_ = v_a_4009_;
v_a_3986_ = v___x_4031_;
goto v___jp_3983_;
}
else
{
lean_object* v___x_4032_; uint8_t v___x_4033_; 
v___x_4032_ = lean_obj_once(&lp_aesop_Aesop_ForwardRuleMatch_apply___lam__1___closed__1, &lp_aesop_Aesop_ForwardRuleMatch_apply___lam__1___closed__1_once, _init_lp_aesop_Aesop_ForwardRuleMatch_apply___lam__1___closed__1);
v___x_4033_ = lean_unbox(v_a_4027_);
if (v___x_4033_ == 0)
{
lean_object* v___x_4034_; uint8_t v___x_4035_; 
v___x_4034_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_apply___lam__1___closed__2));
v___x_4035_ = lean_unbox(v_a_4027_);
lean_dec(v_a_4027_);
v___y_3990_ = v___x_4032_;
v___y_3991_ = v___x_4025_;
v___y_3992_ = v___x_4035_;
v___y_3993_ = v_a_4009_;
v___y_3994_ = v___x_4034_;
goto v___jp_3989_;
}
else
{
lean_object* v___x_4036_; uint8_t v___x_4037_; 
v___x_4036_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_apply___lam__1___closed__3));
v___x_4037_ = lean_unbox(v_a_4027_);
lean_dec(v_a_4027_);
v___y_3990_ = v___x_4032_;
v___y_3991_ = v___x_4025_;
v___y_3992_ = v___x_4037_;
v___y_3993_ = v_a_4009_;
v___y_3994_ = v___x_4036_;
goto v___jp_3989_;
}
}
}
else
{
v___y_4001_ = v___x_4025_;
v___y_4002_ = v_a_4009_;
v___y_4003_ = v___x_4026_;
goto v___jp_4000_;
}
}
}
}
}
v___jp_3786_:
{
uint8_t v___x_3793_; uint8_t v___x_3794_; lean_object* v___x_3795_; uint8_t v___x_3796_; lean_object* v___x_3797_; 
v___x_3793_ = 0;
v___x_3794_ = 0;
v___x_3795_ = lean_alloc_ctor(0, 3, 2);
lean_ctor_set(v___x_3795_, 0, v_a_3771_);
lean_ctor_set(v___x_3795_, 1, v_a_3785_);
lean_ctor_set(v___x_3795_, 2, v_val_3780_);
lean_ctor_set_uint8(v___x_3795_, sizeof(void*)*3, v___x_3793_);
lean_ctor_set_uint8(v___x_3795_, sizeof(void*)*3 + 1, v___x_3794_);
v___x_3796_ = 1;
v___x_3797_ = lp_aesop_Aesop_assertHypothesisS(v_goal_3759_, v___x_3795_, v___x_3796_, v___y_3787_, v___y_3788_, v___y_3789_, v___y_3790_, v___y_3791_, v___y_3792_);
if (lean_obj_tag(v___x_3797_) == 0)
{
lean_object* v_a_3798_; lean_object* v___x_3800_; uint8_t v_isShared_3801_; uint8_t v_isSharedCheck_3877_; 
v_a_3798_ = lean_ctor_get(v___x_3797_, 0);
v_isSharedCheck_3877_ = !lean_is_exclusive(v___x_3797_);
if (v_isSharedCheck_3877_ == 0)
{
v___x_3800_ = v___x_3797_;
v_isShared_3801_ = v_isSharedCheck_3877_;
goto v_resetjp_3799_;
}
else
{
lean_inc(v_a_3798_);
lean_dec(v___x_3797_);
v___x_3800_ = lean_box(0);
v_isShared_3801_ = v_isSharedCheck_3877_;
goto v_resetjp_3799_;
}
v_resetjp_3799_:
{
lean_object* v_fst_3802_; lean_object* v_snd_3803_; lean_object* v___x_3805_; uint8_t v_isShared_3806_; uint8_t v_isSharedCheck_3876_; 
v_fst_3802_ = lean_ctor_get(v_a_3798_, 0);
v_snd_3803_ = lean_ctor_get(v_a_3798_, 1);
v_isSharedCheck_3876_ = !lean_is_exclusive(v_a_3798_);
if (v_isSharedCheck_3876_ == 0)
{
v___x_3805_ = v_a_3798_;
v_isShared_3806_ = v_isSharedCheck_3876_;
goto v_resetjp_3804_;
}
else
{
lean_inc(v_snd_3803_);
lean_inc(v_fst_3802_);
lean_dec(v_a_3798_);
v___x_3805_ = lean_box(0);
v_isShared_3806_ = v_isSharedCheck_3876_;
goto v_resetjp_3804_;
}
v_resetjp_3804_:
{
lean_object* v___x_3807_; lean_object* v___x_3808_; uint8_t v___x_3809_; 
v___x_3807_ = lean_array_get_size(v_snd_3803_);
v___x_3808_ = lean_unsigned_to_nat(1u);
v___x_3809_ = lean_nat_dec_eq(v___x_3807_, v___x_3808_);
if (v___x_3809_ == 0)
{
lean_object* v___x_3810_; lean_object* v___x_3811_; 
lean_del_object(v___x_3805_);
lean_dec(v_snd_3803_);
lean_dec(v_fst_3802_);
lean_del_object(v___x_3800_);
lean_del_object(v___x_3782_);
lean_dec_ref(v___f_3761_);
lean_dec_ref(v_m_3760_);
v___x_3810_ = lean_obj_once(&lp_aesop_Aesop_ForwardRuleMatch_apply___lam__3___closed__2, &lp_aesop_Aesop_ForwardRuleMatch_apply___lam__3___closed__2_once, _init_lp_aesop_Aesop_ForwardRuleMatch_apply___lam__3___closed__2);
v___x_3811_ = lp_aesop_panic___at___00Aesop_ForwardRuleMatch_apply_spec__1(v___x_3810_, v___y_3787_, v___y_3788_, v___y_3789_, v___y_3790_, v___y_3791_, v___y_3792_);
return v___x_3811_;
}
else
{
lean_object* v_rule_3812_; lean_object* v___x_3814_; uint8_t v_isShared_3815_; uint8_t v_isSharedCheck_3874_; 
v_rule_3812_ = lean_ctor_get(v_m_3760_, 0);
v_isSharedCheck_3874_ = !lean_is_exclusive(v_m_3760_);
if (v_isSharedCheck_3874_ == 0)
{
lean_object* v_unused_3875_; 
v_unused_3875_ = lean_ctor_get(v_m_3760_, 1);
lean_dec(v_unused_3875_);
v___x_3814_ = v_m_3760_;
v_isShared_3815_ = v_isSharedCheck_3874_;
goto v_resetjp_3813_;
}
else
{
lean_inc(v_rule_3812_);
lean_dec(v_m_3760_);
v___x_3814_ = lean_box(0);
v_isShared_3815_ = v_isSharedCheck_3874_;
goto v_resetjp_3813_;
}
v_resetjp_3813_:
{
lean_object* v___x_3816_; lean_object* v___x_3817_; uint8_t v___x_3818_; 
v___x_3816_ = lean_unsigned_to_nat(0u);
v___x_3817_ = lean_array_fget(v_snd_3803_, v___x_3816_);
lean_dec(v_snd_3803_);
v___x_3818_ = lp_aesop_Aesop_ForwardRule_destruct(v_rule_3812_);
lean_dec_ref(v_rule_3812_);
if (v___x_3818_ == 0)
{
lean_object* v___x_3819_; lean_object* v___x_3821_; 
lean_dec_ref(v___f_3761_);
v___x_3819_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_apply___lam__3___closed__3));
if (v_isShared_3806_ == 0)
{
lean_ctor_set(v___x_3805_, 1, v___x_3819_);
lean_ctor_set(v___x_3805_, 0, v___x_3817_);
v___x_3821_ = v___x_3805_;
goto v_reusejp_3820_;
}
else
{
lean_object* v_reuseFailAlloc_3831_; 
v_reuseFailAlloc_3831_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3831_, 0, v___x_3817_);
lean_ctor_set(v_reuseFailAlloc_3831_, 1, v___x_3819_);
v___x_3821_ = v_reuseFailAlloc_3831_;
goto v_reusejp_3820_;
}
v_reusejp_3820_:
{
lean_object* v___x_3823_; 
if (v_isShared_3815_ == 0)
{
lean_ctor_set(v___x_3814_, 1, v___x_3821_);
lean_ctor_set(v___x_3814_, 0, v_fst_3802_);
v___x_3823_ = v___x_3814_;
goto v_reusejp_3822_;
}
else
{
lean_object* v_reuseFailAlloc_3830_; 
v_reuseFailAlloc_3830_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3830_, 0, v_fst_3802_);
lean_ctor_set(v_reuseFailAlloc_3830_, 1, v___x_3821_);
v___x_3823_ = v_reuseFailAlloc_3830_;
goto v_reusejp_3822_;
}
v_reusejp_3822_:
{
lean_object* v___x_3825_; 
if (v_isShared_3783_ == 0)
{
lean_ctor_set(v___x_3782_, 0, v___x_3823_);
v___x_3825_ = v___x_3782_;
goto v_reusejp_3824_;
}
else
{
lean_object* v_reuseFailAlloc_3829_; 
v_reuseFailAlloc_3829_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3829_, 0, v___x_3823_);
v___x_3825_ = v_reuseFailAlloc_3829_;
goto v_reusejp_3824_;
}
v_reusejp_3824_:
{
lean_object* v___x_3827_; 
if (v_isShared_3801_ == 0)
{
lean_ctor_set(v___x_3800_, 0, v___x_3825_);
v___x_3827_ = v___x_3800_;
goto v_reusejp_3826_;
}
else
{
lean_object* v_reuseFailAlloc_3828_; 
v_reuseFailAlloc_3828_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3828_, 0, v___x_3825_);
v___x_3827_ = v_reuseFailAlloc_3828_;
goto v_reusejp_3826_;
}
v_reusejp_3826_:
{
return v___x_3827_;
}
}
}
}
}
else
{
lean_object* v___x_3832_; 
lean_del_object(v___x_3814_);
lean_del_object(v___x_3800_);
lean_inc(v_fst_3802_);
v___x_3832_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_ForwardRuleMatch_apply_spec__2___redArg(v_fst_3802_, v___f_3761_, v___y_3787_, v___y_3788_, v___y_3789_, v___y_3790_, v___y_3791_, v___y_3792_);
if (lean_obj_tag(v___x_3832_) == 0)
{
lean_object* v_a_3833_; lean_object* v___x_3834_; 
v_a_3833_ = lean_ctor_get(v___x_3832_, 0);
lean_inc_n(v_a_3833_, 2);
lean_dec_ref_known(v___x_3832_, 1);
v___x_3834_ = lp_aesop_Aesop_tryClearManyS(v_fst_3802_, v_a_3833_, v___y_3787_, v___y_3788_, v___y_3789_, v___y_3790_, v___y_3791_, v___y_3792_);
if (lean_obj_tag(v___x_3834_) == 0)
{
lean_object* v_a_3835_; lean_object* v___x_3837_; uint8_t v_isShared_3838_; uint8_t v_isSharedCheck_3857_; 
v_a_3835_ = lean_ctor_get(v___x_3834_, 0);
v_isSharedCheck_3857_ = !lean_is_exclusive(v___x_3834_);
if (v_isSharedCheck_3857_ == 0)
{
v___x_3837_ = v___x_3834_;
v_isShared_3838_ = v_isSharedCheck_3857_;
goto v_resetjp_3836_;
}
else
{
lean_inc(v_a_3835_);
lean_dec(v___x_3834_);
v___x_3837_ = lean_box(0);
v_isShared_3838_ = v_isSharedCheck_3857_;
goto v_resetjp_3836_;
}
v_resetjp_3836_:
{
lean_object* v_fst_3839_; lean_object* v___x_3841_; uint8_t v_isShared_3842_; uint8_t v_isSharedCheck_3855_; 
v_fst_3839_ = lean_ctor_get(v_a_3835_, 0);
v_isSharedCheck_3855_ = !lean_is_exclusive(v_a_3835_);
if (v_isSharedCheck_3855_ == 0)
{
lean_object* v_unused_3856_; 
v_unused_3856_ = lean_ctor_get(v_a_3835_, 1);
lean_dec(v_unused_3856_);
v___x_3841_ = v_a_3835_;
v_isShared_3842_ = v_isSharedCheck_3855_;
goto v_resetjp_3840_;
}
else
{
lean_inc(v_fst_3839_);
lean_dec(v_a_3835_);
v___x_3841_ = lean_box(0);
v_isShared_3842_ = v_isSharedCheck_3855_;
goto v_resetjp_3840_;
}
v_resetjp_3840_:
{
lean_object* v___x_3844_; 
if (v_isShared_3842_ == 0)
{
lean_ctor_set(v___x_3841_, 1, v_a_3833_);
lean_ctor_set(v___x_3841_, 0, v___x_3817_);
v___x_3844_ = v___x_3841_;
goto v_reusejp_3843_;
}
else
{
lean_object* v_reuseFailAlloc_3854_; 
v_reuseFailAlloc_3854_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3854_, 0, v___x_3817_);
lean_ctor_set(v_reuseFailAlloc_3854_, 1, v_a_3833_);
v___x_3844_ = v_reuseFailAlloc_3854_;
goto v_reusejp_3843_;
}
v_reusejp_3843_:
{
lean_object* v___x_3846_; 
if (v_isShared_3806_ == 0)
{
lean_ctor_set(v___x_3805_, 1, v___x_3844_);
lean_ctor_set(v___x_3805_, 0, v_fst_3839_);
v___x_3846_ = v___x_3805_;
goto v_reusejp_3845_;
}
else
{
lean_object* v_reuseFailAlloc_3853_; 
v_reuseFailAlloc_3853_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3853_, 0, v_fst_3839_);
lean_ctor_set(v_reuseFailAlloc_3853_, 1, v___x_3844_);
v___x_3846_ = v_reuseFailAlloc_3853_;
goto v_reusejp_3845_;
}
v_reusejp_3845_:
{
lean_object* v___x_3848_; 
if (v_isShared_3783_ == 0)
{
lean_ctor_set(v___x_3782_, 0, v___x_3846_);
v___x_3848_ = v___x_3782_;
goto v_reusejp_3847_;
}
else
{
lean_object* v_reuseFailAlloc_3852_; 
v_reuseFailAlloc_3852_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3852_, 0, v___x_3846_);
v___x_3848_ = v_reuseFailAlloc_3852_;
goto v_reusejp_3847_;
}
v_reusejp_3847_:
{
lean_object* v___x_3850_; 
if (v_isShared_3838_ == 0)
{
lean_ctor_set(v___x_3837_, 0, v___x_3848_);
v___x_3850_ = v___x_3837_;
goto v_reusejp_3849_;
}
else
{
lean_object* v_reuseFailAlloc_3851_; 
v_reuseFailAlloc_3851_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3851_, 0, v___x_3848_);
v___x_3850_ = v_reuseFailAlloc_3851_;
goto v_reusejp_3849_;
}
v_reusejp_3849_:
{
return v___x_3850_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_3858_; lean_object* v___x_3860_; uint8_t v_isShared_3861_; uint8_t v_isSharedCheck_3865_; 
lean_dec(v_a_3833_);
lean_dec(v___x_3817_);
lean_del_object(v___x_3805_);
lean_del_object(v___x_3782_);
v_a_3858_ = lean_ctor_get(v___x_3834_, 0);
v_isSharedCheck_3865_ = !lean_is_exclusive(v___x_3834_);
if (v_isSharedCheck_3865_ == 0)
{
v___x_3860_ = v___x_3834_;
v_isShared_3861_ = v_isSharedCheck_3865_;
goto v_resetjp_3859_;
}
else
{
lean_inc(v_a_3858_);
lean_dec(v___x_3834_);
v___x_3860_ = lean_box(0);
v_isShared_3861_ = v_isSharedCheck_3865_;
goto v_resetjp_3859_;
}
v_resetjp_3859_:
{
lean_object* v___x_3863_; 
if (v_isShared_3861_ == 0)
{
v___x_3863_ = v___x_3860_;
goto v_reusejp_3862_;
}
else
{
lean_object* v_reuseFailAlloc_3864_; 
v_reuseFailAlloc_3864_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3864_, 0, v_a_3858_);
v___x_3863_ = v_reuseFailAlloc_3864_;
goto v_reusejp_3862_;
}
v_reusejp_3862_:
{
return v___x_3863_;
}
}
}
}
else
{
lean_object* v_a_3866_; lean_object* v___x_3868_; uint8_t v_isShared_3869_; uint8_t v_isSharedCheck_3873_; 
lean_dec(v___x_3817_);
lean_del_object(v___x_3805_);
lean_dec(v_fst_3802_);
lean_del_object(v___x_3782_);
v_a_3866_ = lean_ctor_get(v___x_3832_, 0);
v_isSharedCheck_3873_ = !lean_is_exclusive(v___x_3832_);
if (v_isSharedCheck_3873_ == 0)
{
v___x_3868_ = v___x_3832_;
v_isShared_3869_ = v_isSharedCheck_3873_;
goto v_resetjp_3867_;
}
else
{
lean_inc(v_a_3866_);
lean_dec(v___x_3832_);
v___x_3868_ = lean_box(0);
v_isShared_3869_ = v_isSharedCheck_3873_;
goto v_resetjp_3867_;
}
v_resetjp_3867_:
{
lean_object* v___x_3871_; 
if (v_isShared_3869_ == 0)
{
v___x_3871_ = v___x_3868_;
goto v_reusejp_3870_;
}
else
{
lean_object* v_reuseFailAlloc_3872_; 
v_reuseFailAlloc_3872_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3872_, 0, v_a_3866_);
v___x_3871_ = v_reuseFailAlloc_3872_;
goto v_reusejp_3870_;
}
v_reusejp_3870_:
{
return v___x_3871_;
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
lean_object* v_a_3878_; lean_object* v___x_3880_; uint8_t v_isShared_3881_; uint8_t v_isSharedCheck_3885_; 
lean_del_object(v___x_3782_);
lean_dec_ref(v___f_3761_);
lean_dec_ref(v_m_3760_);
v_a_3878_ = lean_ctor_get(v___x_3797_, 0);
v_isSharedCheck_3885_ = !lean_is_exclusive(v___x_3797_);
if (v_isSharedCheck_3885_ == 0)
{
v___x_3880_ = v___x_3797_;
v_isShared_3881_ = v_isSharedCheck_3885_;
goto v_resetjp_3879_;
}
else
{
lean_inc(v_a_3878_);
lean_dec(v___x_3797_);
v___x_3880_ = lean_box(0);
v_isShared_3881_ = v_isSharedCheck_3885_;
goto v_resetjp_3879_;
}
v_resetjp_3879_:
{
lean_object* v___x_3883_; 
if (v_isShared_3881_ == 0)
{
v___x_3883_ = v___x_3880_;
goto v_reusejp_3882_;
}
else
{
lean_object* v_reuseFailAlloc_3884_; 
v_reuseFailAlloc_3884_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3884_, 0, v_a_3878_);
v___x_3883_ = v_reuseFailAlloc_3884_;
goto v_reusejp_3882_;
}
v_reusejp_3882_:
{
return v___x_3883_;
}
}
}
}
v___jp_3886_:
{
if (lean_obj_tag(v___y_3887_) == 0)
{
lean_object* v_a_3888_; lean_object* v___x_3890_; uint8_t v_isShared_3891_; uint8_t v_isSharedCheck_3897_; 
v_a_3888_ = lean_ctor_get(v___y_3887_, 0);
v_isSharedCheck_3897_ = !lean_is_exclusive(v___y_3887_);
if (v_isSharedCheck_3897_ == 0)
{
v___x_3890_ = v___y_3887_;
v_isShared_3891_ = v_isSharedCheck_3897_;
goto v_resetjp_3889_;
}
else
{
lean_inc(v_a_3888_);
lean_dec(v___y_3887_);
v___x_3890_ = lean_box(0);
v_isShared_3891_ = v_isSharedCheck_3897_;
goto v_resetjp_3889_;
}
v_resetjp_3889_:
{
uint8_t v___x_3892_; 
v___x_3892_ = lean_unbox(v_a_3888_);
lean_dec(v_a_3888_);
if (v___x_3892_ == 0)
{
lean_del_object(v___x_3890_);
v___y_3787_ = v___y_3763_;
v___y_3788_ = v___y_3764_;
v___y_3789_ = v___y_3765_;
v___y_3790_ = v___y_3766_;
v___y_3791_ = v___y_3767_;
v___y_3792_ = v___y_3768_;
goto v___jp_3786_;
}
else
{
lean_object* v___x_3893_; lean_object* v___x_3895_; 
lean_dec(v_a_3785_);
lean_del_object(v___x_3782_);
lean_dec(v_val_3780_);
lean_dec(v_a_3771_);
lean_dec_ref(v___f_3761_);
lean_dec_ref(v_m_3760_);
lean_dec(v_goal_3759_);
v___x_3893_ = lean_box(0);
if (v_isShared_3891_ == 0)
{
lean_ctor_set(v___x_3890_, 0, v___x_3893_);
v___x_3895_ = v___x_3890_;
goto v_reusejp_3894_;
}
else
{
lean_object* v_reuseFailAlloc_3896_; 
v_reuseFailAlloc_3896_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3896_, 0, v___x_3893_);
v___x_3895_ = v_reuseFailAlloc_3896_;
goto v_reusejp_3894_;
}
v_reusejp_3894_:
{
return v___x_3895_;
}
}
}
}
else
{
lean_object* v_a_3898_; lean_object* v___x_3900_; uint8_t v_isShared_3901_; uint8_t v_isSharedCheck_3905_; 
lean_dec(v_a_3785_);
lean_del_object(v___x_3782_);
lean_dec(v_val_3780_);
lean_dec(v_a_3771_);
lean_dec_ref(v___f_3761_);
lean_dec_ref(v_m_3760_);
lean_dec(v_goal_3759_);
v_a_3898_ = lean_ctor_get(v___y_3887_, 0);
v_isSharedCheck_3905_ = !lean_is_exclusive(v___y_3887_);
if (v_isSharedCheck_3905_ == 0)
{
v___x_3900_ = v___y_3887_;
v_isShared_3901_ = v_isSharedCheck_3905_;
goto v_resetjp_3899_;
}
else
{
lean_inc(v_a_3898_);
lean_dec(v___y_3887_);
v___x_3900_ = lean_box(0);
v_isShared_3901_ = v_isSharedCheck_3905_;
goto v_resetjp_3899_;
}
v_resetjp_3899_:
{
lean_object* v___x_3903_; 
if (v_isShared_3901_ == 0)
{
v___x_3903_ = v___x_3900_;
goto v_reusejp_3902_;
}
else
{
lean_object* v_reuseFailAlloc_3904_; 
v_reuseFailAlloc_3904_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3904_, 0, v_a_3898_);
v___x_3903_ = v_reuseFailAlloc_3904_;
goto v_reusejp_3902_;
}
v_reusejp_3902_:
{
return v___x_3903_;
}
}
}
}
}
else
{
lean_object* v_a_4044_; lean_object* v___x_4046_; uint8_t v_isShared_4047_; uint8_t v_isSharedCheck_4051_; 
lean_del_object(v___x_3782_);
lean_dec(v_val_3780_);
lean_del_object(v___x_3773_);
lean_dec(v_a_3771_);
lean_dec_ref(v___f_3761_);
lean_dec_ref(v_m_3760_);
lean_dec(v_goal_3759_);
v_a_4044_ = lean_ctor_get(v___x_3784_, 0);
v_isSharedCheck_4051_ = !lean_is_exclusive(v___x_3784_);
if (v_isSharedCheck_4051_ == 0)
{
v___x_4046_ = v___x_3784_;
v_isShared_4047_ = v_isSharedCheck_4051_;
goto v_resetjp_4045_;
}
else
{
lean_inc(v_a_4044_);
lean_dec(v___x_3784_);
v___x_4046_ = lean_box(0);
v_isShared_4047_ = v_isSharedCheck_4051_;
goto v_resetjp_4045_;
}
v_resetjp_4045_:
{
lean_object* v___x_4049_; 
if (v_isShared_4047_ == 0)
{
v___x_4049_ = v___x_4046_;
goto v_reusejp_4048_;
}
else
{
lean_object* v_reuseFailAlloc_4050_; 
v_reuseFailAlloc_4050_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4050_, 0, v_a_4044_);
v___x_4049_ = v_reuseFailAlloc_4050_;
goto v_reusejp_4048_;
}
v_reusejp_4048_:
{
return v___x_4049_;
}
}
}
}
}
else
{
lean_object* v___x_4053_; lean_object* v___x_4055_; 
lean_dec(v_a_3776_);
lean_del_object(v___x_3773_);
lean_dec(v_a_3771_);
lean_dec_ref(v___f_3761_);
lean_dec_ref(v_m_3760_);
lean_dec(v_goal_3759_);
v___x_4053_ = lean_box(0);
if (v_isShared_3779_ == 0)
{
lean_ctor_set(v___x_3778_, 0, v___x_4053_);
v___x_4055_ = v___x_3778_;
goto v_reusejp_4054_;
}
else
{
lean_object* v_reuseFailAlloc_4056_; 
v_reuseFailAlloc_4056_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4056_, 0, v___x_4053_);
v___x_4055_ = v_reuseFailAlloc_4056_;
goto v_reusejp_4054_;
}
v_reusejp_4054_:
{
return v___x_4055_;
}
}
}
}
else
{
lean_object* v_a_4058_; lean_object* v___x_4060_; uint8_t v_isShared_4061_; uint8_t v_isSharedCheck_4065_; 
lean_del_object(v___x_3773_);
lean_dec(v_a_3771_);
lean_dec_ref(v___f_3761_);
lean_dec_ref(v_m_3760_);
lean_dec(v_goal_3759_);
v_a_4058_ = lean_ctor_get(v___x_3775_, 0);
v_isSharedCheck_4065_ = !lean_is_exclusive(v___x_3775_);
if (v_isSharedCheck_4065_ == 0)
{
v___x_4060_ = v___x_3775_;
v_isShared_4061_ = v_isSharedCheck_4065_;
goto v_resetjp_4059_;
}
else
{
lean_inc(v_a_4058_);
lean_dec(v___x_3775_);
v___x_4060_ = lean_box(0);
v_isShared_4061_ = v_isSharedCheck_4065_;
goto v_resetjp_4059_;
}
v_resetjp_4059_:
{
lean_object* v___x_4063_; 
if (v_isShared_4061_ == 0)
{
v___x_4063_ = v___x_4060_;
goto v_reusejp_4062_;
}
else
{
lean_object* v_reuseFailAlloc_4064_; 
v_reuseFailAlloc_4064_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4064_, 0, v_a_4058_);
v___x_4063_ = v_reuseFailAlloc_4064_;
goto v_reusejp_4062_;
}
v_reusejp_4062_:
{
return v___x_4063_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_apply___lam__3___boxed(lean_object* v___x_4067_, lean_object* v_goal_4068_, lean_object* v_m_4069_, lean_object* v___f_4070_, lean_object* v_skipExistingProps_4071_, lean_object* v___y_4072_, lean_object* v___y_4073_, lean_object* v___y_4074_, lean_object* v___y_4075_, lean_object* v___y_4076_, lean_object* v___y_4077_, lean_object* v___y_4078_){
_start:
{
uint8_t v_skipExistingProps_boxed_4079_; lean_object* v_res_4080_; 
v_skipExistingProps_boxed_4079_ = lean_unbox(v_skipExistingProps_4071_);
v_res_4080_ = lp_aesop_Aesop_ForwardRuleMatch_apply___lam__3(v___x_4067_, v_goal_4068_, v_m_4069_, v___f_4070_, v_skipExistingProps_boxed_4079_, v___y_4072_, v___y_4073_, v___y_4074_, v___y_4075_, v___y_4076_, v___y_4077_);
lean_dec(v___y_4077_);
lean_dec_ref(v___y_4076_);
lean_dec(v___y_4075_);
lean_dec_ref(v___y_4074_);
lean_dec(v___y_4073_);
lean_dec(v___y_4072_);
lean_dec(v___x_4067_);
return v_res_4080_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_apply___lam__4(lean_object* v___x_4081_, lean_object* v_x_4082_, lean_object* v___y_4083_, lean_object* v___y_4084_, lean_object* v___y_4085_, lean_object* v___y_4086_, lean_object* v___y_4087_, lean_object* v___y_4088_){
_start:
{
lean_object* v___x_4090_; 
v___x_4090_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4090_, 0, v___x_4081_);
return v___x_4090_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_apply___lam__4___boxed(lean_object* v___x_4091_, lean_object* v_x_4092_, lean_object* v___y_4093_, lean_object* v___y_4094_, lean_object* v___y_4095_, lean_object* v___y_4096_, lean_object* v___y_4097_, lean_object* v___y_4098_, lean_object* v___y_4099_){
_start:
{
lean_object* v_res_4100_; 
v_res_4100_ = lp_aesop_Aesop_ForwardRuleMatch_apply___lam__4(v___x_4091_, v_x_4092_, v___y_4093_, v___y_4094_, v___y_4095_, v___y_4096_, v___y_4097_, v___y_4098_);
lean_dec(v___y_4098_);
lean_dec_ref(v___y_4097_);
lean_dec(v___y_4096_);
lean_dec_ref(v___y_4095_);
lean_dec(v___y_4094_);
lean_dec(v___y_4093_);
lean_dec_ref(v_x_4092_);
return v_res_4100_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_apply_spec__7_spec__10(lean_object* v_e_4101_){
_start:
{
if (lean_obj_tag(v_e_4101_) == 0)
{
uint8_t v___x_4102_; 
v___x_4102_ = 2;
return v___x_4102_;
}
else
{
lean_object* v_a_4103_; 
v_a_4103_ = lean_ctor_get(v_e_4101_, 0);
if (lean_obj_tag(v_a_4103_) == 0)
{
uint8_t v___x_4104_; 
v___x_4104_ = 1;
return v___x_4104_;
}
else
{
uint8_t v___x_4105_; 
v___x_4105_ = 0;
return v___x_4105_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_apply_spec__7_spec__10___boxed(lean_object* v_e_4106_){
_start:
{
uint8_t v_res_4107_; lean_object* v_r_4108_; 
v_res_4107_ = lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_apply_spec__7_spec__10(v_e_4106_);
lean_dec_ref(v_e_4106_);
v_r_4108_ = lean_box(v_res_4107_);
return v_r_4108_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_apply_spec__7(lean_object* v_cls_4109_, uint8_t v_collapsed_4110_, lean_object* v_tag_4111_, lean_object* v_opts_4112_, uint8_t v_clsEnabled_4113_, lean_object* v_oldTraces_4114_, lean_object* v_msg_4115_, lean_object* v_resStartStop_4116_, lean_object* v___y_4117_, lean_object* v___y_4118_, lean_object* v___y_4119_, lean_object* v___y_4120_, lean_object* v___y_4121_, lean_object* v___y_4122_){
_start:
{
lean_object* v_fst_4124_; lean_object* v_snd_4125_; lean_object* v___y_4127_; lean_object* v___y_4128_; lean_object* v_data_4129_; lean_object* v_fst_4140_; lean_object* v_snd_4141_; lean_object* v___x_4142_; uint8_t v___x_4143_; lean_object* v___y_4145_; lean_object* v_a_4146_; uint8_t v___y_4161_; double v___y_4192_; 
v_fst_4124_ = lean_ctor_get(v_resStartStop_4116_, 0);
lean_inc(v_fst_4124_);
v_snd_4125_ = lean_ctor_get(v_resStartStop_4116_, 1);
lean_inc(v_snd_4125_);
lean_dec_ref(v_resStartStop_4116_);
v_fst_4140_ = lean_ctor_get(v_snd_4125_, 0);
lean_inc(v_fst_4140_);
v_snd_4141_ = lean_ctor_get(v_snd_4125_, 1);
lean_inc(v_snd_4141_);
lean_dec(v_snd_4125_);
v___x_4142_ = l_Lean_trace_profiler;
v___x_4143_ = lp_aesop_Lean_Option_get___at___00Aesop_ForwardRuleMatch_getProof_spec__12(v_opts_4112_, v___x_4142_);
if (v___x_4143_ == 0)
{
v___y_4161_ = v___x_4143_;
goto v___jp_4160_;
}
else
{
lean_object* v___x_4197_; uint8_t v___x_4198_; 
v___x_4197_ = l_Lean_trace_profiler_useHeartbeats;
v___x_4198_ = lp_aesop_Lean_Option_get___at___00Aesop_ForwardRuleMatch_getProof_spec__12(v_opts_4112_, v___x_4197_);
if (v___x_4198_ == 0)
{
lean_object* v___x_4199_; lean_object* v___x_4200_; double v___x_4201_; double v___x_4202_; double v___x_4203_; 
v___x_4199_ = l_Lean_trace_profiler_threshold;
v___x_4200_ = lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13_spec__19(v_opts_4112_, v___x_4199_);
v___x_4201_ = lean_float_of_nat(v___x_4200_);
v___x_4202_ = lean_float_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13___closed__2, &lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13___closed__2_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13___closed__2);
v___x_4203_ = lean_float_div(v___x_4201_, v___x_4202_);
v___y_4192_ = v___x_4203_;
goto v___jp_4191_;
}
else
{
lean_object* v___x_4204_; lean_object* v___x_4205_; double v___x_4206_; 
v___x_4204_ = l_Lean_trace_profiler_threshold;
v___x_4205_ = lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13_spec__19(v_opts_4112_, v___x_4204_);
v___x_4206_ = lean_float_of_nat(v___x_4205_);
v___y_4192_ = v___x_4206_;
goto v___jp_4191_;
}
}
v___jp_4126_:
{
lean_object* v___x_4130_; 
lean_inc(v___y_4127_);
v___x_4130_ = lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_apply_spec__6_spec__6___redArg(v_oldTraces_4114_, v_data_4129_, v___y_4127_, v___y_4128_, v___y_4119_, v___y_4120_, v___y_4121_, v___y_4122_);
if (lean_obj_tag(v___x_4130_) == 0)
{
lean_object* v___x_4131_; 
lean_dec_ref_known(v___x_4130_, 1);
v___x_4131_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_apply_spec__6_spec__7___redArg(v_fst_4124_);
return v___x_4131_;
}
else
{
lean_object* v_a_4132_; lean_object* v___x_4134_; uint8_t v_isShared_4135_; uint8_t v_isSharedCheck_4139_; 
lean_dec(v_fst_4124_);
v_a_4132_ = lean_ctor_get(v___x_4130_, 0);
v_isSharedCheck_4139_ = !lean_is_exclusive(v___x_4130_);
if (v_isSharedCheck_4139_ == 0)
{
v___x_4134_ = v___x_4130_;
v_isShared_4135_ = v_isSharedCheck_4139_;
goto v_resetjp_4133_;
}
else
{
lean_inc(v_a_4132_);
lean_dec(v___x_4130_);
v___x_4134_ = lean_box(0);
v_isShared_4135_ = v_isSharedCheck_4139_;
goto v_resetjp_4133_;
}
v_resetjp_4133_:
{
lean_object* v___x_4137_; 
if (v_isShared_4135_ == 0)
{
v___x_4137_ = v___x_4134_;
goto v_reusejp_4136_;
}
else
{
lean_object* v_reuseFailAlloc_4138_; 
v_reuseFailAlloc_4138_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4138_, 0, v_a_4132_);
v___x_4137_ = v_reuseFailAlloc_4138_;
goto v_reusejp_4136_;
}
v_reusejp_4136_:
{
return v___x_4137_;
}
}
}
}
v___jp_4144_:
{
uint8_t v_result_4147_; lean_object* v___x_4148_; lean_object* v___x_4149_; double v___x_4150_; lean_object* v_data_4151_; 
v_result_4147_ = lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_apply_spec__7_spec__10(v_fst_4124_);
v___x_4148_ = lean_box(v_result_4147_);
v___x_4149_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4149_, 0, v___x_4148_);
v___x_4150_ = lean_float_once(&lp_aesop_Lean_addTrace___at___00Aesop_ForwardRuleMatch_getProof_spec__4___closed__0, &lp_aesop_Lean_addTrace___at___00Aesop_ForwardRuleMatch_getProof_spec__4___closed__0_once, _init_lp_aesop_Lean_addTrace___at___00Aesop_ForwardRuleMatch_getProof_spec__4___closed__0);
lean_inc_ref(v_tag_4111_);
lean_inc_ref(v___x_4149_);
lean_inc(v_cls_4109_);
v_data_4151_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v_data_4151_, 0, v_cls_4109_);
lean_ctor_set(v_data_4151_, 1, v___x_4149_);
lean_ctor_set(v_data_4151_, 2, v_tag_4111_);
lean_ctor_set_float(v_data_4151_, sizeof(void*)*3, v___x_4150_);
lean_ctor_set_float(v_data_4151_, sizeof(void*)*3 + 8, v___x_4150_);
lean_ctor_set_uint8(v_data_4151_, sizeof(void*)*3 + 16, v_collapsed_4110_);
if (v___x_4143_ == 0)
{
lean_dec_ref_known(v___x_4149_, 1);
lean_dec(v_snd_4141_);
lean_dec(v_fst_4140_);
lean_dec_ref(v_tag_4111_);
lean_dec(v_cls_4109_);
v___y_4127_ = v___y_4145_;
v___y_4128_ = v_a_4146_;
v_data_4129_ = v_data_4151_;
goto v___jp_4126_;
}
else
{
lean_object* v_data_4152_; double v___x_4153_; double v___x_4154_; 
lean_dec_ref_known(v_data_4151_, 3);
v_data_4152_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v_data_4152_, 0, v_cls_4109_);
lean_ctor_set(v_data_4152_, 1, v___x_4149_);
lean_ctor_set(v_data_4152_, 2, v_tag_4111_);
v___x_4153_ = lean_unbox_float(v_fst_4140_);
lean_dec(v_fst_4140_);
lean_ctor_set_float(v_data_4152_, sizeof(void*)*3, v___x_4153_);
v___x_4154_ = lean_unbox_float(v_snd_4141_);
lean_dec(v_snd_4141_);
lean_ctor_set_float(v_data_4152_, sizeof(void*)*3 + 8, v___x_4154_);
lean_ctor_set_uint8(v_data_4152_, sizeof(void*)*3 + 16, v_collapsed_4110_);
v___y_4127_ = v___y_4145_;
v___y_4128_ = v_a_4146_;
v_data_4129_ = v_data_4152_;
goto v___jp_4126_;
}
}
v___jp_4155_:
{
lean_object* v_ref_4156_; lean_object* v___x_4157_; 
v_ref_4156_ = lean_ctor_get(v___y_4121_, 5);
lean_inc(v___y_4122_);
lean_inc_ref(v___y_4121_);
lean_inc(v___y_4120_);
lean_inc_ref(v___y_4119_);
lean_inc(v___y_4118_);
lean_inc(v___y_4117_);
lean_inc(v_fst_4124_);
v___x_4157_ = lean_apply_8(v_msg_4115_, v_fst_4124_, v___y_4117_, v___y_4118_, v___y_4119_, v___y_4120_, v___y_4121_, v___y_4122_, lean_box(0));
if (lean_obj_tag(v___x_4157_) == 0)
{
lean_object* v_a_4158_; 
v_a_4158_ = lean_ctor_get(v___x_4157_, 0);
lean_inc(v_a_4158_);
lean_dec_ref_known(v___x_4157_, 1);
v___y_4145_ = v_ref_4156_;
v_a_4146_ = v_a_4158_;
goto v___jp_4144_;
}
else
{
lean_object* v___x_4159_; 
lean_dec_ref_known(v___x_4157_, 1);
v___x_4159_ = lean_obj_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13___closed__1, &lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13___closed__1_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_getProof_spec__13___closed__1);
v___y_4145_ = v_ref_4156_;
v_a_4146_ = v___x_4159_;
goto v___jp_4144_;
}
}
v___jp_4160_:
{
if (v_clsEnabled_4113_ == 0)
{
if (v___y_4161_ == 0)
{
lean_object* v___x_4162_; lean_object* v_traceState_4163_; lean_object* v_env_4164_; lean_object* v_nextMacroScope_4165_; lean_object* v_ngen_4166_; lean_object* v_auxDeclNGen_4167_; lean_object* v_cache_4168_; lean_object* v_messages_4169_; lean_object* v_infoState_4170_; lean_object* v_snapshotTasks_4171_; lean_object* v___x_4173_; uint8_t v_isShared_4174_; uint8_t v_isSharedCheck_4190_; 
lean_dec(v_snd_4141_);
lean_dec(v_fst_4140_);
lean_dec_ref(v_msg_4115_);
lean_dec_ref(v_tag_4111_);
lean_dec(v_cls_4109_);
v___x_4162_ = lean_st_ref_take(v___y_4122_);
v_traceState_4163_ = lean_ctor_get(v___x_4162_, 4);
v_env_4164_ = lean_ctor_get(v___x_4162_, 0);
v_nextMacroScope_4165_ = lean_ctor_get(v___x_4162_, 1);
v_ngen_4166_ = lean_ctor_get(v___x_4162_, 2);
v_auxDeclNGen_4167_ = lean_ctor_get(v___x_4162_, 3);
v_cache_4168_ = lean_ctor_get(v___x_4162_, 5);
v_messages_4169_ = lean_ctor_get(v___x_4162_, 6);
v_infoState_4170_ = lean_ctor_get(v___x_4162_, 7);
v_snapshotTasks_4171_ = lean_ctor_get(v___x_4162_, 8);
v_isSharedCheck_4190_ = !lean_is_exclusive(v___x_4162_);
if (v_isSharedCheck_4190_ == 0)
{
v___x_4173_ = v___x_4162_;
v_isShared_4174_ = v_isSharedCheck_4190_;
goto v_resetjp_4172_;
}
else
{
lean_inc(v_snapshotTasks_4171_);
lean_inc(v_infoState_4170_);
lean_inc(v_messages_4169_);
lean_inc(v_cache_4168_);
lean_inc(v_traceState_4163_);
lean_inc(v_auxDeclNGen_4167_);
lean_inc(v_ngen_4166_);
lean_inc(v_nextMacroScope_4165_);
lean_inc(v_env_4164_);
lean_dec(v___x_4162_);
v___x_4173_ = lean_box(0);
v_isShared_4174_ = v_isSharedCheck_4190_;
goto v_resetjp_4172_;
}
v_resetjp_4172_:
{
uint64_t v_tid_4175_; lean_object* v_traces_4176_; lean_object* v___x_4178_; uint8_t v_isShared_4179_; uint8_t v_isSharedCheck_4189_; 
v_tid_4175_ = lean_ctor_get_uint64(v_traceState_4163_, sizeof(void*)*1);
v_traces_4176_ = lean_ctor_get(v_traceState_4163_, 0);
v_isSharedCheck_4189_ = !lean_is_exclusive(v_traceState_4163_);
if (v_isSharedCheck_4189_ == 0)
{
v___x_4178_ = v_traceState_4163_;
v_isShared_4179_ = v_isSharedCheck_4189_;
goto v_resetjp_4177_;
}
else
{
lean_inc(v_traces_4176_);
lean_dec(v_traceState_4163_);
v___x_4178_ = lean_box(0);
v_isShared_4179_ = v_isSharedCheck_4189_;
goto v_resetjp_4177_;
}
v_resetjp_4177_:
{
lean_object* v___x_4180_; lean_object* v___x_4182_; 
v___x_4180_ = l_Lean_PersistentArray_append___redArg(v_oldTraces_4114_, v_traces_4176_);
lean_dec_ref(v_traces_4176_);
if (v_isShared_4179_ == 0)
{
lean_ctor_set(v___x_4178_, 0, v___x_4180_);
v___x_4182_ = v___x_4178_;
goto v_reusejp_4181_;
}
else
{
lean_object* v_reuseFailAlloc_4188_; 
v_reuseFailAlloc_4188_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_4188_, 0, v___x_4180_);
lean_ctor_set_uint64(v_reuseFailAlloc_4188_, sizeof(void*)*1, v_tid_4175_);
v___x_4182_ = v_reuseFailAlloc_4188_;
goto v_reusejp_4181_;
}
v_reusejp_4181_:
{
lean_object* v___x_4184_; 
if (v_isShared_4174_ == 0)
{
lean_ctor_set(v___x_4173_, 4, v___x_4182_);
v___x_4184_ = v___x_4173_;
goto v_reusejp_4183_;
}
else
{
lean_object* v_reuseFailAlloc_4187_; 
v_reuseFailAlloc_4187_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_4187_, 0, v_env_4164_);
lean_ctor_set(v_reuseFailAlloc_4187_, 1, v_nextMacroScope_4165_);
lean_ctor_set(v_reuseFailAlloc_4187_, 2, v_ngen_4166_);
lean_ctor_set(v_reuseFailAlloc_4187_, 3, v_auxDeclNGen_4167_);
lean_ctor_set(v_reuseFailAlloc_4187_, 4, v___x_4182_);
lean_ctor_set(v_reuseFailAlloc_4187_, 5, v_cache_4168_);
lean_ctor_set(v_reuseFailAlloc_4187_, 6, v_messages_4169_);
lean_ctor_set(v_reuseFailAlloc_4187_, 7, v_infoState_4170_);
lean_ctor_set(v_reuseFailAlloc_4187_, 8, v_snapshotTasks_4171_);
v___x_4184_ = v_reuseFailAlloc_4187_;
goto v_reusejp_4183_;
}
v_reusejp_4183_:
{
lean_object* v___x_4185_; lean_object* v___x_4186_; 
v___x_4185_ = lean_st_ref_set(v___y_4122_, v___x_4184_);
v___x_4186_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_apply_spec__6_spec__7___redArg(v_fst_4124_);
return v___x_4186_;
}
}
}
}
}
else
{
goto v___jp_4155_;
}
}
else
{
goto v___jp_4155_;
}
}
v___jp_4191_:
{
double v___x_4193_; double v___x_4194_; double v___x_4195_; uint8_t v___x_4196_; 
v___x_4193_ = lean_unbox_float(v_snd_4141_);
v___x_4194_ = lean_unbox_float(v_fst_4140_);
v___x_4195_ = lean_float_sub(v___x_4193_, v___x_4194_);
v___x_4196_ = lean_float_decLt(v___y_4192_, v___x_4195_);
v___y_4161_ = v___x_4196_;
goto v___jp_4160_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_apply_spec__7___boxed(lean_object* v_cls_4207_, lean_object* v_collapsed_4208_, lean_object* v_tag_4209_, lean_object* v_opts_4210_, lean_object* v_clsEnabled_4211_, lean_object* v_oldTraces_4212_, lean_object* v_msg_4213_, lean_object* v_resStartStop_4214_, lean_object* v___y_4215_, lean_object* v___y_4216_, lean_object* v___y_4217_, lean_object* v___y_4218_, lean_object* v___y_4219_, lean_object* v___y_4220_, lean_object* v___y_4221_){
_start:
{
uint8_t v_collapsed_boxed_4222_; uint8_t v_clsEnabled_boxed_4223_; lean_object* v_res_4224_; 
v_collapsed_boxed_4222_ = lean_unbox(v_collapsed_4208_);
v_clsEnabled_boxed_4223_ = lean_unbox(v_clsEnabled_4211_);
v_res_4224_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_apply_spec__7(v_cls_4207_, v_collapsed_boxed_4222_, v_tag_4209_, v_opts_4210_, v_clsEnabled_boxed_4223_, v_oldTraces_4212_, v_msg_4213_, v_resStartStop_4214_, v___y_4215_, v___y_4216_, v___y_4217_, v___y_4218_, v___y_4219_, v___y_4220_);
lean_dec(v___y_4220_);
lean_dec_ref(v___y_4219_);
lean_dec(v___y_4218_);
lean_dec_ref(v___y_4217_);
lean_dec(v___y_4216_);
lean_dec(v___y_4215_);
lean_dec_ref(v_opts_4210_);
return v_res_4224_;
}
}
static lean_object* _init_lp_aesop_Aesop_ForwardRuleMatch_apply___closed__1(void){
_start:
{
lean_object* v___x_4226_; lean_object* v___x_4227_; 
v___x_4226_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_apply___closed__0));
v___x_4227_ = l_Lean_stringToMessageData(v___x_4226_);
return v___x_4227_;
}
}
static lean_object* _init_lp_aesop_Aesop_ForwardRuleMatch_apply___closed__2(void){
_start:
{
lean_object* v___x_4228_; lean_object* v___f_4229_; 
v___x_4228_ = lean_obj_once(&lp_aesop_Aesop_ForwardRuleMatch_apply___closed__1, &lp_aesop_Aesop_ForwardRuleMatch_apply___closed__1_once, _init_lp_aesop_Aesop_ForwardRuleMatch_apply___closed__1);
v___f_4229_ = lean_alloc_closure((void*)(lp_aesop_Aesop_ForwardRuleMatch_apply___lam__4___boxed), 9, 1);
lean_closure_set(v___f_4229_, 0, v___x_4228_);
return v___f_4229_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_apply(lean_object* v_goal_4230_, lean_object* v_m_4231_, uint8_t v_skipExistingProps_4232_, lean_object* v_a_4233_, lean_object* v_a_4234_, lean_object* v_a_4235_, lean_object* v_a_4236_, lean_object* v_a_4237_, lean_object* v_a_4238_){
_start:
{
lean_object* v_options_4240_; lean_object* v_inheritedTraceOptions_4241_; uint8_t v_hasTrace_4242_; lean_object* v___f_4243_; lean_object* v___x_4244_; lean_object* v___x_4245_; lean_object* v___f_4246_; 
v_options_4240_ = lean_ctor_get(v_a_4237_, 2);
v_inheritedTraceOptions_4241_ = lean_ctor_get(v_a_4237_, 13);
v_hasTrace_4242_ = lean_ctor_get_uint8(v_options_4240_, sizeof(void*)*1);
lean_inc_ref(v_m_4231_);
v___f_4243_ = lean_alloc_closure((void*)(lp_aesop_Aesop_ForwardRuleMatch_apply___lam__0___boxed), 8, 1);
lean_closure_set(v___f_4243_, 0, v_m_4231_);
v___x_4244_ = lp_aesop_Aesop_forwardHypPrefix;
v___x_4245_ = lean_box(v_skipExistingProps_4232_);
lean_inc(v_goal_4230_);
v___f_4246_ = lean_alloc_closure((void*)(lp_aesop_Aesop_ForwardRuleMatch_apply___lam__3___boxed), 12, 5);
lean_closure_set(v___f_4246_, 0, v___x_4244_);
lean_closure_set(v___f_4246_, 1, v_goal_4230_);
lean_closure_set(v___f_4246_, 2, v_m_4231_);
lean_closure_set(v___f_4246_, 3, v___f_4243_);
lean_closure_set(v___f_4246_, 4, v___x_4245_);
if (v_hasTrace_4242_ == 0)
{
lean_object* v___x_4247_; 
v___x_4247_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_ForwardRuleMatch_apply_spec__2___redArg(v_goal_4230_, v___f_4246_, v_a_4233_, v_a_4234_, v_a_4235_, v_a_4236_, v_a_4237_, v_a_4238_);
return v___x_4247_;
}
else
{
lean_object* v___x_4248_; lean_object* v_traceClass_4249_; lean_object* v___f_4250_; lean_object* v___x_4251_; lean_object* v___x_4252_; lean_object* v___x_4253_; uint8_t v___x_4254_; lean_object* v___y_4256_; lean_object* v___y_4257_; lean_object* v_a_4258_; lean_object* v___y_4271_; lean_object* v___y_4272_; lean_object* v_a_4273_; 
v___x_4248_ = lp_aesop_Aesop_TraceOption_forward;
v_traceClass_4249_ = lean_ctor_get(v___x_4248_, 0);
v___f_4250_ = lean_obj_once(&lp_aesop_Aesop_ForwardRuleMatch_apply___closed__2, &lp_aesop_Aesop_ForwardRuleMatch_apply___closed__2_once, _init_lp_aesop_Aesop_ForwardRuleMatch_apply___closed__2);
v___x_4251_ = ((lean_object*)(lp_aesop_Lean_addTrace___at___00Aesop_ForwardRuleMatch_getProof_spec__4___closed__1));
v___x_4252_ = ((lean_object*)(lp_aesop_Aesop_ForwardRuleMatch_getProof___closed__7));
lean_inc(v_traceClass_4249_);
v___x_4253_ = l_Lean_Name_append(v___x_4252_, v_traceClass_4249_);
v___x_4254_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_4241_, v_options_4240_, v___x_4253_);
lean_dec(v___x_4253_);
if (v___x_4254_ == 0)
{
lean_object* v___x_4323_; uint8_t v___x_4324_; 
v___x_4323_ = l_Lean_trace_profiler;
v___x_4324_ = lp_aesop_Lean_Option_get___at___00Aesop_ForwardRuleMatch_getProof_spec__12(v_options_4240_, v___x_4323_);
if (v___x_4324_ == 0)
{
lean_object* v___x_4325_; 
v___x_4325_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_ForwardRuleMatch_apply_spec__2___redArg(v_goal_4230_, v___f_4246_, v_a_4233_, v_a_4234_, v_a_4235_, v_a_4236_, v_a_4237_, v_a_4238_);
return v___x_4325_;
}
else
{
goto v___jp_4282_;
}
}
else
{
goto v___jp_4282_;
}
v___jp_4255_:
{
lean_object* v___x_4259_; double v___x_4260_; double v___x_4261_; double v___x_4262_; double v___x_4263_; double v___x_4264_; lean_object* v___x_4265_; lean_object* v___x_4266_; lean_object* v___x_4267_; lean_object* v___x_4268_; lean_object* v___x_4269_; 
v___x_4259_ = lean_io_mono_nanos_now();
v___x_4260_ = lean_float_of_nat(v___y_4257_);
v___x_4261_ = lean_float_once(&lp_aesop_Aesop_ForwardRuleMatch_getProof___closed__8, &lp_aesop_Aesop_ForwardRuleMatch_getProof___closed__8_once, _init_lp_aesop_Aesop_ForwardRuleMatch_getProof___closed__8);
v___x_4262_ = lean_float_div(v___x_4260_, v___x_4261_);
v___x_4263_ = lean_float_of_nat(v___x_4259_);
v___x_4264_ = lean_float_div(v___x_4263_, v___x_4261_);
v___x_4265_ = lean_box_float(v___x_4262_);
v___x_4266_ = lean_box_float(v___x_4264_);
v___x_4267_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4267_, 0, v___x_4265_);
lean_ctor_set(v___x_4267_, 1, v___x_4266_);
v___x_4268_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4268_, 0, v_a_4258_);
lean_ctor_set(v___x_4268_, 1, v___x_4267_);
lean_inc(v_traceClass_4249_);
v___x_4269_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_apply_spec__7(v_traceClass_4249_, v_hasTrace_4242_, v___x_4251_, v_options_4240_, v___x_4254_, v___y_4256_, v___f_4250_, v___x_4268_, v_a_4233_, v_a_4234_, v_a_4235_, v_a_4236_, v_a_4237_, v_a_4238_);
return v___x_4269_;
}
v___jp_4270_:
{
lean_object* v___x_4274_; double v___x_4275_; double v___x_4276_; lean_object* v___x_4277_; lean_object* v___x_4278_; lean_object* v___x_4279_; lean_object* v___x_4280_; lean_object* v___x_4281_; 
v___x_4274_ = lean_io_get_num_heartbeats();
v___x_4275_ = lean_float_of_nat(v___y_4272_);
v___x_4276_ = lean_float_of_nat(v___x_4274_);
v___x_4277_ = lean_box_float(v___x_4275_);
v___x_4278_ = lean_box_float(v___x_4276_);
v___x_4279_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4279_, 0, v___x_4277_);
lean_ctor_set(v___x_4279_, 1, v___x_4278_);
v___x_4280_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4280_, 0, v_a_4273_);
lean_ctor_set(v___x_4280_, 1, v___x_4279_);
lean_inc(v_traceClass_4249_);
v___x_4281_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_apply_spec__7(v_traceClass_4249_, v_hasTrace_4242_, v___x_4251_, v_options_4240_, v___x_4254_, v___y_4271_, v___f_4250_, v___x_4280_, v_a_4233_, v_a_4234_, v_a_4235_, v_a_4236_, v_a_4237_, v_a_4238_);
return v___x_4281_;
}
v___jp_4282_:
{
lean_object* v___x_4283_; lean_object* v_a_4284_; lean_object* v___x_4285_; uint8_t v___x_4286_; 
v___x_4283_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_ForwardRuleMatch_apply_spec__5___redArg(v_a_4238_);
v_a_4284_ = lean_ctor_get(v___x_4283_, 0);
lean_inc(v_a_4284_);
lean_dec_ref(v___x_4283_);
v___x_4285_ = l_Lean_trace_profiler_useHeartbeats;
v___x_4286_ = lp_aesop_Lean_Option_get___at___00Aesop_ForwardRuleMatch_getProof_spec__12(v_options_4240_, v___x_4285_);
if (v___x_4286_ == 0)
{
lean_object* v___x_4287_; lean_object* v___x_4288_; 
v___x_4287_ = lean_io_mono_nanos_now();
v___x_4288_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_ForwardRuleMatch_apply_spec__2___redArg(v_goal_4230_, v___f_4246_, v_a_4233_, v_a_4234_, v_a_4235_, v_a_4236_, v_a_4237_, v_a_4238_);
if (lean_obj_tag(v___x_4288_) == 0)
{
lean_object* v_a_4289_; lean_object* v___x_4291_; uint8_t v_isShared_4292_; uint8_t v_isSharedCheck_4296_; 
v_a_4289_ = lean_ctor_get(v___x_4288_, 0);
v_isSharedCheck_4296_ = !lean_is_exclusive(v___x_4288_);
if (v_isSharedCheck_4296_ == 0)
{
v___x_4291_ = v___x_4288_;
v_isShared_4292_ = v_isSharedCheck_4296_;
goto v_resetjp_4290_;
}
else
{
lean_inc(v_a_4289_);
lean_dec(v___x_4288_);
v___x_4291_ = lean_box(0);
v_isShared_4292_ = v_isSharedCheck_4296_;
goto v_resetjp_4290_;
}
v_resetjp_4290_:
{
lean_object* v___x_4294_; 
if (v_isShared_4292_ == 0)
{
lean_ctor_set_tag(v___x_4291_, 1);
v___x_4294_ = v___x_4291_;
goto v_reusejp_4293_;
}
else
{
lean_object* v_reuseFailAlloc_4295_; 
v_reuseFailAlloc_4295_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4295_, 0, v_a_4289_);
v___x_4294_ = v_reuseFailAlloc_4295_;
goto v_reusejp_4293_;
}
v_reusejp_4293_:
{
v___y_4256_ = v_a_4284_;
v___y_4257_ = v___x_4287_;
v_a_4258_ = v___x_4294_;
goto v___jp_4255_;
}
}
}
else
{
lean_object* v_a_4297_; lean_object* v___x_4299_; uint8_t v_isShared_4300_; uint8_t v_isSharedCheck_4304_; 
v_a_4297_ = lean_ctor_get(v___x_4288_, 0);
v_isSharedCheck_4304_ = !lean_is_exclusive(v___x_4288_);
if (v_isSharedCheck_4304_ == 0)
{
v___x_4299_ = v___x_4288_;
v_isShared_4300_ = v_isSharedCheck_4304_;
goto v_resetjp_4298_;
}
else
{
lean_inc(v_a_4297_);
lean_dec(v___x_4288_);
v___x_4299_ = lean_box(0);
v_isShared_4300_ = v_isSharedCheck_4304_;
goto v_resetjp_4298_;
}
v_resetjp_4298_:
{
lean_object* v___x_4302_; 
if (v_isShared_4300_ == 0)
{
lean_ctor_set_tag(v___x_4299_, 0);
v___x_4302_ = v___x_4299_;
goto v_reusejp_4301_;
}
else
{
lean_object* v_reuseFailAlloc_4303_; 
v_reuseFailAlloc_4303_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4303_, 0, v_a_4297_);
v___x_4302_ = v_reuseFailAlloc_4303_;
goto v_reusejp_4301_;
}
v_reusejp_4301_:
{
v___y_4256_ = v_a_4284_;
v___y_4257_ = v___x_4287_;
v_a_4258_ = v___x_4302_;
goto v___jp_4255_;
}
}
}
}
else
{
lean_object* v___x_4305_; lean_object* v___x_4306_; 
v___x_4305_ = lean_io_get_num_heartbeats();
v___x_4306_ = lp_aesop_Lean_MVarId_withContext___at___00Aesop_ForwardRuleMatch_apply_spec__2___redArg(v_goal_4230_, v___f_4246_, v_a_4233_, v_a_4234_, v_a_4235_, v_a_4236_, v_a_4237_, v_a_4238_);
if (lean_obj_tag(v___x_4306_) == 0)
{
lean_object* v_a_4307_; lean_object* v___x_4309_; uint8_t v_isShared_4310_; uint8_t v_isSharedCheck_4314_; 
v_a_4307_ = lean_ctor_get(v___x_4306_, 0);
v_isSharedCheck_4314_ = !lean_is_exclusive(v___x_4306_);
if (v_isSharedCheck_4314_ == 0)
{
v___x_4309_ = v___x_4306_;
v_isShared_4310_ = v_isSharedCheck_4314_;
goto v_resetjp_4308_;
}
else
{
lean_inc(v_a_4307_);
lean_dec(v___x_4306_);
v___x_4309_ = lean_box(0);
v_isShared_4310_ = v_isSharedCheck_4314_;
goto v_resetjp_4308_;
}
v_resetjp_4308_:
{
lean_object* v___x_4312_; 
if (v_isShared_4310_ == 0)
{
lean_ctor_set_tag(v___x_4309_, 1);
v___x_4312_ = v___x_4309_;
goto v_reusejp_4311_;
}
else
{
lean_object* v_reuseFailAlloc_4313_; 
v_reuseFailAlloc_4313_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4313_, 0, v_a_4307_);
v___x_4312_ = v_reuseFailAlloc_4313_;
goto v_reusejp_4311_;
}
v_reusejp_4311_:
{
v___y_4271_ = v_a_4284_;
v___y_4272_ = v___x_4305_;
v_a_4273_ = v___x_4312_;
goto v___jp_4270_;
}
}
}
else
{
lean_object* v_a_4315_; lean_object* v___x_4317_; uint8_t v_isShared_4318_; uint8_t v_isSharedCheck_4322_; 
v_a_4315_ = lean_ctor_get(v___x_4306_, 0);
v_isSharedCheck_4322_ = !lean_is_exclusive(v___x_4306_);
if (v_isSharedCheck_4322_ == 0)
{
v___x_4317_ = v___x_4306_;
v_isShared_4318_ = v_isSharedCheck_4322_;
goto v_resetjp_4316_;
}
else
{
lean_inc(v_a_4315_);
lean_dec(v___x_4306_);
v___x_4317_ = lean_box(0);
v_isShared_4318_ = v_isSharedCheck_4322_;
goto v_resetjp_4316_;
}
v_resetjp_4316_:
{
lean_object* v___x_4320_; 
if (v_isShared_4318_ == 0)
{
lean_ctor_set_tag(v___x_4317_, 0);
v___x_4320_ = v___x_4317_;
goto v_reusejp_4319_;
}
else
{
lean_object* v_reuseFailAlloc_4321_; 
v_reuseFailAlloc_4321_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4321_, 0, v_a_4315_);
v___x_4320_ = v_reuseFailAlloc_4321_;
goto v_reusejp_4319_;
}
v_reusejp_4319_:
{
v___y_4271_ = v_a_4284_;
v___y_4272_ = v___x_4305_;
v_a_4273_ = v___x_4320_;
goto v___jp_4270_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ForwardRuleMatch_apply___boxed(lean_object* v_goal_4326_, lean_object* v_m_4327_, lean_object* v_skipExistingProps_4328_, lean_object* v_a_4329_, lean_object* v_a_4330_, lean_object* v_a_4331_, lean_object* v_a_4332_, lean_object* v_a_4333_, lean_object* v_a_4334_, lean_object* v_a_4335_){
_start:
{
uint8_t v_skipExistingProps_boxed_4336_; lean_object* v_res_4337_; 
v_skipExistingProps_boxed_4336_ = lean_unbox(v_skipExistingProps_4328_);
v_res_4337_ = lp_aesop_Aesop_ForwardRuleMatch_apply(v_goal_4326_, v_m_4327_, v_skipExistingProps_boxed_4336_, v_a_4329_, v_a_4330_, v_a_4331_, v_a_4332_, v_a_4333_, v_a_4334_);
lean_dec(v_a_4334_);
lean_dec_ref(v_a_4333_);
lean_dec(v_a_4332_);
lean_dec_ref(v_a_4331_);
lean_dec(v_a_4330_);
lean_dec(v_a_4329_);
return v_res_4337_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_ForwardRuleMatch_apply_spec__3(lean_object* v_opt_4338_, lean_object* v___y_4339_, lean_object* v___y_4340_, lean_object* v___y_4341_, lean_object* v___y_4342_, lean_object* v___y_4343_, lean_object* v___y_4344_){
_start:
{
lean_object* v___x_4346_; 
v___x_4346_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_ForwardRuleMatch_apply_spec__3___redArg(v_opt_4338_, v___y_4343_);
return v___x_4346_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_ForwardRuleMatch_apply_spec__3___boxed(lean_object* v_opt_4347_, lean_object* v___y_4348_, lean_object* v___y_4349_, lean_object* v___y_4350_, lean_object* v___y_4351_, lean_object* v___y_4352_, lean_object* v___y_4353_, lean_object* v___y_4354_){
_start:
{
lean_object* v_res_4355_; 
v_res_4355_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_ForwardRuleMatch_apply_spec__3(v_opt_4347_, v___y_4348_, v___y_4349_, v___y_4350_, v___y_4351_, v___y_4352_, v___y_4353_);
lean_dec(v___y_4353_);
lean_dec_ref(v___y_4352_);
lean_dec(v___y_4351_);
lean_dec_ref(v___y_4350_);
lean_dec(v___y_4349_);
lean_dec(v___y_4348_);
lean_dec_ref(v_opt_4347_);
return v_res_4355_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00Aesop_ForwardRuleMatch_apply_spec__4(lean_object* v_cls_4356_, lean_object* v_msg_4357_, lean_object* v___y_4358_, lean_object* v___y_4359_, lean_object* v___y_4360_, lean_object* v___y_4361_, lean_object* v___y_4362_, lean_object* v___y_4363_){
_start:
{
lean_object* v___x_4365_; 
v___x_4365_ = lp_aesop_Lean_addTrace___at___00Aesop_ForwardRuleMatch_apply_spec__4___redArg(v_cls_4356_, v_msg_4357_, v___y_4360_, v___y_4361_, v___y_4362_, v___y_4363_);
return v___x_4365_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00Aesop_ForwardRuleMatch_apply_spec__4___boxed(lean_object* v_cls_4366_, lean_object* v_msg_4367_, lean_object* v___y_4368_, lean_object* v___y_4369_, lean_object* v___y_4370_, lean_object* v___y_4371_, lean_object* v___y_4372_, lean_object* v___y_4373_, lean_object* v___y_4374_){
_start:
{
lean_object* v_res_4375_; 
v_res_4375_ = lp_aesop_Lean_addTrace___at___00Aesop_ForwardRuleMatch_apply_spec__4(v_cls_4366_, v_msg_4367_, v___y_4368_, v___y_4369_, v___y_4370_, v___y_4371_, v___y_4372_, v___y_4373_);
lean_dec(v___y_4373_);
lean_dec_ref(v___y_4372_);
lean_dec(v___y_4371_);
lean_dec_ref(v___y_4370_);
lean_dec(v___y_4369_);
lean_dec(v___y_4368_);
return v_res_4375_;
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_apply_spec__6_spec__7(lean_object* v_00_u03b1_4376_, lean_object* v_x_4377_, lean_object* v___y_4378_, lean_object* v___y_4379_, lean_object* v___y_4380_, lean_object* v___y_4381_, lean_object* v___y_4382_, lean_object* v___y_4383_){
_start:
{
lean_object* v___x_4385_; 
v___x_4385_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_apply_spec__6_spec__7___redArg(v_x_4377_);
return v___x_4385_;
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_apply_spec__6_spec__7___boxed(lean_object* v_00_u03b1_4386_, lean_object* v_x_4387_, lean_object* v___y_4388_, lean_object* v___y_4389_, lean_object* v___y_4390_, lean_object* v___y_4391_, lean_object* v___y_4392_, lean_object* v___y_4393_, lean_object* v___y_4394_){
_start:
{
lean_object* v_res_4395_; 
v_res_4395_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_apply_spec__6_spec__7(v_00_u03b1_4386_, v_x_4387_, v___y_4388_, v___y_4389_, v___y_4390_, v___y_4391_, v___y_4392_, v___y_4393_);
lean_dec(v___y_4393_);
lean_dec_ref(v___y_4392_);
lean_dec(v___y_4391_);
lean_dec_ref(v___y_4390_);
lean_dec(v___y_4389_);
lean_dec(v___y_4388_);
return v_res_4395_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_apply_spec__6_spec__6(lean_object* v_oldTraces_4396_, lean_object* v_data_4397_, lean_object* v_ref_4398_, lean_object* v_msg_4399_, lean_object* v___y_4400_, lean_object* v___y_4401_, lean_object* v___y_4402_, lean_object* v___y_4403_, lean_object* v___y_4404_, lean_object* v___y_4405_){
_start:
{
lean_object* v___x_4407_; 
v___x_4407_ = lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_apply_spec__6_spec__6___redArg(v_oldTraces_4396_, v_data_4397_, v_ref_4398_, v_msg_4399_, v___y_4402_, v___y_4403_, v___y_4404_, v___y_4405_);
return v___x_4407_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_apply_spec__6_spec__6___boxed(lean_object* v_oldTraces_4408_, lean_object* v_data_4409_, lean_object* v_ref_4410_, lean_object* v_msg_4411_, lean_object* v___y_4412_, lean_object* v___y_4413_, lean_object* v___y_4414_, lean_object* v___y_4415_, lean_object* v___y_4416_, lean_object* v___y_4417_, lean_object* v___y_4418_){
_start:
{
lean_object* v_res_4419_; 
v_res_4419_ = lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_ForwardRuleMatch_apply_spec__6_spec__6(v_oldTraces_4408_, v_data_4409_, v_ref_4410_, v_msg_4411_, v___y_4412_, v___y_4413_, v___y_4414_, v___y_4415_, v___y_4416_, v___y_4417_);
lean_dec(v___y_4417_);
lean_dec_ref(v___y_4416_);
lean_dec(v___y_4415_);
lean_dec_ref(v___y_4414_);
lean_dec(v___y_4413_);
lean_dec(v___y_4412_);
return v_res_4419_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__1_spec__3_spec__4_spec__8___redArg(lean_object* v_x_4420_, lean_object* v_x_4421_){
_start:
{
if (lean_obj_tag(v_x_4421_) == 0)
{
return v_x_4420_;
}
else
{
lean_object* v_key_4422_; lean_object* v_value_4423_; lean_object* v_tail_4424_; lean_object* v___x_4426_; uint8_t v_isShared_4427_; uint8_t v_isSharedCheck_4447_; 
v_key_4422_ = lean_ctor_get(v_x_4421_, 0);
v_value_4423_ = lean_ctor_get(v_x_4421_, 1);
v_tail_4424_ = lean_ctor_get(v_x_4421_, 2);
v_isSharedCheck_4447_ = !lean_is_exclusive(v_x_4421_);
if (v_isSharedCheck_4447_ == 0)
{
v___x_4426_ = v_x_4421_;
v_isShared_4427_ = v_isSharedCheck_4447_;
goto v_resetjp_4425_;
}
else
{
lean_inc(v_tail_4424_);
lean_inc(v_value_4423_);
lean_inc(v_key_4422_);
lean_dec(v_x_4421_);
v___x_4426_ = lean_box(0);
v_isShared_4427_ = v_isSharedCheck_4447_;
goto v_resetjp_4425_;
}
v_resetjp_4425_:
{
uint64_t v_hash_4428_; lean_object* v___x_4429_; uint64_t v___x_4430_; uint64_t v___x_4431_; uint64_t v_fold_4432_; uint64_t v___x_4433_; uint64_t v___x_4434_; uint64_t v___x_4435_; size_t v___x_4436_; size_t v___x_4437_; size_t v___x_4438_; size_t v___x_4439_; size_t v___x_4440_; lean_object* v___x_4441_; lean_object* v___x_4443_; 
v_hash_4428_ = lean_ctor_get_uint64(v_key_4422_, sizeof(void*)*1);
v___x_4429_ = lean_array_get_size(v_x_4420_);
v___x_4430_ = 32ULL;
v___x_4431_ = lean_uint64_shift_right(v_hash_4428_, v___x_4430_);
v_fold_4432_ = lean_uint64_xor(v_hash_4428_, v___x_4431_);
v___x_4433_ = 16ULL;
v___x_4434_ = lean_uint64_shift_right(v_fold_4432_, v___x_4433_);
v___x_4435_ = lean_uint64_xor(v_fold_4432_, v___x_4434_);
v___x_4436_ = lean_uint64_to_usize(v___x_4435_);
v___x_4437_ = lean_usize_of_nat(v___x_4429_);
v___x_4438_ = ((size_t)1ULL);
v___x_4439_ = lean_usize_sub(v___x_4437_, v___x_4438_);
v___x_4440_ = lean_usize_land(v___x_4436_, v___x_4439_);
v___x_4441_ = lean_array_uget_borrowed(v_x_4420_, v___x_4440_);
lean_inc(v___x_4441_);
if (v_isShared_4427_ == 0)
{
lean_ctor_set(v___x_4426_, 2, v___x_4441_);
v___x_4443_ = v___x_4426_;
goto v_reusejp_4442_;
}
else
{
lean_object* v_reuseFailAlloc_4446_; 
v_reuseFailAlloc_4446_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_4446_, 0, v_key_4422_);
lean_ctor_set(v_reuseFailAlloc_4446_, 1, v_value_4423_);
lean_ctor_set(v_reuseFailAlloc_4446_, 2, v___x_4441_);
v___x_4443_ = v_reuseFailAlloc_4446_;
goto v_reusejp_4442_;
}
v_reusejp_4442_:
{
lean_object* v___x_4444_; 
v___x_4444_ = lean_array_uset(v_x_4420_, v___x_4440_, v___x_4443_);
v_x_4420_ = v___x_4444_;
v_x_4421_ = v_tail_4424_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__1_spec__3_spec__4___redArg(lean_object* v_i_4448_, lean_object* v_source_4449_, lean_object* v_target_4450_){
_start:
{
lean_object* v___x_4451_; uint8_t v___x_4452_; 
v___x_4451_ = lean_array_get_size(v_source_4449_);
v___x_4452_ = lean_nat_dec_lt(v_i_4448_, v___x_4451_);
if (v___x_4452_ == 0)
{
lean_dec_ref(v_source_4449_);
lean_dec(v_i_4448_);
return v_target_4450_;
}
else
{
lean_object* v_es_4453_; lean_object* v___x_4454_; lean_object* v_source_4455_; lean_object* v_target_4456_; lean_object* v___x_4457_; lean_object* v___x_4458_; 
v_es_4453_ = lean_array_fget(v_source_4449_, v_i_4448_);
v___x_4454_ = lean_box(0);
v_source_4455_ = lean_array_fset(v_source_4449_, v_i_4448_, v___x_4454_);
v_target_4456_ = lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__1_spec__3_spec__4_spec__8___redArg(v_target_4450_, v_es_4453_);
v___x_4457_ = lean_unsigned_to_nat(1u);
v___x_4458_ = lean_nat_add(v_i_4448_, v___x_4457_);
lean_dec(v_i_4448_);
v_i_4448_ = v___x_4458_;
v_source_4449_ = v_source_4455_;
v_target_4450_ = v_target_4456_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__1_spec__3___redArg(lean_object* v_data_4460_){
_start:
{
lean_object* v___x_4461_; lean_object* v___x_4462_; lean_object* v_nbuckets_4463_; lean_object* v___x_4464_; lean_object* v___x_4465_; lean_object* v___x_4466_; lean_object* v___x_4467_; 
v___x_4461_ = lean_array_get_size(v_data_4460_);
v___x_4462_ = lean_unsigned_to_nat(2u);
v_nbuckets_4463_ = lean_nat_mul(v___x_4461_, v___x_4462_);
v___x_4464_ = lean_unsigned_to_nat(0u);
v___x_4465_ = lean_box(0);
v___x_4466_ = lean_mk_array(v_nbuckets_4463_, v___x_4465_);
v___x_4467_ = lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__1_spec__3_spec__4___redArg(v___x_4464_, v_data_4460_, v___x_4466_);
return v___x_4467_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__1_spec__2___redArg(lean_object* v_a_4468_, lean_object* v_x_4469_){
_start:
{
if (lean_obj_tag(v_x_4469_) == 0)
{
uint8_t v___x_4470_; 
v___x_4470_ = 0;
return v___x_4470_;
}
else
{
lean_object* v_key_4471_; lean_object* v_tail_4472_; uint8_t v___y_4474_; lean_object* v_name_4476_; uint8_t v_builder_4477_; uint8_t v_phase_4478_; uint8_t v_scope_4479_; uint64_t v_hash_4480_; lean_object* v_name_4481_; uint8_t v_builder_4482_; uint8_t v_phase_4483_; uint8_t v_scope_4484_; uint64_t v_hash_4485_; uint8_t v___y_4487_; uint8_t v___x_4492_; 
v_key_4471_ = lean_ctor_get(v_x_4469_, 0);
v_tail_4472_ = lean_ctor_get(v_x_4469_, 2);
v_name_4476_ = lean_ctor_get(v_key_4471_, 0);
v_builder_4477_ = lean_ctor_get_uint8(v_key_4471_, sizeof(void*)*1 + 8);
v_phase_4478_ = lean_ctor_get_uint8(v_key_4471_, sizeof(void*)*1 + 9);
v_scope_4479_ = lean_ctor_get_uint8(v_key_4471_, sizeof(void*)*1 + 10);
v_hash_4480_ = lean_ctor_get_uint64(v_key_4471_, sizeof(void*)*1);
v_name_4481_ = lean_ctor_get(v_a_4468_, 0);
v_builder_4482_ = lean_ctor_get_uint8(v_a_4468_, sizeof(void*)*1 + 8);
v_phase_4483_ = lean_ctor_get_uint8(v_a_4468_, sizeof(void*)*1 + 9);
v_scope_4484_ = lean_ctor_get_uint8(v_a_4468_, sizeof(void*)*1 + 10);
v_hash_4485_ = lean_ctor_get_uint64(v_a_4468_, sizeof(void*)*1);
v___x_4492_ = lean_uint64_dec_eq(v_hash_4480_, v_hash_4485_);
if (v___x_4492_ == 0)
{
v___y_4487_ = v___x_4492_;
goto v___jp_4486_;
}
else
{
uint8_t v___x_4493_; 
v___x_4493_ = lp_aesop_Aesop_instBEqBuilderName_beq(v_builder_4477_, v_builder_4482_);
v___y_4487_ = v___x_4493_;
goto v___jp_4486_;
}
v___jp_4473_:
{
if (v___y_4474_ == 0)
{
v_x_4469_ = v_tail_4472_;
goto _start;
}
else
{
return v___y_4474_;
}
}
v___jp_4486_:
{
if (v___y_4487_ == 0)
{
v_x_4469_ = v_tail_4472_;
goto _start;
}
else
{
uint8_t v___x_4489_; 
v___x_4489_ = lp_aesop_Aesop_instBEqPhaseName_beq(v_phase_4478_, v_phase_4483_);
if (v___x_4489_ == 0)
{
v___y_4474_ = v___x_4489_;
goto v___jp_4473_;
}
else
{
uint8_t v___x_4490_; 
v___x_4490_ = lp_aesop_Aesop_instBEqScopeName_beq(v_scope_4479_, v_scope_4484_);
if (v___x_4490_ == 0)
{
v___y_4474_ = v___x_4490_;
goto v___jp_4473_;
}
else
{
uint8_t v___x_4491_; 
v___x_4491_ = lean_name_eq(v_name_4476_, v_name_4481_);
v___y_4474_ = v___x_4491_;
goto v___jp_4473_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__1_spec__2___redArg___boxed(lean_object* v_a_4494_, lean_object* v_x_4495_){
_start:
{
uint8_t v_res_4496_; lean_object* v_r_4497_; 
v_res_4496_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__1_spec__2___redArg(v_a_4494_, v_x_4495_);
lean_dec(v_x_4495_);
lean_dec_ref(v_a_4494_);
v_r_4497_ = lean_box(v_res_4496_);
return v_r_4497_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__1_spec__4___redArg(lean_object* v_a_4498_, lean_object* v_b_4499_, lean_object* v_x_4500_){
_start:
{
if (lean_obj_tag(v_x_4500_) == 0)
{
lean_dec(v_b_4499_);
lean_dec_ref(v_a_4498_);
return v_x_4500_;
}
else
{
lean_object* v_key_4501_; lean_object* v_value_4502_; lean_object* v_tail_4503_; lean_object* v___x_4505_; uint8_t v_isShared_4506_; uint8_t v_isSharedCheck_4532_; 
v_key_4501_ = lean_ctor_get(v_x_4500_, 0);
v_value_4502_ = lean_ctor_get(v_x_4500_, 1);
v_tail_4503_ = lean_ctor_get(v_x_4500_, 2);
v_isSharedCheck_4532_ = !lean_is_exclusive(v_x_4500_);
if (v_isSharedCheck_4532_ == 0)
{
v___x_4505_ = v_x_4500_;
v_isShared_4506_ = v_isSharedCheck_4532_;
goto v_resetjp_4504_;
}
else
{
lean_inc(v_tail_4503_);
lean_inc(v_value_4502_);
lean_inc(v_key_4501_);
lean_dec(v_x_4500_);
v___x_4505_ = lean_box(0);
v_isShared_4506_ = v_isSharedCheck_4532_;
goto v_resetjp_4504_;
}
v_resetjp_4504_:
{
uint8_t v___y_4513_; lean_object* v_name_4515_; uint8_t v_builder_4516_; uint8_t v_phase_4517_; uint8_t v_scope_4518_; uint64_t v_hash_4519_; lean_object* v_name_4520_; uint8_t v_builder_4521_; uint8_t v_phase_4522_; uint8_t v_scope_4523_; uint64_t v_hash_4524_; uint8_t v___y_4526_; uint8_t v___x_4530_; 
v_name_4515_ = lean_ctor_get(v_key_4501_, 0);
v_builder_4516_ = lean_ctor_get_uint8(v_key_4501_, sizeof(void*)*1 + 8);
v_phase_4517_ = lean_ctor_get_uint8(v_key_4501_, sizeof(void*)*1 + 9);
v_scope_4518_ = lean_ctor_get_uint8(v_key_4501_, sizeof(void*)*1 + 10);
v_hash_4519_ = lean_ctor_get_uint64(v_key_4501_, sizeof(void*)*1);
v_name_4520_ = lean_ctor_get(v_a_4498_, 0);
v_builder_4521_ = lean_ctor_get_uint8(v_a_4498_, sizeof(void*)*1 + 8);
v_phase_4522_ = lean_ctor_get_uint8(v_a_4498_, sizeof(void*)*1 + 9);
v_scope_4523_ = lean_ctor_get_uint8(v_a_4498_, sizeof(void*)*1 + 10);
v_hash_4524_ = lean_ctor_get_uint64(v_a_4498_, sizeof(void*)*1);
v___x_4530_ = lean_uint64_dec_eq(v_hash_4519_, v_hash_4524_);
if (v___x_4530_ == 0)
{
v___y_4526_ = v___x_4530_;
goto v___jp_4525_;
}
else
{
uint8_t v___x_4531_; 
v___x_4531_ = lp_aesop_Aesop_instBEqBuilderName_beq(v_builder_4516_, v_builder_4521_);
v___y_4526_ = v___x_4531_;
goto v___jp_4525_;
}
v___jp_4507_:
{
lean_object* v___x_4508_; lean_object* v___x_4510_; 
v___x_4508_ = lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__1_spec__4___redArg(v_a_4498_, v_b_4499_, v_tail_4503_);
if (v_isShared_4506_ == 0)
{
lean_ctor_set(v___x_4505_, 2, v___x_4508_);
v___x_4510_ = v___x_4505_;
goto v_reusejp_4509_;
}
else
{
lean_object* v_reuseFailAlloc_4511_; 
v_reuseFailAlloc_4511_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_4511_, 0, v_key_4501_);
lean_ctor_set(v_reuseFailAlloc_4511_, 1, v_value_4502_);
lean_ctor_set(v_reuseFailAlloc_4511_, 2, v___x_4508_);
v___x_4510_ = v_reuseFailAlloc_4511_;
goto v_reusejp_4509_;
}
v_reusejp_4509_:
{
return v___x_4510_;
}
}
v___jp_4512_:
{
if (v___y_4513_ == 0)
{
goto v___jp_4507_;
}
else
{
lean_object* v___x_4514_; 
lean_del_object(v___x_4505_);
lean_dec(v_value_4502_);
lean_dec(v_key_4501_);
v___x_4514_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_4514_, 0, v_a_4498_);
lean_ctor_set(v___x_4514_, 1, v_b_4499_);
lean_ctor_set(v___x_4514_, 2, v_tail_4503_);
return v___x_4514_;
}
}
v___jp_4525_:
{
if (v___y_4526_ == 0)
{
goto v___jp_4507_;
}
else
{
uint8_t v___x_4527_; 
v___x_4527_ = lp_aesop_Aesop_instBEqPhaseName_beq(v_phase_4517_, v_phase_4522_);
if (v___x_4527_ == 0)
{
v___y_4513_ = v___x_4527_;
goto v___jp_4512_;
}
else
{
uint8_t v___x_4528_; 
v___x_4528_ = lp_aesop_Aesop_instBEqScopeName_beq(v_scope_4518_, v_scope_4523_);
if (v___x_4528_ == 0)
{
v___y_4513_ = v___x_4528_;
goto v___jp_4512_;
}
else
{
uint8_t v___x_4529_; 
v___x_4529_ = lean_name_eq(v_name_4515_, v_name_4520_);
v___y_4513_ = v___x_4529_;
goto v___jp_4512_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__1___redArg(lean_object* v_m_4533_, lean_object* v_a_4534_, lean_object* v_b_4535_){
_start:
{
lean_object* v_size_4536_; lean_object* v_buckets_4537_; lean_object* v___x_4539_; uint8_t v_isShared_4540_; uint8_t v_isSharedCheck_4580_; 
v_size_4536_ = lean_ctor_get(v_m_4533_, 0);
v_buckets_4537_ = lean_ctor_get(v_m_4533_, 1);
v_isSharedCheck_4580_ = !lean_is_exclusive(v_m_4533_);
if (v_isSharedCheck_4580_ == 0)
{
v___x_4539_ = v_m_4533_;
v_isShared_4540_ = v_isSharedCheck_4580_;
goto v_resetjp_4538_;
}
else
{
lean_inc(v_buckets_4537_);
lean_inc(v_size_4536_);
lean_dec(v_m_4533_);
v___x_4539_ = lean_box(0);
v_isShared_4540_ = v_isSharedCheck_4580_;
goto v_resetjp_4538_;
}
v_resetjp_4538_:
{
uint64_t v_hash_4541_; lean_object* v___x_4542_; uint64_t v___x_4543_; uint64_t v___x_4544_; uint64_t v_fold_4545_; uint64_t v___x_4546_; uint64_t v___x_4547_; uint64_t v___x_4548_; size_t v___x_4549_; size_t v___x_4550_; size_t v___x_4551_; size_t v___x_4552_; size_t v___x_4553_; lean_object* v_bkt_4554_; uint8_t v___x_4555_; 
v_hash_4541_ = lean_ctor_get_uint64(v_a_4534_, sizeof(void*)*1);
v___x_4542_ = lean_array_get_size(v_buckets_4537_);
v___x_4543_ = 32ULL;
v___x_4544_ = lean_uint64_shift_right(v_hash_4541_, v___x_4543_);
v_fold_4545_ = lean_uint64_xor(v_hash_4541_, v___x_4544_);
v___x_4546_ = 16ULL;
v___x_4547_ = lean_uint64_shift_right(v_fold_4545_, v___x_4546_);
v___x_4548_ = lean_uint64_xor(v_fold_4545_, v___x_4547_);
v___x_4549_ = lean_uint64_to_usize(v___x_4548_);
v___x_4550_ = lean_usize_of_nat(v___x_4542_);
v___x_4551_ = ((size_t)1ULL);
v___x_4552_ = lean_usize_sub(v___x_4550_, v___x_4551_);
v___x_4553_ = lean_usize_land(v___x_4549_, v___x_4552_);
v_bkt_4554_ = lean_array_uget_borrowed(v_buckets_4537_, v___x_4553_);
v___x_4555_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__1_spec__2___redArg(v_a_4534_, v_bkt_4554_);
if (v___x_4555_ == 0)
{
lean_object* v___x_4556_; lean_object* v_size_x27_4557_; lean_object* v___x_4558_; lean_object* v_buckets_x27_4559_; lean_object* v___x_4560_; lean_object* v___x_4561_; lean_object* v___x_4562_; lean_object* v___x_4563_; lean_object* v___x_4564_; uint8_t v___x_4565_; 
v___x_4556_ = lean_unsigned_to_nat(1u);
v_size_x27_4557_ = lean_nat_add(v_size_4536_, v___x_4556_);
lean_dec(v_size_4536_);
lean_inc(v_bkt_4554_);
v___x_4558_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_4558_, 0, v_a_4534_);
lean_ctor_set(v___x_4558_, 1, v_b_4535_);
lean_ctor_set(v___x_4558_, 2, v_bkt_4554_);
v_buckets_x27_4559_ = lean_array_uset(v_buckets_4537_, v___x_4553_, v___x_4558_);
v___x_4560_ = lean_unsigned_to_nat(4u);
v___x_4561_ = lean_nat_mul(v_size_x27_4557_, v___x_4560_);
v___x_4562_ = lean_unsigned_to_nat(3u);
v___x_4563_ = lean_nat_div(v___x_4561_, v___x_4562_);
lean_dec(v___x_4561_);
v___x_4564_ = lean_array_get_size(v_buckets_x27_4559_);
v___x_4565_ = lean_nat_dec_le(v___x_4563_, v___x_4564_);
lean_dec(v___x_4563_);
if (v___x_4565_ == 0)
{
lean_object* v_val_4566_; lean_object* v___x_4568_; 
v_val_4566_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__1_spec__3___redArg(v_buckets_x27_4559_);
if (v_isShared_4540_ == 0)
{
lean_ctor_set(v___x_4539_, 1, v_val_4566_);
lean_ctor_set(v___x_4539_, 0, v_size_x27_4557_);
v___x_4568_ = v___x_4539_;
goto v_reusejp_4567_;
}
else
{
lean_object* v_reuseFailAlloc_4569_; 
v_reuseFailAlloc_4569_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4569_, 0, v_size_x27_4557_);
lean_ctor_set(v_reuseFailAlloc_4569_, 1, v_val_4566_);
v___x_4568_ = v_reuseFailAlloc_4569_;
goto v_reusejp_4567_;
}
v_reusejp_4567_:
{
return v___x_4568_;
}
}
else
{
lean_object* v___x_4571_; 
if (v_isShared_4540_ == 0)
{
lean_ctor_set(v___x_4539_, 1, v_buckets_x27_4559_);
lean_ctor_set(v___x_4539_, 0, v_size_x27_4557_);
v___x_4571_ = v___x_4539_;
goto v_reusejp_4570_;
}
else
{
lean_object* v_reuseFailAlloc_4572_; 
v_reuseFailAlloc_4572_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4572_, 0, v_size_x27_4557_);
lean_ctor_set(v_reuseFailAlloc_4572_, 1, v_buckets_x27_4559_);
v___x_4571_ = v_reuseFailAlloc_4572_;
goto v_reusejp_4570_;
}
v_reusejp_4570_:
{
return v___x_4571_;
}
}
}
else
{
lean_object* v___x_4573_; lean_object* v_buckets_x27_4574_; lean_object* v___x_4575_; lean_object* v___x_4576_; lean_object* v___x_4578_; 
lean_inc(v_bkt_4554_);
v___x_4573_ = lean_box(0);
v_buckets_x27_4574_ = lean_array_uset(v_buckets_4537_, v___x_4553_, v___x_4573_);
v___x_4575_ = lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__1_spec__4___redArg(v_a_4534_, v_b_4535_, v_bkt_4554_);
v___x_4576_ = lean_array_uset(v_buckets_x27_4574_, v___x_4553_, v___x_4575_);
if (v_isShared_4540_ == 0)
{
lean_ctor_set(v___x_4539_, 1, v___x_4576_);
v___x_4578_ = v___x_4539_;
goto v_reusejp_4577_;
}
else
{
lean_object* v_reuseFailAlloc_4579_; 
v_reuseFailAlloc_4579_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4579_, 0, v_size_4536_);
lean_ctor_set(v_reuseFailAlloc_4579_, 1, v___x_4576_);
v___x_4578_ = v_reuseFailAlloc_4579_;
goto v_reusejp_4577_;
}
v_reusejp_4577_:
{
return v___x_4578_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__0_spec__0___redArg(lean_object* v_a_4581_, lean_object* v_x_4582_){
_start:
{
if (lean_obj_tag(v_x_4582_) == 0)
{
lean_object* v___x_4583_; 
v___x_4583_ = lean_box(0);
return v___x_4583_;
}
else
{
lean_object* v_key_4584_; lean_object* v_value_4585_; lean_object* v_tail_4586_; uint8_t v___y_4588_; lean_object* v_name_4591_; uint8_t v_builder_4592_; uint8_t v_phase_4593_; uint8_t v_scope_4594_; uint64_t v_hash_4595_; lean_object* v_name_4596_; uint8_t v_builder_4597_; uint8_t v_phase_4598_; uint8_t v_scope_4599_; uint64_t v_hash_4600_; uint8_t v___y_4602_; uint8_t v___x_4607_; 
v_key_4584_ = lean_ctor_get(v_x_4582_, 0);
v_value_4585_ = lean_ctor_get(v_x_4582_, 1);
v_tail_4586_ = lean_ctor_get(v_x_4582_, 2);
v_name_4591_ = lean_ctor_get(v_key_4584_, 0);
v_builder_4592_ = lean_ctor_get_uint8(v_key_4584_, sizeof(void*)*1 + 8);
v_phase_4593_ = lean_ctor_get_uint8(v_key_4584_, sizeof(void*)*1 + 9);
v_scope_4594_ = lean_ctor_get_uint8(v_key_4584_, sizeof(void*)*1 + 10);
v_hash_4595_ = lean_ctor_get_uint64(v_key_4584_, sizeof(void*)*1);
v_name_4596_ = lean_ctor_get(v_a_4581_, 0);
v_builder_4597_ = lean_ctor_get_uint8(v_a_4581_, sizeof(void*)*1 + 8);
v_phase_4598_ = lean_ctor_get_uint8(v_a_4581_, sizeof(void*)*1 + 9);
v_scope_4599_ = lean_ctor_get_uint8(v_a_4581_, sizeof(void*)*1 + 10);
v_hash_4600_ = lean_ctor_get_uint64(v_a_4581_, sizeof(void*)*1);
v___x_4607_ = lean_uint64_dec_eq(v_hash_4595_, v_hash_4600_);
if (v___x_4607_ == 0)
{
v___y_4602_ = v___x_4607_;
goto v___jp_4601_;
}
else
{
uint8_t v___x_4608_; 
v___x_4608_ = lp_aesop_Aesop_instBEqBuilderName_beq(v_builder_4592_, v_builder_4597_);
v___y_4602_ = v___x_4608_;
goto v___jp_4601_;
}
v___jp_4587_:
{
if (v___y_4588_ == 0)
{
v_x_4582_ = v_tail_4586_;
goto _start;
}
else
{
lean_object* v___x_4590_; 
lean_inc(v_value_4585_);
v___x_4590_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4590_, 0, v_value_4585_);
return v___x_4590_;
}
}
v___jp_4601_:
{
if (v___y_4602_ == 0)
{
v_x_4582_ = v_tail_4586_;
goto _start;
}
else
{
uint8_t v___x_4604_; 
v___x_4604_ = lp_aesop_Aesop_instBEqPhaseName_beq(v_phase_4593_, v_phase_4598_);
if (v___x_4604_ == 0)
{
v___y_4588_ = v___x_4604_;
goto v___jp_4587_;
}
else
{
uint8_t v___x_4605_; 
v___x_4605_ = lp_aesop_Aesop_instBEqScopeName_beq(v_scope_4594_, v_scope_4599_);
if (v___x_4605_ == 0)
{
v___y_4588_ = v___x_4605_;
goto v___jp_4587_;
}
else
{
uint8_t v___x_4606_; 
v___x_4606_ = lean_name_eq(v_name_4591_, v_name_4596_);
v___y_4588_ = v___x_4606_;
goto v___jp_4587_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__0_spec__0___redArg___boxed(lean_object* v_a_4609_, lean_object* v_x_4610_){
_start:
{
lean_object* v_res_4611_; 
v_res_4611_ = lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__0_spec__0___redArg(v_a_4609_, v_x_4610_);
lean_dec(v_x_4610_);
lean_dec_ref(v_a_4609_);
return v_res_4611_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__0___redArg(lean_object* v_m_4612_, lean_object* v_a_4613_){
_start:
{
lean_object* v_buckets_4614_; uint64_t v_hash_4615_; lean_object* v___x_4616_; uint64_t v___x_4617_; uint64_t v___x_4618_; uint64_t v_fold_4619_; uint64_t v___x_4620_; uint64_t v___x_4621_; uint64_t v___x_4622_; size_t v___x_4623_; size_t v___x_4624_; size_t v___x_4625_; size_t v___x_4626_; size_t v___x_4627_; lean_object* v___x_4628_; lean_object* v___x_4629_; 
v_buckets_4614_ = lean_ctor_get(v_m_4612_, 1);
v_hash_4615_ = lean_ctor_get_uint64(v_a_4613_, sizeof(void*)*1);
v___x_4616_ = lean_array_get_size(v_buckets_4614_);
v___x_4617_ = 32ULL;
v___x_4618_ = lean_uint64_shift_right(v_hash_4615_, v___x_4617_);
v_fold_4619_ = lean_uint64_xor(v_hash_4615_, v___x_4618_);
v___x_4620_ = 16ULL;
v___x_4621_ = lean_uint64_shift_right(v_fold_4619_, v___x_4620_);
v___x_4622_ = lean_uint64_xor(v_fold_4619_, v___x_4621_);
v___x_4623_ = lean_uint64_to_usize(v___x_4622_);
v___x_4624_ = lean_usize_of_nat(v___x_4616_);
v___x_4625_ = ((size_t)1ULL);
v___x_4626_ = lean_usize_sub(v___x_4624_, v___x_4625_);
v___x_4627_ = lean_usize_land(v___x_4623_, v___x_4626_);
v___x_4628_ = lean_array_uget_borrowed(v_buckets_4614_, v___x_4627_);
v___x_4629_ = lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__0_spec__0___redArg(v_a_4613_, v___x_4628_);
return v___x_4629_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__0___redArg___boxed(lean_object* v_m_4630_, lean_object* v_a_4631_){
_start:
{
lean_object* v_res_4632_; 
v_res_4632_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__0___redArg(v_m_4630_, v_a_4631_);
lean_dec_ref(v_a_4631_);
lean_dec_ref(v_m_4630_);
return v_res_4632_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__2(lean_object* v_as_4633_, size_t v_sz_4634_, size_t v_i_4635_, lean_object* v_b_4636_){
_start:
{
lean_object* v_a_4638_; uint8_t v___x_4642_; 
v___x_4642_ = lean_usize_dec_lt(v_i_4635_, v_sz_4634_);
if (v___x_4642_ == 0)
{
return v_b_4636_;
}
else
{
lean_object* v_a_4643_; lean_object* v_rule_4644_; lean_object* v_name_4645_; lean_object* v___x_4646_; 
v_a_4643_ = lean_array_uget_borrowed(v_as_4633_, v_i_4635_);
v_rule_4644_ = lean_ctor_get(v_a_4643_, 0);
v_name_4645_ = lean_ctor_get(v_rule_4644_, 1);
v___x_4646_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__0___redArg(v_b_4636_, v_name_4645_);
if (lean_obj_tag(v___x_4646_) == 1)
{
lean_object* v_val_4647_; lean_object* v___x_4648_; lean_object* v___x_4649_; 
v_val_4647_ = lean_ctor_get(v___x_4646_, 0);
lean_inc(v_val_4647_);
lean_dec_ref_known(v___x_4646_, 1);
lean_inc(v_a_4643_);
v___x_4648_ = lean_array_push(v_val_4647_, v_a_4643_);
lean_inc_ref(v_name_4645_);
v___x_4649_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__1___redArg(v_b_4636_, v_name_4645_, v___x_4648_);
v_a_4638_ = v___x_4649_;
goto v___jp_4637_;
}
else
{
lean_object* v___x_4650_; lean_object* v___x_4651_; lean_object* v___x_4652_; lean_object* v___x_4653_; 
lean_dec(v___x_4646_);
v___x_4650_ = lean_unsigned_to_nat(1u);
v___x_4651_ = lean_mk_empty_array_with_capacity(v___x_4650_);
lean_inc(v_a_4643_);
v___x_4652_ = lean_array_push(v___x_4651_, v_a_4643_);
lean_inc_ref(v_name_4645_);
v___x_4653_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__1___redArg(v_b_4636_, v_name_4645_, v___x_4652_);
v_a_4638_ = v___x_4653_;
goto v___jp_4637_;
}
}
v___jp_4637_:
{
size_t v___x_4639_; size_t v___x_4640_; 
v___x_4639_ = ((size_t)1ULL);
v___x_4640_ = lean_usize_add(v_i_4635_, v___x_4639_);
v_i_4635_ = v___x_4640_;
v_b_4636_ = v_a_4638_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__2___boxed(lean_object* v_as_4654_, lean_object* v_sz_4655_, lean_object* v_i_4656_, lean_object* v_b_4657_){
_start:
{
size_t v_sz_boxed_4658_; size_t v_i_boxed_4659_; lean_object* v_res_4660_; 
v_sz_boxed_4658_ = lean_unbox_usize(v_sz_4655_);
lean_dec(v_sz_4655_);
v_i_boxed_4659_ = lean_unbox_usize(v_i_4656_);
lean_dec(v_i_4656_);
v_res_4660_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__2(v_as_4654_, v_sz_boxed_4658_, v_i_boxed_4659_, v_b_4657_);
lean_dec_ref(v_as_4654_);
return v_res_4660_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__3___redArg(lean_object* v_mkExtra_x3f_4663_, lean_object* v_a_4664_, lean_object* v_a_4665_){
_start:
{
if (lean_obj_tag(v_a_4664_) == 0)
{
lean_object* v___x_4666_; 
lean_dec_ref(v_mkExtra_x3f_4663_);
v___x_4666_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4666_, 0, v_a_4665_);
return v___x_4666_;
}
else
{
lean_object* v_key_4667_; lean_object* v_value_4668_; lean_object* v_tail_4669_; lean_object* v_snd_4670_; lean_object* v___x_4672_; uint8_t v_isShared_4673_; uint8_t v_isSharedCheck_4699_; 
v_key_4667_ = lean_ctor_get(v_a_4664_, 0);
v_value_4668_ = lean_ctor_get(v_a_4664_, 1);
v_tail_4669_ = lean_ctor_get(v_a_4664_, 2);
v_snd_4670_ = lean_ctor_get(v_a_4665_, 1);
v_isSharedCheck_4699_ = !lean_is_exclusive(v_a_4665_);
if (v_isSharedCheck_4699_ == 0)
{
lean_object* v_unused_4700_; 
v_unused_4700_ = lean_ctor_get(v_a_4665_, 0);
lean_dec(v_unused_4700_);
v___x_4672_ = v_a_4665_;
v_isShared_4673_ = v_isSharedCheck_4699_;
goto v_resetjp_4671_;
}
else
{
lean_inc(v_snd_4670_);
lean_dec(v_a_4665_);
v___x_4672_ = lean_box(0);
v_isShared_4673_ = v_isSharedCheck_4699_;
goto v_resetjp_4671_;
}
v_resetjp_4671_:
{
lean_object* v___x_4674_; lean_object* v___x_4675_; lean_object* v___x_4676_; lean_object* v___x_4677_; 
v___x_4674_ = lp_aesop_Aesop_instInhabitedForwardRuleMatch_default;
v___x_4675_ = lean_unsigned_to_nat(0u);
v___x_4676_ = lean_array_get_borrowed(v___x_4674_, v_value_4668_, v___x_4675_);
lean_inc_ref(v_mkExtra_x3f_4663_);
lean_inc(v___x_4676_);
v___x_4677_ = lean_apply_1(v_mkExtra_x3f_4663_, v___x_4676_);
if (lean_obj_tag(v___x_4677_) == 1)
{
lean_object* v_val_4678_; lean_object* v___x_4680_; uint8_t v_isShared_4681_; uint8_t v_isSharedCheck_4693_; 
v_val_4678_ = lean_ctor_get(v___x_4677_, 0);
v_isSharedCheck_4693_ = !lean_is_exclusive(v___x_4677_);
if (v_isSharedCheck_4693_ == 0)
{
v___x_4680_ = v___x_4677_;
v_isShared_4681_ = v_isSharedCheck_4693_;
goto v_resetjp_4679_;
}
else
{
lean_inc(v_val_4678_);
lean_dec(v___x_4677_);
v___x_4680_ = lean_box(0);
v_isShared_4681_ = v_isSharedCheck_4693_;
goto v_resetjp_4679_;
}
v_resetjp_4679_:
{
lean_object* v___x_4682_; lean_object* v___x_4683_; lean_object* v___x_4685_; 
v___x_4682_ = lean_box(0);
v___x_4683_ = lean_box(0);
lean_inc(v_value_4668_);
if (v_isShared_4681_ == 0)
{
lean_ctor_set_tag(v___x_4680_, 10);
lean_ctor_set(v___x_4680_, 0, v_value_4668_);
v___x_4685_ = v___x_4680_;
goto v_reusejp_4684_;
}
else
{
lean_object* v_reuseFailAlloc_4692_; 
v_reuseFailAlloc_4692_ = lean_alloc_ctor(10, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4692_, 0, v_value_4668_);
v___x_4685_ = v_reuseFailAlloc_4692_;
goto v_reusejp_4684_;
}
v_reusejp_4684_:
{
lean_object* v___x_4686_; lean_object* v___x_4687_; lean_object* v___x_4689_; 
lean_inc(v_key_4667_);
v___x_4686_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_4686_, 0, v_key_4667_);
lean_ctor_set(v___x_4686_, 1, v___x_4683_);
lean_ctor_set(v___x_4686_, 2, v___x_4682_);
lean_ctor_set(v___x_4686_, 3, v_val_4678_);
lean_ctor_set(v___x_4686_, 4, v___x_4685_);
v___x_4687_ = lean_array_push(v_snd_4670_, v___x_4686_);
if (v_isShared_4673_ == 0)
{
lean_ctor_set(v___x_4672_, 1, v___x_4687_);
lean_ctor_set(v___x_4672_, 0, v___x_4682_);
v___x_4689_ = v___x_4672_;
goto v_reusejp_4688_;
}
else
{
lean_object* v_reuseFailAlloc_4691_; 
v_reuseFailAlloc_4691_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4691_, 0, v___x_4682_);
lean_ctor_set(v_reuseFailAlloc_4691_, 1, v___x_4687_);
v___x_4689_ = v_reuseFailAlloc_4691_;
goto v_reusejp_4688_;
}
v_reusejp_4688_:
{
v_a_4664_ = v_tail_4669_;
v_a_4665_ = v___x_4689_;
goto _start;
}
}
}
}
else
{
lean_object* v___x_4694_; lean_object* v___x_4696_; 
lean_dec(v___x_4677_);
lean_dec_ref(v_mkExtra_x3f_4663_);
v___x_4694_ = ((lean_object*)(lp_aesop___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__3___redArg___closed__0));
if (v_isShared_4673_ == 0)
{
lean_ctor_set(v___x_4672_, 0, v___x_4694_);
v___x_4696_ = v___x_4672_;
goto v_reusejp_4695_;
}
else
{
lean_object* v_reuseFailAlloc_4698_; 
v_reuseFailAlloc_4698_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4698_, 0, v___x_4694_);
lean_ctor_set(v_reuseFailAlloc_4698_, 1, v_snd_4670_);
v___x_4696_ = v_reuseFailAlloc_4698_;
goto v_reusejp_4695_;
}
v_reusejp_4695_:
{
lean_object* v___x_4697_; 
v___x_4697_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_4697_, 0, v___x_4696_);
return v___x_4697_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__3___redArg___boxed(lean_object* v_mkExtra_x3f_4701_, lean_object* v_a_4702_, lean_object* v_a_4703_){
_start:
{
lean_object* v_res_4704_; 
v_res_4704_ = lp_aesop___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__3___redArg(v_mkExtra_x3f_4701_, v_a_4702_, v_a_4703_);
lean_dec(v_a_4702_);
return v_res_4704_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__4___redArg(lean_object* v_mkExtra_x3f_4705_, lean_object* v_as_4706_, size_t v_sz_4707_, size_t v_i_4708_, lean_object* v_b_4709_){
_start:
{
uint8_t v___x_4710_; 
v___x_4710_ = lean_usize_dec_lt(v_i_4708_, v_sz_4707_);
if (v___x_4710_ == 0)
{
lean_dec_ref(v_mkExtra_x3f_4705_);
return v_b_4709_;
}
else
{
lean_object* v_a_4711_; lean_object* v___x_4712_; 
v_a_4711_ = lean_array_uget_borrowed(v_as_4706_, v_i_4708_);
lean_inc_ref(v_mkExtra_x3f_4705_);
v___x_4712_ = lp_aesop___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__3___redArg(v_mkExtra_x3f_4705_, v_a_4711_, v_b_4709_);
if (lean_obj_tag(v___x_4712_) == 0)
{
lean_object* v_a_4713_; 
lean_dec_ref(v_mkExtra_x3f_4705_);
v_a_4713_ = lean_ctor_get(v___x_4712_, 0);
lean_inc(v_a_4713_);
lean_dec_ref_known(v___x_4712_, 1);
return v_a_4713_;
}
else
{
lean_object* v_a_4714_; size_t v___x_4715_; size_t v___x_4716_; 
v_a_4714_ = lean_ctor_get(v___x_4712_, 0);
lean_inc(v_a_4714_);
lean_dec_ref_known(v___x_4712_, 1);
v___x_4715_ = ((size_t)1ULL);
v___x_4716_ = lean_usize_add(v_i_4708_, v___x_4715_);
v_i_4708_ = v___x_4716_;
v_b_4709_ = v_a_4714_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__4___redArg___boxed(lean_object* v_mkExtra_x3f_4718_, lean_object* v_as_4719_, lean_object* v_sz_4720_, lean_object* v_i_4721_, lean_object* v_b_4722_){
_start:
{
size_t v_sz_boxed_4723_; size_t v_i_boxed_4724_; lean_object* v_res_4725_; 
v_sz_boxed_4723_ = lean_unbox_usize(v_sz_4720_);
lean_dec(v_sz_4720_);
v_i_boxed_4724_ = lean_unbox_usize(v_i_4721_);
lean_dec(v_i_4721_);
v_res_4725_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__4___redArg(v_mkExtra_x3f_4718_, v_as_4719_, v_sz_boxed_4723_, v_i_boxed_4724_, v_b_4722_);
lean_dec_ref(v_as_4719_);
return v_res_4725_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f___redArg___closed__0(void){
_start:
{
lean_object* v___x_4726_; lean_object* v___x_4727_; lean_object* v___x_4728_; 
v___x_4726_ = lean_box(0);
v___x_4727_ = lean_unsigned_to_nat(16u);
v___x_4728_ = lean_mk_array(v___x_4727_, v___x_4726_);
return v___x_4728_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f___redArg___closed__1(void){
_start:
{
lean_object* v___x_4729_; lean_object* v___x_4730_; lean_object* v_ruleMap_4731_; 
v___x_4729_ = lean_obj_once(&lp_aesop___private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f___redArg___closed__0, &lp_aesop___private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f___redArg___closed__0_once, _init_lp_aesop___private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f___redArg___closed__0);
v___x_4730_ = lean_unsigned_to_nat(0u);
v_ruleMap_4731_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_ruleMap_4731_, 0, v___x_4730_);
lean_ctor_set(v_ruleMap_4731_, 1, v___x_4729_);
return v_ruleMap_4731_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f___redArg(lean_object* v_ms_4732_, lean_object* v_mkExtra_x3f_4733_){
_start:
{
lean_object* v_ruleMap_4734_; size_t v_sz_4735_; size_t v___x_4736_; lean_object* v___x_4737_; lean_object* v_size_4738_; lean_object* v_buckets_4739_; lean_object* v___x_4741_; uint8_t v_isShared_4742_; uint8_t v_isSharedCheck_4754_; 
v_ruleMap_4734_ = lean_obj_once(&lp_aesop___private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f___redArg___closed__1, &lp_aesop___private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f___redArg___closed__1_once, _init_lp_aesop___private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f___redArg___closed__1);
v_sz_4735_ = lean_array_size(v_ms_4732_);
v___x_4736_ = ((size_t)0ULL);
v___x_4737_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__2(v_ms_4732_, v_sz_4735_, v___x_4736_, v_ruleMap_4734_);
v_size_4738_ = lean_ctor_get(v___x_4737_, 0);
v_buckets_4739_ = lean_ctor_get(v___x_4737_, 1);
v_isSharedCheck_4754_ = !lean_is_exclusive(v___x_4737_);
if (v_isSharedCheck_4754_ == 0)
{
v___x_4741_ = v___x_4737_;
v_isShared_4742_ = v_isSharedCheck_4754_;
goto v_resetjp_4740_;
}
else
{
lean_inc(v_buckets_4739_);
lean_inc(v_size_4738_);
lean_dec(v___x_4737_);
v___x_4741_ = lean_box(0);
v_isShared_4742_ = v_isSharedCheck_4754_;
goto v_resetjp_4740_;
}
v_resetjp_4740_:
{
lean_object* v___x_4743_; lean_object* v___x_4744_; lean_object* v___x_4746_; 
v___x_4743_ = lean_mk_empty_array_with_capacity(v_size_4738_);
lean_dec(v_size_4738_);
v___x_4744_ = lean_box(0);
if (v_isShared_4742_ == 0)
{
lean_ctor_set(v___x_4741_, 1, v___x_4743_);
lean_ctor_set(v___x_4741_, 0, v___x_4744_);
v___x_4746_ = v___x_4741_;
goto v_reusejp_4745_;
}
else
{
lean_object* v_reuseFailAlloc_4753_; 
v_reuseFailAlloc_4753_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_4753_, 0, v___x_4744_);
lean_ctor_set(v_reuseFailAlloc_4753_, 1, v___x_4743_);
v___x_4746_ = v_reuseFailAlloc_4753_;
goto v_reusejp_4745_;
}
v_reusejp_4745_:
{
size_t v_sz_4747_; lean_object* v___x_4748_; lean_object* v_fst_4749_; 
v_sz_4747_ = lean_array_size(v_buckets_4739_);
v___x_4748_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__4___redArg(v_mkExtra_x3f_4733_, v_buckets_4739_, v_sz_4747_, v___x_4736_, v___x_4746_);
lean_dec_ref(v_buckets_4739_);
v_fst_4749_ = lean_ctor_get(v___x_4748_, 0);
lean_inc(v_fst_4749_);
if (lean_obj_tag(v_fst_4749_) == 0)
{
lean_object* v_snd_4750_; lean_object* v___x_4751_; 
v_snd_4750_ = lean_ctor_get(v___x_4748_, 1);
lean_inc(v_snd_4750_);
lean_dec_ref(v___x_4748_);
v___x_4751_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4751_, 0, v_snd_4750_);
return v___x_4751_;
}
else
{
lean_object* v_val_4752_; 
lean_dec_ref(v___x_4748_);
v_val_4752_ = lean_ctor_get(v_fst_4749_, 0);
lean_inc(v_val_4752_);
lean_dec_ref_known(v_fst_4749_, 1);
return v_val_4752_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f___redArg___boxed(lean_object* v_ms_4755_, lean_object* v_mkExtra_x3f_4756_){
_start:
{
lean_object* v_res_4757_; 
v_res_4757_ = lp_aesop___private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f___redArg(v_ms_4755_, v_mkExtra_x3f_4756_);
lean_dec_ref(v_ms_4755_);
return v_res_4757_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f(lean_object* v_00_u03b1_4758_, lean_object* v_ms_4759_, lean_object* v_mkExtra_x3f_4760_){
_start:
{
lean_object* v___x_4761_; 
v___x_4761_ = lp_aesop___private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f___redArg(v_ms_4759_, v_mkExtra_x3f_4760_);
return v___x_4761_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f___boxed(lean_object* v_00_u03b1_4762_, lean_object* v_ms_4763_, lean_object* v_mkExtra_x3f_4764_){
_start:
{
lean_object* v_res_4765_; 
v_res_4765_ = lp_aesop___private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f(v_00_u03b1_4762_, v_ms_4763_, v_mkExtra_x3f_4764_);
lean_dec_ref(v_ms_4763_);
return v_res_4765_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__0(lean_object* v_00_u03b2_4766_, lean_object* v_m_4767_, lean_object* v_a_4768_){
_start:
{
lean_object* v___x_4769_; 
v___x_4769_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__0___redArg(v_m_4767_, v_a_4768_);
return v___x_4769_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__0___boxed(lean_object* v_00_u03b2_4770_, lean_object* v_m_4771_, lean_object* v_a_4772_){
_start:
{
lean_object* v_res_4773_; 
v_res_4773_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__0(v_00_u03b2_4770_, v_m_4771_, v_a_4772_);
lean_dec_ref(v_a_4772_);
lean_dec_ref(v_m_4771_);
return v_res_4773_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__1(lean_object* v_00_u03b2_4774_, lean_object* v_m_4775_, lean_object* v_a_4776_, lean_object* v_b_4777_){
_start:
{
lean_object* v___x_4778_; 
v___x_4778_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__1___redArg(v_m_4775_, v_a_4776_, v_b_4777_);
return v___x_4778_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__3(lean_object* v_00_u03b1_4779_, lean_object* v_mkExtra_x3f_4780_, lean_object* v_a_4781_, lean_object* v_a_4782_){
_start:
{
lean_object* v___x_4783_; 
v___x_4783_ = lp_aesop___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__3___redArg(v_mkExtra_x3f_4780_, v_a_4781_, v_a_4782_);
return v___x_4783_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__3___boxed(lean_object* v_00_u03b1_4784_, lean_object* v_mkExtra_x3f_4785_, lean_object* v_a_4786_, lean_object* v_a_4787_){
_start:
{
lean_object* v_res_4788_; 
v_res_4788_ = lp_aesop___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__3(v_00_u03b1_4784_, v_mkExtra_x3f_4785_, v_a_4786_, v_a_4787_);
lean_dec(v_a_4786_);
return v_res_4788_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__4(lean_object* v_00_u03b1_4789_, lean_object* v_mkExtra_x3f_4790_, lean_object* v_as_4791_, size_t v_sz_4792_, size_t v_i_4793_, lean_object* v_b_4794_){
_start:
{
lean_object* v___x_4795_; 
v___x_4795_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__4___redArg(v_mkExtra_x3f_4790_, v_as_4791_, v_sz_4792_, v_i_4793_, v_b_4794_);
return v___x_4795_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__4___boxed(lean_object* v_00_u03b1_4796_, lean_object* v_mkExtra_x3f_4797_, lean_object* v_as_4798_, lean_object* v_sz_4799_, lean_object* v_i_4800_, lean_object* v_b_4801_){
_start:
{
size_t v_sz_boxed_4802_; size_t v_i_boxed_4803_; lean_object* v_res_4804_; 
v_sz_boxed_4802_ = lean_unbox_usize(v_sz_4799_);
lean_dec(v_sz_4799_);
v_i_boxed_4803_ = lean_unbox_usize(v_i_4800_);
lean_dec(v_i_4800_);
v_res_4804_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__4(v_00_u03b1_4796_, v_mkExtra_x3f_4797_, v_as_4798_, v_sz_boxed_4802_, v_i_boxed_4803_, v_b_4801_);
lean_dec_ref(v_as_4798_);
return v_res_4804_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__0_spec__0(lean_object* v_00_u03b2_4805_, lean_object* v_a_4806_, lean_object* v_x_4807_){
_start:
{
lean_object* v___x_4808_; 
v___x_4808_ = lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__0_spec__0___redArg(v_a_4806_, v_x_4807_);
return v___x_4808_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__0_spec__0___boxed(lean_object* v_00_u03b2_4809_, lean_object* v_a_4810_, lean_object* v_x_4811_){
_start:
{
lean_object* v_res_4812_; 
v_res_4812_ = lp_aesop_Std_DHashMap_Internal_AssocList_get_x3f___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__0_spec__0(v_00_u03b2_4809_, v_a_4810_, v_x_4811_);
lean_dec(v_x_4811_);
lean_dec_ref(v_a_4810_);
return v_res_4812_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__1_spec__2(lean_object* v_00_u03b2_4813_, lean_object* v_a_4814_, lean_object* v_x_4815_){
_start:
{
uint8_t v___x_4816_; 
v___x_4816_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__1_spec__2___redArg(v_a_4814_, v_x_4815_);
return v___x_4816_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__1_spec__2___boxed(lean_object* v_00_u03b2_4817_, lean_object* v_a_4818_, lean_object* v_x_4819_){
_start:
{
uint8_t v_res_4820_; lean_object* v_r_4821_; 
v_res_4820_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__1_spec__2(v_00_u03b2_4817_, v_a_4818_, v_x_4819_);
lean_dec(v_x_4819_);
lean_dec_ref(v_a_4818_);
v_r_4821_ = lean_box(v_res_4820_);
return v_r_4821_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__1_spec__3(lean_object* v_00_u03b2_4822_, lean_object* v_data_4823_){
_start:
{
lean_object* v___x_4824_; 
v___x_4824_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__1_spec__3___redArg(v_data_4823_);
return v___x_4824_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__1_spec__4(lean_object* v_00_u03b2_4825_, lean_object* v_a_4826_, lean_object* v_b_4827_, lean_object* v_x_4828_){
_start:
{
lean_object* v___x_4829_; 
v___x_4829_ = lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__1_spec__4___redArg(v_a_4826_, v_b_4827_, v_x_4828_);
return v___x_4829_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__1_spec__3_spec__4(lean_object* v_00_u03b2_4830_, lean_object* v_i_4831_, lean_object* v_source_4832_, lean_object* v_target_4833_){
_start:
{
lean_object* v___x_4834_; 
v___x_4834_ = lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__1_spec__3_spec__4___redArg(v_i_4831_, v_source_4832_, v_target_4833_);
return v___x_4834_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__1_spec__3_spec__4_spec__8(lean_object* v_00_u03b2_4835_, lean_object* v_x_4836_, lean_object* v_x_4837_){
_start:
{
lean_object* v___x_4838_; 
v___x_4838_ = lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00__private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f_spec__1_spec__3_spec__4_spec__8___redArg(v_x_4836_, v_x_4837_);
return v___x_4838_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_forwardRuleMatchesToNormRules_x3f___lam__0(lean_object* v_x_4839_){
_start:
{
lean_object* v_rule_4840_; lean_object* v_prio_4841_; lean_object* v___x_4842_; 
v_rule_4840_ = lean_ctor_get(v_x_4839_, 0);
lean_inc_ref(v_rule_4840_);
lean_dec_ref(v_x_4839_);
v_prio_4841_ = lean_ctor_get(v_rule_4840_, 3);
lean_inc_ref(v_prio_4841_);
lean_dec_ref(v_rule_4840_);
v___x_4842_ = lp_aesop_Aesop_ForwardRulePriority_penalty_x3f(v_prio_4841_);
if (lean_obj_tag(v___x_4842_) == 0)
{
lean_object* v___x_4843_; 
v___x_4843_ = lean_box(0);
return v___x_4843_;
}
else
{
lean_object* v_val_4844_; lean_object* v___x_4846_; uint8_t v_isShared_4847_; uint8_t v_isSharedCheck_4851_; 
v_val_4844_ = lean_ctor_get(v___x_4842_, 0);
v_isSharedCheck_4851_ = !lean_is_exclusive(v___x_4842_);
if (v_isSharedCheck_4851_ == 0)
{
v___x_4846_ = v___x_4842_;
v_isShared_4847_ = v_isSharedCheck_4851_;
goto v_resetjp_4845_;
}
else
{
lean_inc(v_val_4844_);
lean_dec(v___x_4842_);
v___x_4846_ = lean_box(0);
v_isShared_4847_ = v_isSharedCheck_4851_;
goto v_resetjp_4845_;
}
v_resetjp_4845_:
{
lean_object* v___x_4849_; 
if (v_isShared_4847_ == 0)
{
v___x_4849_ = v___x_4846_;
goto v_reusejp_4848_;
}
else
{
lean_object* v_reuseFailAlloc_4850_; 
v_reuseFailAlloc_4850_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4850_, 0, v_val_4844_);
v___x_4849_ = v_reuseFailAlloc_4850_;
goto v_reusejp_4848_;
}
v_reusejp_4848_:
{
return v___x_4849_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_forwardRuleMatchesToNormRules_x3f(lean_object* v_ms_4853_){
_start:
{
lean_object* v___f_4854_; lean_object* v___x_4855_; 
v___f_4854_ = ((lean_object*)(lp_aesop_Aesop_forwardRuleMatchesToNormRules_x3f___closed__0));
v___x_4855_ = lp_aesop___private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f___redArg(v_ms_4853_, v___f_4854_);
return v___x_4855_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_forwardRuleMatchesToNormRules_x3f___boxed(lean_object* v_ms_4856_){
_start:
{
lean_object* v_res_4857_; 
v_res_4857_ = lp_aesop_Aesop_forwardRuleMatchesToNormRules_x3f(v_ms_4856_);
lean_dec_ref(v_ms_4856_);
return v_res_4857_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_forwardRuleMatchesToSafeRules_x3f___lam__0(lean_object* v_x_4858_){
_start:
{
lean_object* v_rule_4859_; lean_object* v_prio_4860_; lean_object* v___x_4861_; 
v_rule_4859_ = lean_ctor_get(v_x_4858_, 0);
lean_inc_ref(v_rule_4859_);
lean_dec_ref(v_x_4858_);
v_prio_4860_ = lean_ctor_get(v_rule_4859_, 3);
lean_inc_ref(v_prio_4860_);
lean_dec_ref(v_rule_4859_);
v___x_4861_ = lp_aesop_Aesop_ForwardRulePriority_penalty_x3f(v_prio_4860_);
if (lean_obj_tag(v___x_4861_) == 0)
{
lean_object* v___x_4862_; 
v___x_4862_ = lean_box(0);
return v___x_4862_;
}
else
{
lean_object* v_val_4863_; lean_object* v___x_4865_; uint8_t v_isShared_4866_; uint8_t v_isSharedCheck_4872_; 
v_val_4863_ = lean_ctor_get(v___x_4861_, 0);
v_isSharedCheck_4872_ = !lean_is_exclusive(v___x_4861_);
if (v_isSharedCheck_4872_ == 0)
{
v___x_4865_ = v___x_4861_;
v_isShared_4866_ = v_isSharedCheck_4872_;
goto v_resetjp_4864_;
}
else
{
lean_inc(v_val_4863_);
lean_dec(v___x_4861_);
v___x_4865_ = lean_box(0);
v_isShared_4866_ = v_isSharedCheck_4872_;
goto v_resetjp_4864_;
}
v_resetjp_4864_:
{
uint8_t v___x_4867_; lean_object* v___x_4868_; lean_object* v___x_4870_; 
v___x_4867_ = 0;
v___x_4868_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v___x_4868_, 0, v_val_4863_);
lean_ctor_set_uint8(v___x_4868_, sizeof(void*)*1, v___x_4867_);
if (v_isShared_4866_ == 0)
{
lean_ctor_set(v___x_4865_, 0, v___x_4868_);
v___x_4870_ = v___x_4865_;
goto v_reusejp_4869_;
}
else
{
lean_object* v_reuseFailAlloc_4871_; 
v_reuseFailAlloc_4871_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4871_, 0, v___x_4868_);
v___x_4870_ = v_reuseFailAlloc_4871_;
goto v_reusejp_4869_;
}
v_reusejp_4869_:
{
return v___x_4870_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_forwardRuleMatchesToSafeRules_x3f(lean_object* v_ms_4874_){
_start:
{
lean_object* v___f_4875_; lean_object* v___x_4876_; 
v___f_4875_ = ((lean_object*)(lp_aesop_Aesop_forwardRuleMatchesToSafeRules_x3f___closed__0));
v___x_4876_ = lp_aesop___private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f___redArg(v_ms_4874_, v___f_4875_);
return v___x_4876_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_forwardRuleMatchesToSafeRules_x3f___boxed(lean_object* v_ms_4877_){
_start:
{
lean_object* v_res_4878_; 
v_res_4878_ = lp_aesop_Aesop_forwardRuleMatchesToSafeRules_x3f(v_ms_4877_);
lean_dec_ref(v_ms_4877_);
return v_res_4878_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_forwardRuleMatchesToUnsafeRules_x3f___lam__0(lean_object* v_x_4879_){
_start:
{
lean_object* v_rule_4880_; lean_object* v_prio_4881_; lean_object* v___x_4882_; 
v_rule_4880_ = lean_ctor_get(v_x_4879_, 0);
v_prio_4881_ = lean_ctor_get(v_rule_4880_, 3);
v___x_4882_ = lp_aesop_Aesop_ForwardRulePriority_successProbability_x3f(v_prio_4881_);
if (lean_obj_tag(v___x_4882_) == 0)
{
lean_object* v___x_4883_; 
v___x_4883_ = lean_box(0);
return v___x_4883_;
}
else
{
lean_object* v_val_4884_; lean_object* v___x_4886_; uint8_t v_isShared_4887_; uint8_t v_isSharedCheck_4891_; 
v_val_4884_ = lean_ctor_get(v___x_4882_, 0);
v_isSharedCheck_4891_ = !lean_is_exclusive(v___x_4882_);
if (v_isSharedCheck_4891_ == 0)
{
v___x_4886_ = v___x_4882_;
v_isShared_4887_ = v_isSharedCheck_4891_;
goto v_resetjp_4885_;
}
else
{
lean_inc(v_val_4884_);
lean_dec(v___x_4882_);
v___x_4886_ = lean_box(0);
v_isShared_4887_ = v_isSharedCheck_4891_;
goto v_resetjp_4885_;
}
v_resetjp_4885_:
{
lean_object* v___x_4889_; 
if (v_isShared_4887_ == 0)
{
v___x_4889_ = v___x_4886_;
goto v_reusejp_4888_;
}
else
{
lean_object* v_reuseFailAlloc_4890_; 
v_reuseFailAlloc_4890_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_4890_, 0, v_val_4884_);
v___x_4889_ = v_reuseFailAlloc_4890_;
goto v_reusejp_4888_;
}
v_reusejp_4888_:
{
return v___x_4889_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_forwardRuleMatchesToUnsafeRules_x3f___lam__0___boxed(lean_object* v_x_4892_){
_start:
{
lean_object* v_res_4893_; 
v_res_4893_ = lp_aesop_Aesop_forwardRuleMatchesToUnsafeRules_x3f___lam__0(v_x_4892_);
lean_dec_ref(v_x_4892_);
return v_res_4893_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_forwardRuleMatchesToUnsafeRules_x3f(lean_object* v_ms_4895_){
_start:
{
lean_object* v___f_4896_; lean_object* v___x_4897_; 
v___f_4896_ = ((lean_object*)(lp_aesop_Aesop_forwardRuleMatchesToUnsafeRules_x3f___closed__0));
v___x_4897_ = lp_aesop___private_Aesop_Forward_Match_0__Aesop_forwardRuleMatchesToRules_x3f___redArg(v_ms_4895_, v___f_4896_);
return v___x_4897_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_forwardRuleMatchesToUnsafeRules_x3f___boxed(lean_object* v_ms_4898_){
_start:
{
lean_object* v_res_4899_; 
v_res_4899_ = lp_aesop_Aesop_forwardRuleMatchesToUnsafeRules_x3f(v_ms_4898_);
lean_dec_ref(v_ms_4898_);
return v_res_4899_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Tactic_Apply(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Forward_Match_Types(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Rule(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_RuleTac_ElabRuleTerm(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Script_SpecificTactics(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_RuleTac_Forward_Basic(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Lean_Meta_UnusedNames(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Forward_Match(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Tactic_Apply(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Forward_Match_Types(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Rule(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_RuleTac_ElabRuleTerm(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Script_SpecificTactics(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_RuleTac_Forward_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Lean_Meta_UnusedNames(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Forward_Match(uint8_t builtin) {
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
lean_object* initialize_Lean_Meta_Tactic_Apply(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Forward_Match_Types(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Rule(uint8_t builtin);
lean_object* initialize_aesop_Aesop_RuleTac_ElabRuleTerm(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Script_SpecificTactics(uint8_t builtin);
lean_object* initialize_aesop_Aesop_RuleTac_Forward_Basic(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Lean_Meta_UnusedNames(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Forward_Match(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Tactic_Apply(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Forward_Match_Types(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Rule(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_RuleTac_ElabRuleTerm(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Script_SpecificTactics(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_RuleTac_Forward_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Lean_Meta_UnusedNames(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Forward_Match(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Forward_Match(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Forward_Match(builtin);
}
#ifdef __cplusplus
}
#endif
