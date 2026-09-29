// Lean compiler output
// Module: Mathlib.Tactic.Subsingleton
// Imports: public import Init public meta import Init public meta import Lean.Meta.Tactic.Refl public import Mathlib.Basic.Logic.Basic
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
lean_object* lean_st_ref_get(lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
uint8_t l_Lean_instBEqMVarId_beq(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_replaceMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_getMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_intros(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_heqOfEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MVarId_getType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_whnfR(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
uint8_t l_Lean_Expr_isAppOfArity(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_appFn_x21(lean_object*);
lean_object* l_Lean_Expr_appArg_x21(lean_object*);
lean_object* l_Lean_Expr_const___override(lean_object*, lean_object*);
lean_object* l_Lean_mkApp4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
uint64_t l_Lean_instHashableMVarId_hash(lean_object*);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_usize_to_nat(size_t);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkCollisionNode___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_usize_shift_right(size_t, size_t);
size_t lean_usize_add(size_t, size_t);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntries(lean_object*, lean_object*);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_mul(size_t, size_t);
uint8_t lean_usize_dec_le(size_t, size_t);
lean_object* l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_Meta_isProp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_getLevel(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_app___override(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
lean_object* l_Lean_indentD(lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* l_Lean_Level_dec(lean_object*);
lean_object* l_Lean_mkAppB(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_synthInstance___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withNewMCtxDepthImp(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkApp5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* l_Lean_Meta_mkFreshLevelMVar(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* l_Lean_Expr_instantiateLevelParamsArray(lean_object*, lean_object*, lean_object*);
lean_object* l_Array_unzip___redArg(lean_object*);
lean_object* l_Lean_Meta_synthInstance(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* lean_expr_abstract(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
uint8_t lean_expr_has_loose_bvar(lean_object*, lean_object*);
lean_object* l_Array_instInhabited(lean_object*);
lean_object* lean_array_get_borrowed(lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lean_instantiate_level_mvars(lean_object*, lean_object*);
uint8_t l_Lean_Level_isMVar(lean_object*);
extern lean_object* l_Lean_Meta_instInhabitedAbstractMVarsResult_default;
lean_object* lean_array_get(lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_instInhabitedExpr;
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* lean_expr_instantiate_rev(lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withNewLocalInstancesImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
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
lean_object* l_instInhabitedOfMonad___redArg(lean_object*, lean_object*);
lean_object* l_Pi_instInhabited___redArg___lam__0(lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDeclImp(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_mkApp3(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_saveState___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Meta_SavedState_restore___redArg(lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_hrefl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_refl(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_isClass_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_fvarId_x21(lean_object*);
lean_object* l_Lean_Elab_Term_withoutErrToSorryImp___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_elabTerm(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_synthesizeSyntheticMVars(uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_abstractMVars(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_beta(lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkLambdaFVars(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withNewBinderInfosImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_AbstractMVarsResult_numMVars(lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_forallTelescopeReducingAux(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_getBetterRef(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_pp_macroStack;
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_MessageData_ofSyntax(lean_object*);
lean_object* l_Lean_Elab_Term_withoutModifyingElabMetaStateWithInfo___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_to_list(lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
extern lean_object* l_Lean_MessageData_nil;
lean_object* l_Lean_Meta_Tactic_TryThis_addSuggestion(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_withMainContext___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isNone(lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArgs(lean_object*);
static const lean_string_object lp_mathlib_Lean_Meta_mkSubsingleton___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "Subsingleton"};
static const lean_object* lp_mathlib_Lean_Meta_mkSubsingleton___closed__0 = (const lean_object*)&lp_mathlib_Lean_Meta_mkSubsingleton___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Meta_mkSubsingleton___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Meta_mkSubsingleton___closed__0_value),LEAN_SCALAR_PTR_LITERAL(23, 130, 42, 228, 248, 162, 23, 186)}};
static const lean_object* lp_mathlib_Lean_Meta_mkSubsingleton___closed__1 = (const lean_object*)&lp_mathlib_Lean_Meta_mkSubsingleton___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_mkSubsingleton(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_mkSubsingleton___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Meta_synthSubsingletonInst_spec__2___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Meta_synthSubsingletonInst_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Meta_synthSubsingletonInst_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Meta_synthSubsingletonInst_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLevelMVars___at___00Lean_Meta_synthSubsingletonInst_spec__3___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLevelMVars___at___00Lean_Meta_synthSubsingletonInst_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLevelMVars___at___00Lean_Meta_synthSubsingletonInst_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLevelMVars___at___00Lean_Meta_synthSubsingletonInst_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Lean_Meta_synthSubsingletonInst_spec__10___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Lean_Meta_synthSubsingletonInst_spec__10___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Lean_Meta_synthSubsingletonInst_spec__10(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Lean_Meta_synthSubsingletonInst_spec__10___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Meta_synthSubsingletonInst_spec__4_spec__4_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Meta_synthSubsingletonInst_spec__4_spec__4_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Meta_synthSubsingletonInst_spec__4_spec__4___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Meta_synthSubsingletonInst_spec__4_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_Meta_synthSubsingletonInst_spec__4___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_Meta_synthSubsingletonInst_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_synthSubsingletonInst_spec__5___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_synthSubsingletonInst_spec__5___closed__0;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_synthSubsingletonInst_spec__5___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 79, .m_capacity = 79, .m_length = 78, .m_data = "Instance provided to 'subsingleton' has unassigned universe level metavariable"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_synthSubsingletonInst_spec__5___closed__1 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_synthSubsingletonInst_spec__5___closed__1_value;
static lean_once_cell_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_synthSubsingletonInst_spec__5___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_synthSubsingletonInst_spec__5___closed__2;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_synthSubsingletonInst_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_synthSubsingletonInst_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_synthSubsingletonInst_spec__6_spec__7___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_synthSubsingletonInst_spec__6_spec__7___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_synthSubsingletonInst_spec__6_spec__7___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_synthSubsingletonInst_spec__6_spec__7___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_synthSubsingletonInst_spec__6___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_synthSubsingletonInst_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_synthSubsingletonInst___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_synthSubsingletonInst___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Basic_0__Lean_Meta_withNewLocalInstancesImpAux___at___00Lean_Meta_withNewLocalInstances___at___00Lean_Meta_synthSubsingletonInst_spec__7_spec__9___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Basic_0__Lean_Meta_withNewLocalInstancesImpAux___at___00Lean_Meta_withNewLocalInstances___at___00Lean_Meta_synthSubsingletonInst_spec__7_spec__9___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewLocalInstances___at___00Lean_Meta_synthSubsingletonInst_spec__7___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewLocalInstances___at___00Lean_Meta_synthSubsingletonInst_spec__7___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_synthSubsingletonInst___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_synthSubsingletonInst___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Meta_synthSubsingletonInst_spec__0(size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Meta_synthSubsingletonInst_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Meta_synthSubsingletonInst_spec__1(size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Meta_synthSubsingletonInst_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__12(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__12___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13_spec__16_spec__17___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13_spec__16_spec__17___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13_spec__16_spec__17___redArg(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13_spec__16_spec__17___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13_spec__16___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13_spec__16___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13_spec__16___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13_spec__16___closed__0;
static lean_once_cell_t lp_mathlib___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13_spec__16___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13_spec__16___closed__1;
static const lean_closure_object lp_mathlib___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13_spec__16___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadCoreM___lam__0___boxed, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13_spec__16___closed__2 = (const lean_object*)&lp_mathlib___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13_spec__16___closed__2_value;
static const lean_closure_object lp_mathlib___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13_spec__16___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadCoreM___lam__1___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13_spec__16___closed__3 = (const lean_object*)&lp_mathlib___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13_spec__16___closed__3_value;
static const lean_closure_object lp_mathlib___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13_spec__16___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instMonadMetaM___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13_spec__16___closed__4 = (const lean_object*)&lp_mathlib___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13_spec__16___closed__4_value;
static const lean_closure_object lp_mathlib___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13_spec__16___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instMonadMetaM___lam__1___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13_spec__16___closed__5 = (const lean_object*)&lp_mathlib___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13_spec__16___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13_spec__16___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13_spec__16(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13_spec__16___lam__1(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13_spec__16___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib_Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13___closed__0 = (const lean_object*)&lp_mathlib_Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Meta_synthSubsingletonInst_spec__8___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Meta_synthSubsingletonInst_spec__8___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Meta_synthSubsingletonInst_spec__8___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "inst"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Meta_synthSubsingletonInst_spec__8___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Meta_synthSubsingletonInst_spec__8___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Meta_synthSubsingletonInst_spec__8___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Meta_synthSubsingletonInst_spec__8___closed__0_value),LEAN_SCALAR_PTR_LITERAL(170, 188, 240, 205, 110, 63, 170, 91)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Meta_synthSubsingletonInst_spec__8___closed__1 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Meta_synthSubsingletonInst_spec__8___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Meta_synthSubsingletonInst_spec__8(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Meta_synthSubsingletonInst_spec__8___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_synthSubsingletonInst___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_synthSubsingletonInst___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_synthSubsingletonInst(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_synthSubsingletonInst___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_Meta_synthSubsingletonInst_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_Meta_synthSubsingletonInst_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_synthSubsingletonInst_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_synthSubsingletonInst_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Basic_0__Lean_Meta_withNewLocalInstancesImpAux___at___00Lean_Meta_withNewLocalInstances___at___00Lean_Meta_synthSubsingletonInst_spec__7_spec__9(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Basic_0__Lean_Meta_withNewLocalInstancesImpAux___at___00Lean_Meta_withNewLocalInstances___at___00Lean_Meta_synthSubsingletonInst_spec__7_spec__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewLocalInstances___at___00Lean_Meta_synthSubsingletonInst_spec__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewLocalInstances___at___00Lean_Meta_synthSubsingletonInst_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Meta_synthSubsingletonInst_spec__4_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Meta_synthSubsingletonInst_spec__4_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_synthSubsingletonInst_spec__6_spec__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_synthSubsingletonInst_spec__6_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13_spec__16_spec__17(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13_spec__16_spec__17___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_MVarId_subsingleton_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_MVarId_subsingleton_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_MVarId_subsingleton_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_MVarId_subsingleton_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_commitIfNoEx___at___00Lean_MVarId_subsingleton_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_commitIfNoEx___at___00Lean_MVarId_subsingleton_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_commitIfNoEx___at___00Lean_MVarId_subsingleton_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_commitIfNoEx___at___00Lean_MVarId_subsingleton_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_subsingleton_spec__0_spec__0_spec__3_spec__4_spec__5___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_subsingleton_spec__0_spec__0_spec__3_spec__4___redArg(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_subsingleton_spec__0_spec__0_spec__3___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_subsingleton_spec__0_spec__0_spec__3___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_subsingleton_spec__0_spec__0_spec__3___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_subsingleton_spec__0_spec__0_spec__3_spec__5___redArg(size_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_subsingleton_spec__0_spec__0_spec__3_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_subsingleton_spec__0_spec__0_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_subsingleton_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Lean_MVarId_subsingleton_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Lean_MVarId_subsingleton_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 87, .m_capacity = 87, .m_length = 86, .m_data = "tactic 'subsingleton' failed, goal is neither an equality nor a heterogeneous equality"};
static const lean_object* lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__1;
static const lean_string_object lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "Eq"};
static const lean_object* lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__2_value;
static const lean_ctor_object lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(143, 37, 101, 248, 9, 246, 191, 223)}};
static const lean_object* lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__3 = (const lean_object*)&lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__3_value;
static const lean_string_object lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "HEq"};
static const lean_object* lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__4 = (const lean_object*)&lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__4_value;
static const lean_ctor_object lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__4_value),LEAN_SCALAR_PTR_LITERAL(67, 180, 169, 191, 74, 196, 152, 188)}};
static const lean_object* lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__5 = (const lean_object*)&lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__5_value;
static const lean_string_object lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 61, .m_capacity = 61, .m_length = 60, .m_data = "tactic 'subsingleton' could not prove heterogeneous equality"};
static const lean_object* lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__6 = (const lean_object*)&lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__6_value;
static lean_once_cell_t lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__7;
static const lean_string_object lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "proof_irrel_heq"};
static const lean_object* lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__8 = (const lean_object*)&lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__8_value;
static const lean_ctor_object lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__8_value),LEAN_SCALAR_PTR_LITERAL(180, 105, 248, 247, 187, 48, 190, 226)}};
static const lean_object* lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__9 = (const lean_object*)&lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__9_value;
static lean_once_cell_t lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__10;
static const lean_string_object lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 77, .m_capacity = 77, .m_length = 76, .m_data = "tactic 'subsingleton' could not prove equality since it could not synthesize"};
static const lean_object* lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__11 = (const lean_object*)&lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__11_value;
static lean_once_cell_t lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__12;
static const lean_string_object lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "BEq"};
static const lean_object* lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__13 = (const lean_object*)&lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__13_value;
static const lean_ctor_object lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__13_value),LEAN_SCALAR_PTR_LITERAL(195, 188, 39, 55, 57, 152, 88, 223)}};
static const lean_object* lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__14 = (const lean_object*)&lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__14_value;
static const lean_string_object lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "LawfulBEq"};
static const lean_object* lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__15 = (const lean_object*)&lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__15_value;
static const lean_ctor_object lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__15_value),LEAN_SCALAR_PTR_LITERAL(198, 131, 20, 143, 70, 69, 65, 69)}};
static const lean_object* lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__16 = (const lean_object*)&lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__16_value;
static const lean_string_object lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "lawful_beq_subsingleton"};
static const lean_object* lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__17 = (const lean_object*)&lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__17_value;
static const lean_ctor_object lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__17_value),LEAN_SCALAR_PTR_LITERAL(161, 15, 5, 156, 31, 229, 166, 243)}};
static const lean_object* lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__18 = (const lean_object*)&lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__18_value;
static const lean_string_object lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "failed"};
static const lean_object* lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__19 = (const lean_object*)&lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__19_value;
static lean_once_cell_t lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__20;
static const lean_string_object lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "elim"};
static const lean_object* lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__21 = (const lean_object*)&lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__21_value;
static const lean_ctor_object lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__22_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Meta_mkSubsingleton___closed__0_value),LEAN_SCALAR_PTR_LITERAL(23, 130, 42, 228, 248, 162, 23, 186)}};
static const lean_ctor_object lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__22_value_aux_0),((lean_object*)&lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__21_value),LEAN_SCALAR_PTR_LITERAL(79, 85, 152, 16, 239, 41, 62, 212)}};
static const lean_object* lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__22 = (const lean_object*)&lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__22_value;
static const lean_string_object lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "proof_irrel"};
static const lean_object* lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__23 = (const lean_object*)&lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__23_value;
static const lean_ctor_object lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__23_value),LEAN_SCALAR_PTR_LITERAL(37, 39, 215, 172, 52, 39, 214, 110)}};
static const lean_object* lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__24 = (const lean_object*)&lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__24_value;
static lean_once_cell_t lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__25_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__25;
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_subsingleton___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_subsingleton___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_subsingleton___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_subsingleton___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_subsingleton(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_subsingleton___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Lean_MVarId_subsingleton_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Lean_MVarId_subsingleton_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_subsingleton_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_subsingleton_spec__0_spec__0_spec__3(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_subsingleton_spec__0_spec__0_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_subsingleton_spec__0_spec__0_spec__3_spec__4(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_subsingleton_spec__0_spec__0_spec__3_spec__5(lean_object*, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_subsingleton_spec__0_spec__0_spec__3_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_subsingleton_spec__0_spec__0_spec__3_spec__4_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "subsingletonStx"};
static const lean_object* lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__2_value),LEAN_SCALAR_PTR_LITERAL(225, 183, 66, 164, 191, 173, 114, 250)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__4_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "subsingleton"};
static const lean_object* lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__6_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "optional"};
static const lean_object* lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__8_value),LEAN_SCALAR_PTR_LITERAL(233, 141, 154, 50, 143, 135, 42, 252)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "ppSpace"};
static const lean_object* lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__10_value),LEAN_SCALAR_PTR_LITERAL(207, 47, 58, 43, 30, 240, 125, 246)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__11_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__11_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "["};
static const lean_object* lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__13_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__13_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__12_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__14_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__15_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__16_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__17_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__17_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__18_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__19 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__19_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ", "};
static const lean_object* lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__20_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__20_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__21_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 8, .m_other = 3, .m_tag = 10}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__18_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__19_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__21_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__22_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__15_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__22_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__23 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__23_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__24 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__24_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__24_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__25 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__25_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__23_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__25_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__26 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__26_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__9_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__26_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__27 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__27_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__5_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__7_value),((lean_object*)&lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__27_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__28 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__28_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__3_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__28_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__29 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__29_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_subsingletonStx = (const lean_object*)&lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__29_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewBinderInfos___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__2___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewBinderInfos___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__2___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewBinderInfos___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewBinderInfos___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewBinderInfos___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewBinderInfos___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__3___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__3___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__3___redArg(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__5___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__6___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__6(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go___lam__0(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__1_spec__1___redArg(uint8_t, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib_Array_filterMapM___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Array_filterMapM___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__1___closed__0 = (const lean_object*)&lp_mathlib_Array_filterMapM___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__1___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Array_filterMapM___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__1(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Array_filterMapM___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go___lam__1(uint8_t, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__4_spec__5_spec__9___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__4_spec__5_spec__9___closed__0;
static const lean_string_object lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__4_spec__5_spec__9___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "while expanding"};
static const lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__4_spec__5_spec__9___closed__1 = (const lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__4_spec__5_spec__9___closed__1_value;
static const lean_ctor_object lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__4_spec__5_spec__9___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__4_spec__5_spec__9___closed__1_value)}};
static const lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__4_spec__5_spec__9___closed__2 = (const lean_object*)&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__4_spec__5_spec__9___closed__2_value;
static lean_once_cell_t lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__4_spec__5_spec__9___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__4_spec__5_spec__9___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__4_spec__5_spec__9(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__4_spec__5_spec__8(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__4_spec__5_spec__8___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__4_spec__5___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "with resulting expansion"};
static const lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__4_spec__5___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__4_spec__5___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__4_spec__5___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__4_spec__5___redArg___closed__0_value)}};
static const lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__4_spec__5___redArg___closed__1 = (const lean_object*)&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__4_spec__5___redArg___closed__1_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__4_spec__5___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__4_spec__5___redArg___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__4_spec__5___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__4_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__4___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib___private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go___lam__2___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go___lam__2___closed__0_value;
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go___lam__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 31, .m_capacity = 31, .m_length = 30, .m_data = "Not an instance. Term has type"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go___lam__2___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go___lam__2___closed__1_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go___lam__2___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go___lam__2___closed__2;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go___lam__2(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__1_spec__1(uint8_t, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__4_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__4_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib_Mathlib_Tactic_elabSubsingletonInsts___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic_elabSubsingletonInsts___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_elabSubsingletonInsts___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_elabSubsingletonInsts(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_elabSubsingletonInsts___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "tactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(99, 76, 33, 121, 85, 143, 17, 224)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__0___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Try this:"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__0___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__0___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__0___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__1___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__1___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "paren"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__1___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__1___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__1___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__1___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__1___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__1___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeq1Indented"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__1___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__1___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__1___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__1___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__1___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__1___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "intros"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__1___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__1___closed__8_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__1___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__1___closed__9;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ";"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__1___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__1___closed__10_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticRfl"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__1___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__1___closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "rfl"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__1___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__1___closed__12_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__1___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__1___closed__13_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__1(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__2(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1_spec__2(uint8_t, uint8_t, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1_spec__1(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1_spec__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__0___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___closed__0_value;
static const lean_array_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_mkSubsingleton(lean_object* v_ty_4_, lean_object* v_a_5_, lean_object* v_a_6_, lean_object* v_a_7_, lean_object* v_a_8_){
_start:
{
lean_object* v___x_10_; 
lean_inc_ref(v_ty_4_);
v___x_10_ = l_Lean_Meta_getLevel(v_ty_4_, v_a_5_, v_a_6_, v_a_7_, v_a_8_);
if (lean_obj_tag(v___x_10_) == 0)
{
lean_object* v_a_11_; lean_object* v___x_13_; uint8_t v_isShared_14_; uint8_t v_isSharedCheck_23_; 
v_a_11_ = lean_ctor_get(v___x_10_, 0);
v_isSharedCheck_23_ = !lean_is_exclusive(v___x_10_);
if (v_isSharedCheck_23_ == 0)
{
v___x_13_ = v___x_10_;
v_isShared_14_ = v_isSharedCheck_23_;
goto v_resetjp_12_;
}
else
{
lean_inc(v_a_11_);
lean_dec(v___x_10_);
v___x_13_ = lean_box(0);
v_isShared_14_ = v_isSharedCheck_23_;
goto v_resetjp_12_;
}
v_resetjp_12_:
{
lean_object* v___x_15_; lean_object* v___x_16_; lean_object* v___x_17_; lean_object* v___x_18_; lean_object* v___x_19_; lean_object* v___x_21_; 
v___x_15_ = ((lean_object*)(lp_mathlib_Lean_Meta_mkSubsingleton___closed__1));
v___x_16_ = lean_box(0);
v___x_17_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_17_, 0, v_a_11_);
lean_ctor_set(v___x_17_, 1, v___x_16_);
v___x_18_ = l_Lean_Expr_const___override(v___x_15_, v___x_17_);
v___x_19_ = l_Lean_Expr_app___override(v___x_18_, v_ty_4_);
if (v_isShared_14_ == 0)
{
lean_ctor_set(v___x_13_, 0, v___x_19_);
v___x_21_ = v___x_13_;
goto v_reusejp_20_;
}
else
{
lean_object* v_reuseFailAlloc_22_; 
v_reuseFailAlloc_22_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_22_, 0, v___x_19_);
v___x_21_ = v_reuseFailAlloc_22_;
goto v_reusejp_20_;
}
v_reusejp_20_:
{
return v___x_21_;
}
}
}
else
{
lean_object* v_a_24_; lean_object* v___x_26_; uint8_t v_isShared_27_; uint8_t v_isSharedCheck_31_; 
lean_dec_ref(v_ty_4_);
v_a_24_ = lean_ctor_get(v___x_10_, 0);
v_isSharedCheck_31_ = !lean_is_exclusive(v___x_10_);
if (v_isSharedCheck_31_ == 0)
{
v___x_26_ = v___x_10_;
v_isShared_27_ = v_isSharedCheck_31_;
goto v_resetjp_25_;
}
else
{
lean_inc(v_a_24_);
lean_dec(v___x_10_);
v___x_26_ = lean_box(0);
v_isShared_27_ = v_isSharedCheck_31_;
goto v_resetjp_25_;
}
v_resetjp_25_:
{
lean_object* v___x_29_; 
if (v_isShared_27_ == 0)
{
v___x_29_ = v___x_26_;
goto v_reusejp_28_;
}
else
{
lean_object* v_reuseFailAlloc_30_; 
v_reuseFailAlloc_30_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_30_, 0, v_a_24_);
v___x_29_ = v_reuseFailAlloc_30_;
goto v_reusejp_28_;
}
v_reusejp_28_:
{
return v___x_29_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_mkSubsingleton___boxed(lean_object* v_ty_32_, lean_object* v_a_33_, lean_object* v_a_34_, lean_object* v_a_35_, lean_object* v_a_36_, lean_object* v_a_37_){
_start:
{
lean_object* v_res_38_; 
v_res_38_ = lp_mathlib_Lean_Meta_mkSubsingleton(v_ty_32_, v_a_33_, v_a_34_, v_a_35_, v_a_36_);
lean_dec(v_a_36_);
lean_dec_ref(v_a_35_);
lean_dec(v_a_34_);
lean_dec_ref(v_a_33_);
return v_res_38_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Meta_synthSubsingletonInst_spec__2___redArg(lean_object* v_e_39_, lean_object* v___y_40_){
_start:
{
uint8_t v___x_42_; 
v___x_42_ = l_Lean_Expr_hasMVar(v_e_39_);
if (v___x_42_ == 0)
{
lean_object* v___x_43_; 
v___x_43_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_43_, 0, v_e_39_);
return v___x_43_;
}
else
{
lean_object* v___x_44_; lean_object* v_mctx_45_; lean_object* v___x_46_; lean_object* v_fst_47_; lean_object* v_snd_48_; lean_object* v___x_49_; lean_object* v_cache_50_; lean_object* v_zetaDeltaFVarIds_51_; lean_object* v_postponed_52_; lean_object* v_diag_53_; lean_object* v___x_55_; uint8_t v_isShared_56_; uint8_t v_isSharedCheck_62_; 
v___x_44_ = lean_st_ref_get(v___y_40_);
v_mctx_45_ = lean_ctor_get(v___x_44_, 0);
lean_inc_ref(v_mctx_45_);
lean_dec(v___x_44_);
v___x_46_ = l_Lean_instantiateMVarsCore(v_mctx_45_, v_e_39_);
v_fst_47_ = lean_ctor_get(v___x_46_, 0);
lean_inc(v_fst_47_);
v_snd_48_ = lean_ctor_get(v___x_46_, 1);
lean_inc(v_snd_48_);
lean_dec_ref(v___x_46_);
v___x_49_ = lean_st_ref_take(v___y_40_);
v_cache_50_ = lean_ctor_get(v___x_49_, 1);
v_zetaDeltaFVarIds_51_ = lean_ctor_get(v___x_49_, 2);
v_postponed_52_ = lean_ctor_get(v___x_49_, 3);
v_diag_53_ = lean_ctor_get(v___x_49_, 4);
v_isSharedCheck_62_ = !lean_is_exclusive(v___x_49_);
if (v_isSharedCheck_62_ == 0)
{
lean_object* v_unused_63_; 
v_unused_63_ = lean_ctor_get(v___x_49_, 0);
lean_dec(v_unused_63_);
v___x_55_ = v___x_49_;
v_isShared_56_ = v_isSharedCheck_62_;
goto v_resetjp_54_;
}
else
{
lean_inc(v_diag_53_);
lean_inc(v_postponed_52_);
lean_inc(v_zetaDeltaFVarIds_51_);
lean_inc(v_cache_50_);
lean_dec(v___x_49_);
v___x_55_ = lean_box(0);
v_isShared_56_ = v_isSharedCheck_62_;
goto v_resetjp_54_;
}
v_resetjp_54_:
{
lean_object* v___x_58_; 
if (v_isShared_56_ == 0)
{
lean_ctor_set(v___x_55_, 0, v_snd_48_);
v___x_58_ = v___x_55_;
goto v_reusejp_57_;
}
else
{
lean_object* v_reuseFailAlloc_61_; 
v_reuseFailAlloc_61_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_61_, 0, v_snd_48_);
lean_ctor_set(v_reuseFailAlloc_61_, 1, v_cache_50_);
lean_ctor_set(v_reuseFailAlloc_61_, 2, v_zetaDeltaFVarIds_51_);
lean_ctor_set(v_reuseFailAlloc_61_, 3, v_postponed_52_);
lean_ctor_set(v_reuseFailAlloc_61_, 4, v_diag_53_);
v___x_58_ = v_reuseFailAlloc_61_;
goto v_reusejp_57_;
}
v_reusejp_57_:
{
lean_object* v___x_59_; lean_object* v___x_60_; 
v___x_59_ = lean_st_ref_set(v___y_40_, v___x_58_);
v___x_60_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_60_, 0, v_fst_47_);
return v___x_60_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Meta_synthSubsingletonInst_spec__2___redArg___boxed(lean_object* v_e_64_, lean_object* v___y_65_, lean_object* v___y_66_){
_start:
{
lean_object* v_res_67_; 
v_res_67_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_Meta_synthSubsingletonInst_spec__2___redArg(v_e_64_, v___y_65_);
lean_dec(v___y_65_);
return v_res_67_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Meta_synthSubsingletonInst_spec__2(lean_object* v_e_68_, lean_object* v___y_69_, lean_object* v___y_70_, lean_object* v___y_71_, lean_object* v___y_72_){
_start:
{
lean_object* v___x_74_; 
v___x_74_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_Meta_synthSubsingletonInst_spec__2___redArg(v_e_68_, v___y_70_);
return v___x_74_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_Meta_synthSubsingletonInst_spec__2___boxed(lean_object* v_e_75_, lean_object* v___y_76_, lean_object* v___y_77_, lean_object* v___y_78_, lean_object* v___y_79_, lean_object* v___y_80_){
_start:
{
lean_object* v_res_81_; 
v_res_81_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_Meta_synthSubsingletonInst_spec__2(v_e_75_, v___y_76_, v___y_77_, v___y_78_, v___y_79_);
lean_dec(v___y_79_);
lean_dec_ref(v___y_78_);
lean_dec(v___y_77_);
lean_dec_ref(v___y_76_);
return v_res_81_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLevelMVars___at___00Lean_Meta_synthSubsingletonInst_spec__3___redArg(lean_object* v_l_82_, lean_object* v___y_83_){
_start:
{
lean_object* v___x_85_; lean_object* v_mctx_86_; lean_object* v___x_87_; lean_object* v_fst_88_; lean_object* v_snd_89_; lean_object* v___x_90_; lean_object* v_cache_91_; lean_object* v_zetaDeltaFVarIds_92_; lean_object* v_postponed_93_; lean_object* v_diag_94_; lean_object* v___x_96_; uint8_t v_isShared_97_; uint8_t v_isSharedCheck_103_; 
v___x_85_ = lean_st_ref_get(v___y_83_);
v_mctx_86_ = lean_ctor_get(v___x_85_, 0);
lean_inc_ref(v_mctx_86_);
lean_dec(v___x_85_);
v___x_87_ = lean_instantiate_level_mvars(v_mctx_86_, v_l_82_);
v_fst_88_ = lean_ctor_get(v___x_87_, 0);
lean_inc(v_fst_88_);
v_snd_89_ = lean_ctor_get(v___x_87_, 1);
lean_inc(v_snd_89_);
lean_dec_ref(v___x_87_);
v___x_90_ = lean_st_ref_take(v___y_83_);
v_cache_91_ = lean_ctor_get(v___x_90_, 1);
v_zetaDeltaFVarIds_92_ = lean_ctor_get(v___x_90_, 2);
v_postponed_93_ = lean_ctor_get(v___x_90_, 3);
v_diag_94_ = lean_ctor_get(v___x_90_, 4);
v_isSharedCheck_103_ = !lean_is_exclusive(v___x_90_);
if (v_isSharedCheck_103_ == 0)
{
lean_object* v_unused_104_; 
v_unused_104_ = lean_ctor_get(v___x_90_, 0);
lean_dec(v_unused_104_);
v___x_96_ = v___x_90_;
v_isShared_97_ = v_isSharedCheck_103_;
goto v_resetjp_95_;
}
else
{
lean_inc(v_diag_94_);
lean_inc(v_postponed_93_);
lean_inc(v_zetaDeltaFVarIds_92_);
lean_inc(v_cache_91_);
lean_dec(v___x_90_);
v___x_96_ = lean_box(0);
v_isShared_97_ = v_isSharedCheck_103_;
goto v_resetjp_95_;
}
v_resetjp_95_:
{
lean_object* v___x_99_; 
if (v_isShared_97_ == 0)
{
lean_ctor_set(v___x_96_, 0, v_fst_88_);
v___x_99_ = v___x_96_;
goto v_reusejp_98_;
}
else
{
lean_object* v_reuseFailAlloc_102_; 
v_reuseFailAlloc_102_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_102_, 0, v_fst_88_);
lean_ctor_set(v_reuseFailAlloc_102_, 1, v_cache_91_);
lean_ctor_set(v_reuseFailAlloc_102_, 2, v_zetaDeltaFVarIds_92_);
lean_ctor_set(v_reuseFailAlloc_102_, 3, v_postponed_93_);
lean_ctor_set(v_reuseFailAlloc_102_, 4, v_diag_94_);
v___x_99_ = v_reuseFailAlloc_102_;
goto v_reusejp_98_;
}
v_reusejp_98_:
{
lean_object* v___x_100_; lean_object* v___x_101_; 
v___x_100_ = lean_st_ref_set(v___y_83_, v___x_99_);
v___x_101_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_101_, 0, v_snd_89_);
return v___x_101_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLevelMVars___at___00Lean_Meta_synthSubsingletonInst_spec__3___redArg___boxed(lean_object* v_l_105_, lean_object* v___y_106_, lean_object* v___y_107_){
_start:
{
lean_object* v_res_108_; 
v_res_108_ = lp_mathlib_Lean_instantiateLevelMVars___at___00Lean_Meta_synthSubsingletonInst_spec__3___redArg(v_l_105_, v___y_106_);
lean_dec(v___y_106_);
return v_res_108_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLevelMVars___at___00Lean_Meta_synthSubsingletonInst_spec__3(lean_object* v_l_109_, lean_object* v___y_110_, lean_object* v___y_111_, lean_object* v___y_112_, lean_object* v___y_113_){
_start:
{
lean_object* v___x_115_; 
v___x_115_ = lp_mathlib_Lean_instantiateLevelMVars___at___00Lean_Meta_synthSubsingletonInst_spec__3___redArg(v_l_109_, v___y_111_);
return v___x_115_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLevelMVars___at___00Lean_Meta_synthSubsingletonInst_spec__3___boxed(lean_object* v_l_116_, lean_object* v___y_117_, lean_object* v___y_118_, lean_object* v___y_119_, lean_object* v___y_120_, lean_object* v___y_121_){
_start:
{
lean_object* v_res_122_; 
v_res_122_ = lp_mathlib_Lean_instantiateLevelMVars___at___00Lean_Meta_synthSubsingletonInst_spec__3(v_l_116_, v___y_117_, v___y_118_, v___y_119_, v___y_120_);
lean_dec(v___y_120_);
lean_dec_ref(v___y_119_);
lean_dec(v___y_118_);
lean_dec_ref(v___y_117_);
return v_res_122_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Lean_Meta_synthSubsingletonInst_spec__10___redArg(lean_object* v_k_123_, uint8_t v_allowLevelAssignments_124_, lean_object* v___y_125_, lean_object* v___y_126_, lean_object* v___y_127_, lean_object* v___y_128_){
_start:
{
lean_object* v___x_130_; 
v___x_130_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withNewMCtxDepthImp(lean_box(0), v_allowLevelAssignments_124_, v_k_123_, v___y_125_, v___y_126_, v___y_127_, v___y_128_);
if (lean_obj_tag(v___x_130_) == 0)
{
lean_object* v_a_131_; lean_object* v___x_133_; uint8_t v_isShared_134_; uint8_t v_isSharedCheck_138_; 
v_a_131_ = lean_ctor_get(v___x_130_, 0);
v_isSharedCheck_138_ = !lean_is_exclusive(v___x_130_);
if (v_isSharedCheck_138_ == 0)
{
v___x_133_ = v___x_130_;
v_isShared_134_ = v_isSharedCheck_138_;
goto v_resetjp_132_;
}
else
{
lean_inc(v_a_131_);
lean_dec(v___x_130_);
v___x_133_ = lean_box(0);
v_isShared_134_ = v_isSharedCheck_138_;
goto v_resetjp_132_;
}
v_resetjp_132_:
{
lean_object* v___x_136_; 
if (v_isShared_134_ == 0)
{
v___x_136_ = v___x_133_;
goto v_reusejp_135_;
}
else
{
lean_object* v_reuseFailAlloc_137_; 
v_reuseFailAlloc_137_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_137_, 0, v_a_131_);
v___x_136_ = v_reuseFailAlloc_137_;
goto v_reusejp_135_;
}
v_reusejp_135_:
{
return v___x_136_;
}
}
}
else
{
lean_object* v_a_139_; lean_object* v___x_141_; uint8_t v_isShared_142_; uint8_t v_isSharedCheck_146_; 
v_a_139_ = lean_ctor_get(v___x_130_, 0);
v_isSharedCheck_146_ = !lean_is_exclusive(v___x_130_);
if (v_isSharedCheck_146_ == 0)
{
v___x_141_ = v___x_130_;
v_isShared_142_ = v_isSharedCheck_146_;
goto v_resetjp_140_;
}
else
{
lean_inc(v_a_139_);
lean_dec(v___x_130_);
v___x_141_ = lean_box(0);
v_isShared_142_ = v_isSharedCheck_146_;
goto v_resetjp_140_;
}
v_resetjp_140_:
{
lean_object* v___x_144_; 
if (v_isShared_142_ == 0)
{
v___x_144_ = v___x_141_;
goto v_reusejp_143_;
}
else
{
lean_object* v_reuseFailAlloc_145_; 
v_reuseFailAlloc_145_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_145_, 0, v_a_139_);
v___x_144_ = v_reuseFailAlloc_145_;
goto v_reusejp_143_;
}
v_reusejp_143_:
{
return v___x_144_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Lean_Meta_synthSubsingletonInst_spec__10___redArg___boxed(lean_object* v_k_147_, lean_object* v_allowLevelAssignments_148_, lean_object* v___y_149_, lean_object* v___y_150_, lean_object* v___y_151_, lean_object* v___y_152_, lean_object* v___y_153_){
_start:
{
uint8_t v_allowLevelAssignments_boxed_154_; lean_object* v_res_155_; 
v_allowLevelAssignments_boxed_154_ = lean_unbox(v_allowLevelAssignments_148_);
v_res_155_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Lean_Meta_synthSubsingletonInst_spec__10___redArg(v_k_147_, v_allowLevelAssignments_boxed_154_, v___y_149_, v___y_150_, v___y_151_, v___y_152_);
lean_dec(v___y_152_);
lean_dec_ref(v___y_151_);
lean_dec(v___y_150_);
lean_dec_ref(v___y_149_);
return v_res_155_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Lean_Meta_synthSubsingletonInst_spec__10(lean_object* v_00_u03b1_156_, lean_object* v_k_157_, uint8_t v_allowLevelAssignments_158_, lean_object* v___y_159_, lean_object* v___y_160_, lean_object* v___y_161_, lean_object* v___y_162_){
_start:
{
lean_object* v___x_164_; 
v___x_164_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Lean_Meta_synthSubsingletonInst_spec__10___redArg(v_k_157_, v_allowLevelAssignments_158_, v___y_159_, v___y_160_, v___y_161_, v___y_162_);
return v___x_164_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Lean_Meta_synthSubsingletonInst_spec__10___boxed(lean_object* v_00_u03b1_165_, lean_object* v_k_166_, lean_object* v_allowLevelAssignments_167_, lean_object* v___y_168_, lean_object* v___y_169_, lean_object* v___y_170_, lean_object* v___y_171_, lean_object* v___y_172_){
_start:
{
uint8_t v_allowLevelAssignments_boxed_173_; lean_object* v_res_174_; 
v_allowLevelAssignments_boxed_173_ = lean_unbox(v_allowLevelAssignments_167_);
v_res_174_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Lean_Meta_synthSubsingletonInst_spec__10(v_00_u03b1_165_, v_k_166_, v_allowLevelAssignments_boxed_173_, v___y_168_, v___y_169_, v___y_170_, v___y_171_);
lean_dec(v___y_171_);
lean_dec_ref(v___y_170_);
lean_dec(v___y_169_);
lean_dec_ref(v___y_168_);
return v_res_174_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Meta_synthSubsingletonInst_spec__4_spec__4_spec__6(lean_object* v_msgData_175_, lean_object* v___y_176_, lean_object* v___y_177_, lean_object* v___y_178_, lean_object* v___y_179_){
_start:
{
lean_object* v___x_181_; lean_object* v_env_182_; lean_object* v___x_183_; lean_object* v_mctx_184_; lean_object* v_lctx_185_; lean_object* v_options_186_; lean_object* v___x_187_; lean_object* v___x_188_; lean_object* v___x_189_; 
v___x_181_ = lean_st_ref_get(v___y_179_);
v_env_182_ = lean_ctor_get(v___x_181_, 0);
lean_inc_ref(v_env_182_);
lean_dec(v___x_181_);
v___x_183_ = lean_st_ref_get(v___y_177_);
v_mctx_184_ = lean_ctor_get(v___x_183_, 0);
lean_inc_ref(v_mctx_184_);
lean_dec(v___x_183_);
v_lctx_185_ = lean_ctor_get(v___y_176_, 2);
v_options_186_ = lean_ctor_get(v___y_178_, 2);
lean_inc_ref(v_options_186_);
lean_inc_ref(v_lctx_185_);
v___x_187_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_187_, 0, v_env_182_);
lean_ctor_set(v___x_187_, 1, v_mctx_184_);
lean_ctor_set(v___x_187_, 2, v_lctx_185_);
lean_ctor_set(v___x_187_, 3, v_options_186_);
v___x_188_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_188_, 0, v___x_187_);
lean_ctor_set(v___x_188_, 1, v_msgData_175_);
v___x_189_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_189_, 0, v___x_188_);
return v___x_189_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Meta_synthSubsingletonInst_spec__4_spec__4_spec__6___boxed(lean_object* v_msgData_190_, lean_object* v___y_191_, lean_object* v___y_192_, lean_object* v___y_193_, lean_object* v___y_194_, lean_object* v___y_195_){
_start:
{
lean_object* v_res_196_; 
v_res_196_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Meta_synthSubsingletonInst_spec__4_spec__4_spec__6(v_msgData_190_, v___y_191_, v___y_192_, v___y_193_, v___y_194_);
lean_dec(v___y_194_);
lean_dec_ref(v___y_193_);
lean_dec(v___y_192_);
lean_dec_ref(v___y_191_);
return v_res_196_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Meta_synthSubsingletonInst_spec__4_spec__4___redArg(lean_object* v_msg_197_, lean_object* v___y_198_, lean_object* v___y_199_, lean_object* v___y_200_, lean_object* v___y_201_){
_start:
{
lean_object* v_ref_203_; lean_object* v___x_204_; lean_object* v_a_205_; lean_object* v___x_207_; uint8_t v_isShared_208_; uint8_t v_isSharedCheck_213_; 
v_ref_203_ = lean_ctor_get(v___y_200_, 5);
v___x_204_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Meta_synthSubsingletonInst_spec__4_spec__4_spec__6(v_msg_197_, v___y_198_, v___y_199_, v___y_200_, v___y_201_);
v_a_205_ = lean_ctor_get(v___x_204_, 0);
v_isSharedCheck_213_ = !lean_is_exclusive(v___x_204_);
if (v_isSharedCheck_213_ == 0)
{
v___x_207_ = v___x_204_;
v_isShared_208_ = v_isSharedCheck_213_;
goto v_resetjp_206_;
}
else
{
lean_inc(v_a_205_);
lean_dec(v___x_204_);
v___x_207_ = lean_box(0);
v_isShared_208_ = v_isSharedCheck_213_;
goto v_resetjp_206_;
}
v_resetjp_206_:
{
lean_object* v___x_209_; lean_object* v___x_211_; 
lean_inc(v_ref_203_);
v___x_209_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_209_, 0, v_ref_203_);
lean_ctor_set(v___x_209_, 1, v_a_205_);
if (v_isShared_208_ == 0)
{
lean_ctor_set_tag(v___x_207_, 1);
lean_ctor_set(v___x_207_, 0, v___x_209_);
v___x_211_ = v___x_207_;
goto v_reusejp_210_;
}
else
{
lean_object* v_reuseFailAlloc_212_; 
v_reuseFailAlloc_212_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_212_, 0, v___x_209_);
v___x_211_ = v_reuseFailAlloc_212_;
goto v_reusejp_210_;
}
v_reusejp_210_:
{
return v___x_211_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Meta_synthSubsingletonInst_spec__4_spec__4___redArg___boxed(lean_object* v_msg_214_, lean_object* v___y_215_, lean_object* v___y_216_, lean_object* v___y_217_, lean_object* v___y_218_, lean_object* v___y_219_){
_start:
{
lean_object* v_res_220_; 
v_res_220_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Meta_synthSubsingletonInst_spec__4_spec__4___redArg(v_msg_214_, v___y_215_, v___y_216_, v___y_217_, v___y_218_);
lean_dec(v___y_218_);
lean_dec_ref(v___y_217_);
lean_dec(v___y_216_);
lean_dec_ref(v___y_215_);
return v_res_220_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_Meta_synthSubsingletonInst_spec__4___redArg(lean_object* v_ref_221_, lean_object* v_msg_222_, lean_object* v___y_223_, lean_object* v___y_224_, lean_object* v___y_225_, lean_object* v___y_226_){
_start:
{
lean_object* v_fileName_228_; lean_object* v_fileMap_229_; lean_object* v_options_230_; lean_object* v_currRecDepth_231_; lean_object* v_maxRecDepth_232_; lean_object* v_ref_233_; lean_object* v_currNamespace_234_; lean_object* v_openDecls_235_; lean_object* v_initHeartbeats_236_; lean_object* v_maxHeartbeats_237_; lean_object* v_quotContext_238_; lean_object* v_currMacroScope_239_; uint8_t v_diag_240_; lean_object* v_cancelTk_x3f_241_; uint8_t v_suppressElabErrors_242_; lean_object* v_inheritedTraceOptions_243_; lean_object* v_ref_244_; lean_object* v___x_245_; lean_object* v___x_246_; 
v_fileName_228_ = lean_ctor_get(v___y_225_, 0);
v_fileMap_229_ = lean_ctor_get(v___y_225_, 1);
v_options_230_ = lean_ctor_get(v___y_225_, 2);
v_currRecDepth_231_ = lean_ctor_get(v___y_225_, 3);
v_maxRecDepth_232_ = lean_ctor_get(v___y_225_, 4);
v_ref_233_ = lean_ctor_get(v___y_225_, 5);
v_currNamespace_234_ = lean_ctor_get(v___y_225_, 6);
v_openDecls_235_ = lean_ctor_get(v___y_225_, 7);
v_initHeartbeats_236_ = lean_ctor_get(v___y_225_, 8);
v_maxHeartbeats_237_ = lean_ctor_get(v___y_225_, 9);
v_quotContext_238_ = lean_ctor_get(v___y_225_, 10);
v_currMacroScope_239_ = lean_ctor_get(v___y_225_, 11);
v_diag_240_ = lean_ctor_get_uint8(v___y_225_, sizeof(void*)*14);
v_cancelTk_x3f_241_ = lean_ctor_get(v___y_225_, 12);
v_suppressElabErrors_242_ = lean_ctor_get_uint8(v___y_225_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_243_ = lean_ctor_get(v___y_225_, 13);
v_ref_244_ = l_Lean_replaceRef(v_ref_221_, v_ref_233_);
lean_inc_ref(v_inheritedTraceOptions_243_);
lean_inc(v_cancelTk_x3f_241_);
lean_inc(v_currMacroScope_239_);
lean_inc(v_quotContext_238_);
lean_inc(v_maxHeartbeats_237_);
lean_inc(v_initHeartbeats_236_);
lean_inc(v_openDecls_235_);
lean_inc(v_currNamespace_234_);
lean_inc(v_maxRecDepth_232_);
lean_inc(v_currRecDepth_231_);
lean_inc_ref(v_options_230_);
lean_inc_ref(v_fileMap_229_);
lean_inc_ref(v_fileName_228_);
v___x_245_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_245_, 0, v_fileName_228_);
lean_ctor_set(v___x_245_, 1, v_fileMap_229_);
lean_ctor_set(v___x_245_, 2, v_options_230_);
lean_ctor_set(v___x_245_, 3, v_currRecDepth_231_);
lean_ctor_set(v___x_245_, 4, v_maxRecDepth_232_);
lean_ctor_set(v___x_245_, 5, v_ref_244_);
lean_ctor_set(v___x_245_, 6, v_currNamespace_234_);
lean_ctor_set(v___x_245_, 7, v_openDecls_235_);
lean_ctor_set(v___x_245_, 8, v_initHeartbeats_236_);
lean_ctor_set(v___x_245_, 9, v_maxHeartbeats_237_);
lean_ctor_set(v___x_245_, 10, v_quotContext_238_);
lean_ctor_set(v___x_245_, 11, v_currMacroScope_239_);
lean_ctor_set(v___x_245_, 12, v_cancelTk_x3f_241_);
lean_ctor_set(v___x_245_, 13, v_inheritedTraceOptions_243_);
lean_ctor_set_uint8(v___x_245_, sizeof(void*)*14, v_diag_240_);
lean_ctor_set_uint8(v___x_245_, sizeof(void*)*14 + 1, v_suppressElabErrors_242_);
v___x_246_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Meta_synthSubsingletonInst_spec__4_spec__4___redArg(v_msg_222_, v___y_223_, v___y_224_, v___x_245_, v___y_226_);
lean_dec_ref_known(v___x_245_, 14);
return v___x_246_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_Meta_synthSubsingletonInst_spec__4___redArg___boxed(lean_object* v_ref_247_, lean_object* v_msg_248_, lean_object* v___y_249_, lean_object* v___y_250_, lean_object* v___y_251_, lean_object* v___y_252_, lean_object* v___y_253_){
_start:
{
lean_object* v_res_254_; 
v_res_254_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_Meta_synthSubsingletonInst_spec__4___redArg(v_ref_247_, v_msg_248_, v___y_249_, v___y_250_, v___y_251_, v___y_252_);
lean_dec(v___y_252_);
lean_dec_ref(v___y_251_);
lean_dec(v___y_250_);
lean_dec_ref(v___y_249_);
lean_dec(v_ref_247_);
return v_res_254_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_synthSubsingletonInst_spec__5___closed__0(void){
_start:
{
lean_object* v___x_255_; lean_object* v___x_256_; lean_object* v___x_257_; 
v___x_255_ = l_Lean_Meta_instInhabitedAbstractMVarsResult_default;
v___x_256_ = lean_box(0);
v___x_257_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_257_, 0, v___x_256_);
lean_ctor_set(v___x_257_, 1, v___x_255_);
return v___x_257_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_synthSubsingletonInst_spec__5___closed__2(void){
_start:
{
lean_object* v___x_259_; lean_object* v___x_260_; 
v___x_259_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_synthSubsingletonInst_spec__5___closed__1));
v___x_260_ = l_Lean_stringToMessageData(v___x_259_);
return v___x_260_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_synthSubsingletonInst_spec__5(lean_object* v_insts_261_, lean_object* v_a_262_, lean_object* v_fst_263_, lean_object* v_as_264_, size_t v_i_265_, size_t v_stop_266_, lean_object* v_b_267_, lean_object* v___y_268_, lean_object* v___y_269_, lean_object* v___y_270_, lean_object* v___y_271_){
_start:
{
lean_object* v_a_274_; uint8_t v___x_278_; 
v___x_278_ = lean_usize_dec_eq(v_i_265_, v_stop_266_);
if (v___x_278_ == 0)
{
lean_object* v___x_279_; lean_object* v___x_280_; 
v___x_279_ = lean_array_uget_borrowed(v_as_264_, v_i_265_);
lean_inc(v___x_279_);
v___x_280_ = lp_mathlib_Lean_instantiateLevelMVars___at___00Lean_Meta_synthSubsingletonInst_spec__3___redArg(v___x_279_, v___y_269_);
if (lean_obj_tag(v___x_280_) == 0)
{
lean_object* v_a_281_; uint8_t v___x_282_; 
v_a_281_ = lean_ctor_get(v___x_280_, 0);
lean_inc(v_a_281_);
lean_dec_ref_known(v___x_280_, 1);
v___x_282_ = l_Lean_Level_isMVar(v_a_281_);
lean_dec(v_a_281_);
if (v___x_282_ == 0)
{
lean_object* v___x_283_; 
v___x_283_ = lean_box(0);
v_a_274_ = v___x_283_;
goto v___jp_273_;
}
else
{
lean_object* v___x_284_; lean_object* v___x_285_; lean_object* v_fst_286_; lean_object* v___x_288_; uint8_t v_isShared_289_; uint8_t v_isSharedCheck_300_; 
v___x_284_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_synthSubsingletonInst_spec__5___closed__0, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_synthSubsingletonInst_spec__5___closed__0_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_synthSubsingletonInst_spec__5___closed__0);
v___x_285_ = lean_array_get(v___x_284_, v_insts_261_, v_a_262_);
v_fst_286_ = lean_ctor_get(v___x_285_, 0);
v_isSharedCheck_300_ = !lean_is_exclusive(v___x_285_);
if (v_isSharedCheck_300_ == 0)
{
lean_object* v_unused_301_; 
v_unused_301_ = lean_ctor_get(v___x_285_, 1);
lean_dec(v_unused_301_);
v___x_288_ = v___x_285_;
v_isShared_289_ = v_isSharedCheck_300_;
goto v_resetjp_287_;
}
else
{
lean_inc(v_fst_286_);
lean_dec(v___x_285_);
v___x_288_ = lean_box(0);
v_isShared_289_ = v_isSharedCheck_300_;
goto v_resetjp_287_;
}
v_resetjp_287_:
{
lean_object* v___x_290_; lean_object* v___x_291_; lean_object* v___x_292_; lean_object* v___x_293_; lean_object* v___x_294_; lean_object* v___x_296_; 
v___x_290_ = l_Lean_instInhabitedExpr;
v___x_291_ = lean_obj_once(&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_synthSubsingletonInst_spec__5___closed__2, &lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_synthSubsingletonInst_spec__5___closed__2_once, _init_lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_synthSubsingletonInst_spec__5___closed__2);
v___x_292_ = lean_array_get_borrowed(v___x_290_, v_fst_263_, v_a_262_);
lean_inc(v___x_292_);
v___x_293_ = l_Lean_MessageData_ofExpr(v___x_292_);
v___x_294_ = l_Lean_indentD(v___x_293_);
if (v_isShared_289_ == 0)
{
lean_ctor_set_tag(v___x_288_, 7);
lean_ctor_set(v___x_288_, 1, v___x_294_);
lean_ctor_set(v___x_288_, 0, v___x_291_);
v___x_296_ = v___x_288_;
goto v_reusejp_295_;
}
else
{
lean_object* v_reuseFailAlloc_299_; 
v_reuseFailAlloc_299_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_299_, 0, v___x_291_);
lean_ctor_set(v_reuseFailAlloc_299_, 1, v___x_294_);
v___x_296_ = v_reuseFailAlloc_299_;
goto v_reusejp_295_;
}
v_reusejp_295_:
{
lean_object* v___x_297_; 
v___x_297_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_Meta_synthSubsingletonInst_spec__4___redArg(v_fst_286_, v___x_296_, v___y_268_, v___y_269_, v___y_270_, v___y_271_);
lean_dec(v_fst_286_);
if (lean_obj_tag(v___x_297_) == 0)
{
lean_object* v_a_298_; 
v_a_298_ = lean_ctor_get(v___x_297_, 0);
lean_inc(v_a_298_);
lean_dec_ref_known(v___x_297_, 1);
v_a_274_ = v_a_298_;
goto v___jp_273_;
}
else
{
return v___x_297_;
}
}
}
}
}
else
{
lean_object* v_a_302_; lean_object* v___x_304_; uint8_t v_isShared_305_; uint8_t v_isSharedCheck_309_; 
v_a_302_ = lean_ctor_get(v___x_280_, 0);
v_isSharedCheck_309_ = !lean_is_exclusive(v___x_280_);
if (v_isSharedCheck_309_ == 0)
{
v___x_304_ = v___x_280_;
v_isShared_305_ = v_isSharedCheck_309_;
goto v_resetjp_303_;
}
else
{
lean_inc(v_a_302_);
lean_dec(v___x_280_);
v___x_304_ = lean_box(0);
v_isShared_305_ = v_isSharedCheck_309_;
goto v_resetjp_303_;
}
v_resetjp_303_:
{
lean_object* v___x_307_; 
if (v_isShared_305_ == 0)
{
v___x_307_ = v___x_304_;
goto v_reusejp_306_;
}
else
{
lean_object* v_reuseFailAlloc_308_; 
v_reuseFailAlloc_308_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_308_, 0, v_a_302_);
v___x_307_ = v_reuseFailAlloc_308_;
goto v_reusejp_306_;
}
v_reusejp_306_:
{
return v___x_307_;
}
}
}
}
else
{
lean_object* v___x_310_; 
v___x_310_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_310_, 0, v_b_267_);
return v___x_310_;
}
v___jp_273_:
{
size_t v___x_275_; size_t v___x_276_; 
v___x_275_ = ((size_t)1ULL);
v___x_276_ = lean_usize_add(v_i_265_, v___x_275_);
v_i_265_ = v___x_276_;
v_b_267_ = v_a_274_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_synthSubsingletonInst_spec__5___boxed(lean_object* v_insts_311_, lean_object* v_a_312_, lean_object* v_fst_313_, lean_object* v_as_314_, lean_object* v_i_315_, lean_object* v_stop_316_, lean_object* v_b_317_, lean_object* v___y_318_, lean_object* v___y_319_, lean_object* v___y_320_, lean_object* v___y_321_, lean_object* v___y_322_){
_start:
{
size_t v_i_boxed_323_; size_t v_stop_boxed_324_; lean_object* v_res_325_; 
v_i_boxed_323_ = lean_unbox_usize(v_i_315_);
lean_dec(v_i_315_);
v_stop_boxed_324_ = lean_unbox_usize(v_stop_316_);
lean_dec(v_stop_316_);
v_res_325_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_synthSubsingletonInst_spec__5(v_insts_311_, v_a_312_, v_fst_313_, v_as_314_, v_i_boxed_323_, v_stop_boxed_324_, v_b_317_, v___y_318_, v___y_319_, v___y_320_, v___y_321_);
lean_dec(v___y_321_);
lean_dec_ref(v___y_320_);
lean_dec(v___y_319_);
lean_dec_ref(v___y_318_);
lean_dec_ref(v_as_314_);
lean_dec_ref(v_fst_313_);
lean_dec(v_a_312_);
lean_dec_ref(v_insts_311_);
return v_res_325_;
}
}
static lean_object* _init_lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_synthSubsingletonInst_spec__6_spec__7___redArg___closed__0(void){
_start:
{
lean_object* v___x_326_; 
v___x_326_ = l_Array_instInhabited(lean_box(0));
return v___x_326_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_synthSubsingletonInst_spec__6_spec__7___redArg(lean_object* v___x_327_, lean_object* v___x_328_, lean_object* v_snd_329_, lean_object* v_insts_330_, lean_object* v_fst_331_, lean_object* v_range_332_, lean_object* v_b_333_, lean_object* v_i_334_, lean_object* v___y_335_, lean_object* v___y_336_, lean_object* v___y_337_, lean_object* v___y_338_){
_start:
{
lean_object* v_stop_340_; lean_object* v_step_341_; uint8_t v___x_342_; 
v_stop_340_ = lean_ctor_get(v_range_332_, 1);
v_step_341_ = lean_ctor_get(v_range_332_, 2);
v___x_342_ = lean_nat_dec_lt(v_i_334_, v_stop_340_);
if (v___x_342_ == 0)
{
lean_object* v___x_343_; 
lean_dec(v_i_334_);
v___x_343_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_343_, 0, v_b_333_);
return v___x_343_;
}
else
{
lean_object* v___x_344_; lean_object* v___y_349_; lean_object* v___x_350_; lean_object* v___x_351_; lean_object* v___x_352_; uint8_t v___x_353_; 
v___x_344_ = lean_box(0);
v___x_350_ = lean_unsigned_to_nat(1u);
v___x_351_ = lean_nat_sub(v___x_327_, v_i_334_);
v___x_352_ = lean_nat_sub(v___x_351_, v___x_350_);
lean_dec(v___x_351_);
v___x_353_ = lean_expr_has_loose_bvar(v___x_328_, v___x_352_);
lean_dec(v___x_352_);
if (v___x_353_ == 0)
{
goto v___jp_345_;
}
else
{
lean_object* v___x_354_; lean_object* v___x_355_; lean_object* v___x_356_; lean_object* v___x_357_; uint8_t v___x_358_; 
v___x_354_ = lean_unsigned_to_nat(0u);
v___x_355_ = lean_obj_once(&lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_synthSubsingletonInst_spec__6_spec__7___redArg___closed__0, &lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_synthSubsingletonInst_spec__6_spec__7___redArg___closed__0_once, _init_lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_synthSubsingletonInst_spec__6_spec__7___redArg___closed__0);
v___x_356_ = lean_array_get_borrowed(v___x_355_, v_snd_329_, v_i_334_);
v___x_357_ = lean_array_get_size(v___x_356_);
v___x_358_ = lean_nat_dec_lt(v___x_354_, v___x_357_);
if (v___x_358_ == 0)
{
goto v___jp_345_;
}
else
{
uint8_t v___x_359_; 
v___x_359_ = lean_nat_dec_le(v___x_357_, v___x_357_);
if (v___x_359_ == 0)
{
if (v___x_358_ == 0)
{
goto v___jp_345_;
}
else
{
size_t v___x_360_; size_t v___x_361_; lean_object* v___x_362_; 
v___x_360_ = ((size_t)0ULL);
v___x_361_ = lean_usize_of_nat(v___x_357_);
v___x_362_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_synthSubsingletonInst_spec__5(v_insts_330_, v_i_334_, v_fst_331_, v___x_356_, v___x_360_, v___x_361_, v___x_344_, v___y_335_, v___y_336_, v___y_337_, v___y_338_);
v___y_349_ = v___x_362_;
goto v___jp_348_;
}
}
else
{
size_t v___x_363_; size_t v___x_364_; lean_object* v___x_365_; 
v___x_363_ = ((size_t)0ULL);
v___x_364_ = lean_usize_of_nat(v___x_357_);
v___x_365_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_synthSubsingletonInst_spec__5(v_insts_330_, v_i_334_, v_fst_331_, v___x_356_, v___x_363_, v___x_364_, v___x_344_, v___y_335_, v___y_336_, v___y_337_, v___y_338_);
v___y_349_ = v___x_365_;
goto v___jp_348_;
}
}
}
v___jp_345_:
{
lean_object* v___x_346_; 
v___x_346_ = lean_nat_add(v_i_334_, v_step_341_);
lean_dec(v_i_334_);
v_b_333_ = v___x_344_;
v_i_334_ = v___x_346_;
goto _start;
}
v___jp_348_:
{
if (lean_obj_tag(v___y_349_) == 0)
{
lean_dec_ref_known(v___y_349_, 1);
goto v___jp_345_;
}
else
{
lean_dec(v_i_334_);
return v___y_349_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_synthSubsingletonInst_spec__6_spec__7___redArg___boxed(lean_object* v___x_366_, lean_object* v___x_367_, lean_object* v_snd_368_, lean_object* v_insts_369_, lean_object* v_fst_370_, lean_object* v_range_371_, lean_object* v_b_372_, lean_object* v_i_373_, lean_object* v___y_374_, lean_object* v___y_375_, lean_object* v___y_376_, lean_object* v___y_377_, lean_object* v___y_378_){
_start:
{
lean_object* v_res_379_; 
v_res_379_ = lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_synthSubsingletonInst_spec__6_spec__7___redArg(v___x_366_, v___x_367_, v_snd_368_, v_insts_369_, v_fst_370_, v_range_371_, v_b_372_, v_i_373_, v___y_374_, v___y_375_, v___y_376_, v___y_377_);
lean_dec(v___y_377_);
lean_dec_ref(v___y_376_);
lean_dec(v___y_375_);
lean_dec_ref(v___y_374_);
lean_dec_ref(v_range_371_);
lean_dec_ref(v_fst_370_);
lean_dec_ref(v_insts_369_);
lean_dec_ref(v_snd_368_);
lean_dec_ref(v___x_367_);
lean_dec(v___x_366_);
return v_res_379_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_synthSubsingletonInst_spec__6___redArg(lean_object* v_insts_380_, lean_object* v_fst_381_, lean_object* v___x_382_, lean_object* v___x_383_, lean_object* v_snd_384_, lean_object* v_range_385_, lean_object* v_b_386_, lean_object* v_i_387_, lean_object* v___y_388_, lean_object* v___y_389_, lean_object* v___y_390_, lean_object* v___y_391_){
_start:
{
lean_object* v_stop_393_; lean_object* v_step_394_; uint8_t v___x_395_; 
v_stop_393_ = lean_ctor_get(v_range_385_, 1);
v_step_394_ = lean_ctor_get(v_range_385_, 2);
v___x_395_ = lean_nat_dec_lt(v_i_387_, v_stop_393_);
if (v___x_395_ == 0)
{
lean_object* v___x_396_; 
v___x_396_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_396_, 0, v_b_386_);
return v___x_396_;
}
else
{
lean_object* v___x_397_; lean_object* v___y_402_; lean_object* v___x_403_; lean_object* v___x_404_; lean_object* v___x_405_; uint8_t v___x_406_; 
v___x_397_ = lean_box(0);
v___x_403_ = lean_unsigned_to_nat(1u);
v___x_404_ = lean_nat_sub(v___x_382_, v_i_387_);
v___x_405_ = lean_nat_sub(v___x_404_, v___x_403_);
lean_dec(v___x_404_);
v___x_406_ = lean_expr_has_loose_bvar(v___x_383_, v___x_405_);
lean_dec(v___x_405_);
if (v___x_406_ == 0)
{
goto v___jp_398_;
}
else
{
lean_object* v___x_407_; lean_object* v___x_408_; lean_object* v___x_409_; lean_object* v___x_410_; uint8_t v___x_411_; 
v___x_407_ = lean_unsigned_to_nat(0u);
v___x_408_ = lean_obj_once(&lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_synthSubsingletonInst_spec__6_spec__7___redArg___closed__0, &lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_synthSubsingletonInst_spec__6_spec__7___redArg___closed__0_once, _init_lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_synthSubsingletonInst_spec__6_spec__7___redArg___closed__0);
v___x_409_ = lean_array_get_borrowed(v___x_408_, v_snd_384_, v_i_387_);
v___x_410_ = lean_array_get_size(v___x_409_);
v___x_411_ = lean_nat_dec_lt(v___x_407_, v___x_410_);
if (v___x_411_ == 0)
{
goto v___jp_398_;
}
else
{
uint8_t v___x_412_; 
v___x_412_ = lean_nat_dec_le(v___x_410_, v___x_410_);
if (v___x_412_ == 0)
{
if (v___x_411_ == 0)
{
goto v___jp_398_;
}
else
{
size_t v___x_413_; size_t v___x_414_; lean_object* v___x_415_; 
v___x_413_ = ((size_t)0ULL);
v___x_414_ = lean_usize_of_nat(v___x_410_);
v___x_415_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_synthSubsingletonInst_spec__5(v_insts_380_, v_i_387_, v_fst_381_, v___x_409_, v___x_413_, v___x_414_, v___x_397_, v___y_388_, v___y_389_, v___y_390_, v___y_391_);
v___y_402_ = v___x_415_;
goto v___jp_401_;
}
}
else
{
size_t v___x_416_; size_t v___x_417_; lean_object* v___x_418_; 
v___x_416_ = ((size_t)0ULL);
v___x_417_ = lean_usize_of_nat(v___x_410_);
v___x_418_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Meta_synthSubsingletonInst_spec__5(v_insts_380_, v_i_387_, v_fst_381_, v___x_409_, v___x_416_, v___x_417_, v___x_397_, v___y_388_, v___y_389_, v___y_390_, v___y_391_);
v___y_402_ = v___x_418_;
goto v___jp_401_;
}
}
}
v___jp_398_:
{
lean_object* v___x_399_; lean_object* v___x_400_; 
v___x_399_ = lean_nat_add(v_i_387_, v_step_394_);
v___x_400_ = lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_synthSubsingletonInst_spec__6_spec__7___redArg(v___x_382_, v___x_383_, v_snd_384_, v_insts_380_, v_fst_381_, v_range_385_, v___x_397_, v___x_399_, v___y_388_, v___y_389_, v___y_390_, v___y_391_);
return v___x_400_;
}
v___jp_401_:
{
if (lean_obj_tag(v___y_402_) == 0)
{
lean_dec_ref_known(v___y_402_, 1);
goto v___jp_398_;
}
else
{
return v___y_402_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_synthSubsingletonInst_spec__6___redArg___boxed(lean_object* v_insts_419_, lean_object* v_fst_420_, lean_object* v___x_421_, lean_object* v___x_422_, lean_object* v_snd_423_, lean_object* v_range_424_, lean_object* v_b_425_, lean_object* v_i_426_, lean_object* v___y_427_, lean_object* v___y_428_, lean_object* v___y_429_, lean_object* v___y_430_, lean_object* v___y_431_){
_start:
{
lean_object* v_res_432_; 
v_res_432_ = lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_synthSubsingletonInst_spec__6___redArg(v_insts_419_, v_fst_420_, v___x_421_, v___x_422_, v_snd_423_, v_range_424_, v_b_425_, v_i_426_, v___y_427_, v___y_428_, v___y_429_, v___y_430_);
lean_dec(v___y_430_);
lean_dec_ref(v___y_429_);
lean_dec(v___y_428_);
lean_dec_ref(v___y_427_);
lean_dec(v_i_426_);
lean_dec_ref(v_range_424_);
lean_dec_ref(v_snd_423_);
lean_dec_ref(v___x_422_);
lean_dec(v___x_421_);
lean_dec_ref(v_fst_420_);
lean_dec_ref(v_insts_419_);
return v_res_432_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_synthSubsingletonInst___lam__0(lean_object* v_ty_433_, lean_object* v_fvars_434_, lean_object* v___x_435_, lean_object* v_insts_436_, lean_object* v_fst_437_, lean_object* v_snd_438_, lean_object* v___y_439_, lean_object* v___y_440_, lean_object* v___y_441_, lean_object* v___y_442_){
_start:
{
lean_object* v___x_444_; 
v___x_444_ = lp_mathlib_Lean_Meta_mkSubsingleton(v_ty_433_, v___y_439_, v___y_440_, v___y_441_, v___y_442_);
if (lean_obj_tag(v___x_444_) == 0)
{
lean_object* v_a_445_; lean_object* v___x_446_; lean_object* v___x_447_; 
v_a_445_ = lean_ctor_get(v___x_444_, 0);
lean_inc(v_a_445_);
lean_dec_ref_known(v___x_444_, 1);
v___x_446_ = lean_box(0);
v___x_447_ = l_Lean_Meta_synthInstance(v_a_445_, v___x_446_, v___y_439_, v___y_440_, v___y_441_, v___y_442_);
if (lean_obj_tag(v___x_447_) == 0)
{
lean_object* v_a_448_; lean_object* v___x_449_; lean_object* v_a_450_; lean_object* v___x_451_; lean_object* v___x_452_; lean_object* v___x_453_; lean_object* v___x_454_; lean_object* v___x_455_; lean_object* v___x_456_; 
v_a_448_ = lean_ctor_get(v___x_447_, 0);
lean_inc(v_a_448_);
lean_dec_ref_known(v___x_447_, 1);
v___x_449_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_Meta_synthSubsingletonInst_spec__2___redArg(v_a_448_, v___y_440_);
v_a_450_ = lean_ctor_get(v___x_449_, 0);
lean_inc(v_a_450_);
lean_dec_ref(v___x_449_);
v___x_451_ = lean_expr_abstract(v_a_450_, v_fvars_434_);
lean_dec(v_a_450_);
v___x_452_ = lean_array_get_size(v_fvars_434_);
v___x_453_ = lean_unsigned_to_nat(1u);
lean_inc(v___x_435_);
v___x_454_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_454_, 0, v___x_435_);
lean_ctor_set(v___x_454_, 1, v___x_452_);
lean_ctor_set(v___x_454_, 2, v___x_453_);
v___x_455_ = lean_box(0);
v___x_456_ = lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_synthSubsingletonInst_spec__6___redArg(v_insts_436_, v_fst_437_, v___x_452_, v___x_451_, v_snd_438_, v___x_454_, v___x_455_, v___x_435_, v___y_439_, v___y_440_, v___y_441_, v___y_442_);
lean_dec(v___x_435_);
lean_dec_ref_known(v___x_454_, 3);
if (lean_obj_tag(v___x_456_) == 0)
{
lean_object* v___x_457_; lean_object* v___x_458_; 
lean_dec_ref_known(v___x_456_, 1);
v___x_457_ = lean_expr_instantiate_rev(v___x_451_, v_fst_437_);
lean_dec_ref(v___x_451_);
v___x_458_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_Meta_synthSubsingletonInst_spec__2___redArg(v___x_457_, v___y_440_);
return v___x_458_;
}
else
{
lean_object* v_a_459_; lean_object* v___x_461_; uint8_t v_isShared_462_; uint8_t v_isSharedCheck_466_; 
lean_dec_ref(v___x_451_);
v_a_459_ = lean_ctor_get(v___x_456_, 0);
v_isSharedCheck_466_ = !lean_is_exclusive(v___x_456_);
if (v_isSharedCheck_466_ == 0)
{
v___x_461_ = v___x_456_;
v_isShared_462_ = v_isSharedCheck_466_;
goto v_resetjp_460_;
}
else
{
lean_inc(v_a_459_);
lean_dec(v___x_456_);
v___x_461_ = lean_box(0);
v_isShared_462_ = v_isSharedCheck_466_;
goto v_resetjp_460_;
}
v_resetjp_460_:
{
lean_object* v___x_464_; 
if (v_isShared_462_ == 0)
{
v___x_464_ = v___x_461_;
goto v_reusejp_463_;
}
else
{
lean_object* v_reuseFailAlloc_465_; 
v_reuseFailAlloc_465_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_465_, 0, v_a_459_);
v___x_464_ = v_reuseFailAlloc_465_;
goto v_reusejp_463_;
}
v_reusejp_463_:
{
return v___x_464_;
}
}
}
}
else
{
lean_dec(v___x_435_);
return v___x_447_;
}
}
else
{
lean_dec(v___x_435_);
return v___x_444_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_synthSubsingletonInst___lam__0___boxed(lean_object* v_ty_467_, lean_object* v_fvars_468_, lean_object* v___x_469_, lean_object* v_insts_470_, lean_object* v_fst_471_, lean_object* v_snd_472_, lean_object* v___y_473_, lean_object* v___y_474_, lean_object* v___y_475_, lean_object* v___y_476_, lean_object* v___y_477_){
_start:
{
lean_object* v_res_478_; 
v_res_478_ = lp_mathlib_Lean_Meta_synthSubsingletonInst___lam__0(v_ty_467_, v_fvars_468_, v___x_469_, v_insts_470_, v_fst_471_, v_snd_472_, v___y_473_, v___y_474_, v___y_475_, v___y_476_);
lean_dec(v___y_476_);
lean_dec_ref(v___y_475_);
lean_dec(v___y_474_);
lean_dec_ref(v___y_473_);
lean_dec_ref(v_snd_472_);
lean_dec_ref(v_fst_471_);
lean_dec_ref(v_insts_470_);
lean_dec_ref(v_fvars_468_);
return v_res_478_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Basic_0__Lean_Meta_withNewLocalInstancesImpAux___at___00Lean_Meta_withNewLocalInstances___at___00Lean_Meta_synthSubsingletonInst_spec__7_spec__9___redArg(lean_object* v_fvars_479_, lean_object* v_j_480_, lean_object* v_x_481_, lean_object* v___y_482_, lean_object* v___y_483_, lean_object* v___y_484_, lean_object* v___y_485_){
_start:
{
lean_object* v___x_487_; 
v___x_487_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withNewLocalInstancesImp(lean_box(0), v_fvars_479_, v_j_480_, v_x_481_, v___y_482_, v___y_483_, v___y_484_, v___y_485_);
if (lean_obj_tag(v___x_487_) == 0)
{
lean_object* v_a_488_; lean_object* v___x_490_; uint8_t v_isShared_491_; uint8_t v_isSharedCheck_495_; 
v_a_488_ = lean_ctor_get(v___x_487_, 0);
v_isSharedCheck_495_ = !lean_is_exclusive(v___x_487_);
if (v_isSharedCheck_495_ == 0)
{
v___x_490_ = v___x_487_;
v_isShared_491_ = v_isSharedCheck_495_;
goto v_resetjp_489_;
}
else
{
lean_inc(v_a_488_);
lean_dec(v___x_487_);
v___x_490_ = lean_box(0);
v_isShared_491_ = v_isSharedCheck_495_;
goto v_resetjp_489_;
}
v_resetjp_489_:
{
lean_object* v___x_493_; 
if (v_isShared_491_ == 0)
{
v___x_493_ = v___x_490_;
goto v_reusejp_492_;
}
else
{
lean_object* v_reuseFailAlloc_494_; 
v_reuseFailAlloc_494_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_494_, 0, v_a_488_);
v___x_493_ = v_reuseFailAlloc_494_;
goto v_reusejp_492_;
}
v_reusejp_492_:
{
return v___x_493_;
}
}
}
else
{
lean_object* v_a_496_; lean_object* v___x_498_; uint8_t v_isShared_499_; uint8_t v_isSharedCheck_503_; 
v_a_496_ = lean_ctor_get(v___x_487_, 0);
v_isSharedCheck_503_ = !lean_is_exclusive(v___x_487_);
if (v_isSharedCheck_503_ == 0)
{
v___x_498_ = v___x_487_;
v_isShared_499_ = v_isSharedCheck_503_;
goto v_resetjp_497_;
}
else
{
lean_inc(v_a_496_);
lean_dec(v___x_487_);
v___x_498_ = lean_box(0);
v_isShared_499_ = v_isSharedCheck_503_;
goto v_resetjp_497_;
}
v_resetjp_497_:
{
lean_object* v___x_501_; 
if (v_isShared_499_ == 0)
{
v___x_501_ = v___x_498_;
goto v_reusejp_500_;
}
else
{
lean_object* v_reuseFailAlloc_502_; 
v_reuseFailAlloc_502_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_502_, 0, v_a_496_);
v___x_501_ = v_reuseFailAlloc_502_;
goto v_reusejp_500_;
}
v_reusejp_500_:
{
return v___x_501_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Basic_0__Lean_Meta_withNewLocalInstancesImpAux___at___00Lean_Meta_withNewLocalInstances___at___00Lean_Meta_synthSubsingletonInst_spec__7_spec__9___redArg___boxed(lean_object* v_fvars_504_, lean_object* v_j_505_, lean_object* v_x_506_, lean_object* v___y_507_, lean_object* v___y_508_, lean_object* v___y_509_, lean_object* v___y_510_, lean_object* v___y_511_){
_start:
{
lean_object* v_res_512_; 
v_res_512_ = lp_mathlib___private_Lean_Meta_Basic_0__Lean_Meta_withNewLocalInstancesImpAux___at___00Lean_Meta_withNewLocalInstances___at___00Lean_Meta_synthSubsingletonInst_spec__7_spec__9___redArg(v_fvars_504_, v_j_505_, v_x_506_, v___y_507_, v___y_508_, v___y_509_, v___y_510_);
lean_dec(v___y_510_);
lean_dec_ref(v___y_509_);
lean_dec(v___y_508_);
lean_dec_ref(v___y_507_);
return v_res_512_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewLocalInstances___at___00Lean_Meta_synthSubsingletonInst_spec__7___redArg(lean_object* v_fvars_513_, lean_object* v_j_514_, lean_object* v_x_515_, lean_object* v___y_516_, lean_object* v___y_517_, lean_object* v___y_518_, lean_object* v___y_519_){
_start:
{
lean_object* v___x_521_; 
v___x_521_ = lp_mathlib___private_Lean_Meta_Basic_0__Lean_Meta_withNewLocalInstancesImpAux___at___00Lean_Meta_withNewLocalInstances___at___00Lean_Meta_synthSubsingletonInst_spec__7_spec__9___redArg(v_fvars_513_, v_j_514_, v_x_515_, v___y_516_, v___y_517_, v___y_518_, v___y_519_);
if (lean_obj_tag(v___x_521_) == 0)
{
lean_object* v_a_522_; lean_object* v___x_524_; uint8_t v_isShared_525_; uint8_t v_isSharedCheck_529_; 
v_a_522_ = lean_ctor_get(v___x_521_, 0);
v_isSharedCheck_529_ = !lean_is_exclusive(v___x_521_);
if (v_isSharedCheck_529_ == 0)
{
v___x_524_ = v___x_521_;
v_isShared_525_ = v_isSharedCheck_529_;
goto v_resetjp_523_;
}
else
{
lean_inc(v_a_522_);
lean_dec(v___x_521_);
v___x_524_ = lean_box(0);
v_isShared_525_ = v_isSharedCheck_529_;
goto v_resetjp_523_;
}
v_resetjp_523_:
{
lean_object* v___x_527_; 
if (v_isShared_525_ == 0)
{
v___x_527_ = v___x_524_;
goto v_reusejp_526_;
}
else
{
lean_object* v_reuseFailAlloc_528_; 
v_reuseFailAlloc_528_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_528_, 0, v_a_522_);
v___x_527_ = v_reuseFailAlloc_528_;
goto v_reusejp_526_;
}
v_reusejp_526_:
{
return v___x_527_;
}
}
}
else
{
lean_object* v_a_530_; lean_object* v___x_532_; uint8_t v_isShared_533_; uint8_t v_isSharedCheck_537_; 
v_a_530_ = lean_ctor_get(v___x_521_, 0);
v_isSharedCheck_537_ = !lean_is_exclusive(v___x_521_);
if (v_isSharedCheck_537_ == 0)
{
v___x_532_ = v___x_521_;
v_isShared_533_ = v_isSharedCheck_537_;
goto v_resetjp_531_;
}
else
{
lean_inc(v_a_530_);
lean_dec(v___x_521_);
v___x_532_ = lean_box(0);
v_isShared_533_ = v_isSharedCheck_537_;
goto v_resetjp_531_;
}
v_resetjp_531_:
{
lean_object* v___x_535_; 
if (v_isShared_533_ == 0)
{
v___x_535_ = v___x_532_;
goto v_reusejp_534_;
}
else
{
lean_object* v_reuseFailAlloc_536_; 
v_reuseFailAlloc_536_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_536_, 0, v_a_530_);
v___x_535_ = v_reuseFailAlloc_536_;
goto v_reusejp_534_;
}
v_reusejp_534_:
{
return v___x_535_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewLocalInstances___at___00Lean_Meta_synthSubsingletonInst_spec__7___redArg___boxed(lean_object* v_fvars_538_, lean_object* v_j_539_, lean_object* v_x_540_, lean_object* v___y_541_, lean_object* v___y_542_, lean_object* v___y_543_, lean_object* v___y_544_, lean_object* v___y_545_){
_start:
{
lean_object* v_res_546_; 
v_res_546_ = lp_mathlib_Lean_Meta_withNewLocalInstances___at___00Lean_Meta_synthSubsingletonInst_spec__7___redArg(v_fvars_538_, v_j_539_, v_x_540_, v___y_541_, v___y_542_, v___y_543_, v___y_544_);
lean_dec(v___y_544_);
lean_dec_ref(v___y_543_);
lean_dec(v___y_542_);
lean_dec_ref(v___y_541_);
return v_res_546_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_synthSubsingletonInst___lam__1(lean_object* v_ty_547_, lean_object* v_insts_548_, lean_object* v_fst_549_, lean_object* v_snd_550_, lean_object* v_fvars_551_, lean_object* v___y_552_, lean_object* v___y_553_, lean_object* v___y_554_, lean_object* v___y_555_){
_start:
{
lean_object* v___x_557_; lean_object* v___f_558_; lean_object* v___x_559_; 
v___x_557_ = lean_unsigned_to_nat(0u);
lean_inc_ref(v_fvars_551_);
v___f_558_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_synthSubsingletonInst___lam__0___boxed), 11, 6);
lean_closure_set(v___f_558_, 0, v_ty_547_);
lean_closure_set(v___f_558_, 1, v_fvars_551_);
lean_closure_set(v___f_558_, 2, v___x_557_);
lean_closure_set(v___f_558_, 3, v_insts_548_);
lean_closure_set(v___f_558_, 4, v_fst_549_);
lean_closure_set(v___f_558_, 5, v_snd_550_);
v___x_559_ = lp_mathlib_Lean_Meta_withNewLocalInstances___at___00Lean_Meta_synthSubsingletonInst_spec__7___redArg(v_fvars_551_, v___x_557_, v___f_558_, v___y_552_, v___y_553_, v___y_554_, v___y_555_);
return v___x_559_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_synthSubsingletonInst___lam__1___boxed(lean_object* v_ty_560_, lean_object* v_insts_561_, lean_object* v_fst_562_, lean_object* v_snd_563_, lean_object* v_fvars_564_, lean_object* v___y_565_, lean_object* v___y_566_, lean_object* v___y_567_, lean_object* v___y_568_, lean_object* v___y_569_){
_start:
{
lean_object* v_res_570_; 
v_res_570_ = lp_mathlib_Lean_Meta_synthSubsingletonInst___lam__1(v_ty_560_, v_insts_561_, v_fst_562_, v_snd_563_, v_fvars_564_, v___y_565_, v___y_566_, v___y_567_, v___y_568_);
lean_dec(v___y_568_);
lean_dec_ref(v___y_567_);
lean_dec(v___y_566_);
lean_dec_ref(v___y_565_);
return v_res_570_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Meta_synthSubsingletonInst_spec__0(size_t v_sz_571_, size_t v_i_572_, lean_object* v_bs_573_, lean_object* v___y_574_, lean_object* v___y_575_, lean_object* v___y_576_, lean_object* v___y_577_){
_start:
{
uint8_t v___x_579_; 
v___x_579_ = lean_usize_dec_lt(v_i_572_, v_sz_571_);
if (v___x_579_ == 0)
{
lean_object* v___x_580_; 
v___x_580_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_580_, 0, v_bs_573_);
return v___x_580_;
}
else
{
lean_object* v___x_581_; 
v___x_581_ = l_Lean_Meta_mkFreshLevelMVar(v___y_574_, v___y_575_, v___y_576_, v___y_577_);
if (lean_obj_tag(v___x_581_) == 0)
{
lean_object* v_a_582_; lean_object* v___x_583_; lean_object* v_bs_x27_584_; size_t v___x_585_; size_t v___x_586_; lean_object* v___x_587_; 
v_a_582_ = lean_ctor_get(v___x_581_, 0);
lean_inc(v_a_582_);
lean_dec_ref_known(v___x_581_, 1);
v___x_583_ = lean_unsigned_to_nat(0u);
v_bs_x27_584_ = lean_array_uset(v_bs_573_, v_i_572_, v___x_583_);
v___x_585_ = ((size_t)1ULL);
v___x_586_ = lean_usize_add(v_i_572_, v___x_585_);
v___x_587_ = lean_array_uset(v_bs_x27_584_, v_i_572_, v_a_582_);
v_i_572_ = v___x_586_;
v_bs_573_ = v___x_587_;
goto _start;
}
else
{
lean_object* v_a_589_; lean_object* v___x_591_; uint8_t v_isShared_592_; uint8_t v_isSharedCheck_596_; 
lean_dec_ref(v_bs_573_);
v_a_589_ = lean_ctor_get(v___x_581_, 0);
v_isSharedCheck_596_ = !lean_is_exclusive(v___x_581_);
if (v_isSharedCheck_596_ == 0)
{
v___x_591_ = v___x_581_;
v_isShared_592_ = v_isSharedCheck_596_;
goto v_resetjp_590_;
}
else
{
lean_inc(v_a_589_);
lean_dec(v___x_581_);
v___x_591_ = lean_box(0);
v_isShared_592_ = v_isSharedCheck_596_;
goto v_resetjp_590_;
}
v_resetjp_590_:
{
lean_object* v___x_594_; 
if (v_isShared_592_ == 0)
{
v___x_594_ = v___x_591_;
goto v_reusejp_593_;
}
else
{
lean_object* v_reuseFailAlloc_595_; 
v_reuseFailAlloc_595_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_595_, 0, v_a_589_);
v___x_594_ = v_reuseFailAlloc_595_;
goto v_reusejp_593_;
}
v_reusejp_593_:
{
return v___x_594_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Meta_synthSubsingletonInst_spec__0___boxed(lean_object* v_sz_597_, lean_object* v_i_598_, lean_object* v_bs_599_, lean_object* v___y_600_, lean_object* v___y_601_, lean_object* v___y_602_, lean_object* v___y_603_, lean_object* v___y_604_){
_start:
{
size_t v_sz_boxed_605_; size_t v_i_boxed_606_; lean_object* v_res_607_; 
v_sz_boxed_605_ = lean_unbox_usize(v_sz_597_);
lean_dec(v_sz_597_);
v_i_boxed_606_ = lean_unbox_usize(v_i_598_);
lean_dec(v_i_598_);
v_res_607_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Meta_synthSubsingletonInst_spec__0(v_sz_boxed_605_, v_i_boxed_606_, v_bs_599_, v___y_600_, v___y_601_, v___y_602_, v___y_603_);
lean_dec(v___y_603_);
lean_dec_ref(v___y_602_);
lean_dec(v___y_601_);
lean_dec_ref(v___y_600_);
return v_res_607_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Meta_synthSubsingletonInst_spec__1(size_t v_sz_608_, size_t v_i_609_, lean_object* v_bs_610_, lean_object* v___y_611_, lean_object* v___y_612_, lean_object* v___y_613_, lean_object* v___y_614_){
_start:
{
uint8_t v___x_616_; 
v___x_616_ = lean_usize_dec_lt(v_i_609_, v_sz_608_);
if (v___x_616_ == 0)
{
lean_object* v___x_617_; 
v___x_617_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_617_, 0, v_bs_610_);
return v___x_617_;
}
else
{
lean_object* v_v_618_; lean_object* v_snd_619_; lean_object* v___x_621_; uint8_t v_isShared_622_; uint8_t v_isSharedCheck_639_; 
v_v_618_ = lean_array_uget(v_bs_610_, v_i_609_);
v_snd_619_ = lean_ctor_get(v_v_618_, 1);
v_isSharedCheck_639_ = !lean_is_exclusive(v_v_618_);
if (v_isSharedCheck_639_ == 0)
{
lean_object* v_unused_640_; 
v_unused_640_ = lean_ctor_get(v_v_618_, 0);
lean_dec(v_unused_640_);
v___x_621_ = v_v_618_;
v_isShared_622_ = v_isSharedCheck_639_;
goto v_resetjp_620_;
}
else
{
lean_inc(v_snd_619_);
lean_dec(v_v_618_);
v___x_621_ = lean_box(0);
v_isShared_622_ = v_isSharedCheck_639_;
goto v_resetjp_620_;
}
v_resetjp_620_:
{
lean_object* v_paramNames_623_; lean_object* v_expr_624_; size_t v_sz_625_; size_t v___x_626_; lean_object* v___x_627_; 
v_paramNames_623_ = lean_ctor_get(v_snd_619_, 0);
lean_inc_ref_n(v_paramNames_623_, 2);
v_expr_624_ = lean_ctor_get(v_snd_619_, 2);
lean_inc_ref(v_expr_624_);
lean_dec(v_snd_619_);
v_sz_625_ = lean_array_size(v_paramNames_623_);
v___x_626_ = ((size_t)0ULL);
v___x_627_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Meta_synthSubsingletonInst_spec__0(v_sz_625_, v___x_626_, v_paramNames_623_, v___y_611_, v___y_612_, v___y_613_, v___y_614_);
if (lean_obj_tag(v___x_627_) == 0)
{
lean_object* v_a_628_; lean_object* v___x_629_; lean_object* v_bs_x27_630_; lean_object* v___x_631_; lean_object* v___x_633_; 
v_a_628_ = lean_ctor_get(v___x_627_, 0);
lean_inc_n(v_a_628_, 2);
lean_dec_ref_known(v___x_627_, 1);
v___x_629_ = lean_unsigned_to_nat(0u);
v_bs_x27_630_ = lean_array_uset(v_bs_610_, v_i_609_, v___x_629_);
v___x_631_ = l_Lean_Expr_instantiateLevelParamsArray(v_expr_624_, v_paramNames_623_, v_a_628_);
lean_dec_ref(v_expr_624_);
if (v_isShared_622_ == 0)
{
lean_ctor_set(v___x_621_, 1, v_a_628_);
lean_ctor_set(v___x_621_, 0, v___x_631_);
v___x_633_ = v___x_621_;
goto v_reusejp_632_;
}
else
{
lean_object* v_reuseFailAlloc_638_; 
v_reuseFailAlloc_638_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_638_, 0, v___x_631_);
lean_ctor_set(v_reuseFailAlloc_638_, 1, v_a_628_);
v___x_633_ = v_reuseFailAlloc_638_;
goto v_reusejp_632_;
}
v_reusejp_632_:
{
size_t v___x_634_; size_t v___x_635_; lean_object* v___x_636_; 
v___x_634_ = ((size_t)1ULL);
v___x_635_ = lean_usize_add(v_i_609_, v___x_634_);
v___x_636_ = lean_array_uset(v_bs_x27_630_, v_i_609_, v___x_633_);
v_i_609_ = v___x_635_;
v_bs_610_ = v___x_636_;
goto _start;
}
}
else
{
lean_dec_ref(v_expr_624_);
lean_dec_ref(v_paramNames_623_);
lean_del_object(v___x_621_);
lean_dec_ref(v_bs_610_);
return v___x_627_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Meta_synthSubsingletonInst_spec__1___boxed(lean_object* v_sz_641_, lean_object* v_i_642_, lean_object* v_bs_643_, lean_object* v___y_644_, lean_object* v___y_645_, lean_object* v___y_646_, lean_object* v___y_647_, lean_object* v___y_648_){
_start:
{
size_t v_sz_boxed_649_; size_t v_i_boxed_650_; lean_object* v_res_651_; 
v_sz_boxed_649_ = lean_unbox_usize(v_sz_641_);
lean_dec(v_sz_641_);
v_i_boxed_650_ = lean_unbox_usize(v_i_642_);
lean_dec(v_i_642_);
v_res_651_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Meta_synthSubsingletonInst_spec__1(v_sz_boxed_649_, v_i_boxed_650_, v_bs_643_, v___y_644_, v___y_645_, v___y_646_, v___y_647_);
lean_dec(v___y_647_);
lean_dec_ref(v___y_646_);
lean_dec(v___y_645_);
lean_dec_ref(v___y_644_);
return v_res_651_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__12(size_t v_sz_652_, size_t v_i_653_, lean_object* v_bs_654_){
_start:
{
uint8_t v___x_655_; 
v___x_655_ = lean_usize_dec_lt(v_i_653_, v_sz_652_);
if (v___x_655_ == 0)
{
return v_bs_654_;
}
else
{
lean_object* v_v_656_; lean_object* v_fst_657_; lean_object* v_snd_658_; lean_object* v___x_660_; uint8_t v_isShared_661_; uint8_t v_isSharedCheck_674_; 
v_v_656_ = lean_array_uget(v_bs_654_, v_i_653_);
v_fst_657_ = lean_ctor_get(v_v_656_, 0);
v_snd_658_ = lean_ctor_get(v_v_656_, 1);
v_isSharedCheck_674_ = !lean_is_exclusive(v_v_656_);
if (v_isSharedCheck_674_ == 0)
{
v___x_660_ = v_v_656_;
v_isShared_661_ = v_isSharedCheck_674_;
goto v_resetjp_659_;
}
else
{
lean_inc(v_snd_658_);
lean_inc(v_fst_657_);
lean_dec(v_v_656_);
v___x_660_ = lean_box(0);
v_isShared_661_ = v_isSharedCheck_674_;
goto v_resetjp_659_;
}
v_resetjp_659_:
{
lean_object* v___x_662_; lean_object* v_bs_x27_663_; uint8_t v___x_664_; lean_object* v___x_665_; lean_object* v___x_667_; 
v___x_662_ = lean_unsigned_to_nat(0u);
v_bs_x27_663_ = lean_array_uset(v_bs_654_, v_i_653_, v___x_662_);
v___x_664_ = 0;
v___x_665_ = lean_box(v___x_664_);
if (v_isShared_661_ == 0)
{
lean_ctor_set(v___x_660_, 0, v___x_665_);
v___x_667_ = v___x_660_;
goto v_reusejp_666_;
}
else
{
lean_object* v_reuseFailAlloc_673_; 
v_reuseFailAlloc_673_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_673_, 0, v___x_665_);
lean_ctor_set(v_reuseFailAlloc_673_, 1, v_snd_658_);
v___x_667_ = v_reuseFailAlloc_673_;
goto v_reusejp_666_;
}
v_reusejp_666_:
{
lean_object* v___x_668_; size_t v___x_669_; size_t v___x_670_; lean_object* v___x_671_; 
v___x_668_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_668_, 0, v_fst_657_);
lean_ctor_set(v___x_668_, 1, v___x_667_);
v___x_669_ = ((size_t)1ULL);
v___x_670_ = lean_usize_add(v_i_653_, v___x_669_);
v___x_671_ = lean_array_uset(v_bs_x27_663_, v_i_653_, v___x_668_);
v_i_653_ = v___x_670_;
v_bs_654_ = v___x_671_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__12___boxed(lean_object* v_sz_675_, lean_object* v_i_676_, lean_object* v_bs_677_){
_start:
{
size_t v_sz_boxed_678_; size_t v_i_boxed_679_; lean_object* v_res_680_; 
v_sz_boxed_678_ = lean_unbox_usize(v_sz_675_);
lean_dec(v_sz_675_);
v_i_boxed_679_ = lean_unbox_usize(v_i_676_);
lean_dec(v_i_676_);
v_res_680_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__12(v_sz_boxed_678_, v_i_boxed_679_, v_bs_677_);
return v_res_680_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13_spec__16_spec__17___redArg___lam__0(lean_object* v_k_681_, lean_object* v_b_682_, lean_object* v___y_683_, lean_object* v___y_684_, lean_object* v___y_685_, lean_object* v___y_686_){
_start:
{
lean_object* v___x_688_; 
lean_inc(v___y_686_);
lean_inc_ref(v___y_685_);
lean_inc(v___y_684_);
lean_inc_ref(v___y_683_);
v___x_688_ = lean_apply_6(v_k_681_, v_b_682_, v___y_683_, v___y_684_, v___y_685_, v___y_686_, lean_box(0));
return v___x_688_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13_spec__16_spec__17___redArg___lam__0___boxed(lean_object* v_k_689_, lean_object* v_b_690_, lean_object* v___y_691_, lean_object* v___y_692_, lean_object* v___y_693_, lean_object* v___y_694_, lean_object* v___y_695_){
_start:
{
lean_object* v_res_696_; 
v_res_696_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13_spec__16_spec__17___redArg___lam__0(v_k_689_, v_b_690_, v___y_691_, v___y_692_, v___y_693_, v___y_694_);
lean_dec(v___y_694_);
lean_dec_ref(v___y_693_);
lean_dec(v___y_692_);
lean_dec_ref(v___y_691_);
return v_res_696_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13_spec__16_spec__17___redArg(lean_object* v_name_697_, uint8_t v_bi_698_, lean_object* v_type_699_, lean_object* v_k_700_, uint8_t v_kind_701_, lean_object* v___y_702_, lean_object* v___y_703_, lean_object* v___y_704_, lean_object* v___y_705_){
_start:
{
lean_object* v___f_707_; lean_object* v___x_708_; 
v___f_707_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13_spec__16_spec__17___redArg___lam__0___boxed), 7, 1);
lean_closure_set(v___f_707_, 0, v_k_700_);
v___x_708_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDeclImp(lean_box(0), v_name_697_, v_bi_698_, v_type_699_, v___f_707_, v_kind_701_, v___y_702_, v___y_703_, v___y_704_, v___y_705_);
if (lean_obj_tag(v___x_708_) == 0)
{
lean_object* v_a_709_; lean_object* v___x_711_; uint8_t v_isShared_712_; uint8_t v_isSharedCheck_716_; 
v_a_709_ = lean_ctor_get(v___x_708_, 0);
v_isSharedCheck_716_ = !lean_is_exclusive(v___x_708_);
if (v_isSharedCheck_716_ == 0)
{
v___x_711_ = v___x_708_;
v_isShared_712_ = v_isSharedCheck_716_;
goto v_resetjp_710_;
}
else
{
lean_inc(v_a_709_);
lean_dec(v___x_708_);
v___x_711_ = lean_box(0);
v_isShared_712_ = v_isSharedCheck_716_;
goto v_resetjp_710_;
}
v_resetjp_710_:
{
lean_object* v___x_714_; 
if (v_isShared_712_ == 0)
{
v___x_714_ = v___x_711_;
goto v_reusejp_713_;
}
else
{
lean_object* v_reuseFailAlloc_715_; 
v_reuseFailAlloc_715_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_715_, 0, v_a_709_);
v___x_714_ = v_reuseFailAlloc_715_;
goto v_reusejp_713_;
}
v_reusejp_713_:
{
return v___x_714_;
}
}
}
else
{
lean_object* v_a_717_; lean_object* v___x_719_; uint8_t v_isShared_720_; uint8_t v_isSharedCheck_724_; 
v_a_717_ = lean_ctor_get(v___x_708_, 0);
v_isSharedCheck_724_ = !lean_is_exclusive(v___x_708_);
if (v_isSharedCheck_724_ == 0)
{
v___x_719_ = v___x_708_;
v_isShared_720_ = v_isSharedCheck_724_;
goto v_resetjp_718_;
}
else
{
lean_inc(v_a_717_);
lean_dec(v___x_708_);
v___x_719_ = lean_box(0);
v_isShared_720_ = v_isSharedCheck_724_;
goto v_resetjp_718_;
}
v_resetjp_718_:
{
lean_object* v___x_722_; 
if (v_isShared_720_ == 0)
{
v___x_722_ = v___x_719_;
goto v_reusejp_721_;
}
else
{
lean_object* v_reuseFailAlloc_723_; 
v_reuseFailAlloc_723_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_723_, 0, v_a_717_);
v___x_722_ = v_reuseFailAlloc_723_;
goto v_reusejp_721_;
}
v_reusejp_721_:
{
return v___x_722_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13_spec__16_spec__17___redArg___boxed(lean_object* v_name_725_, lean_object* v_bi_726_, lean_object* v_type_727_, lean_object* v_k_728_, lean_object* v_kind_729_, lean_object* v___y_730_, lean_object* v___y_731_, lean_object* v___y_732_, lean_object* v___y_733_, lean_object* v___y_734_){
_start:
{
uint8_t v_bi_boxed_735_; uint8_t v_kind_boxed_736_; lean_object* v_res_737_; 
v_bi_boxed_735_ = lean_unbox(v_bi_726_);
v_kind_boxed_736_ = lean_unbox(v_kind_729_);
v_res_737_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13_spec__16_spec__17___redArg(v_name_725_, v_bi_boxed_735_, v_type_727_, v_k_728_, v_kind_boxed_736_, v___y_730_, v___y_731_, v___y_732_, v___y_733_);
lean_dec(v___y_733_);
lean_dec_ref(v___y_732_);
lean_dec(v___y_731_);
lean_dec_ref(v___y_730_);
return v_res_737_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13_spec__16___lam__0(lean_object* v___x_738_, lean_object* v_a_739_, lean_object* v___y_740_, lean_object* v___y_741_, lean_object* v___y_742_, lean_object* v___y_743_){
_start:
{
lean_object* v___x_745_; lean_object* v___x_8860__overap_746_; lean_object* v___x_747_; 
v___x_745_ = l_Lean_instInhabitedExpr;
v___x_8860__overap_746_ = l_instInhabitedOfMonad___redArg(v___x_738_, v___x_745_);
lean_inc(v___y_743_);
lean_inc_ref(v___y_742_);
lean_inc(v___y_741_);
lean_inc_ref(v___y_740_);
v___x_747_ = lean_apply_5(v___x_8860__overap_746_, v___y_740_, v___y_741_, v___y_742_, v___y_743_, lean_box(0));
return v___x_747_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13_spec__16___lam__0___boxed(lean_object* v___x_748_, lean_object* v_a_749_, lean_object* v___y_750_, lean_object* v___y_751_, lean_object* v___y_752_, lean_object* v___y_753_, lean_object* v___y_754_){
_start:
{
lean_object* v_res_755_; 
v_res_755_ = lp_mathlib___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13_spec__16___lam__0(v___x_748_, v_a_749_, v___y_750_, v___y_751_, v___y_752_, v___y_753_);
lean_dec(v___y_753_);
lean_dec_ref(v___y_752_);
lean_dec(v___y_751_);
lean_dec_ref(v___y_750_);
lean_dec_ref(v_a_749_);
return v_res_755_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13_spec__16___closed__0(void){
_start:
{
lean_object* v___x_756_; 
v___x_756_ = l_instMonadEIO(lean_box(0));
return v___x_756_;
}
}
static lean_object* _init_lp_mathlib___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13_spec__16___closed__1(void){
_start:
{
lean_object* v___x_757_; lean_object* v___x_758_; 
v___x_757_ = lean_obj_once(&lp_mathlib___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13_spec__16___closed__0, &lp_mathlib___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13_spec__16___closed__0_once, _init_lp_mathlib___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13_spec__16___closed__0);
v___x_758_ = l_StateRefT_x27_instMonad___redArg(v___x_757_);
return v___x_758_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13_spec__16___lam__1___boxed(lean_object* v_acc_763_, lean_object* v_declInfos_764_, lean_object* v_k_765_, lean_object* v_kind_766_, lean_object* v_x_767_, lean_object* v___y_768_, lean_object* v___y_769_, lean_object* v___y_770_, lean_object* v___y_771_, lean_object* v___y_772_){
_start:
{
uint8_t v_kind_boxed_773_; lean_object* v_res_774_; 
v_kind_boxed_773_ = lean_unbox(v_kind_766_);
v_res_774_ = lp_mathlib___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13_spec__16___lam__1(v_acc_763_, v_declInfos_764_, v_k_765_, v_kind_boxed_773_, v_x_767_, v___y_768_, v___y_769_, v___y_770_, v___y_771_);
lean_dec(v___y_771_);
lean_dec_ref(v___y_770_);
lean_dec(v___y_769_);
lean_dec_ref(v___y_768_);
return v_res_774_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13_spec__16(lean_object* v_declInfos_775_, lean_object* v_k_776_, uint8_t v_kind_777_, lean_object* v_acc_778_, lean_object* v___y_779_, lean_object* v___y_780_, lean_object* v___y_781_, lean_object* v___y_782_){
_start:
{
lean_object* v___x_784_; lean_object* v_toApplicative_785_; lean_object* v_toFunctor_786_; lean_object* v_toSeq_787_; lean_object* v_toSeqLeft_788_; lean_object* v_toSeqRight_789_; lean_object* v___f_790_; lean_object* v___f_791_; lean_object* v___f_792_; lean_object* v___f_793_; lean_object* v___x_794_; lean_object* v___f_795_; lean_object* v___f_796_; lean_object* v___f_797_; lean_object* v___x_798_; lean_object* v___x_799_; lean_object* v___x_800_; lean_object* v_toApplicative_801_; lean_object* v___x_803_; uint8_t v_isShared_804_; uint8_t v_isSharedCheck_850_; 
v___x_784_ = lean_obj_once(&lp_mathlib___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13_spec__16___closed__1, &lp_mathlib___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13_spec__16___closed__1_once, _init_lp_mathlib___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13_spec__16___closed__1);
v_toApplicative_785_ = lean_ctor_get(v___x_784_, 0);
v_toFunctor_786_ = lean_ctor_get(v_toApplicative_785_, 0);
v_toSeq_787_ = lean_ctor_get(v_toApplicative_785_, 2);
v_toSeqLeft_788_ = lean_ctor_get(v_toApplicative_785_, 3);
v_toSeqRight_789_ = lean_ctor_get(v_toApplicative_785_, 4);
v___f_790_ = ((lean_object*)(lp_mathlib___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13_spec__16___closed__2));
v___f_791_ = ((lean_object*)(lp_mathlib___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13_spec__16___closed__3));
lean_inc_ref_n(v_toFunctor_786_, 2);
v___f_792_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_792_, 0, v_toFunctor_786_);
v___f_793_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_793_, 0, v_toFunctor_786_);
v___x_794_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_794_, 0, v___f_792_);
lean_ctor_set(v___x_794_, 1, v___f_793_);
lean_inc(v_toSeqRight_789_);
v___f_795_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_795_, 0, v_toSeqRight_789_);
lean_inc(v_toSeqLeft_788_);
v___f_796_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_796_, 0, v_toSeqLeft_788_);
lean_inc(v_toSeq_787_);
v___f_797_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_797_, 0, v_toSeq_787_);
v___x_798_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_798_, 0, v___x_794_);
lean_ctor_set(v___x_798_, 1, v___f_790_);
lean_ctor_set(v___x_798_, 2, v___f_797_);
lean_ctor_set(v___x_798_, 3, v___f_796_);
lean_ctor_set(v___x_798_, 4, v___f_795_);
v___x_799_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_799_, 0, v___x_798_);
lean_ctor_set(v___x_799_, 1, v___f_791_);
v___x_800_ = l_StateRefT_x27_instMonad___redArg(v___x_799_);
v_toApplicative_801_ = lean_ctor_get(v___x_800_, 0);
v_isSharedCheck_850_ = !lean_is_exclusive(v___x_800_);
if (v_isSharedCheck_850_ == 0)
{
lean_object* v_unused_851_; 
v_unused_851_ = lean_ctor_get(v___x_800_, 1);
lean_dec(v_unused_851_);
v___x_803_ = v___x_800_;
v_isShared_804_ = v_isSharedCheck_850_;
goto v_resetjp_802_;
}
else
{
lean_inc(v_toApplicative_801_);
lean_dec(v___x_800_);
v___x_803_ = lean_box(0);
v_isShared_804_ = v_isSharedCheck_850_;
goto v_resetjp_802_;
}
v_resetjp_802_:
{
lean_object* v_toFunctor_805_; lean_object* v_toSeq_806_; lean_object* v_toSeqLeft_807_; lean_object* v_toSeqRight_808_; lean_object* v___x_810_; uint8_t v_isShared_811_; uint8_t v_isSharedCheck_848_; 
v_toFunctor_805_ = lean_ctor_get(v_toApplicative_801_, 0);
v_toSeq_806_ = lean_ctor_get(v_toApplicative_801_, 2);
v_toSeqLeft_807_ = lean_ctor_get(v_toApplicative_801_, 3);
v_toSeqRight_808_ = lean_ctor_get(v_toApplicative_801_, 4);
v_isSharedCheck_848_ = !lean_is_exclusive(v_toApplicative_801_);
if (v_isSharedCheck_848_ == 0)
{
lean_object* v_unused_849_; 
v_unused_849_ = lean_ctor_get(v_toApplicative_801_, 1);
lean_dec(v_unused_849_);
v___x_810_ = v_toApplicative_801_;
v_isShared_811_ = v_isSharedCheck_848_;
goto v_resetjp_809_;
}
else
{
lean_inc(v_toSeqRight_808_);
lean_inc(v_toSeqLeft_807_);
lean_inc(v_toSeq_806_);
lean_inc(v_toFunctor_805_);
lean_dec(v_toApplicative_801_);
v___x_810_ = lean_box(0);
v_isShared_811_ = v_isSharedCheck_848_;
goto v_resetjp_809_;
}
v_resetjp_809_:
{
lean_object* v___f_812_; lean_object* v___f_813_; lean_object* v___f_814_; lean_object* v___f_815_; lean_object* v___x_816_; lean_object* v___f_817_; lean_object* v___f_818_; lean_object* v___f_819_; lean_object* v___x_821_; 
v___f_812_ = ((lean_object*)(lp_mathlib___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13_spec__16___closed__4));
v___f_813_ = ((lean_object*)(lp_mathlib___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13_spec__16___closed__5));
lean_inc_ref(v_toFunctor_805_);
v___f_814_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_814_, 0, v_toFunctor_805_);
v___f_815_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_815_, 0, v_toFunctor_805_);
v___x_816_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_816_, 0, v___f_814_);
lean_ctor_set(v___x_816_, 1, v___f_815_);
v___f_817_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_817_, 0, v_toSeqRight_808_);
v___f_818_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_818_, 0, v_toSeqLeft_807_);
v___f_819_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_819_, 0, v_toSeq_806_);
if (v_isShared_811_ == 0)
{
lean_ctor_set(v___x_810_, 4, v___f_817_);
lean_ctor_set(v___x_810_, 3, v___f_818_);
lean_ctor_set(v___x_810_, 2, v___f_819_);
lean_ctor_set(v___x_810_, 1, v___f_812_);
lean_ctor_set(v___x_810_, 0, v___x_816_);
v___x_821_ = v___x_810_;
goto v_reusejp_820_;
}
else
{
lean_object* v_reuseFailAlloc_847_; 
v_reuseFailAlloc_847_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_847_, 0, v___x_816_);
lean_ctor_set(v_reuseFailAlloc_847_, 1, v___f_812_);
lean_ctor_set(v_reuseFailAlloc_847_, 2, v___f_819_);
lean_ctor_set(v_reuseFailAlloc_847_, 3, v___f_818_);
lean_ctor_set(v_reuseFailAlloc_847_, 4, v___f_817_);
v___x_821_ = v_reuseFailAlloc_847_;
goto v_reusejp_820_;
}
v_reusejp_820_:
{
lean_object* v___x_823_; 
if (v_isShared_804_ == 0)
{
lean_ctor_set(v___x_803_, 1, v___f_813_);
lean_ctor_set(v___x_803_, 0, v___x_821_);
v___x_823_ = v___x_803_;
goto v_reusejp_822_;
}
else
{
lean_object* v_reuseFailAlloc_846_; 
v_reuseFailAlloc_846_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_846_, 0, v___x_821_);
lean_ctor_set(v_reuseFailAlloc_846_, 1, v___f_813_);
v___x_823_ = v_reuseFailAlloc_846_;
goto v_reusejp_822_;
}
v_reusejp_822_:
{
lean_object* v___x_824_; lean_object* v___x_825_; uint8_t v___x_826_; 
v___x_824_ = lean_array_get_size(v_acc_778_);
v___x_825_ = lean_array_get_size(v_declInfos_775_);
v___x_826_ = lean_nat_dec_lt(v___x_824_, v___x_825_);
if (v___x_826_ == 0)
{
lean_object* v___x_827_; 
lean_dec_ref(v___x_823_);
lean_dec_ref(v_declInfos_775_);
lean_inc(v___y_782_);
lean_inc_ref(v___y_781_);
lean_inc(v___y_780_);
lean_inc_ref(v___y_779_);
v___x_827_ = lean_apply_6(v_k_776_, v_acc_778_, v___y_779_, v___y_780_, v___y_781_, v___y_782_, lean_box(0));
return v___x_827_;
}
else
{
lean_object* v___f_828_; lean_object* v___x_829_; uint8_t v___x_830_; lean_object* v___f_831_; lean_object* v___x_832_; lean_object* v___x_833_; lean_object* v___x_834_; lean_object* v___x_835_; lean_object* v_snd_836_; lean_object* v_fst_837_; lean_object* v_fst_838_; lean_object* v_snd_839_; lean_object* v___x_840_; 
v___f_828_ = lean_alloc_closure((void*)(lp_mathlib___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13_spec__16___lam__0___boxed), 7, 1);
lean_closure_set(v___f_828_, 0, v___x_823_);
v___x_829_ = lean_box(0);
v___x_830_ = 0;
v___f_831_ = lean_alloc_closure((void*)(l_Pi_instInhabited___redArg___lam__0), 2, 1);
lean_closure_set(v___f_831_, 0, v___f_828_);
v___x_832_ = lean_box(v___x_830_);
v___x_833_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_833_, 0, v___x_832_);
lean_ctor_set(v___x_833_, 1, v___f_831_);
v___x_834_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_834_, 0, v___x_829_);
lean_ctor_set(v___x_834_, 1, v___x_833_);
v___x_835_ = lean_array_get(v___x_834_, v_declInfos_775_, v___x_824_);
lean_dec_ref_known(v___x_834_, 2);
v_snd_836_ = lean_ctor_get(v___x_835_, 1);
lean_inc(v_snd_836_);
v_fst_837_ = lean_ctor_get(v___x_835_, 0);
lean_inc(v_fst_837_);
lean_dec(v___x_835_);
v_fst_838_ = lean_ctor_get(v_snd_836_, 0);
lean_inc(v_fst_838_);
v_snd_839_ = lean_ctor_get(v_snd_836_, 1);
lean_inc(v_snd_839_);
lean_dec(v_snd_836_);
lean_inc(v___y_782_);
lean_inc_ref(v___y_781_);
lean_inc(v___y_780_);
lean_inc_ref(v___y_779_);
lean_inc_ref(v_acc_778_);
v___x_840_ = lean_apply_6(v_snd_839_, v_acc_778_, v___y_779_, v___y_780_, v___y_781_, v___y_782_, lean_box(0));
if (lean_obj_tag(v___x_840_) == 0)
{
lean_object* v_a_841_; lean_object* v___x_842_; lean_object* v___f_843_; uint8_t v___x_844_; lean_object* v___x_845_; 
v_a_841_ = lean_ctor_get(v___x_840_, 0);
lean_inc(v_a_841_);
lean_dec_ref_known(v___x_840_, 1);
v___x_842_ = lean_box(v_kind_777_);
v___f_843_ = lean_alloc_closure((void*)(lp_mathlib___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13_spec__16___lam__1___boxed), 10, 4);
lean_closure_set(v___f_843_, 0, v_acc_778_);
lean_closure_set(v___f_843_, 1, v_declInfos_775_);
lean_closure_set(v___f_843_, 2, v_k_776_);
lean_closure_set(v___f_843_, 3, v___x_842_);
v___x_844_ = lean_unbox(v_fst_838_);
lean_dec(v_fst_838_);
v___x_845_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13_spec__16_spec__17___redArg(v_fst_837_, v___x_844_, v_a_841_, v___f_843_, v_kind_777_, v___y_779_, v___y_780_, v___y_781_, v___y_782_);
return v___x_845_;
}
else
{
lean_dec(v_fst_838_);
lean_dec(v_fst_837_);
lean_dec_ref(v_acc_778_);
lean_dec_ref(v_k_776_);
lean_dec_ref(v_declInfos_775_);
return v___x_840_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13_spec__16___lam__1(lean_object* v_acc_852_, lean_object* v_declInfos_853_, lean_object* v_k_854_, uint8_t v_kind_855_, lean_object* v_x_856_, lean_object* v___y_857_, lean_object* v___y_858_, lean_object* v___y_859_, lean_object* v___y_860_){
_start:
{
lean_object* v___x_862_; lean_object* v___x_863_; 
v___x_862_ = lean_array_push(v_acc_852_, v_x_856_);
v___x_863_ = lp_mathlib___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13_spec__16(v_declInfos_853_, v_k_854_, v_kind_855_, v___x_862_, v___y_857_, v___y_858_, v___y_859_, v___y_860_);
return v___x_863_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13_spec__16___boxed(lean_object* v_declInfos_864_, lean_object* v_k_865_, lean_object* v_kind_866_, lean_object* v_acc_867_, lean_object* v___y_868_, lean_object* v___y_869_, lean_object* v___y_870_, lean_object* v___y_871_, lean_object* v___y_872_){
_start:
{
uint8_t v_kind_boxed_873_; lean_object* v_res_874_; 
v_kind_boxed_873_ = lean_unbox(v_kind_866_);
v_res_874_ = lp_mathlib___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13_spec__16(v_declInfos_864_, v_k_865_, v_kind_boxed_873_, v_acc_867_, v___y_868_, v___y_869_, v___y_870_, v___y_871_);
lean_dec(v___y_871_);
lean_dec_ref(v___y_870_);
lean_dec(v___y_869_);
lean_dec_ref(v___y_868_);
return v_res_874_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13(lean_object* v_declInfos_877_, lean_object* v_k_878_, uint8_t v_kind_879_, lean_object* v___y_880_, lean_object* v___y_881_, lean_object* v___y_882_, lean_object* v___y_883_){
_start:
{
lean_object* v___x_885_; lean_object* v___x_886_; 
v___x_885_ = ((lean_object*)(lp_mathlib_Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13___closed__0));
v___x_886_ = lp_mathlib___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13_spec__16(v_declInfos_877_, v_k_878_, v_kind_879_, v___x_885_, v___y_880_, v___y_881_, v___y_882_, v___y_883_);
return v___x_886_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13___boxed(lean_object* v_declInfos_887_, lean_object* v_k_888_, lean_object* v_kind_889_, lean_object* v___y_890_, lean_object* v___y_891_, lean_object* v___y_892_, lean_object* v___y_893_, lean_object* v___y_894_){
_start:
{
uint8_t v_kind_boxed_895_; lean_object* v_res_896_; 
v_kind_boxed_895_ = lean_unbox(v_kind_889_);
v_res_896_ = lp_mathlib_Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13(v_declInfos_887_, v_k_888_, v_kind_boxed_895_, v___y_890_, v___y_891_, v___y_892_, v___y_893_);
lean_dec(v___y_893_);
lean_dec_ref(v___y_892_);
lean_dec(v___y_891_);
lean_dec_ref(v___y_890_);
return v_res_896_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9(lean_object* v_declInfos_897_, lean_object* v_k_898_, uint8_t v_kind_899_, lean_object* v___y_900_, lean_object* v___y_901_, lean_object* v___y_902_, lean_object* v___y_903_){
_start:
{
size_t v_sz_905_; size_t v___x_906_; lean_object* v___x_907_; lean_object* v___x_908_; 
v_sz_905_ = lean_array_size(v_declInfos_897_);
v___x_906_ = ((size_t)0ULL);
v___x_907_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__12(v_sz_905_, v___x_906_, v_declInfos_897_);
v___x_908_ = lp_mathlib_Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13(v___x_907_, v_k_898_, v_kind_899_, v___y_900_, v___y_901_, v___y_902_, v___y_903_);
return v___x_908_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9___boxed(lean_object* v_declInfos_909_, lean_object* v_k_910_, lean_object* v_kind_911_, lean_object* v___y_912_, lean_object* v___y_913_, lean_object* v___y_914_, lean_object* v___y_915_, lean_object* v___y_916_){
_start:
{
uint8_t v_kind_boxed_917_; lean_object* v_res_918_; 
v_kind_boxed_917_ = lean_unbox(v_kind_911_);
v_res_918_ = lp_mathlib_Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9(v_declInfos_909_, v_k_910_, v_kind_boxed_917_, v___y_912_, v___y_913_, v___y_914_, v___y_915_);
lean_dec(v___y_915_);
lean_dec_ref(v___y_914_);
lean_dec(v___y_913_);
lean_dec_ref(v___y_912_);
return v_res_918_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Meta_synthSubsingletonInst_spec__8___lam__0(lean_object* v_v_919_, lean_object* v_x_920_, lean_object* v___y_921_, lean_object* v___y_922_, lean_object* v___y_923_, lean_object* v___y_924_){
_start:
{
lean_object* v___x_926_; 
lean_inc(v___y_924_);
lean_inc_ref(v___y_923_);
lean_inc(v___y_922_);
lean_inc_ref(v___y_921_);
v___x_926_ = lean_infer_type(v_v_919_, v___y_921_, v___y_922_, v___y_923_, v___y_924_);
return v___x_926_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Meta_synthSubsingletonInst_spec__8___lam__0___boxed(lean_object* v_v_927_, lean_object* v_x_928_, lean_object* v___y_929_, lean_object* v___y_930_, lean_object* v___y_931_, lean_object* v___y_932_, lean_object* v___y_933_){
_start:
{
lean_object* v_res_934_; 
v_res_934_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Meta_synthSubsingletonInst_spec__8___lam__0(v_v_927_, v_x_928_, v___y_929_, v___y_930_, v___y_931_, v___y_932_);
lean_dec(v___y_932_);
lean_dec_ref(v___y_931_);
lean_dec(v___y_930_);
lean_dec_ref(v___y_929_);
lean_dec_ref(v_x_928_);
return v_res_934_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Meta_synthSubsingletonInst_spec__8(size_t v_sz_938_, size_t v_i_939_, lean_object* v_bs_940_){
_start:
{
uint8_t v___x_941_; 
v___x_941_ = lean_usize_dec_lt(v_i_939_, v_sz_938_);
if (v___x_941_ == 0)
{
return v_bs_940_;
}
else
{
lean_object* v_v_942_; lean_object* v___f_943_; lean_object* v___x_944_; lean_object* v_bs_x27_945_; lean_object* v___x_946_; lean_object* v___x_947_; size_t v___x_948_; size_t v___x_949_; lean_object* v___x_950_; 
v_v_942_ = lean_array_uget_borrowed(v_bs_940_, v_i_939_);
lean_inc(v_v_942_);
v___f_943_ = lean_alloc_closure((void*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Meta_synthSubsingletonInst_spec__8___lam__0___boxed), 7, 1);
lean_closure_set(v___f_943_, 0, v_v_942_);
v___x_944_ = lean_unsigned_to_nat(0u);
v_bs_x27_945_ = lean_array_uset(v_bs_940_, v_i_939_, v___x_944_);
v___x_946_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Meta_synthSubsingletonInst_spec__8___closed__1));
v___x_947_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_947_, 0, v___x_946_);
lean_ctor_set(v___x_947_, 1, v___f_943_);
v___x_948_ = ((size_t)1ULL);
v___x_949_ = lean_usize_add(v_i_939_, v___x_948_);
v___x_950_ = lean_array_uset(v_bs_x27_945_, v_i_939_, v___x_947_);
v_i_939_ = v___x_949_;
v_bs_940_ = v___x_950_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Meta_synthSubsingletonInst_spec__8___boxed(lean_object* v_sz_952_, lean_object* v_i_953_, lean_object* v_bs_954_){
_start:
{
size_t v_sz_boxed_955_; size_t v_i_boxed_956_; lean_object* v_res_957_; 
v_sz_boxed_955_ = lean_unbox_usize(v_sz_952_);
lean_dec(v_sz_952_);
v_i_boxed_956_ = lean_unbox_usize(v_i_953_);
lean_dec(v_i_953_);
v_res_957_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Meta_synthSubsingletonInst_spec__8(v_sz_boxed_955_, v_i_boxed_956_, v_bs_954_);
return v_res_957_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_synthSubsingletonInst___lam__2(lean_object* v_insts_958_, lean_object* v_ty_959_, lean_object* v___y_960_, lean_object* v___y_961_, lean_object* v___y_962_, lean_object* v___y_963_){
_start:
{
size_t v_sz_965_; size_t v___x_966_; lean_object* v___x_967_; 
v_sz_965_ = lean_array_size(v_insts_958_);
v___x_966_ = ((size_t)0ULL);
lean_inc_ref(v_insts_958_);
v___x_967_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Meta_synthSubsingletonInst_spec__1(v_sz_965_, v___x_966_, v_insts_958_, v___y_960_, v___y_961_, v___y_962_, v___y_963_);
if (lean_obj_tag(v___x_967_) == 0)
{
lean_object* v_a_968_; lean_object* v___x_969_; lean_object* v_fst_970_; lean_object* v_snd_971_; lean_object* v___f_972_; size_t v_sz_973_; lean_object* v___x_974_; uint8_t v___x_975_; lean_object* v___x_976_; 
v_a_968_ = lean_ctor_get(v___x_967_, 0);
lean_inc(v_a_968_);
lean_dec_ref_known(v___x_967_, 1);
v___x_969_ = l_Array_unzip___redArg(v_a_968_);
lean_dec(v_a_968_);
v_fst_970_ = lean_ctor_get(v___x_969_, 0);
lean_inc_n(v_fst_970_, 2);
v_snd_971_ = lean_ctor_get(v___x_969_, 1);
lean_inc(v_snd_971_);
lean_dec_ref(v___x_969_);
v___f_972_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_synthSubsingletonInst___lam__1___boxed), 10, 4);
lean_closure_set(v___f_972_, 0, v_ty_959_);
lean_closure_set(v___f_972_, 1, v_insts_958_);
lean_closure_set(v___f_972_, 2, v_fst_970_);
lean_closure_set(v___f_972_, 3, v_snd_971_);
v_sz_973_ = lean_array_size(v_fst_970_);
v___x_974_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Meta_synthSubsingletonInst_spec__8(v_sz_973_, v___x_966_, v_fst_970_);
v___x_975_ = 0;
v___x_976_ = lp_mathlib_Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9(v___x_974_, v___f_972_, v___x_975_, v___y_960_, v___y_961_, v___y_962_, v___y_963_);
return v___x_976_;
}
else
{
lean_object* v_a_977_; lean_object* v___x_979_; uint8_t v_isShared_980_; uint8_t v_isSharedCheck_984_; 
lean_dec_ref(v_ty_959_);
lean_dec_ref(v_insts_958_);
v_a_977_ = lean_ctor_get(v___x_967_, 0);
v_isSharedCheck_984_ = !lean_is_exclusive(v___x_967_);
if (v_isSharedCheck_984_ == 0)
{
v___x_979_ = v___x_967_;
v_isShared_980_ = v_isSharedCheck_984_;
goto v_resetjp_978_;
}
else
{
lean_inc(v_a_977_);
lean_dec(v___x_967_);
v___x_979_ = lean_box(0);
v_isShared_980_ = v_isSharedCheck_984_;
goto v_resetjp_978_;
}
v_resetjp_978_:
{
lean_object* v___x_982_; 
if (v_isShared_980_ == 0)
{
v___x_982_ = v___x_979_;
goto v_reusejp_981_;
}
else
{
lean_object* v_reuseFailAlloc_983_; 
v_reuseFailAlloc_983_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_983_, 0, v_a_977_);
v___x_982_ = v_reuseFailAlloc_983_;
goto v_reusejp_981_;
}
v_reusejp_981_:
{
return v___x_982_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_synthSubsingletonInst___lam__2___boxed(lean_object* v_insts_985_, lean_object* v_ty_986_, lean_object* v___y_987_, lean_object* v___y_988_, lean_object* v___y_989_, lean_object* v___y_990_, lean_object* v___y_991_){
_start:
{
lean_object* v_res_992_; 
v_res_992_ = lp_mathlib_Lean_Meta_synthSubsingletonInst___lam__2(v_insts_985_, v_ty_986_, v___y_987_, v___y_988_, v___y_989_, v___y_990_);
lean_dec(v___y_990_);
lean_dec_ref(v___y_989_);
lean_dec(v___y_988_);
lean_dec_ref(v___y_987_);
return v_res_992_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_synthSubsingletonInst(lean_object* v_ty_993_, lean_object* v_insts_994_, lean_object* v_a_995_, lean_object* v_a_996_, lean_object* v_a_997_, lean_object* v_a_998_){
_start:
{
lean_object* v___f_1000_; uint8_t v___x_1001_; lean_object* v___x_1002_; 
v___f_1000_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_synthSubsingletonInst___lam__2___boxed), 7, 2);
lean_closure_set(v___f_1000_, 0, v_insts_994_);
lean_closure_set(v___f_1000_, 1, v_ty_993_);
v___x_1001_ = 0;
v___x_1002_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Lean_Meta_synthSubsingletonInst_spec__10___redArg(v___f_1000_, v___x_1001_, v_a_995_, v_a_996_, v_a_997_, v_a_998_);
return v___x_1002_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_synthSubsingletonInst___boxed(lean_object* v_ty_1003_, lean_object* v_insts_1004_, lean_object* v_a_1005_, lean_object* v_a_1006_, lean_object* v_a_1007_, lean_object* v_a_1008_, lean_object* v_a_1009_){
_start:
{
lean_object* v_res_1010_; 
v_res_1010_ = lp_mathlib_Lean_Meta_synthSubsingletonInst(v_ty_1003_, v_insts_1004_, v_a_1005_, v_a_1006_, v_a_1007_, v_a_1008_);
lean_dec(v_a_1008_);
lean_dec_ref(v_a_1007_);
lean_dec(v_a_1006_);
lean_dec_ref(v_a_1005_);
return v_res_1010_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_Meta_synthSubsingletonInst_spec__4(lean_object* v_00_u03b1_1011_, lean_object* v_ref_1012_, lean_object* v_msg_1013_, lean_object* v___y_1014_, lean_object* v___y_1015_, lean_object* v___y_1016_, lean_object* v___y_1017_){
_start:
{
lean_object* v___x_1019_; 
v___x_1019_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_Meta_synthSubsingletonInst_spec__4___redArg(v_ref_1012_, v_msg_1013_, v___y_1014_, v___y_1015_, v___y_1016_, v___y_1017_);
return v___x_1019_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwErrorAt___at___00Lean_Meta_synthSubsingletonInst_spec__4___boxed(lean_object* v_00_u03b1_1020_, lean_object* v_ref_1021_, lean_object* v_msg_1022_, lean_object* v___y_1023_, lean_object* v___y_1024_, lean_object* v___y_1025_, lean_object* v___y_1026_, lean_object* v___y_1027_){
_start:
{
lean_object* v_res_1028_; 
v_res_1028_ = lp_mathlib_Lean_throwErrorAt___at___00Lean_Meta_synthSubsingletonInst_spec__4(v_00_u03b1_1020_, v_ref_1021_, v_msg_1022_, v___y_1023_, v___y_1024_, v___y_1025_, v___y_1026_);
lean_dec(v___y_1026_);
lean_dec_ref(v___y_1025_);
lean_dec(v___y_1024_);
lean_dec_ref(v___y_1023_);
lean_dec(v_ref_1021_);
return v_res_1028_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_synthSubsingletonInst_spec__6(lean_object* v_insts_1029_, lean_object* v_fst_1030_, lean_object* v___x_1031_, lean_object* v___x_1032_, lean_object* v_snd_1033_, lean_object* v_range_1034_, lean_object* v_b_1035_, lean_object* v_i_1036_, lean_object* v_hs_1037_, lean_object* v_hl_1038_, lean_object* v___y_1039_, lean_object* v___y_1040_, lean_object* v___y_1041_, lean_object* v___y_1042_){
_start:
{
lean_object* v___x_1044_; 
v___x_1044_ = lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_synthSubsingletonInst_spec__6___redArg(v_insts_1029_, v_fst_1030_, v___x_1031_, v___x_1032_, v_snd_1033_, v_range_1034_, v_b_1035_, v_i_1036_, v___y_1039_, v___y_1040_, v___y_1041_, v___y_1042_);
return v___x_1044_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_synthSubsingletonInst_spec__6___boxed(lean_object* v_insts_1045_, lean_object* v_fst_1046_, lean_object* v___x_1047_, lean_object* v___x_1048_, lean_object* v_snd_1049_, lean_object* v_range_1050_, lean_object* v_b_1051_, lean_object* v_i_1052_, lean_object* v_hs_1053_, lean_object* v_hl_1054_, lean_object* v___y_1055_, lean_object* v___y_1056_, lean_object* v___y_1057_, lean_object* v___y_1058_, lean_object* v___y_1059_){
_start:
{
lean_object* v_res_1060_; 
v_res_1060_ = lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_synthSubsingletonInst_spec__6(v_insts_1045_, v_fst_1046_, v___x_1047_, v___x_1048_, v_snd_1049_, v_range_1050_, v_b_1051_, v_i_1052_, v_hs_1053_, v_hl_1054_, v___y_1055_, v___y_1056_, v___y_1057_, v___y_1058_);
lean_dec(v___y_1058_);
lean_dec_ref(v___y_1057_);
lean_dec(v___y_1056_);
lean_dec_ref(v___y_1055_);
lean_dec(v_i_1052_);
lean_dec_ref(v_range_1050_);
lean_dec_ref(v_snd_1049_);
lean_dec_ref(v___x_1048_);
lean_dec(v___x_1047_);
lean_dec_ref(v_fst_1046_);
lean_dec_ref(v_insts_1045_);
return v_res_1060_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Basic_0__Lean_Meta_withNewLocalInstancesImpAux___at___00Lean_Meta_withNewLocalInstances___at___00Lean_Meta_synthSubsingletonInst_spec__7_spec__9(lean_object* v_00_u03b1_1061_, lean_object* v_fvars_1062_, lean_object* v_j_1063_, lean_object* v_x_1064_, lean_object* v___y_1065_, lean_object* v___y_1066_, lean_object* v___y_1067_, lean_object* v___y_1068_){
_start:
{
lean_object* v___x_1070_; 
v___x_1070_ = lp_mathlib___private_Lean_Meta_Basic_0__Lean_Meta_withNewLocalInstancesImpAux___at___00Lean_Meta_withNewLocalInstances___at___00Lean_Meta_synthSubsingletonInst_spec__7_spec__9___redArg(v_fvars_1062_, v_j_1063_, v_x_1064_, v___y_1065_, v___y_1066_, v___y_1067_, v___y_1068_);
return v___x_1070_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Meta_Basic_0__Lean_Meta_withNewLocalInstancesImpAux___at___00Lean_Meta_withNewLocalInstances___at___00Lean_Meta_synthSubsingletonInst_spec__7_spec__9___boxed(lean_object* v_00_u03b1_1071_, lean_object* v_fvars_1072_, lean_object* v_j_1073_, lean_object* v_x_1074_, lean_object* v___y_1075_, lean_object* v___y_1076_, lean_object* v___y_1077_, lean_object* v___y_1078_, lean_object* v___y_1079_){
_start:
{
lean_object* v_res_1080_; 
v_res_1080_ = lp_mathlib___private_Lean_Meta_Basic_0__Lean_Meta_withNewLocalInstancesImpAux___at___00Lean_Meta_withNewLocalInstances___at___00Lean_Meta_synthSubsingletonInst_spec__7_spec__9(v_00_u03b1_1071_, v_fvars_1072_, v_j_1073_, v_x_1074_, v___y_1075_, v___y_1076_, v___y_1077_, v___y_1078_);
lean_dec(v___y_1078_);
lean_dec_ref(v___y_1077_);
lean_dec(v___y_1076_);
lean_dec_ref(v___y_1075_);
return v_res_1080_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewLocalInstances___at___00Lean_Meta_synthSubsingletonInst_spec__7(lean_object* v_00_u03b1_1081_, lean_object* v_fvars_1082_, lean_object* v_j_1083_, lean_object* v_x_1084_, lean_object* v___y_1085_, lean_object* v___y_1086_, lean_object* v___y_1087_, lean_object* v___y_1088_){
_start:
{
lean_object* v___x_1090_; 
v___x_1090_ = lp_mathlib_Lean_Meta_withNewLocalInstances___at___00Lean_Meta_synthSubsingletonInst_spec__7___redArg(v_fvars_1082_, v_j_1083_, v_x_1084_, v___y_1085_, v___y_1086_, v___y_1087_, v___y_1088_);
return v___x_1090_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewLocalInstances___at___00Lean_Meta_synthSubsingletonInst_spec__7___boxed(lean_object* v_00_u03b1_1091_, lean_object* v_fvars_1092_, lean_object* v_j_1093_, lean_object* v_x_1094_, lean_object* v___y_1095_, lean_object* v___y_1096_, lean_object* v___y_1097_, lean_object* v___y_1098_, lean_object* v___y_1099_){
_start:
{
lean_object* v_res_1100_; 
v_res_1100_ = lp_mathlib_Lean_Meta_withNewLocalInstances___at___00Lean_Meta_synthSubsingletonInst_spec__7(v_00_u03b1_1091_, v_fvars_1092_, v_j_1093_, v_x_1094_, v___y_1095_, v___y_1096_, v___y_1097_, v___y_1098_);
lean_dec(v___y_1098_);
lean_dec_ref(v___y_1097_);
lean_dec(v___y_1096_);
lean_dec_ref(v___y_1095_);
return v_res_1100_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Meta_synthSubsingletonInst_spec__4_spec__4(lean_object* v_00_u03b1_1101_, lean_object* v_msg_1102_, lean_object* v___y_1103_, lean_object* v___y_1104_, lean_object* v___y_1105_, lean_object* v___y_1106_){
_start:
{
lean_object* v___x_1108_; 
v___x_1108_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Meta_synthSubsingletonInst_spec__4_spec__4___redArg(v_msg_1102_, v___y_1103_, v___y_1104_, v___y_1105_, v___y_1106_);
return v___x_1108_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Meta_synthSubsingletonInst_spec__4_spec__4___boxed(lean_object* v_00_u03b1_1109_, lean_object* v_msg_1110_, lean_object* v___y_1111_, lean_object* v___y_1112_, lean_object* v___y_1113_, lean_object* v___y_1114_, lean_object* v___y_1115_){
_start:
{
lean_object* v_res_1116_; 
v_res_1116_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Meta_synthSubsingletonInst_spec__4_spec__4(v_00_u03b1_1109_, v_msg_1110_, v___y_1111_, v___y_1112_, v___y_1113_, v___y_1114_);
lean_dec(v___y_1114_);
lean_dec_ref(v___y_1113_);
lean_dec(v___y_1112_);
lean_dec_ref(v___y_1111_);
return v_res_1116_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_synthSubsingletonInst_spec__6_spec__7(lean_object* v___x_1117_, lean_object* v___x_1118_, lean_object* v_snd_1119_, lean_object* v_insts_1120_, lean_object* v_fst_1121_, lean_object* v_range_1122_, lean_object* v_b_1123_, lean_object* v_i_1124_, lean_object* v_hs_1125_, lean_object* v_hl_1126_, lean_object* v___y_1127_, lean_object* v___y_1128_, lean_object* v___y_1129_, lean_object* v___y_1130_){
_start:
{
lean_object* v___x_1132_; 
v___x_1132_ = lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_synthSubsingletonInst_spec__6_spec__7___redArg(v___x_1117_, v___x_1118_, v_snd_1119_, v_insts_1120_, v_fst_1121_, v_range_1122_, v_b_1123_, v_i_1124_, v___y_1127_, v___y_1128_, v___y_1129_, v___y_1130_);
return v___x_1132_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_synthSubsingletonInst_spec__6_spec__7___boxed(lean_object* v___x_1133_, lean_object* v___x_1134_, lean_object* v_snd_1135_, lean_object* v_insts_1136_, lean_object* v_fst_1137_, lean_object* v_range_1138_, lean_object* v_b_1139_, lean_object* v_i_1140_, lean_object* v_hs_1141_, lean_object* v_hl_1142_, lean_object* v___y_1143_, lean_object* v___y_1144_, lean_object* v___y_1145_, lean_object* v___y_1146_, lean_object* v___y_1147_){
_start:
{
lean_object* v_res_1148_; 
v_res_1148_ = lp_mathlib___private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00__private_Init_Data_Range_Basic_0__Std_Legacy_Range_forIn_x27_loop___at___00Lean_Meta_synthSubsingletonInst_spec__6_spec__7(v___x_1133_, v___x_1134_, v_snd_1135_, v_insts_1136_, v_fst_1137_, v_range_1138_, v_b_1139_, v_i_1140_, v_hs_1141_, v_hl_1142_, v___y_1143_, v___y_1144_, v___y_1145_, v___y_1146_);
lean_dec(v___y_1146_);
lean_dec_ref(v___y_1145_);
lean_dec(v___y_1144_);
lean_dec_ref(v___y_1143_);
lean_dec_ref(v_range_1138_);
lean_dec_ref(v_fst_1137_);
lean_dec_ref(v_insts_1136_);
lean_dec_ref(v_snd_1135_);
lean_dec_ref(v___x_1134_);
lean_dec(v___x_1133_);
return v_res_1148_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13_spec__16_spec__17(lean_object* v_00_u03b1_1149_, lean_object* v_name_1150_, uint8_t v_bi_1151_, lean_object* v_type_1152_, lean_object* v_k_1153_, uint8_t v_kind_1154_, lean_object* v___y_1155_, lean_object* v___y_1156_, lean_object* v___y_1157_, lean_object* v___y_1158_){
_start:
{
lean_object* v___x_1160_; 
v___x_1160_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13_spec__16_spec__17___redArg(v_name_1150_, v_bi_1151_, v_type_1152_, v_k_1153_, v_kind_1154_, v___y_1155_, v___y_1156_, v___y_1157_, v___y_1158_);
return v___x_1160_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13_spec__16_spec__17___boxed(lean_object* v_00_u03b1_1161_, lean_object* v_name_1162_, lean_object* v_bi_1163_, lean_object* v_type_1164_, lean_object* v_k_1165_, lean_object* v_kind_1166_, lean_object* v___y_1167_, lean_object* v___y_1168_, lean_object* v___y_1169_, lean_object* v___y_1170_, lean_object* v___y_1171_){
_start:
{
uint8_t v_bi_boxed_1172_; uint8_t v_kind_boxed_1173_; lean_object* v_res_1174_; 
v_bi_boxed_1172_ = lean_unbox(v_bi_1163_);
v_kind_boxed_1173_ = lean_unbox(v_kind_1166_);
v_res_1174_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00__private_Lean_Meta_Basic_0__Lean_Meta_withLocalDecls_loop___at___00Lean_Meta_withLocalDecls___at___00Lean_Meta_withLocalDeclsD___at___00Lean_Meta_synthSubsingletonInst_spec__9_spec__13_spec__16_spec__17(v_00_u03b1_1161_, v_name_1162_, v_bi_boxed_1172_, v_type_1164_, v_k_1165_, v_kind_boxed_1173_, v___y_1167_, v___y_1168_, v___y_1169_, v___y_1170_);
lean_dec(v___y_1170_);
lean_dec_ref(v___y_1169_);
lean_dec(v___y_1168_);
lean_dec_ref(v___y_1167_);
return v_res_1174_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_MVarId_subsingleton_spec__1___redArg(lean_object* v_mvarId_1175_, lean_object* v_x_1176_, lean_object* v___y_1177_, lean_object* v___y_1178_, lean_object* v___y_1179_, lean_object* v___y_1180_){
_start:
{
lean_object* v___x_1182_; 
v___x_1182_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_box(0), v_mvarId_1175_, v_x_1176_, v___y_1177_, v___y_1178_, v___y_1179_, v___y_1180_);
if (lean_obj_tag(v___x_1182_) == 0)
{
lean_object* v_a_1183_; lean_object* v___x_1185_; uint8_t v_isShared_1186_; uint8_t v_isSharedCheck_1190_; 
v_a_1183_ = lean_ctor_get(v___x_1182_, 0);
v_isSharedCheck_1190_ = !lean_is_exclusive(v___x_1182_);
if (v_isSharedCheck_1190_ == 0)
{
v___x_1185_ = v___x_1182_;
v_isShared_1186_ = v_isSharedCheck_1190_;
goto v_resetjp_1184_;
}
else
{
lean_inc(v_a_1183_);
lean_dec(v___x_1182_);
v___x_1185_ = lean_box(0);
v_isShared_1186_ = v_isSharedCheck_1190_;
goto v_resetjp_1184_;
}
v_resetjp_1184_:
{
lean_object* v___x_1188_; 
if (v_isShared_1186_ == 0)
{
v___x_1188_ = v___x_1185_;
goto v_reusejp_1187_;
}
else
{
lean_object* v_reuseFailAlloc_1189_; 
v_reuseFailAlloc_1189_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1189_, 0, v_a_1183_);
v___x_1188_ = v_reuseFailAlloc_1189_;
goto v_reusejp_1187_;
}
v_reusejp_1187_:
{
return v___x_1188_;
}
}
}
else
{
lean_object* v_a_1191_; lean_object* v___x_1193_; uint8_t v_isShared_1194_; uint8_t v_isSharedCheck_1198_; 
v_a_1191_ = lean_ctor_get(v___x_1182_, 0);
v_isSharedCheck_1198_ = !lean_is_exclusive(v___x_1182_);
if (v_isSharedCheck_1198_ == 0)
{
v___x_1193_ = v___x_1182_;
v_isShared_1194_ = v_isSharedCheck_1198_;
goto v_resetjp_1192_;
}
else
{
lean_inc(v_a_1191_);
lean_dec(v___x_1182_);
v___x_1193_ = lean_box(0);
v_isShared_1194_ = v_isSharedCheck_1198_;
goto v_resetjp_1192_;
}
v_resetjp_1192_:
{
lean_object* v___x_1196_; 
if (v_isShared_1194_ == 0)
{
v___x_1196_ = v___x_1193_;
goto v_reusejp_1195_;
}
else
{
lean_object* v_reuseFailAlloc_1197_; 
v_reuseFailAlloc_1197_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1197_, 0, v_a_1191_);
v___x_1196_ = v_reuseFailAlloc_1197_;
goto v_reusejp_1195_;
}
v_reusejp_1195_:
{
return v___x_1196_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_MVarId_subsingleton_spec__1___redArg___boxed(lean_object* v_mvarId_1199_, lean_object* v_x_1200_, lean_object* v___y_1201_, lean_object* v___y_1202_, lean_object* v___y_1203_, lean_object* v___y_1204_, lean_object* v___y_1205_){
_start:
{
lean_object* v_res_1206_; 
v_res_1206_ = lp_mathlib_Lean_MVarId_withContext___at___00Lean_MVarId_subsingleton_spec__1___redArg(v_mvarId_1199_, v_x_1200_, v___y_1201_, v___y_1202_, v___y_1203_, v___y_1204_);
lean_dec(v___y_1204_);
lean_dec_ref(v___y_1203_);
lean_dec(v___y_1202_);
lean_dec_ref(v___y_1201_);
return v_res_1206_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_MVarId_subsingleton_spec__1(lean_object* v_00_u03b1_1207_, lean_object* v_mvarId_1208_, lean_object* v_x_1209_, lean_object* v___y_1210_, lean_object* v___y_1211_, lean_object* v___y_1212_, lean_object* v___y_1213_){
_start:
{
lean_object* v___x_1215_; 
v___x_1215_ = lp_mathlib_Lean_MVarId_withContext___at___00Lean_MVarId_subsingleton_spec__1___redArg(v_mvarId_1208_, v_x_1209_, v___y_1210_, v___y_1211_, v___y_1212_, v___y_1213_);
return v___x_1215_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_MVarId_subsingleton_spec__1___boxed(lean_object* v_00_u03b1_1216_, lean_object* v_mvarId_1217_, lean_object* v_x_1218_, lean_object* v___y_1219_, lean_object* v___y_1220_, lean_object* v___y_1221_, lean_object* v___y_1222_, lean_object* v___y_1223_){
_start:
{
lean_object* v_res_1224_; 
v_res_1224_ = lp_mathlib_Lean_MVarId_withContext___at___00Lean_MVarId_subsingleton_spec__1(v_00_u03b1_1216_, v_mvarId_1217_, v_x_1218_, v___y_1219_, v___y_1220_, v___y_1221_, v___y_1222_);
lean_dec(v___y_1222_);
lean_dec_ref(v___y_1221_);
lean_dec(v___y_1220_);
lean_dec_ref(v___y_1219_);
return v_res_1224_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_commitIfNoEx___at___00Lean_MVarId_subsingleton_spec__2___redArg(lean_object* v_x_1225_, lean_object* v___y_1226_, lean_object* v___y_1227_, lean_object* v___y_1228_, lean_object* v___y_1229_){
_start:
{
lean_object* v___x_1231_; 
v___x_1231_ = l_Lean_Meta_saveState___redArg(v___y_1227_, v___y_1229_);
if (lean_obj_tag(v___x_1231_) == 0)
{
lean_object* v_a_1232_; lean_object* v___x_1233_; 
v_a_1232_ = lean_ctor_get(v___x_1231_, 0);
lean_inc(v_a_1232_);
lean_dec_ref_known(v___x_1231_, 1);
lean_inc(v___y_1229_);
lean_inc_ref(v___y_1228_);
lean_inc(v___y_1227_);
lean_inc_ref(v___y_1226_);
v___x_1233_ = lean_apply_5(v_x_1225_, v___y_1226_, v___y_1227_, v___y_1228_, v___y_1229_, lean_box(0));
if (lean_obj_tag(v___x_1233_) == 0)
{
lean_dec(v_a_1232_);
return v___x_1233_;
}
else
{
lean_object* v_a_1234_; uint8_t v___y_1236_; uint8_t v___x_1254_; 
v_a_1234_ = lean_ctor_get(v___x_1233_, 0);
lean_inc(v_a_1234_);
v___x_1254_ = l_Lean_Exception_isInterrupt(v_a_1234_);
if (v___x_1254_ == 0)
{
uint8_t v___x_1255_; 
lean_inc(v_a_1234_);
v___x_1255_ = l_Lean_Exception_isRuntime(v_a_1234_);
v___y_1236_ = v___x_1255_;
goto v___jp_1235_;
}
else
{
v___y_1236_ = v___x_1254_;
goto v___jp_1235_;
}
v___jp_1235_:
{
if (v___y_1236_ == 0)
{
lean_object* v___x_1237_; 
lean_dec_ref_known(v___x_1233_, 1);
v___x_1237_ = l_Lean_Meta_SavedState_restore___redArg(v_a_1232_, v___y_1227_, v___y_1229_);
lean_dec(v_a_1232_);
if (lean_obj_tag(v___x_1237_) == 0)
{
lean_object* v___x_1239_; uint8_t v_isShared_1240_; uint8_t v_isSharedCheck_1244_; 
v_isSharedCheck_1244_ = !lean_is_exclusive(v___x_1237_);
if (v_isSharedCheck_1244_ == 0)
{
lean_object* v_unused_1245_; 
v_unused_1245_ = lean_ctor_get(v___x_1237_, 0);
lean_dec(v_unused_1245_);
v___x_1239_ = v___x_1237_;
v_isShared_1240_ = v_isSharedCheck_1244_;
goto v_resetjp_1238_;
}
else
{
lean_dec(v___x_1237_);
v___x_1239_ = lean_box(0);
v_isShared_1240_ = v_isSharedCheck_1244_;
goto v_resetjp_1238_;
}
v_resetjp_1238_:
{
lean_object* v___x_1242_; 
if (v_isShared_1240_ == 0)
{
lean_ctor_set_tag(v___x_1239_, 1);
lean_ctor_set(v___x_1239_, 0, v_a_1234_);
v___x_1242_ = v___x_1239_;
goto v_reusejp_1241_;
}
else
{
lean_object* v_reuseFailAlloc_1243_; 
v_reuseFailAlloc_1243_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1243_, 0, v_a_1234_);
v___x_1242_ = v_reuseFailAlloc_1243_;
goto v_reusejp_1241_;
}
v_reusejp_1241_:
{
return v___x_1242_;
}
}
}
else
{
lean_object* v_a_1246_; lean_object* v___x_1248_; uint8_t v_isShared_1249_; uint8_t v_isSharedCheck_1253_; 
lean_dec(v_a_1234_);
v_a_1246_ = lean_ctor_get(v___x_1237_, 0);
v_isSharedCheck_1253_ = !lean_is_exclusive(v___x_1237_);
if (v_isSharedCheck_1253_ == 0)
{
v___x_1248_ = v___x_1237_;
v_isShared_1249_ = v_isSharedCheck_1253_;
goto v_resetjp_1247_;
}
else
{
lean_inc(v_a_1246_);
lean_dec(v___x_1237_);
v___x_1248_ = lean_box(0);
v_isShared_1249_ = v_isSharedCheck_1253_;
goto v_resetjp_1247_;
}
v_resetjp_1247_:
{
lean_object* v___x_1251_; 
if (v_isShared_1249_ == 0)
{
v___x_1251_ = v___x_1248_;
goto v_reusejp_1250_;
}
else
{
lean_object* v_reuseFailAlloc_1252_; 
v_reuseFailAlloc_1252_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1252_, 0, v_a_1246_);
v___x_1251_ = v_reuseFailAlloc_1252_;
goto v_reusejp_1250_;
}
v_reusejp_1250_:
{
return v___x_1251_;
}
}
}
}
else
{
lean_dec(v_a_1234_);
lean_dec(v_a_1232_);
return v___x_1233_;
}
}
}
}
else
{
lean_object* v_a_1256_; lean_object* v___x_1258_; uint8_t v_isShared_1259_; uint8_t v_isSharedCheck_1263_; 
lean_dec_ref(v_x_1225_);
v_a_1256_ = lean_ctor_get(v___x_1231_, 0);
v_isSharedCheck_1263_ = !lean_is_exclusive(v___x_1231_);
if (v_isSharedCheck_1263_ == 0)
{
v___x_1258_ = v___x_1231_;
v_isShared_1259_ = v_isSharedCheck_1263_;
goto v_resetjp_1257_;
}
else
{
lean_inc(v_a_1256_);
lean_dec(v___x_1231_);
v___x_1258_ = lean_box(0);
v_isShared_1259_ = v_isSharedCheck_1263_;
goto v_resetjp_1257_;
}
v_resetjp_1257_:
{
lean_object* v___x_1261_; 
if (v_isShared_1259_ == 0)
{
v___x_1261_ = v___x_1258_;
goto v_reusejp_1260_;
}
else
{
lean_object* v_reuseFailAlloc_1262_; 
v_reuseFailAlloc_1262_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1262_, 0, v_a_1256_);
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
LEAN_EXPORT lean_object* lp_mathlib_Lean_commitIfNoEx___at___00Lean_MVarId_subsingleton_spec__2___redArg___boxed(lean_object* v_x_1264_, lean_object* v___y_1265_, lean_object* v___y_1266_, lean_object* v___y_1267_, lean_object* v___y_1268_, lean_object* v___y_1269_){
_start:
{
lean_object* v_res_1270_; 
v_res_1270_ = lp_mathlib_Lean_commitIfNoEx___at___00Lean_MVarId_subsingleton_spec__2___redArg(v_x_1264_, v___y_1265_, v___y_1266_, v___y_1267_, v___y_1268_);
lean_dec(v___y_1268_);
lean_dec_ref(v___y_1267_);
lean_dec(v___y_1266_);
lean_dec_ref(v___y_1265_);
return v_res_1270_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_commitIfNoEx___at___00Lean_MVarId_subsingleton_spec__2(lean_object* v_00_u03b1_1271_, lean_object* v_x_1272_, lean_object* v___y_1273_, lean_object* v___y_1274_, lean_object* v___y_1275_, lean_object* v___y_1276_){
_start:
{
lean_object* v___x_1278_; 
v___x_1278_ = lp_mathlib_Lean_commitIfNoEx___at___00Lean_MVarId_subsingleton_spec__2___redArg(v_x_1272_, v___y_1273_, v___y_1274_, v___y_1275_, v___y_1276_);
return v___x_1278_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_commitIfNoEx___at___00Lean_MVarId_subsingleton_spec__2___boxed(lean_object* v_00_u03b1_1279_, lean_object* v_x_1280_, lean_object* v___y_1281_, lean_object* v___y_1282_, lean_object* v___y_1283_, lean_object* v___y_1284_, lean_object* v___y_1285_){
_start:
{
lean_object* v_res_1286_; 
v_res_1286_ = lp_mathlib_Lean_commitIfNoEx___at___00Lean_MVarId_subsingleton_spec__2(v_00_u03b1_1279_, v_x_1280_, v___y_1281_, v___y_1282_, v___y_1283_, v___y_1284_);
lean_dec(v___y_1284_);
lean_dec_ref(v___y_1283_);
lean_dec(v___y_1282_);
lean_dec_ref(v___y_1281_);
return v_res_1286_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_subsingleton_spec__0_spec__0_spec__3_spec__4_spec__5___redArg(lean_object* v_x_1287_, lean_object* v_x_1288_, lean_object* v_x_1289_, lean_object* v_x_1290_){
_start:
{
lean_object* v_ks_1291_; lean_object* v_vs_1292_; lean_object* v___x_1294_; uint8_t v_isShared_1295_; uint8_t v_isSharedCheck_1316_; 
v_ks_1291_ = lean_ctor_get(v_x_1287_, 0);
v_vs_1292_ = lean_ctor_get(v_x_1287_, 1);
v_isSharedCheck_1316_ = !lean_is_exclusive(v_x_1287_);
if (v_isSharedCheck_1316_ == 0)
{
v___x_1294_ = v_x_1287_;
v_isShared_1295_ = v_isSharedCheck_1316_;
goto v_resetjp_1293_;
}
else
{
lean_inc(v_vs_1292_);
lean_inc(v_ks_1291_);
lean_dec(v_x_1287_);
v___x_1294_ = lean_box(0);
v_isShared_1295_ = v_isSharedCheck_1316_;
goto v_resetjp_1293_;
}
v_resetjp_1293_:
{
lean_object* v___x_1296_; uint8_t v___x_1297_; 
v___x_1296_ = lean_array_get_size(v_ks_1291_);
v___x_1297_ = lean_nat_dec_lt(v_x_1288_, v___x_1296_);
if (v___x_1297_ == 0)
{
lean_object* v___x_1298_; lean_object* v___x_1299_; lean_object* v___x_1301_; 
lean_dec(v_x_1288_);
v___x_1298_ = lean_array_push(v_ks_1291_, v_x_1289_);
v___x_1299_ = lean_array_push(v_vs_1292_, v_x_1290_);
if (v_isShared_1295_ == 0)
{
lean_ctor_set(v___x_1294_, 1, v___x_1299_);
lean_ctor_set(v___x_1294_, 0, v___x_1298_);
v___x_1301_ = v___x_1294_;
goto v_reusejp_1300_;
}
else
{
lean_object* v_reuseFailAlloc_1302_; 
v_reuseFailAlloc_1302_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1302_, 0, v___x_1298_);
lean_ctor_set(v_reuseFailAlloc_1302_, 1, v___x_1299_);
v___x_1301_ = v_reuseFailAlloc_1302_;
goto v_reusejp_1300_;
}
v_reusejp_1300_:
{
return v___x_1301_;
}
}
else
{
lean_object* v_k_x27_1303_; uint8_t v___x_1304_; 
v_k_x27_1303_ = lean_array_fget_borrowed(v_ks_1291_, v_x_1288_);
v___x_1304_ = l_Lean_instBEqMVarId_beq(v_x_1289_, v_k_x27_1303_);
if (v___x_1304_ == 0)
{
lean_object* v___x_1306_; 
if (v_isShared_1295_ == 0)
{
v___x_1306_ = v___x_1294_;
goto v_reusejp_1305_;
}
else
{
lean_object* v_reuseFailAlloc_1310_; 
v_reuseFailAlloc_1310_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1310_, 0, v_ks_1291_);
lean_ctor_set(v_reuseFailAlloc_1310_, 1, v_vs_1292_);
v___x_1306_ = v_reuseFailAlloc_1310_;
goto v_reusejp_1305_;
}
v_reusejp_1305_:
{
lean_object* v___x_1307_; lean_object* v___x_1308_; 
v___x_1307_ = lean_unsigned_to_nat(1u);
v___x_1308_ = lean_nat_add(v_x_1288_, v___x_1307_);
lean_dec(v_x_1288_);
v_x_1287_ = v___x_1306_;
v_x_1288_ = v___x_1308_;
goto _start;
}
}
else
{
lean_object* v___x_1311_; lean_object* v___x_1312_; lean_object* v___x_1314_; 
v___x_1311_ = lean_array_fset(v_ks_1291_, v_x_1288_, v_x_1289_);
v___x_1312_ = lean_array_fset(v_vs_1292_, v_x_1288_, v_x_1290_);
lean_dec(v_x_1288_);
if (v_isShared_1295_ == 0)
{
lean_ctor_set(v___x_1294_, 1, v___x_1312_);
lean_ctor_set(v___x_1294_, 0, v___x_1311_);
v___x_1314_ = v___x_1294_;
goto v_reusejp_1313_;
}
else
{
lean_object* v_reuseFailAlloc_1315_; 
v_reuseFailAlloc_1315_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1315_, 0, v___x_1311_);
lean_ctor_set(v_reuseFailAlloc_1315_, 1, v___x_1312_);
v___x_1314_ = v_reuseFailAlloc_1315_;
goto v_reusejp_1313_;
}
v_reusejp_1313_:
{
return v___x_1314_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_subsingleton_spec__0_spec__0_spec__3_spec__4___redArg(lean_object* v_n_1317_, lean_object* v_k_1318_, lean_object* v_v_1319_){
_start:
{
lean_object* v___x_1320_; lean_object* v___x_1321_; 
v___x_1320_ = lean_unsigned_to_nat(0u);
v___x_1321_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_subsingleton_spec__0_spec__0_spec__3_spec__4_spec__5___redArg(v_n_1317_, v___x_1320_, v_k_1318_, v_v_1319_);
return v___x_1321_;
}
}
static lean_object* _init_lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_subsingleton_spec__0_spec__0_spec__3___redArg___closed__0(void){
_start:
{
lean_object* v___x_1322_; 
v___x_1322_ = l_Lean_PersistentHashMap_mkEmptyEntries(lean_box(0), lean_box(0));
return v___x_1322_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_subsingleton_spec__0_spec__0_spec__3___redArg(lean_object* v_x_1323_, size_t v_x_1324_, size_t v_x_1325_, lean_object* v_x_1326_, lean_object* v_x_1327_){
_start:
{
if (lean_obj_tag(v_x_1323_) == 0)
{
lean_object* v_es_1328_; size_t v___x_1329_; size_t v___x_1330_; lean_object* v_j_1331_; lean_object* v___x_1332_; uint8_t v___x_1333_; 
v_es_1328_ = lean_ctor_get(v_x_1323_, 0);
v___x_1329_ = ((size_t)31ULL);
v___x_1330_ = lean_usize_land(v_x_1324_, v___x_1329_);
v_j_1331_ = lean_usize_to_nat(v___x_1330_);
v___x_1332_ = lean_array_get_size(v_es_1328_);
v___x_1333_ = lean_nat_dec_lt(v_j_1331_, v___x_1332_);
if (v___x_1333_ == 0)
{
lean_dec(v_j_1331_);
lean_dec(v_x_1327_);
lean_dec(v_x_1326_);
return v_x_1323_;
}
else
{
lean_object* v___x_1335_; uint8_t v_isShared_1336_; uint8_t v_isSharedCheck_1372_; 
lean_inc_ref(v_es_1328_);
v_isSharedCheck_1372_ = !lean_is_exclusive(v_x_1323_);
if (v_isSharedCheck_1372_ == 0)
{
lean_object* v_unused_1373_; 
v_unused_1373_ = lean_ctor_get(v_x_1323_, 0);
lean_dec(v_unused_1373_);
v___x_1335_ = v_x_1323_;
v_isShared_1336_ = v_isSharedCheck_1372_;
goto v_resetjp_1334_;
}
else
{
lean_dec(v_x_1323_);
v___x_1335_ = lean_box(0);
v_isShared_1336_ = v_isSharedCheck_1372_;
goto v_resetjp_1334_;
}
v_resetjp_1334_:
{
lean_object* v_v_1337_; lean_object* v___x_1338_; lean_object* v_xs_x27_1339_; lean_object* v___y_1341_; 
v_v_1337_ = lean_array_fget(v_es_1328_, v_j_1331_);
v___x_1338_ = lean_box(0);
v_xs_x27_1339_ = lean_array_fset(v_es_1328_, v_j_1331_, v___x_1338_);
switch(lean_obj_tag(v_v_1337_))
{
case 0:
{
lean_object* v_key_1346_; lean_object* v_val_1347_; lean_object* v___x_1349_; uint8_t v_isShared_1350_; uint8_t v_isSharedCheck_1357_; 
v_key_1346_ = lean_ctor_get(v_v_1337_, 0);
v_val_1347_ = lean_ctor_get(v_v_1337_, 1);
v_isSharedCheck_1357_ = !lean_is_exclusive(v_v_1337_);
if (v_isSharedCheck_1357_ == 0)
{
v___x_1349_ = v_v_1337_;
v_isShared_1350_ = v_isSharedCheck_1357_;
goto v_resetjp_1348_;
}
else
{
lean_inc(v_val_1347_);
lean_inc(v_key_1346_);
lean_dec(v_v_1337_);
v___x_1349_ = lean_box(0);
v_isShared_1350_ = v_isSharedCheck_1357_;
goto v_resetjp_1348_;
}
v_resetjp_1348_:
{
uint8_t v___x_1351_; 
v___x_1351_ = l_Lean_instBEqMVarId_beq(v_x_1326_, v_key_1346_);
if (v___x_1351_ == 0)
{
lean_object* v___x_1352_; lean_object* v___x_1353_; 
lean_del_object(v___x_1349_);
v___x_1352_ = l_Lean_PersistentHashMap_mkCollisionNode___redArg(v_key_1346_, v_val_1347_, v_x_1326_, v_x_1327_);
v___x_1353_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1353_, 0, v___x_1352_);
v___y_1341_ = v___x_1353_;
goto v___jp_1340_;
}
else
{
lean_object* v___x_1355_; 
lean_dec(v_val_1347_);
lean_dec(v_key_1346_);
if (v_isShared_1350_ == 0)
{
lean_ctor_set(v___x_1349_, 1, v_x_1327_);
lean_ctor_set(v___x_1349_, 0, v_x_1326_);
v___x_1355_ = v___x_1349_;
goto v_reusejp_1354_;
}
else
{
lean_object* v_reuseFailAlloc_1356_; 
v_reuseFailAlloc_1356_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1356_, 0, v_x_1326_);
lean_ctor_set(v_reuseFailAlloc_1356_, 1, v_x_1327_);
v___x_1355_ = v_reuseFailAlloc_1356_;
goto v_reusejp_1354_;
}
v_reusejp_1354_:
{
v___y_1341_ = v___x_1355_;
goto v___jp_1340_;
}
}
}
}
case 1:
{
lean_object* v_node_1358_; lean_object* v___x_1360_; uint8_t v_isShared_1361_; uint8_t v_isSharedCheck_1370_; 
v_node_1358_ = lean_ctor_get(v_v_1337_, 0);
v_isSharedCheck_1370_ = !lean_is_exclusive(v_v_1337_);
if (v_isSharedCheck_1370_ == 0)
{
v___x_1360_ = v_v_1337_;
v_isShared_1361_ = v_isSharedCheck_1370_;
goto v_resetjp_1359_;
}
else
{
lean_inc(v_node_1358_);
lean_dec(v_v_1337_);
v___x_1360_ = lean_box(0);
v_isShared_1361_ = v_isSharedCheck_1370_;
goto v_resetjp_1359_;
}
v_resetjp_1359_:
{
size_t v___x_1362_; size_t v___x_1363_; size_t v___x_1364_; size_t v___x_1365_; lean_object* v___x_1366_; lean_object* v___x_1368_; 
v___x_1362_ = ((size_t)5ULL);
v___x_1363_ = lean_usize_shift_right(v_x_1324_, v___x_1362_);
v___x_1364_ = ((size_t)1ULL);
v___x_1365_ = lean_usize_add(v_x_1325_, v___x_1364_);
v___x_1366_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_subsingleton_spec__0_spec__0_spec__3___redArg(v_node_1358_, v___x_1363_, v___x_1365_, v_x_1326_, v_x_1327_);
if (v_isShared_1361_ == 0)
{
lean_ctor_set(v___x_1360_, 0, v___x_1366_);
v___x_1368_ = v___x_1360_;
goto v_reusejp_1367_;
}
else
{
lean_object* v_reuseFailAlloc_1369_; 
v_reuseFailAlloc_1369_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1369_, 0, v___x_1366_);
v___x_1368_ = v_reuseFailAlloc_1369_;
goto v_reusejp_1367_;
}
v_reusejp_1367_:
{
v___y_1341_ = v___x_1368_;
goto v___jp_1340_;
}
}
}
default: 
{
lean_object* v___x_1371_; 
v___x_1371_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1371_, 0, v_x_1326_);
lean_ctor_set(v___x_1371_, 1, v_x_1327_);
v___y_1341_ = v___x_1371_;
goto v___jp_1340_;
}
}
v___jp_1340_:
{
lean_object* v___x_1342_; lean_object* v___x_1344_; 
v___x_1342_ = lean_array_fset(v_xs_x27_1339_, v_j_1331_, v___y_1341_);
lean_dec(v_j_1331_);
if (v_isShared_1336_ == 0)
{
lean_ctor_set(v___x_1335_, 0, v___x_1342_);
v___x_1344_ = v___x_1335_;
goto v_reusejp_1343_;
}
else
{
lean_object* v_reuseFailAlloc_1345_; 
v_reuseFailAlloc_1345_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1345_, 0, v___x_1342_);
v___x_1344_ = v_reuseFailAlloc_1345_;
goto v_reusejp_1343_;
}
v_reusejp_1343_:
{
return v___x_1344_;
}
}
}
}
}
else
{
lean_object* v_ks_1374_; lean_object* v_vs_1375_; lean_object* v___x_1377_; uint8_t v_isShared_1378_; uint8_t v_isSharedCheck_1395_; 
v_ks_1374_ = lean_ctor_get(v_x_1323_, 0);
v_vs_1375_ = lean_ctor_get(v_x_1323_, 1);
v_isSharedCheck_1395_ = !lean_is_exclusive(v_x_1323_);
if (v_isSharedCheck_1395_ == 0)
{
v___x_1377_ = v_x_1323_;
v_isShared_1378_ = v_isSharedCheck_1395_;
goto v_resetjp_1376_;
}
else
{
lean_inc(v_vs_1375_);
lean_inc(v_ks_1374_);
lean_dec(v_x_1323_);
v___x_1377_ = lean_box(0);
v_isShared_1378_ = v_isSharedCheck_1395_;
goto v_resetjp_1376_;
}
v_resetjp_1376_:
{
lean_object* v___x_1380_; 
if (v_isShared_1378_ == 0)
{
v___x_1380_ = v___x_1377_;
goto v_reusejp_1379_;
}
else
{
lean_object* v_reuseFailAlloc_1394_; 
v_reuseFailAlloc_1394_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1394_, 0, v_ks_1374_);
lean_ctor_set(v_reuseFailAlloc_1394_, 1, v_vs_1375_);
v___x_1380_ = v_reuseFailAlloc_1394_;
goto v_reusejp_1379_;
}
v_reusejp_1379_:
{
lean_object* v_newNode_1381_; uint8_t v___y_1383_; size_t v___x_1389_; uint8_t v___x_1390_; 
v_newNode_1381_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_subsingleton_spec__0_spec__0_spec__3_spec__4___redArg(v___x_1380_, v_x_1326_, v_x_1327_);
v___x_1389_ = ((size_t)7ULL);
v___x_1390_ = lean_usize_dec_le(v___x_1389_, v_x_1325_);
if (v___x_1390_ == 0)
{
lean_object* v___x_1391_; lean_object* v___x_1392_; uint8_t v___x_1393_; 
v___x_1391_ = l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(v_newNode_1381_);
v___x_1392_ = lean_unsigned_to_nat(4u);
v___x_1393_ = lean_nat_dec_lt(v___x_1391_, v___x_1392_);
lean_dec(v___x_1391_);
v___y_1383_ = v___x_1393_;
goto v___jp_1382_;
}
else
{
v___y_1383_ = v___x_1390_;
goto v___jp_1382_;
}
v___jp_1382_:
{
if (v___y_1383_ == 0)
{
lean_object* v_ks_1384_; lean_object* v_vs_1385_; lean_object* v___x_1386_; lean_object* v___x_1387_; lean_object* v___x_1388_; 
v_ks_1384_ = lean_ctor_get(v_newNode_1381_, 0);
lean_inc_ref(v_ks_1384_);
v_vs_1385_ = lean_ctor_get(v_newNode_1381_, 1);
lean_inc_ref(v_vs_1385_);
lean_dec_ref(v_newNode_1381_);
v___x_1386_ = lean_unsigned_to_nat(0u);
v___x_1387_ = lean_obj_once(&lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_subsingleton_spec__0_spec__0_spec__3___redArg___closed__0, &lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_subsingleton_spec__0_spec__0_spec__3___redArg___closed__0_once, _init_lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_subsingleton_spec__0_spec__0_spec__3___redArg___closed__0);
v___x_1388_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_subsingleton_spec__0_spec__0_spec__3_spec__5___redArg(v_x_1325_, v_ks_1384_, v_vs_1385_, v___x_1386_, v___x_1387_);
lean_dec_ref(v_vs_1385_);
lean_dec_ref(v_ks_1384_);
return v___x_1388_;
}
else
{
return v_newNode_1381_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_subsingleton_spec__0_spec__0_spec__3_spec__5___redArg(size_t v_depth_1396_, lean_object* v_keys_1397_, lean_object* v_vals_1398_, lean_object* v_i_1399_, lean_object* v_entries_1400_){
_start:
{
lean_object* v___x_1401_; uint8_t v___x_1402_; 
v___x_1401_ = lean_array_get_size(v_keys_1397_);
v___x_1402_ = lean_nat_dec_lt(v_i_1399_, v___x_1401_);
if (v___x_1402_ == 0)
{
lean_dec(v_i_1399_);
return v_entries_1400_;
}
else
{
lean_object* v_k_1403_; lean_object* v_v_1404_; uint64_t v___x_1405_; size_t v_h_1406_; size_t v___x_1407_; lean_object* v___x_1408_; size_t v___x_1409_; size_t v___x_1410_; size_t v___x_1411_; size_t v_h_1412_; lean_object* v___x_1413_; lean_object* v___x_1414_; 
v_k_1403_ = lean_array_fget_borrowed(v_keys_1397_, v_i_1399_);
v_v_1404_ = lean_array_fget_borrowed(v_vals_1398_, v_i_1399_);
v___x_1405_ = l_Lean_instHashableMVarId_hash(v_k_1403_);
v_h_1406_ = lean_uint64_to_usize(v___x_1405_);
v___x_1407_ = ((size_t)5ULL);
v___x_1408_ = lean_unsigned_to_nat(1u);
v___x_1409_ = ((size_t)1ULL);
v___x_1410_ = lean_usize_sub(v_depth_1396_, v___x_1409_);
v___x_1411_ = lean_usize_mul(v___x_1407_, v___x_1410_);
v_h_1412_ = lean_usize_shift_right(v_h_1406_, v___x_1411_);
v___x_1413_ = lean_nat_add(v_i_1399_, v___x_1408_);
lean_dec(v_i_1399_);
lean_inc(v_v_1404_);
lean_inc(v_k_1403_);
v___x_1414_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_subsingleton_spec__0_spec__0_spec__3___redArg(v_entries_1400_, v_h_1412_, v_depth_1396_, v_k_1403_, v_v_1404_);
v_i_1399_ = v___x_1413_;
v_entries_1400_ = v___x_1414_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_subsingleton_spec__0_spec__0_spec__3_spec__5___redArg___boxed(lean_object* v_depth_1416_, lean_object* v_keys_1417_, lean_object* v_vals_1418_, lean_object* v_i_1419_, lean_object* v_entries_1420_){
_start:
{
size_t v_depth_boxed_1421_; lean_object* v_res_1422_; 
v_depth_boxed_1421_ = lean_unbox_usize(v_depth_1416_);
lean_dec(v_depth_1416_);
v_res_1422_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_subsingleton_spec__0_spec__0_spec__3_spec__5___redArg(v_depth_boxed_1421_, v_keys_1417_, v_vals_1418_, v_i_1419_, v_entries_1420_);
lean_dec_ref(v_vals_1418_);
lean_dec_ref(v_keys_1417_);
return v_res_1422_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_subsingleton_spec__0_spec__0_spec__3___redArg___boxed(lean_object* v_x_1423_, lean_object* v_x_1424_, lean_object* v_x_1425_, lean_object* v_x_1426_, lean_object* v_x_1427_){
_start:
{
size_t v_x_10033__boxed_1428_; size_t v_x_10034__boxed_1429_; lean_object* v_res_1430_; 
v_x_10033__boxed_1428_ = lean_unbox_usize(v_x_1424_);
lean_dec(v_x_1424_);
v_x_10034__boxed_1429_ = lean_unbox_usize(v_x_1425_);
lean_dec(v_x_1425_);
v_res_1430_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_subsingleton_spec__0_spec__0_spec__3___redArg(v_x_1423_, v_x_10033__boxed_1428_, v_x_10034__boxed_1429_, v_x_1426_, v_x_1427_);
return v_res_1430_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_subsingleton_spec__0_spec__0___redArg(lean_object* v_x_1431_, lean_object* v_x_1432_, lean_object* v_x_1433_){
_start:
{
uint64_t v___x_1434_; size_t v___x_1435_; size_t v___x_1436_; lean_object* v___x_1437_; 
v___x_1434_ = l_Lean_instHashableMVarId_hash(v_x_1432_);
v___x_1435_ = lean_uint64_to_usize(v___x_1434_);
v___x_1436_ = ((size_t)1ULL);
v___x_1437_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_subsingleton_spec__0_spec__0_spec__3___redArg(v_x_1431_, v___x_1435_, v___x_1436_, v_x_1432_, v_x_1433_);
return v___x_1437_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Lean_MVarId_subsingleton_spec__0___redArg(lean_object* v_mvarId_1438_, lean_object* v_val_1439_, lean_object* v___y_1440_){
_start:
{
lean_object* v___x_1442_; lean_object* v_mctx_1443_; lean_object* v_cache_1444_; lean_object* v_zetaDeltaFVarIds_1445_; lean_object* v_postponed_1446_; lean_object* v_diag_1447_; lean_object* v___x_1449_; uint8_t v_isShared_1450_; uint8_t v_isSharedCheck_1475_; 
v___x_1442_ = lean_st_ref_take(v___y_1440_);
v_mctx_1443_ = lean_ctor_get(v___x_1442_, 0);
v_cache_1444_ = lean_ctor_get(v___x_1442_, 1);
v_zetaDeltaFVarIds_1445_ = lean_ctor_get(v___x_1442_, 2);
v_postponed_1446_ = lean_ctor_get(v___x_1442_, 3);
v_diag_1447_ = lean_ctor_get(v___x_1442_, 4);
v_isSharedCheck_1475_ = !lean_is_exclusive(v___x_1442_);
if (v_isSharedCheck_1475_ == 0)
{
v___x_1449_ = v___x_1442_;
v_isShared_1450_ = v_isSharedCheck_1475_;
goto v_resetjp_1448_;
}
else
{
lean_inc(v_diag_1447_);
lean_inc(v_postponed_1446_);
lean_inc(v_zetaDeltaFVarIds_1445_);
lean_inc(v_cache_1444_);
lean_inc(v_mctx_1443_);
lean_dec(v___x_1442_);
v___x_1449_ = lean_box(0);
v_isShared_1450_ = v_isSharedCheck_1475_;
goto v_resetjp_1448_;
}
v_resetjp_1448_:
{
lean_object* v_depth_1451_; lean_object* v_levelAssignDepth_1452_; lean_object* v_lmvarCounter_1453_; lean_object* v_mvarCounter_1454_; lean_object* v_lDecls_1455_; lean_object* v_decls_1456_; lean_object* v_userNames_1457_; lean_object* v_lAssignment_1458_; lean_object* v_eAssignment_1459_; lean_object* v_dAssignment_1460_; lean_object* v___x_1462_; uint8_t v_isShared_1463_; uint8_t v_isSharedCheck_1474_; 
v_depth_1451_ = lean_ctor_get(v_mctx_1443_, 0);
v_levelAssignDepth_1452_ = lean_ctor_get(v_mctx_1443_, 1);
v_lmvarCounter_1453_ = lean_ctor_get(v_mctx_1443_, 2);
v_mvarCounter_1454_ = lean_ctor_get(v_mctx_1443_, 3);
v_lDecls_1455_ = lean_ctor_get(v_mctx_1443_, 4);
v_decls_1456_ = lean_ctor_get(v_mctx_1443_, 5);
v_userNames_1457_ = lean_ctor_get(v_mctx_1443_, 6);
v_lAssignment_1458_ = lean_ctor_get(v_mctx_1443_, 7);
v_eAssignment_1459_ = lean_ctor_get(v_mctx_1443_, 8);
v_dAssignment_1460_ = lean_ctor_get(v_mctx_1443_, 9);
v_isSharedCheck_1474_ = !lean_is_exclusive(v_mctx_1443_);
if (v_isSharedCheck_1474_ == 0)
{
v___x_1462_ = v_mctx_1443_;
v_isShared_1463_ = v_isSharedCheck_1474_;
goto v_resetjp_1461_;
}
else
{
lean_inc(v_dAssignment_1460_);
lean_inc(v_eAssignment_1459_);
lean_inc(v_lAssignment_1458_);
lean_inc(v_userNames_1457_);
lean_inc(v_decls_1456_);
lean_inc(v_lDecls_1455_);
lean_inc(v_mvarCounter_1454_);
lean_inc(v_lmvarCounter_1453_);
lean_inc(v_levelAssignDepth_1452_);
lean_inc(v_depth_1451_);
lean_dec(v_mctx_1443_);
v___x_1462_ = lean_box(0);
v_isShared_1463_ = v_isSharedCheck_1474_;
goto v_resetjp_1461_;
}
v_resetjp_1461_:
{
lean_object* v___x_1464_; lean_object* v___x_1466_; 
v___x_1464_ = lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_subsingleton_spec__0_spec__0___redArg(v_eAssignment_1459_, v_mvarId_1438_, v_val_1439_);
if (v_isShared_1463_ == 0)
{
lean_ctor_set(v___x_1462_, 8, v___x_1464_);
v___x_1466_ = v___x_1462_;
goto v_reusejp_1465_;
}
else
{
lean_object* v_reuseFailAlloc_1473_; 
v_reuseFailAlloc_1473_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_1473_, 0, v_depth_1451_);
lean_ctor_set(v_reuseFailAlloc_1473_, 1, v_levelAssignDepth_1452_);
lean_ctor_set(v_reuseFailAlloc_1473_, 2, v_lmvarCounter_1453_);
lean_ctor_set(v_reuseFailAlloc_1473_, 3, v_mvarCounter_1454_);
lean_ctor_set(v_reuseFailAlloc_1473_, 4, v_lDecls_1455_);
lean_ctor_set(v_reuseFailAlloc_1473_, 5, v_decls_1456_);
lean_ctor_set(v_reuseFailAlloc_1473_, 6, v_userNames_1457_);
lean_ctor_set(v_reuseFailAlloc_1473_, 7, v_lAssignment_1458_);
lean_ctor_set(v_reuseFailAlloc_1473_, 8, v___x_1464_);
lean_ctor_set(v_reuseFailAlloc_1473_, 9, v_dAssignment_1460_);
v___x_1466_ = v_reuseFailAlloc_1473_;
goto v_reusejp_1465_;
}
v_reusejp_1465_:
{
lean_object* v___x_1468_; 
if (v_isShared_1450_ == 0)
{
lean_ctor_set(v___x_1449_, 0, v___x_1466_);
v___x_1468_ = v___x_1449_;
goto v_reusejp_1467_;
}
else
{
lean_object* v_reuseFailAlloc_1472_; 
v_reuseFailAlloc_1472_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1472_, 0, v___x_1466_);
lean_ctor_set(v_reuseFailAlloc_1472_, 1, v_cache_1444_);
lean_ctor_set(v_reuseFailAlloc_1472_, 2, v_zetaDeltaFVarIds_1445_);
lean_ctor_set(v_reuseFailAlloc_1472_, 3, v_postponed_1446_);
lean_ctor_set(v_reuseFailAlloc_1472_, 4, v_diag_1447_);
v___x_1468_ = v_reuseFailAlloc_1472_;
goto v_reusejp_1467_;
}
v_reusejp_1467_:
{
lean_object* v___x_1469_; lean_object* v___x_1470_; lean_object* v___x_1471_; 
v___x_1469_ = lean_st_ref_set(v___y_1440_, v___x_1468_);
v___x_1470_ = lean_box(0);
v___x_1471_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1471_, 0, v___x_1470_);
return v___x_1471_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Lean_MVarId_subsingleton_spec__0___redArg___boxed(lean_object* v_mvarId_1476_, lean_object* v_val_1477_, lean_object* v___y_1478_, lean_object* v___y_1479_){
_start:
{
lean_object* v_res_1480_; 
v_res_1480_ = lp_mathlib_Lean_MVarId_assign___at___00Lean_MVarId_subsingleton_spec__0___redArg(v_mvarId_1476_, v_val_1477_, v___y_1478_);
lean_dec(v___y_1478_);
return v_res_1480_;
}
}
static lean_object* _init_lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__1(void){
_start:
{
lean_object* v___x_1482_; lean_object* v___x_1483_; 
v___x_1482_ = ((lean_object*)(lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__0));
v___x_1483_ = l_Lean_stringToMessageData(v___x_1482_);
return v___x_1483_;
}
}
static lean_object* _init_lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__7(void){
_start:
{
lean_object* v___x_1491_; lean_object* v___x_1492_; 
v___x_1491_ = ((lean_object*)(lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__6));
v___x_1492_ = l_Lean_stringToMessageData(v___x_1491_);
return v___x_1492_;
}
}
static lean_object* _init_lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__10(void){
_start:
{
lean_object* v___x_1496_; lean_object* v___x_1497_; lean_object* v___x_1498_; 
v___x_1496_ = lean_box(0);
v___x_1497_ = ((lean_object*)(lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__9));
v___x_1498_ = l_Lean_Expr_const___override(v___x_1497_, v___x_1496_);
return v___x_1498_;
}
}
static lean_object* _init_lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__12(void){
_start:
{
lean_object* v___x_1500_; lean_object* v___x_1501_; 
v___x_1500_ = ((lean_object*)(lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__11));
v___x_1501_ = l_Lean_stringToMessageData(v___x_1500_);
return v___x_1501_;
}
}
static lean_object* _init_lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__20(void){
_start:
{
lean_object* v___x_1512_; lean_object* v___x_1513_; 
v___x_1512_ = ((lean_object*)(lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__19));
v___x_1513_ = l_Lean_stringToMessageData(v___x_1512_);
return v___x_1513_;
}
}
static lean_object* _init_lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__25(void){
_start:
{
lean_object* v___x_1521_; lean_object* v___x_1522_; lean_object* v___x_1523_; 
v___x_1521_ = lean_box(0);
v___x_1522_ = ((lean_object*)(lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__24));
v___x_1523_ = l_Lean_Expr_const___override(v___x_1522_, v___x_1521_);
return v___x_1523_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_subsingleton___lam__0(lean_object* v_a_1524_, lean_object* v_insts_1525_, lean_object* v___y_1526_, lean_object* v___y_1527_, lean_object* v___y_1528_, lean_object* v___y_1529_){
_start:
{
lean_object* v___y_1532_; lean_object* v___y_1533_; lean_object* v___y_1534_; lean_object* v___y_1535_; lean_object* v___x_1538_; 
lean_inc(v_a_1524_);
v___x_1538_ = l_Lean_MVarId_getType(v_a_1524_, v___y_1526_, v___y_1527_, v___y_1528_, v___y_1529_);
if (lean_obj_tag(v___x_1538_) == 0)
{
lean_object* v_a_1539_; lean_object* v___x_1540_; 
v_a_1539_ = lean_ctor_get(v___x_1538_, 0);
lean_inc(v_a_1539_);
lean_dec_ref_known(v___x_1538_, 1);
v___x_1540_ = l_Lean_Meta_whnfR(v_a_1539_, v___y_1526_, v___y_1527_, v___y_1528_, v___y_1529_);
if (lean_obj_tag(v___x_1540_) == 0)
{
lean_object* v_a_1541_; lean_object* v___x_1543_; uint8_t v_isShared_1544_; uint8_t v_isSharedCheck_1737_; 
v_a_1541_ = lean_ctor_get(v___x_1540_, 0);
v_isSharedCheck_1737_ = !lean_is_exclusive(v___x_1540_);
if (v_isSharedCheck_1737_ == 0)
{
v___x_1543_ = v___x_1540_;
v_isShared_1544_ = v_isSharedCheck_1737_;
goto v_resetjp_1542_;
}
else
{
lean_inc(v_a_1541_);
lean_dec(v___x_1540_);
v___x_1543_ = lean_box(0);
v_isShared_1544_ = v_isSharedCheck_1737_;
goto v_resetjp_1542_;
}
v_resetjp_1542_:
{
lean_object* v___x_1545_; lean_object* v___x_1546_; uint8_t v___x_1547_; 
v___x_1545_ = ((lean_object*)(lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__3));
v___x_1546_ = lean_unsigned_to_nat(3u);
v___x_1547_ = l_Lean_Expr_isAppOfArity(v_a_1541_, v___x_1545_, v___x_1546_);
if (v___x_1547_ == 0)
{
lean_object* v___x_1548_; lean_object* v___x_1549_; uint8_t v___x_1550_; 
lean_del_object(v___x_1543_);
lean_dec_ref(v_insts_1525_);
v___x_1548_ = ((lean_object*)(lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__5));
v___x_1549_ = lean_unsigned_to_nat(4u);
v___x_1550_ = l_Lean_Expr_isAppOfArity(v_a_1541_, v___x_1548_, v___x_1549_);
if (v___x_1550_ == 0)
{
lean_dec(v_a_1541_);
lean_dec(v_a_1524_);
v___y_1532_ = v___y_1526_;
v___y_1533_ = v___y_1527_;
v___y_1534_ = v___y_1528_;
v___y_1535_ = v___y_1529_;
goto v___jp_1531_;
}
else
{
lean_object* v___x_1551_; lean_object* v___x_1552_; lean_object* v___x_1553_; lean_object* v___x_1554_; lean_object* v___x_1555_; lean_object* v___x_1556_; lean_object* v___x_1557_; lean_object* v___y_1559_; lean_object* v___x_1584_; 
v___x_1551_ = l_Lean_Expr_appFn_x21(v_a_1541_);
v___x_1552_ = l_Lean_Expr_appFn_x21(v___x_1551_);
v___x_1553_ = l_Lean_Expr_appFn_x21(v___x_1552_);
v___x_1554_ = l_Lean_Expr_appArg_x21(v___x_1553_);
lean_dec_ref(v___x_1553_);
v___x_1555_ = l_Lean_Expr_appArg_x21(v___x_1552_);
lean_dec_ref(v___x_1552_);
v___x_1556_ = l_Lean_Expr_appArg_x21(v___x_1551_);
lean_dec_ref(v___x_1551_);
v___x_1557_ = l_Lean_Expr_appArg_x21(v_a_1541_);
lean_dec(v_a_1541_);
lean_inc_ref(v___x_1554_);
v___x_1584_ = l_Lean_Meta_isProp(v___x_1554_, v___y_1526_, v___y_1527_, v___y_1528_, v___y_1529_);
if (lean_obj_tag(v___x_1584_) == 0)
{
lean_object* v_a_1585_; uint8_t v___x_1586_; 
v_a_1585_ = lean_ctor_get(v___x_1584_, 0);
lean_inc(v_a_1585_);
v___x_1586_ = lean_unbox(v_a_1585_);
lean_dec(v_a_1585_);
if (v___x_1586_ == 0)
{
v___y_1559_ = v___x_1584_;
goto v___jp_1558_;
}
else
{
lean_object* v___x_1587_; 
lean_dec_ref_known(v___x_1584_, 1);
lean_inc_ref(v___x_1556_);
v___x_1587_ = l_Lean_Meta_isProp(v___x_1556_, v___y_1526_, v___y_1527_, v___y_1528_, v___y_1529_);
v___y_1559_ = v___x_1587_;
goto v___jp_1558_;
}
}
else
{
v___y_1559_ = v___x_1584_;
goto v___jp_1558_;
}
v___jp_1558_:
{
if (lean_obj_tag(v___y_1559_) == 0)
{
lean_object* v_a_1560_; uint8_t v___x_1561_; 
v_a_1560_ = lean_ctor_get(v___y_1559_, 0);
lean_inc(v_a_1560_);
lean_dec_ref_known(v___y_1559_, 1);
v___x_1561_ = lean_unbox(v_a_1560_);
lean_dec(v_a_1560_);
if (v___x_1561_ == 0)
{
lean_object* v___x_1562_; lean_object* v___x_1563_; 
lean_dec_ref(v___x_1557_);
lean_dec_ref(v___x_1556_);
lean_dec_ref(v___x_1555_);
lean_dec_ref(v___x_1554_);
lean_dec(v_a_1524_);
v___x_1562_ = lean_obj_once(&lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__7, &lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__7_once, _init_lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__7);
v___x_1563_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Meta_synthSubsingletonInst_spec__4_spec__4___redArg(v___x_1562_, v___y_1526_, v___y_1527_, v___y_1528_, v___y_1529_);
return v___x_1563_;
}
else
{
lean_object* v___x_1564_; lean_object* v___x_1565_; lean_object* v___x_1566_; lean_object* v___x_1568_; uint8_t v_isShared_1569_; uint8_t v_isSharedCheck_1574_; 
v___x_1564_ = lean_obj_once(&lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__10, &lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__10_once, _init_lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__10);
v___x_1565_ = l_Lean_mkApp4(v___x_1564_, v___x_1554_, v___x_1556_, v___x_1555_, v___x_1557_);
v___x_1566_ = lp_mathlib_Lean_MVarId_assign___at___00Lean_MVarId_subsingleton_spec__0___redArg(v_a_1524_, v___x_1565_, v___y_1527_);
v_isSharedCheck_1574_ = !lean_is_exclusive(v___x_1566_);
if (v_isSharedCheck_1574_ == 0)
{
lean_object* v_unused_1575_; 
v_unused_1575_ = lean_ctor_get(v___x_1566_, 0);
lean_dec(v_unused_1575_);
v___x_1568_ = v___x_1566_;
v_isShared_1569_ = v_isSharedCheck_1574_;
goto v_resetjp_1567_;
}
else
{
lean_dec(v___x_1566_);
v___x_1568_ = lean_box(0);
v_isShared_1569_ = v_isSharedCheck_1574_;
goto v_resetjp_1567_;
}
v_resetjp_1567_:
{
lean_object* v___x_1570_; lean_object* v___x_1572_; 
v___x_1570_ = lean_box(0);
if (v_isShared_1569_ == 0)
{
lean_ctor_set(v___x_1568_, 0, v___x_1570_);
v___x_1572_ = v___x_1568_;
goto v_reusejp_1571_;
}
else
{
lean_object* v_reuseFailAlloc_1573_; 
v_reuseFailAlloc_1573_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1573_, 0, v___x_1570_);
v___x_1572_ = v_reuseFailAlloc_1573_;
goto v_reusejp_1571_;
}
v_reusejp_1571_:
{
return v___x_1572_;
}
}
}
}
else
{
lean_object* v_a_1576_; lean_object* v___x_1578_; uint8_t v_isShared_1579_; uint8_t v_isSharedCheck_1583_; 
lean_dec_ref(v___x_1557_);
lean_dec_ref(v___x_1556_);
lean_dec_ref(v___x_1555_);
lean_dec_ref(v___x_1554_);
lean_dec(v_a_1524_);
v_a_1576_ = lean_ctor_get(v___y_1559_, 0);
v_isSharedCheck_1583_ = !lean_is_exclusive(v___y_1559_);
if (v_isSharedCheck_1583_ == 0)
{
v___x_1578_ = v___y_1559_;
v_isShared_1579_ = v_isSharedCheck_1583_;
goto v_resetjp_1577_;
}
else
{
lean_inc(v_a_1576_);
lean_dec(v___y_1559_);
v___x_1578_ = lean_box(0);
v_isShared_1579_ = v_isSharedCheck_1583_;
goto v_resetjp_1577_;
}
v_resetjp_1577_:
{
lean_object* v___x_1581_; 
if (v_isShared_1579_ == 0)
{
v___x_1581_ = v___x_1578_;
goto v_reusejp_1580_;
}
else
{
lean_object* v_reuseFailAlloc_1582_; 
v_reuseFailAlloc_1582_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1582_, 0, v_a_1576_);
v___x_1581_ = v_reuseFailAlloc_1582_;
goto v_reusejp_1580_;
}
v_reusejp_1580_:
{
return v___x_1581_;
}
}
}
}
}
}
else
{
lean_object* v___x_1588_; lean_object* v___x_1589_; lean_object* v___x_1590_; lean_object* v___y_1592_; lean_object* v___y_1593_; lean_object* v___y_1594_; lean_object* v___y_1595_; lean_object* v___y_1612_; uint8_t v___y_1613_; lean_object* v_a_1618_; lean_object* v___x_1621_; 
v___x_1588_ = l_Lean_Expr_appFn_x21(v_a_1541_);
v___x_1589_ = l_Lean_Expr_appFn_x21(v___x_1588_);
v___x_1590_ = l_Lean_Expr_appArg_x21(v___x_1589_);
lean_dec_ref(v___x_1589_);
lean_inc_ref(v___x_1590_);
v___x_1621_ = l_Lean_Meta_isProp(v___x_1590_, v___y_1526_, v___y_1527_, v___y_1528_, v___y_1529_);
if (lean_obj_tag(v___x_1621_) == 0)
{
lean_object* v_a_1622_; lean_object* v___x_1623_; lean_object* v___x_1624_; uint8_t v___x_1625_; 
v_a_1622_ = lean_ctor_get(v___x_1621_, 0);
lean_inc(v_a_1622_);
lean_dec_ref_known(v___x_1621_, 1);
v___x_1623_ = l_Lean_Expr_appArg_x21(v___x_1588_);
lean_dec_ref(v___x_1588_);
v___x_1624_ = l_Lean_Expr_appArg_x21(v_a_1541_);
lean_dec(v_a_1541_);
v___x_1625_ = lean_unbox(v_a_1622_);
if (v___x_1625_ == 0)
{
lean_object* v___x_1626_; 
lean_inc_ref(v___x_1590_);
v___x_1626_ = l_Lean_Meta_getLevel(v___x_1590_, v___y_1526_, v___y_1527_, v___y_1528_, v___y_1529_);
if (lean_obj_tag(v___x_1626_) == 0)
{
lean_object* v_a_1627_; lean_object* v___x_1629_; uint8_t v_isShared_1630_; uint8_t v_isSharedCheck_1708_; 
v_a_1627_ = lean_ctor_get(v___x_1626_, 0);
v_isSharedCheck_1708_ = !lean_is_exclusive(v___x_1626_);
if (v_isSharedCheck_1708_ == 0)
{
v___x_1629_ = v___x_1626_;
v_isShared_1630_ = v_isSharedCheck_1708_;
goto v_resetjp_1628_;
}
else
{
lean_inc(v_a_1627_);
lean_dec(v___x_1626_);
v___x_1629_ = lean_box(0);
v_isShared_1630_ = v_isSharedCheck_1708_;
goto v_resetjp_1628_;
}
v_resetjp_1628_:
{
lean_object* v___y_1632_; uint8_t v___y_1633_; lean_object* v_a_1687_; lean_object* v___x_1690_; 
lean_inc_ref(v___x_1590_);
v___x_1690_ = lp_mathlib_Lean_Meta_synthSubsingletonInst(v___x_1590_, v_insts_1525_, v___y_1526_, v___y_1527_, v___y_1528_, v___y_1529_);
if (lean_obj_tag(v___x_1690_) == 0)
{
lean_object* v_a_1691_; lean_object* v___x_1692_; lean_object* v___x_1693_; lean_object* v___x_1694_; lean_object* v___x_1695_; lean_object* v___x_1696_; lean_object* v___x_1697_; lean_object* v___x_1699_; uint8_t v_isShared_1700_; uint8_t v_isSharedCheck_1705_; 
lean_del_object(v___x_1629_);
lean_dec(v_a_1622_);
lean_del_object(v___x_1543_);
v_a_1691_ = lean_ctor_get(v___x_1690_, 0);
lean_inc(v_a_1691_);
lean_dec_ref_known(v___x_1690_, 1);
v___x_1692_ = ((lean_object*)(lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__22));
v___x_1693_ = lean_box(0);
v___x_1694_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1694_, 0, v_a_1627_);
lean_ctor_set(v___x_1694_, 1, v___x_1693_);
v___x_1695_ = l_Lean_Expr_const___override(v___x_1692_, v___x_1694_);
v___x_1696_ = l_Lean_mkApp4(v___x_1695_, v___x_1590_, v_a_1691_, v___x_1623_, v___x_1624_);
v___x_1697_ = lp_mathlib_Lean_MVarId_assign___at___00Lean_MVarId_subsingleton_spec__0___redArg(v_a_1524_, v___x_1696_, v___y_1527_);
v_isSharedCheck_1705_ = !lean_is_exclusive(v___x_1697_);
if (v_isSharedCheck_1705_ == 0)
{
lean_object* v_unused_1706_; 
v_unused_1706_ = lean_ctor_get(v___x_1697_, 0);
lean_dec(v_unused_1706_);
v___x_1699_ = v___x_1697_;
v_isShared_1700_ = v_isSharedCheck_1705_;
goto v_resetjp_1698_;
}
else
{
lean_dec(v___x_1697_);
v___x_1699_ = lean_box(0);
v_isShared_1700_ = v_isSharedCheck_1705_;
goto v_resetjp_1698_;
}
v_resetjp_1698_:
{
lean_object* v___x_1701_; lean_object* v___x_1703_; 
v___x_1701_ = lean_box(0);
if (v_isShared_1700_ == 0)
{
lean_ctor_set(v___x_1699_, 0, v___x_1701_);
v___x_1703_ = v___x_1699_;
goto v_reusejp_1702_;
}
else
{
lean_object* v_reuseFailAlloc_1704_; 
v_reuseFailAlloc_1704_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1704_, 0, v___x_1701_);
v___x_1703_ = v_reuseFailAlloc_1704_;
goto v_reusejp_1702_;
}
v_reusejp_1702_:
{
return v___x_1703_;
}
}
}
else
{
lean_object* v_a_1707_; 
v_a_1707_ = lean_ctor_get(v___x_1690_, 0);
lean_inc(v_a_1707_);
lean_dec_ref_known(v___x_1690_, 1);
v_a_1687_ = v_a_1707_;
goto v___jp_1686_;
}
v___jp_1631_:
{
if (v___y_1633_ == 0)
{
lean_object* v___x_1634_; 
lean_dec_ref(v___y_1632_);
lean_del_object(v___x_1629_);
lean_inc_ref(v___x_1590_);
v___x_1634_ = l_Lean_Meta_whnfR(v___x_1590_, v___y_1526_, v___y_1527_, v___y_1528_, v___y_1529_);
if (lean_obj_tag(v___x_1634_) == 0)
{
lean_object* v_a_1635_; lean_object* v___x_1636_; lean_object* v___x_1637_; uint8_t v___x_1638_; 
v_a_1635_ = lean_ctor_get(v___x_1634_, 0);
lean_inc(v_a_1635_);
lean_dec_ref_known(v___x_1634_, 1);
v___x_1636_ = ((lean_object*)(lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__14));
v___x_1637_ = lean_unsigned_to_nat(1u);
v___x_1638_ = l_Lean_Expr_isAppOfArity(v_a_1635_, v___x_1636_, v___x_1637_);
if (v___x_1638_ == 0)
{
lean_dec(v_a_1635_);
lean_dec(v_a_1627_);
lean_dec_ref(v___x_1624_);
lean_dec_ref(v___x_1623_);
lean_dec(v_a_1622_);
lean_del_object(v___x_1543_);
lean_dec(v_a_1524_);
v___y_1592_ = v___y_1526_;
v___y_1593_ = v___y_1527_;
v___y_1594_ = v___y_1528_;
v___y_1595_ = v___y_1529_;
goto v___jp_1591_;
}
else
{
lean_object* v___x_1639_; 
v___x_1639_ = l_Lean_Level_dec(v_a_1627_);
lean_dec(v_a_1627_);
if (lean_obj_tag(v___x_1639_) == 1)
{
lean_object* v_val_1640_; lean_object* v___x_1641_; lean_object* v___x_1642_; lean_object* v___x_1643_; lean_object* v___x_1644_; lean_object* v___x_1645_; lean_object* v___x_1646_; lean_object* v___x_1647_; lean_object* v___x_1648_; uint8_t v___x_1649_; lean_object* v___x_1650_; 
v_val_1640_ = lean_ctor_get(v___x_1639_, 0);
lean_inc(v_val_1640_);
lean_dec_ref_known(v___x_1639_, 1);
v___x_1641_ = l_Lean_Expr_appArg_x21(v_a_1635_);
lean_dec(v_a_1635_);
v___x_1642_ = ((lean_object*)(lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__16));
v___x_1643_ = lean_box(0);
v___x_1644_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1644_, 0, v_val_1640_);
lean_ctor_set(v___x_1644_, 1, v___x_1643_);
lean_inc_ref(v___x_1644_);
v___x_1645_ = l_Lean_Expr_const___override(v___x_1642_, v___x_1644_);
lean_inc_ref(v___x_1623_);
lean_inc_ref(v___x_1641_);
lean_inc_ref(v___x_1645_);
v___x_1646_ = l_Lean_mkAppB(v___x_1645_, v___x_1641_, v___x_1623_);
v___x_1647_ = lean_box(0);
v___x_1648_ = lean_alloc_closure((void*)(l_Lean_Meta_synthInstance___boxed), 7, 2);
lean_closure_set(v___x_1648_, 0, v___x_1646_);
lean_closure_set(v___x_1648_, 1, v___x_1647_);
v___x_1649_ = lean_unbox(v_a_1622_);
v___x_1650_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Lean_Meta_synthSubsingletonInst_spec__10___redArg(v___x_1648_, v___x_1649_, v___y_1526_, v___y_1527_, v___y_1528_, v___y_1529_);
if (lean_obj_tag(v___x_1650_) == 0)
{
lean_object* v_a_1651_; lean_object* v___x_1652_; lean_object* v___x_1653_; uint8_t v___x_1654_; lean_object* v___x_1655_; 
v_a_1651_ = lean_ctor_get(v___x_1650_, 0);
lean_inc(v_a_1651_);
lean_dec_ref_known(v___x_1650_, 1);
lean_inc_ref(v___x_1624_);
lean_inc_ref(v___x_1641_);
v___x_1652_ = l_Lean_mkAppB(v___x_1645_, v___x_1641_, v___x_1624_);
v___x_1653_ = lean_alloc_closure((void*)(l_Lean_Meta_synthInstance___boxed), 7, 2);
lean_closure_set(v___x_1653_, 0, v___x_1652_);
lean_closure_set(v___x_1653_, 1, v___x_1647_);
v___x_1654_ = lean_unbox(v_a_1622_);
lean_dec(v_a_1622_);
v___x_1655_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00Lean_Meta_synthSubsingletonInst_spec__10___redArg(v___x_1653_, v___x_1654_, v___y_1526_, v___y_1527_, v___y_1528_, v___y_1529_);
if (lean_obj_tag(v___x_1655_) == 0)
{
lean_object* v_a_1656_; lean_object* v___x_1657_; lean_object* v___x_1658_; lean_object* v___x_1659_; lean_object* v___x_1660_; lean_object* v___x_1662_; uint8_t v_isShared_1663_; uint8_t v_isSharedCheck_1668_; 
lean_dec_ref(v___x_1590_);
lean_del_object(v___x_1543_);
v_a_1656_ = lean_ctor_get(v___x_1655_, 0);
lean_inc(v_a_1656_);
lean_dec_ref_known(v___x_1655_, 1);
v___x_1657_ = ((lean_object*)(lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__18));
v___x_1658_ = l_Lean_Expr_const___override(v___x_1657_, v___x_1644_);
v___x_1659_ = l_Lean_mkApp5(v___x_1658_, v___x_1641_, v___x_1623_, v___x_1624_, v_a_1651_, v_a_1656_);
v___x_1660_ = lp_mathlib_Lean_MVarId_assign___at___00Lean_MVarId_subsingleton_spec__0___redArg(v_a_1524_, v___x_1659_, v___y_1527_);
v_isSharedCheck_1668_ = !lean_is_exclusive(v___x_1660_);
if (v_isSharedCheck_1668_ == 0)
{
lean_object* v_unused_1669_; 
v_unused_1669_ = lean_ctor_get(v___x_1660_, 0);
lean_dec(v_unused_1669_);
v___x_1662_ = v___x_1660_;
v_isShared_1663_ = v_isSharedCheck_1668_;
goto v_resetjp_1661_;
}
else
{
lean_dec(v___x_1660_);
v___x_1662_ = lean_box(0);
v_isShared_1663_ = v_isSharedCheck_1668_;
goto v_resetjp_1661_;
}
v_resetjp_1661_:
{
lean_object* v___x_1664_; lean_object* v___x_1666_; 
v___x_1664_ = lean_box(0);
if (v_isShared_1663_ == 0)
{
lean_ctor_set(v___x_1662_, 0, v___x_1664_);
v___x_1666_ = v___x_1662_;
goto v_reusejp_1665_;
}
else
{
lean_object* v_reuseFailAlloc_1667_; 
v_reuseFailAlloc_1667_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1667_, 0, v___x_1664_);
v___x_1666_ = v_reuseFailAlloc_1667_;
goto v_reusejp_1665_;
}
v_reusejp_1665_:
{
return v___x_1666_;
}
}
}
else
{
lean_object* v_a_1670_; 
lean_dec(v_a_1651_);
lean_dec_ref_known(v___x_1644_, 2);
lean_dec_ref(v___x_1641_);
lean_dec_ref(v___x_1624_);
lean_dec_ref(v___x_1623_);
lean_dec(v_a_1524_);
v_a_1670_ = lean_ctor_get(v___x_1655_, 0);
lean_inc(v_a_1670_);
lean_dec_ref_known(v___x_1655_, 1);
v_a_1618_ = v_a_1670_;
goto v___jp_1617_;
}
}
else
{
lean_object* v_a_1671_; 
lean_dec_ref(v___x_1645_);
lean_dec_ref_known(v___x_1644_, 2);
lean_dec_ref(v___x_1641_);
lean_dec_ref(v___x_1624_);
lean_dec_ref(v___x_1623_);
lean_dec(v_a_1622_);
lean_dec(v_a_1524_);
v_a_1671_ = lean_ctor_get(v___x_1650_, 0);
lean_inc(v_a_1671_);
lean_dec_ref_known(v___x_1650_, 1);
v_a_1618_ = v_a_1671_;
goto v___jp_1617_;
}
}
else
{
lean_object* v___x_1672_; lean_object* v___x_1673_; lean_object* v_a_1674_; 
lean_dec(v___x_1639_);
lean_dec(v_a_1635_);
lean_dec_ref(v___x_1624_);
lean_dec_ref(v___x_1623_);
lean_dec(v_a_1622_);
lean_dec(v_a_1524_);
v___x_1672_ = lean_obj_once(&lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__20, &lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__20_once, _init_lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__20);
v___x_1673_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Meta_synthSubsingletonInst_spec__4_spec__4___redArg(v___x_1672_, v___y_1526_, v___y_1527_, v___y_1528_, v___y_1529_);
v_a_1674_ = lean_ctor_get(v___x_1673_, 0);
lean_inc(v_a_1674_);
lean_dec_ref(v___x_1673_);
v_a_1618_ = v_a_1674_;
goto v___jp_1617_;
}
}
}
else
{
lean_object* v_a_1675_; lean_object* v___x_1677_; uint8_t v_isShared_1678_; uint8_t v_isSharedCheck_1682_; 
lean_dec(v_a_1627_);
lean_dec_ref(v___x_1624_);
lean_dec_ref(v___x_1623_);
lean_dec(v_a_1622_);
lean_dec_ref(v___x_1590_);
lean_del_object(v___x_1543_);
lean_dec(v_a_1524_);
v_a_1675_ = lean_ctor_get(v___x_1634_, 0);
v_isSharedCheck_1682_ = !lean_is_exclusive(v___x_1634_);
if (v_isSharedCheck_1682_ == 0)
{
v___x_1677_ = v___x_1634_;
v_isShared_1678_ = v_isSharedCheck_1682_;
goto v_resetjp_1676_;
}
else
{
lean_inc(v_a_1675_);
lean_dec(v___x_1634_);
v___x_1677_ = lean_box(0);
v_isShared_1678_ = v_isSharedCheck_1682_;
goto v_resetjp_1676_;
}
v_resetjp_1676_:
{
lean_object* v___x_1680_; 
if (v_isShared_1678_ == 0)
{
v___x_1680_ = v___x_1677_;
goto v_reusejp_1679_;
}
else
{
lean_object* v_reuseFailAlloc_1681_; 
v_reuseFailAlloc_1681_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1681_, 0, v_a_1675_);
v___x_1680_ = v_reuseFailAlloc_1681_;
goto v_reusejp_1679_;
}
v_reusejp_1679_:
{
return v___x_1680_;
}
}
}
}
else
{
lean_object* v___x_1684_; 
lean_dec(v_a_1627_);
lean_dec_ref(v___x_1624_);
lean_dec_ref(v___x_1623_);
lean_dec(v_a_1622_);
lean_dec_ref(v___x_1590_);
lean_del_object(v___x_1543_);
lean_dec(v_a_1524_);
if (v_isShared_1630_ == 0)
{
lean_ctor_set_tag(v___x_1629_, 1);
lean_ctor_set(v___x_1629_, 0, v___y_1632_);
v___x_1684_ = v___x_1629_;
goto v_reusejp_1683_;
}
else
{
lean_object* v_reuseFailAlloc_1685_; 
v_reuseFailAlloc_1685_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1685_, 0, v___y_1632_);
v___x_1684_ = v_reuseFailAlloc_1685_;
goto v_reusejp_1683_;
}
v_reusejp_1683_:
{
return v___x_1684_;
}
}
}
v___jp_1686_:
{
uint8_t v___x_1688_; 
v___x_1688_ = l_Lean_Exception_isInterrupt(v_a_1687_);
if (v___x_1688_ == 0)
{
uint8_t v___x_1689_; 
lean_inc_ref(v_a_1687_);
v___x_1689_ = l_Lean_Exception_isRuntime(v_a_1687_);
v___y_1632_ = v_a_1687_;
v___y_1633_ = v___x_1689_;
goto v___jp_1631_;
}
else
{
v___y_1632_ = v_a_1687_;
v___y_1633_ = v___x_1688_;
goto v___jp_1631_;
}
}
}
}
else
{
lean_object* v_a_1709_; lean_object* v___x_1711_; uint8_t v_isShared_1712_; uint8_t v_isSharedCheck_1716_; 
lean_dec_ref(v___x_1624_);
lean_dec_ref(v___x_1623_);
lean_dec(v_a_1622_);
lean_dec_ref(v___x_1590_);
lean_del_object(v___x_1543_);
lean_dec_ref(v_insts_1525_);
lean_dec(v_a_1524_);
v_a_1709_ = lean_ctor_get(v___x_1626_, 0);
v_isSharedCheck_1716_ = !lean_is_exclusive(v___x_1626_);
if (v_isSharedCheck_1716_ == 0)
{
v___x_1711_ = v___x_1626_;
v_isShared_1712_ = v_isSharedCheck_1716_;
goto v_resetjp_1710_;
}
else
{
lean_inc(v_a_1709_);
lean_dec(v___x_1626_);
v___x_1711_ = lean_box(0);
v_isShared_1712_ = v_isSharedCheck_1716_;
goto v_resetjp_1710_;
}
v_resetjp_1710_:
{
lean_object* v___x_1714_; 
if (v_isShared_1712_ == 0)
{
v___x_1714_ = v___x_1711_;
goto v_reusejp_1713_;
}
else
{
lean_object* v_reuseFailAlloc_1715_; 
v_reuseFailAlloc_1715_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1715_, 0, v_a_1709_);
v___x_1714_ = v_reuseFailAlloc_1715_;
goto v_reusejp_1713_;
}
v_reusejp_1713_:
{
return v___x_1714_;
}
}
}
}
else
{
lean_object* v___x_1717_; lean_object* v___x_1718_; lean_object* v___x_1719_; lean_object* v___x_1721_; uint8_t v_isShared_1722_; uint8_t v_isSharedCheck_1727_; 
lean_dec(v_a_1622_);
lean_del_object(v___x_1543_);
lean_dec_ref(v_insts_1525_);
v___x_1717_ = lean_obj_once(&lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__25, &lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__25_once, _init_lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__25);
v___x_1718_ = l_Lean_mkApp3(v___x_1717_, v___x_1590_, v___x_1623_, v___x_1624_);
v___x_1719_ = lp_mathlib_Lean_MVarId_assign___at___00Lean_MVarId_subsingleton_spec__0___redArg(v_a_1524_, v___x_1718_, v___y_1527_);
v_isSharedCheck_1727_ = !lean_is_exclusive(v___x_1719_);
if (v_isSharedCheck_1727_ == 0)
{
lean_object* v_unused_1728_; 
v_unused_1728_ = lean_ctor_get(v___x_1719_, 0);
lean_dec(v_unused_1728_);
v___x_1721_ = v___x_1719_;
v_isShared_1722_ = v_isSharedCheck_1727_;
goto v_resetjp_1720_;
}
else
{
lean_dec(v___x_1719_);
v___x_1721_ = lean_box(0);
v_isShared_1722_ = v_isSharedCheck_1727_;
goto v_resetjp_1720_;
}
v_resetjp_1720_:
{
lean_object* v___x_1723_; lean_object* v___x_1725_; 
v___x_1723_ = lean_box(0);
if (v_isShared_1722_ == 0)
{
lean_ctor_set(v___x_1721_, 0, v___x_1723_);
v___x_1725_ = v___x_1721_;
goto v_reusejp_1724_;
}
else
{
lean_object* v_reuseFailAlloc_1726_; 
v_reuseFailAlloc_1726_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1726_, 0, v___x_1723_);
v___x_1725_ = v_reuseFailAlloc_1726_;
goto v_reusejp_1724_;
}
v_reusejp_1724_:
{
return v___x_1725_;
}
}
}
}
else
{
lean_object* v_a_1729_; lean_object* v___x_1731_; uint8_t v_isShared_1732_; uint8_t v_isSharedCheck_1736_; 
lean_dec_ref(v___x_1590_);
lean_dec_ref(v___x_1588_);
lean_del_object(v___x_1543_);
lean_dec(v_a_1541_);
lean_dec_ref(v_insts_1525_);
lean_dec(v_a_1524_);
v_a_1729_ = lean_ctor_get(v___x_1621_, 0);
v_isSharedCheck_1736_ = !lean_is_exclusive(v___x_1621_);
if (v_isSharedCheck_1736_ == 0)
{
v___x_1731_ = v___x_1621_;
v_isShared_1732_ = v_isSharedCheck_1736_;
goto v_resetjp_1730_;
}
else
{
lean_inc(v_a_1729_);
lean_dec(v___x_1621_);
v___x_1731_ = lean_box(0);
v_isShared_1732_ = v_isSharedCheck_1736_;
goto v_resetjp_1730_;
}
v_resetjp_1730_:
{
lean_object* v___x_1734_; 
if (v_isShared_1732_ == 0)
{
v___x_1734_ = v___x_1731_;
goto v_reusejp_1733_;
}
else
{
lean_object* v_reuseFailAlloc_1735_; 
v_reuseFailAlloc_1735_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1735_, 0, v_a_1729_);
v___x_1734_ = v_reuseFailAlloc_1735_;
goto v_reusejp_1733_;
}
v_reusejp_1733_:
{
return v___x_1734_;
}
}
}
v___jp_1591_:
{
lean_object* v___x_1596_; 
v___x_1596_ = lp_mathlib_Lean_Meta_mkSubsingleton(v___x_1590_, v___y_1592_, v___y_1593_, v___y_1594_, v___y_1595_);
if (lean_obj_tag(v___x_1596_) == 0)
{
lean_object* v_a_1597_; lean_object* v___x_1598_; lean_object* v___x_1599_; lean_object* v___x_1600_; lean_object* v___x_1601_; lean_object* v___x_1602_; 
v_a_1597_ = lean_ctor_get(v___x_1596_, 0);
lean_inc(v_a_1597_);
lean_dec_ref_known(v___x_1596_, 1);
v___x_1598_ = lean_obj_once(&lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__12, &lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__12_once, _init_lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__12);
v___x_1599_ = l_Lean_MessageData_ofExpr(v_a_1597_);
v___x_1600_ = l_Lean_indentD(v___x_1599_);
v___x_1601_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1601_, 0, v___x_1598_);
lean_ctor_set(v___x_1601_, 1, v___x_1600_);
v___x_1602_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Meta_synthSubsingletonInst_spec__4_spec__4___redArg(v___x_1601_, v___y_1592_, v___y_1593_, v___y_1594_, v___y_1595_);
return v___x_1602_;
}
else
{
lean_object* v_a_1603_; lean_object* v___x_1605_; uint8_t v_isShared_1606_; uint8_t v_isSharedCheck_1610_; 
v_a_1603_ = lean_ctor_get(v___x_1596_, 0);
v_isSharedCheck_1610_ = !lean_is_exclusive(v___x_1596_);
if (v_isSharedCheck_1610_ == 0)
{
v___x_1605_ = v___x_1596_;
v_isShared_1606_ = v_isSharedCheck_1610_;
goto v_resetjp_1604_;
}
else
{
lean_inc(v_a_1603_);
lean_dec(v___x_1596_);
v___x_1605_ = lean_box(0);
v_isShared_1606_ = v_isSharedCheck_1610_;
goto v_resetjp_1604_;
}
v_resetjp_1604_:
{
lean_object* v___x_1608_; 
if (v_isShared_1606_ == 0)
{
v___x_1608_ = v___x_1605_;
goto v_reusejp_1607_;
}
else
{
lean_object* v_reuseFailAlloc_1609_; 
v_reuseFailAlloc_1609_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1609_, 0, v_a_1603_);
v___x_1608_ = v_reuseFailAlloc_1609_;
goto v_reusejp_1607_;
}
v_reusejp_1607_:
{
return v___x_1608_;
}
}
}
}
v___jp_1611_:
{
if (v___y_1613_ == 0)
{
lean_dec_ref(v___y_1612_);
lean_del_object(v___x_1543_);
v___y_1592_ = v___y_1526_;
v___y_1593_ = v___y_1527_;
v___y_1594_ = v___y_1528_;
v___y_1595_ = v___y_1529_;
goto v___jp_1591_;
}
else
{
lean_object* v___x_1615_; 
lean_dec_ref(v___x_1590_);
if (v_isShared_1544_ == 0)
{
lean_ctor_set_tag(v___x_1543_, 1);
lean_ctor_set(v___x_1543_, 0, v___y_1612_);
v___x_1615_ = v___x_1543_;
goto v_reusejp_1614_;
}
else
{
lean_object* v_reuseFailAlloc_1616_; 
v_reuseFailAlloc_1616_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1616_, 0, v___y_1612_);
v___x_1615_ = v_reuseFailAlloc_1616_;
goto v_reusejp_1614_;
}
v_reusejp_1614_:
{
return v___x_1615_;
}
}
}
v___jp_1617_:
{
uint8_t v___x_1619_; 
v___x_1619_ = l_Lean_Exception_isInterrupt(v_a_1618_);
if (v___x_1619_ == 0)
{
uint8_t v___x_1620_; 
lean_inc_ref(v_a_1618_);
v___x_1620_ = l_Lean_Exception_isRuntime(v_a_1618_);
v___y_1612_ = v_a_1618_;
v___y_1613_ = v___x_1620_;
goto v___jp_1611_;
}
else
{
v___y_1612_ = v_a_1618_;
v___y_1613_ = v___x_1619_;
goto v___jp_1611_;
}
}
}
}
}
else
{
lean_object* v_a_1738_; lean_object* v___x_1740_; uint8_t v_isShared_1741_; uint8_t v_isSharedCheck_1745_; 
lean_dec_ref(v_insts_1525_);
lean_dec(v_a_1524_);
v_a_1738_ = lean_ctor_get(v___x_1540_, 0);
v_isSharedCheck_1745_ = !lean_is_exclusive(v___x_1540_);
if (v_isSharedCheck_1745_ == 0)
{
v___x_1740_ = v___x_1540_;
v_isShared_1741_ = v_isSharedCheck_1745_;
goto v_resetjp_1739_;
}
else
{
lean_inc(v_a_1738_);
lean_dec(v___x_1540_);
v___x_1740_ = lean_box(0);
v_isShared_1741_ = v_isSharedCheck_1745_;
goto v_resetjp_1739_;
}
v_resetjp_1739_:
{
lean_object* v___x_1743_; 
if (v_isShared_1741_ == 0)
{
v___x_1743_ = v___x_1740_;
goto v_reusejp_1742_;
}
else
{
lean_object* v_reuseFailAlloc_1744_; 
v_reuseFailAlloc_1744_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1744_, 0, v_a_1738_);
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
else
{
lean_object* v_a_1746_; lean_object* v___x_1748_; uint8_t v_isShared_1749_; uint8_t v_isSharedCheck_1753_; 
lean_dec_ref(v_insts_1525_);
lean_dec(v_a_1524_);
v_a_1746_ = lean_ctor_get(v___x_1538_, 0);
v_isSharedCheck_1753_ = !lean_is_exclusive(v___x_1538_);
if (v_isSharedCheck_1753_ == 0)
{
v___x_1748_ = v___x_1538_;
v_isShared_1749_ = v_isSharedCheck_1753_;
goto v_resetjp_1747_;
}
else
{
lean_inc(v_a_1746_);
lean_dec(v___x_1538_);
v___x_1748_ = lean_box(0);
v_isShared_1749_ = v_isSharedCheck_1753_;
goto v_resetjp_1747_;
}
v_resetjp_1747_:
{
lean_object* v___x_1751_; 
if (v_isShared_1749_ == 0)
{
v___x_1751_ = v___x_1748_;
goto v_reusejp_1750_;
}
else
{
lean_object* v_reuseFailAlloc_1752_; 
v_reuseFailAlloc_1752_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1752_, 0, v_a_1746_);
v___x_1751_ = v_reuseFailAlloc_1752_;
goto v_reusejp_1750_;
}
v_reusejp_1750_:
{
return v___x_1751_;
}
}
}
v___jp_1531_:
{
lean_object* v___x_1536_; lean_object* v___x_1537_; 
v___x_1536_ = lean_obj_once(&lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__1, &lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__1_once, _init_lp_mathlib_Lean_MVarId_subsingleton___lam__0___closed__1);
v___x_1537_ = lp_mathlib_Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Meta_synthSubsingletonInst_spec__4_spec__4___redArg(v___x_1536_, v___y_1532_, v___y_1533_, v___y_1534_, v___y_1535_);
return v___x_1537_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_subsingleton___lam__0___boxed(lean_object* v_a_1754_, lean_object* v_insts_1755_, lean_object* v___y_1756_, lean_object* v___y_1757_, lean_object* v___y_1758_, lean_object* v___y_1759_, lean_object* v___y_1760_){
_start:
{
lean_object* v_res_1761_; 
v_res_1761_ = lp_mathlib_Lean_MVarId_subsingleton___lam__0(v_a_1754_, v_insts_1755_, v___y_1756_, v___y_1757_, v___y_1758_, v___y_1759_);
lean_dec(v___y_1759_);
lean_dec_ref(v___y_1758_);
lean_dec(v___y_1757_);
lean_dec_ref(v___y_1756_);
return v_res_1761_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_subsingleton___lam__1(lean_object* v_g_1762_, lean_object* v_insts_1763_, lean_object* v___y_1764_, lean_object* v___y_1765_, lean_object* v___y_1766_, lean_object* v___y_1767_){
_start:
{
lean_object* v___x_1769_; 
v___x_1769_ = l_Lean_MVarId_heqOfEq(v_g_1762_, v___y_1764_, v___y_1765_, v___y_1766_, v___y_1767_);
if (lean_obj_tag(v___x_1769_) == 0)
{
lean_object* v_a_1770_; lean_object* v___f_1771_; lean_object* v___x_1772_; 
v_a_1770_ = lean_ctor_get(v___x_1769_, 0);
lean_inc_n(v_a_1770_, 2);
lean_dec_ref_known(v___x_1769_, 1);
v___f_1771_ = lean_alloc_closure((void*)(lp_mathlib_Lean_MVarId_subsingleton___lam__0___boxed), 7, 2);
lean_closure_set(v___f_1771_, 0, v_a_1770_);
lean_closure_set(v___f_1771_, 1, v_insts_1763_);
v___x_1772_ = lp_mathlib_Lean_MVarId_withContext___at___00Lean_MVarId_subsingleton_spec__1___redArg(v_a_1770_, v___f_1771_, v___y_1764_, v___y_1765_, v___y_1766_, v___y_1767_);
return v___x_1772_;
}
else
{
lean_object* v_a_1773_; lean_object* v___x_1775_; uint8_t v_isShared_1776_; uint8_t v_isSharedCheck_1780_; 
lean_dec_ref(v_insts_1763_);
v_a_1773_ = lean_ctor_get(v___x_1769_, 0);
v_isSharedCheck_1780_ = !lean_is_exclusive(v___x_1769_);
if (v_isSharedCheck_1780_ == 0)
{
v___x_1775_ = v___x_1769_;
v_isShared_1776_ = v_isSharedCheck_1780_;
goto v_resetjp_1774_;
}
else
{
lean_inc(v_a_1773_);
lean_dec(v___x_1769_);
v___x_1775_ = lean_box(0);
v_isShared_1776_ = v_isSharedCheck_1780_;
goto v_resetjp_1774_;
}
v_resetjp_1774_:
{
lean_object* v___x_1778_; 
if (v_isShared_1776_ == 0)
{
v___x_1778_ = v___x_1775_;
goto v_reusejp_1777_;
}
else
{
lean_object* v_reuseFailAlloc_1779_; 
v_reuseFailAlloc_1779_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1779_, 0, v_a_1773_);
v___x_1778_ = v_reuseFailAlloc_1779_;
goto v_reusejp_1777_;
}
v_reusejp_1777_:
{
return v___x_1778_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_subsingleton___lam__1___boxed(lean_object* v_g_1781_, lean_object* v_insts_1782_, lean_object* v___y_1783_, lean_object* v___y_1784_, lean_object* v___y_1785_, lean_object* v___y_1786_, lean_object* v___y_1787_){
_start:
{
lean_object* v_res_1788_; 
v_res_1788_ = lp_mathlib_Lean_MVarId_subsingleton___lam__1(v_g_1781_, v_insts_1782_, v___y_1783_, v___y_1784_, v___y_1785_, v___y_1786_);
lean_dec(v___y_1786_);
lean_dec_ref(v___y_1785_);
lean_dec(v___y_1784_);
lean_dec_ref(v___y_1783_);
return v_res_1788_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_subsingleton(lean_object* v_g_1789_, lean_object* v_insts_1790_, lean_object* v_a_1791_, lean_object* v_a_1792_, lean_object* v_a_1793_, lean_object* v_a_1794_){
_start:
{
lean_object* v___f_1796_; lean_object* v___x_1797_; 
v___f_1796_ = lean_alloc_closure((void*)(lp_mathlib_Lean_MVarId_subsingleton___lam__1___boxed), 7, 2);
lean_closure_set(v___f_1796_, 0, v_g_1789_);
lean_closure_set(v___f_1796_, 1, v_insts_1790_);
v___x_1797_ = lp_mathlib_Lean_commitIfNoEx___at___00Lean_MVarId_subsingleton_spec__2___redArg(v___f_1796_, v_a_1791_, v_a_1792_, v_a_1793_, v_a_1794_);
return v___x_1797_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_subsingleton___boxed(lean_object* v_g_1798_, lean_object* v_insts_1799_, lean_object* v_a_1800_, lean_object* v_a_1801_, lean_object* v_a_1802_, lean_object* v_a_1803_, lean_object* v_a_1804_){
_start:
{
lean_object* v_res_1805_; 
v_res_1805_ = lp_mathlib_Lean_MVarId_subsingleton(v_g_1798_, v_insts_1799_, v_a_1800_, v_a_1801_, v_a_1802_, v_a_1803_);
lean_dec(v_a_1803_);
lean_dec_ref(v_a_1802_);
lean_dec(v_a_1801_);
lean_dec_ref(v_a_1800_);
return v_res_1805_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Lean_MVarId_subsingleton_spec__0(lean_object* v_mvarId_1806_, lean_object* v_val_1807_, lean_object* v___y_1808_, lean_object* v___y_1809_, lean_object* v___y_1810_, lean_object* v___y_1811_){
_start:
{
lean_object* v___x_1813_; 
v___x_1813_ = lp_mathlib_Lean_MVarId_assign___at___00Lean_MVarId_subsingleton_spec__0___redArg(v_mvarId_1806_, v_val_1807_, v___y_1809_);
return v___x_1813_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Lean_MVarId_subsingleton_spec__0___boxed(lean_object* v_mvarId_1814_, lean_object* v_val_1815_, lean_object* v___y_1816_, lean_object* v___y_1817_, lean_object* v___y_1818_, lean_object* v___y_1819_, lean_object* v___y_1820_){
_start:
{
lean_object* v_res_1821_; 
v_res_1821_ = lp_mathlib_Lean_MVarId_assign___at___00Lean_MVarId_subsingleton_spec__0(v_mvarId_1814_, v_val_1815_, v___y_1816_, v___y_1817_, v___y_1818_, v___y_1819_);
lean_dec(v___y_1819_);
lean_dec_ref(v___y_1818_);
lean_dec(v___y_1817_);
lean_dec_ref(v___y_1816_);
return v_res_1821_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_subsingleton_spec__0_spec__0(lean_object* v_00_u03b2_1822_, lean_object* v_x_1823_, lean_object* v_x_1824_, lean_object* v_x_1825_){
_start:
{
lean_object* v___x_1826_; 
v___x_1826_ = lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_subsingleton_spec__0_spec__0___redArg(v_x_1823_, v_x_1824_, v_x_1825_);
return v___x_1826_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_subsingleton_spec__0_spec__0_spec__3(lean_object* v_00_u03b2_1827_, lean_object* v_x_1828_, size_t v_x_1829_, size_t v_x_1830_, lean_object* v_x_1831_, lean_object* v_x_1832_){
_start:
{
lean_object* v___x_1833_; 
v___x_1833_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_subsingleton_spec__0_spec__0_spec__3___redArg(v_x_1828_, v_x_1829_, v_x_1830_, v_x_1831_, v_x_1832_);
return v___x_1833_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_subsingleton_spec__0_spec__0_spec__3___boxed(lean_object* v_00_u03b2_1834_, lean_object* v_x_1835_, lean_object* v_x_1836_, lean_object* v_x_1837_, lean_object* v_x_1838_, lean_object* v_x_1839_){
_start:
{
size_t v_x_10926__boxed_1840_; size_t v_x_10927__boxed_1841_; lean_object* v_res_1842_; 
v_x_10926__boxed_1840_ = lean_unbox_usize(v_x_1836_);
lean_dec(v_x_1836_);
v_x_10927__boxed_1841_ = lean_unbox_usize(v_x_1837_);
lean_dec(v_x_1837_);
v_res_1842_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_subsingleton_spec__0_spec__0_spec__3(v_00_u03b2_1834_, v_x_1835_, v_x_10926__boxed_1840_, v_x_10927__boxed_1841_, v_x_1838_, v_x_1839_);
return v_res_1842_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_subsingleton_spec__0_spec__0_spec__3_spec__4(lean_object* v_00_u03b2_1843_, lean_object* v_n_1844_, lean_object* v_k_1845_, lean_object* v_v_1846_){
_start:
{
lean_object* v___x_1847_; 
v___x_1847_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_subsingleton_spec__0_spec__0_spec__3_spec__4___redArg(v_n_1844_, v_k_1845_, v_v_1846_);
return v___x_1847_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_subsingleton_spec__0_spec__0_spec__3_spec__5(lean_object* v_00_u03b2_1848_, size_t v_depth_1849_, lean_object* v_keys_1850_, lean_object* v_vals_1851_, lean_object* v_heq_1852_, lean_object* v_i_1853_, lean_object* v_entries_1854_){
_start:
{
lean_object* v___x_1855_; 
v___x_1855_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_subsingleton_spec__0_spec__0_spec__3_spec__5___redArg(v_depth_1849_, v_keys_1850_, v_vals_1851_, v_i_1853_, v_entries_1854_);
return v___x_1855_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_subsingleton_spec__0_spec__0_spec__3_spec__5___boxed(lean_object* v_00_u03b2_1856_, lean_object* v_depth_1857_, lean_object* v_keys_1858_, lean_object* v_vals_1859_, lean_object* v_heq_1860_, lean_object* v_i_1861_, lean_object* v_entries_1862_){
_start:
{
size_t v_depth_boxed_1863_; lean_object* v_res_1864_; 
v_depth_boxed_1863_ = lean_unbox_usize(v_depth_1857_);
lean_dec(v_depth_1857_);
v_res_1864_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_subsingleton_spec__0_spec__0_spec__3_spec__5(v_00_u03b2_1856_, v_depth_boxed_1863_, v_keys_1858_, v_vals_1859_, v_heq_1860_, v_i_1861_, v_entries_1862_);
lean_dec_ref(v_vals_1859_);
lean_dec_ref(v_keys_1858_);
return v_res_1864_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_subsingleton_spec__0_spec__0_spec__3_spec__4_spec__5(lean_object* v_00_u03b2_1865_, lean_object* v_x_1866_, lean_object* v_x_1867_, lean_object* v_x_1868_, lean_object* v_x_1869_){
_start:
{
lean_object* v___x_1870_; 
v___x_1870_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_subsingleton_spec__0_spec__0_spec__3_spec__4_spec__5___redArg(v_x_1866_, v_x_1867_, v_x_1868_, v_x_1869_);
return v___x_1870_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__0___redArg(lean_object* v_e_1938_, lean_object* v___y_1939_){
_start:
{
uint8_t v___x_1941_; 
v___x_1941_ = l_Lean_Expr_hasMVar(v_e_1938_);
if (v___x_1941_ == 0)
{
lean_object* v___x_1942_; 
v___x_1942_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1942_, 0, v_e_1938_);
return v___x_1942_;
}
else
{
lean_object* v___x_1943_; lean_object* v_mctx_1944_; lean_object* v___x_1945_; lean_object* v_fst_1946_; lean_object* v_snd_1947_; lean_object* v___x_1948_; lean_object* v_cache_1949_; lean_object* v_zetaDeltaFVarIds_1950_; lean_object* v_postponed_1951_; lean_object* v_diag_1952_; lean_object* v___x_1954_; uint8_t v_isShared_1955_; uint8_t v_isSharedCheck_1961_; 
v___x_1943_ = lean_st_ref_get(v___y_1939_);
v_mctx_1944_ = lean_ctor_get(v___x_1943_, 0);
lean_inc_ref(v_mctx_1944_);
lean_dec(v___x_1943_);
v___x_1945_ = l_Lean_instantiateMVarsCore(v_mctx_1944_, v_e_1938_);
v_fst_1946_ = lean_ctor_get(v___x_1945_, 0);
lean_inc(v_fst_1946_);
v_snd_1947_ = lean_ctor_get(v___x_1945_, 1);
lean_inc(v_snd_1947_);
lean_dec_ref(v___x_1945_);
v___x_1948_ = lean_st_ref_take(v___y_1939_);
v_cache_1949_ = lean_ctor_get(v___x_1948_, 1);
v_zetaDeltaFVarIds_1950_ = lean_ctor_get(v___x_1948_, 2);
v_postponed_1951_ = lean_ctor_get(v___x_1948_, 3);
v_diag_1952_ = lean_ctor_get(v___x_1948_, 4);
v_isSharedCheck_1961_ = !lean_is_exclusive(v___x_1948_);
if (v_isSharedCheck_1961_ == 0)
{
lean_object* v_unused_1962_; 
v_unused_1962_ = lean_ctor_get(v___x_1948_, 0);
lean_dec(v_unused_1962_);
v___x_1954_ = v___x_1948_;
v_isShared_1955_ = v_isSharedCheck_1961_;
goto v_resetjp_1953_;
}
else
{
lean_inc(v_diag_1952_);
lean_inc(v_postponed_1951_);
lean_inc(v_zetaDeltaFVarIds_1950_);
lean_inc(v_cache_1949_);
lean_dec(v___x_1948_);
v___x_1954_ = lean_box(0);
v_isShared_1955_ = v_isSharedCheck_1961_;
goto v_resetjp_1953_;
}
v_resetjp_1953_:
{
lean_object* v___x_1957_; 
if (v_isShared_1955_ == 0)
{
lean_ctor_set(v___x_1954_, 0, v_snd_1947_);
v___x_1957_ = v___x_1954_;
goto v_reusejp_1956_;
}
else
{
lean_object* v_reuseFailAlloc_1960_; 
v_reuseFailAlloc_1960_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_1960_, 0, v_snd_1947_);
lean_ctor_set(v_reuseFailAlloc_1960_, 1, v_cache_1949_);
lean_ctor_set(v_reuseFailAlloc_1960_, 2, v_zetaDeltaFVarIds_1950_);
lean_ctor_set(v_reuseFailAlloc_1960_, 3, v_postponed_1951_);
lean_ctor_set(v_reuseFailAlloc_1960_, 4, v_diag_1952_);
v___x_1957_ = v_reuseFailAlloc_1960_;
goto v_reusejp_1956_;
}
v_reusejp_1956_:
{
lean_object* v___x_1958_; lean_object* v___x_1959_; 
v___x_1958_ = lean_st_ref_set(v___y_1939_, v___x_1957_);
v___x_1959_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1959_, 0, v_fst_1946_);
return v___x_1959_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__0___redArg___boxed(lean_object* v_e_1963_, lean_object* v___y_1964_, lean_object* v___y_1965_){
_start:
{
lean_object* v_res_1966_; 
v_res_1966_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__0___redArg(v_e_1963_, v___y_1964_);
lean_dec(v___y_1964_);
return v_res_1966_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__0(lean_object* v_e_1967_, lean_object* v___y_1968_, lean_object* v___y_1969_, lean_object* v___y_1970_, lean_object* v___y_1971_, lean_object* v___y_1972_, lean_object* v___y_1973_){
_start:
{
lean_object* v___x_1975_; 
v___x_1975_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__0___redArg(v_e_1967_, v___y_1971_);
return v___x_1975_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__0___boxed(lean_object* v_e_1976_, lean_object* v___y_1977_, lean_object* v___y_1978_, lean_object* v___y_1979_, lean_object* v___y_1980_, lean_object* v___y_1981_, lean_object* v___y_1982_, lean_object* v___y_1983_){
_start:
{
lean_object* v_res_1984_; 
v_res_1984_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__0(v_e_1976_, v___y_1977_, v___y_1978_, v___y_1979_, v___y_1980_, v___y_1981_, v___y_1982_);
lean_dec(v___y_1982_);
lean_dec_ref(v___y_1981_);
lean_dec(v___y_1980_);
lean_dec_ref(v___y_1979_);
lean_dec(v___y_1978_);
lean_dec_ref(v___y_1977_);
return v_res_1984_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewBinderInfos___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__2___redArg___lam__0(lean_object* v_k_1985_, lean_object* v___y_1986_, lean_object* v___y_1987_, lean_object* v___y_1988_, lean_object* v___y_1989_, lean_object* v___y_1990_, lean_object* v___y_1991_){
_start:
{
lean_object* v___x_1993_; 
lean_inc(v___y_1987_);
lean_inc_ref(v___y_1986_);
v___x_1993_ = lean_apply_7(v_k_1985_, v___y_1986_, v___y_1987_, v___y_1988_, v___y_1989_, v___y_1990_, v___y_1991_, lean_box(0));
return v___x_1993_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewBinderInfos___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__2___redArg___lam__0___boxed(lean_object* v_k_1994_, lean_object* v___y_1995_, lean_object* v___y_1996_, lean_object* v___y_1997_, lean_object* v___y_1998_, lean_object* v___y_1999_, lean_object* v___y_2000_, lean_object* v___y_2001_){
_start:
{
lean_object* v_res_2002_; 
v_res_2002_ = lp_mathlib_Lean_Meta_withNewBinderInfos___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__2___redArg___lam__0(v_k_1994_, v___y_1995_, v___y_1996_, v___y_1997_, v___y_1998_, v___y_1999_, v___y_2000_);
lean_dec(v___y_1996_);
lean_dec_ref(v___y_1995_);
return v_res_2002_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewBinderInfos___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__2___redArg(lean_object* v_bs_2003_, lean_object* v_k_2004_, lean_object* v___y_2005_, lean_object* v___y_2006_, lean_object* v___y_2007_, lean_object* v___y_2008_, lean_object* v___y_2009_, lean_object* v___y_2010_){
_start:
{
lean_object* v___f_2012_; lean_object* v___x_2013_; 
lean_inc(v___y_2006_);
lean_inc_ref(v___y_2005_);
v___f_2012_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_withNewBinderInfos___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__2___redArg___lam__0___boxed), 8, 3);
lean_closure_set(v___f_2012_, 0, v_k_2004_);
lean_closure_set(v___f_2012_, 1, v___y_2005_);
lean_closure_set(v___f_2012_, 2, v___y_2006_);
v___x_2013_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withNewBinderInfosImp(lean_box(0), v_bs_2003_, v___f_2012_, v___y_2007_, v___y_2008_, v___y_2009_, v___y_2010_);
if (lean_obj_tag(v___x_2013_) == 0)
{
return v___x_2013_;
}
else
{
lean_object* v_a_2014_; lean_object* v___x_2016_; uint8_t v_isShared_2017_; uint8_t v_isSharedCheck_2021_; 
v_a_2014_ = lean_ctor_get(v___x_2013_, 0);
v_isSharedCheck_2021_ = !lean_is_exclusive(v___x_2013_);
if (v_isSharedCheck_2021_ == 0)
{
v___x_2016_ = v___x_2013_;
v_isShared_2017_ = v_isSharedCheck_2021_;
goto v_resetjp_2015_;
}
else
{
lean_inc(v_a_2014_);
lean_dec(v___x_2013_);
v___x_2016_ = lean_box(0);
v_isShared_2017_ = v_isSharedCheck_2021_;
goto v_resetjp_2015_;
}
v_resetjp_2015_:
{
lean_object* v___x_2019_; 
if (v_isShared_2017_ == 0)
{
v___x_2019_ = v___x_2016_;
goto v_reusejp_2018_;
}
else
{
lean_object* v_reuseFailAlloc_2020_; 
v_reuseFailAlloc_2020_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2020_, 0, v_a_2014_);
v___x_2019_ = v_reuseFailAlloc_2020_;
goto v_reusejp_2018_;
}
v_reusejp_2018_:
{
return v___x_2019_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewBinderInfos___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__2___redArg___boxed(lean_object* v_bs_2022_, lean_object* v_k_2023_, lean_object* v___y_2024_, lean_object* v___y_2025_, lean_object* v___y_2026_, lean_object* v___y_2027_, lean_object* v___y_2028_, lean_object* v___y_2029_, lean_object* v___y_2030_){
_start:
{
lean_object* v_res_2031_; 
v_res_2031_ = lp_mathlib_Lean_Meta_withNewBinderInfos___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__2___redArg(v_bs_2022_, v_k_2023_, v___y_2024_, v___y_2025_, v___y_2026_, v___y_2027_, v___y_2028_, v___y_2029_);
lean_dec(v___y_2029_);
lean_dec_ref(v___y_2028_);
lean_dec(v___y_2027_);
lean_dec_ref(v___y_2026_);
lean_dec(v___y_2025_);
lean_dec_ref(v___y_2024_);
lean_dec_ref(v_bs_2022_);
return v_res_2031_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewBinderInfos___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__2(lean_object* v_00_u03b1_2032_, lean_object* v_bs_2033_, lean_object* v_k_2034_, lean_object* v___y_2035_, lean_object* v___y_2036_, lean_object* v___y_2037_, lean_object* v___y_2038_, lean_object* v___y_2039_, lean_object* v___y_2040_){
_start:
{
lean_object* v___x_2042_; 
v___x_2042_ = lp_mathlib_Lean_Meta_withNewBinderInfos___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__2___redArg(v_bs_2033_, v_k_2034_, v___y_2035_, v___y_2036_, v___y_2037_, v___y_2038_, v___y_2039_, v___y_2040_);
return v___x_2042_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewBinderInfos___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__2___boxed(lean_object* v_00_u03b1_2043_, lean_object* v_bs_2044_, lean_object* v_k_2045_, lean_object* v___y_2046_, lean_object* v___y_2047_, lean_object* v___y_2048_, lean_object* v___y_2049_, lean_object* v___y_2050_, lean_object* v___y_2051_, lean_object* v___y_2052_){
_start:
{
lean_object* v_res_2053_; 
v_res_2053_ = lp_mathlib_Lean_Meta_withNewBinderInfos___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__2(v_00_u03b1_2043_, v_bs_2044_, v_k_2045_, v___y_2046_, v___y_2047_, v___y_2048_, v___y_2049_, v___y_2050_, v___y_2051_);
lean_dec(v___y_2051_);
lean_dec_ref(v___y_2050_);
lean_dec(v___y_2049_);
lean_dec_ref(v___y_2048_);
lean_dec(v___y_2047_);
lean_dec_ref(v___y_2046_);
lean_dec_ref(v_bs_2044_);
return v_res_2053_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__3___redArg___lam__0(lean_object* v_k_2054_, lean_object* v___y_2055_, lean_object* v___y_2056_, lean_object* v_b_2057_, lean_object* v_c_2058_, lean_object* v___y_2059_, lean_object* v___y_2060_, lean_object* v___y_2061_, lean_object* v___y_2062_){
_start:
{
lean_object* v___x_2064_; 
lean_inc(v___y_2062_);
lean_inc_ref(v___y_2061_);
lean_inc(v___y_2060_);
lean_inc_ref(v___y_2059_);
lean_inc(v___y_2056_);
lean_inc_ref(v___y_2055_);
v___x_2064_ = lean_apply_9(v_k_2054_, v_b_2057_, v_c_2058_, v___y_2055_, v___y_2056_, v___y_2059_, v___y_2060_, v___y_2061_, v___y_2062_, lean_box(0));
return v___x_2064_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__3___redArg___lam__0___boxed(lean_object* v_k_2065_, lean_object* v___y_2066_, lean_object* v___y_2067_, lean_object* v_b_2068_, lean_object* v_c_2069_, lean_object* v___y_2070_, lean_object* v___y_2071_, lean_object* v___y_2072_, lean_object* v___y_2073_, lean_object* v___y_2074_){
_start:
{
lean_object* v_res_2075_; 
v_res_2075_ = lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__3___redArg___lam__0(v_k_2065_, v___y_2066_, v___y_2067_, v_b_2068_, v_c_2069_, v___y_2070_, v___y_2071_, v___y_2072_, v___y_2073_);
lean_dec(v___y_2073_);
lean_dec_ref(v___y_2072_);
lean_dec(v___y_2071_);
lean_dec_ref(v___y_2070_);
lean_dec(v___y_2067_);
lean_dec_ref(v___y_2066_);
return v_res_2075_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__3___redArg(lean_object* v_type_2076_, lean_object* v_maxFVars_x3f_2077_, lean_object* v_k_2078_, uint8_t v_cleanupAnnotations_2079_, uint8_t v_whnfType_2080_, lean_object* v___y_2081_, lean_object* v___y_2082_, lean_object* v___y_2083_, lean_object* v___y_2084_, lean_object* v___y_2085_, lean_object* v___y_2086_){
_start:
{
lean_object* v___f_2088_; lean_object* v___x_2089_; 
lean_inc(v___y_2082_);
lean_inc_ref(v___y_2081_);
v___f_2088_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__3___redArg___lam__0___boxed), 10, 3);
lean_closure_set(v___f_2088_, 0, v_k_2078_);
lean_closure_set(v___f_2088_, 1, v___y_2081_);
lean_closure_set(v___f_2088_, 2, v___y_2082_);
v___x_2089_ = l___private_Lean_Meta_Basic_0__Lean_Meta_forallTelescopeReducingAux(lean_box(0), v_type_2076_, v_maxFVars_x3f_2077_, v___f_2088_, v_cleanupAnnotations_2079_, v_whnfType_2080_, v___y_2083_, v___y_2084_, v___y_2085_, v___y_2086_);
if (lean_obj_tag(v___x_2089_) == 0)
{
return v___x_2089_;
}
else
{
lean_object* v_a_2090_; lean_object* v___x_2092_; uint8_t v_isShared_2093_; uint8_t v_isSharedCheck_2097_; 
v_a_2090_ = lean_ctor_get(v___x_2089_, 0);
v_isSharedCheck_2097_ = !lean_is_exclusive(v___x_2089_);
if (v_isSharedCheck_2097_ == 0)
{
v___x_2092_ = v___x_2089_;
v_isShared_2093_ = v_isSharedCheck_2097_;
goto v_resetjp_2091_;
}
else
{
lean_inc(v_a_2090_);
lean_dec(v___x_2089_);
v___x_2092_ = lean_box(0);
v_isShared_2093_ = v_isSharedCheck_2097_;
goto v_resetjp_2091_;
}
v_resetjp_2091_:
{
lean_object* v___x_2095_; 
if (v_isShared_2093_ == 0)
{
v___x_2095_ = v___x_2092_;
goto v_reusejp_2094_;
}
else
{
lean_object* v_reuseFailAlloc_2096_; 
v_reuseFailAlloc_2096_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2096_, 0, v_a_2090_);
v___x_2095_ = v_reuseFailAlloc_2096_;
goto v_reusejp_2094_;
}
v_reusejp_2094_:
{
return v___x_2095_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__3___redArg___boxed(lean_object* v_type_2098_, lean_object* v_maxFVars_x3f_2099_, lean_object* v_k_2100_, lean_object* v_cleanupAnnotations_2101_, lean_object* v_whnfType_2102_, lean_object* v___y_2103_, lean_object* v___y_2104_, lean_object* v___y_2105_, lean_object* v___y_2106_, lean_object* v___y_2107_, lean_object* v___y_2108_, lean_object* v___y_2109_){
_start:
{
uint8_t v_cleanupAnnotations_boxed_2110_; uint8_t v_whnfType_boxed_2111_; lean_object* v_res_2112_; 
v_cleanupAnnotations_boxed_2110_ = lean_unbox(v_cleanupAnnotations_2101_);
v_whnfType_boxed_2111_ = lean_unbox(v_whnfType_2102_);
v_res_2112_ = lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__3___redArg(v_type_2098_, v_maxFVars_x3f_2099_, v_k_2100_, v_cleanupAnnotations_boxed_2110_, v_whnfType_boxed_2111_, v___y_2103_, v___y_2104_, v___y_2105_, v___y_2106_, v___y_2107_, v___y_2108_);
lean_dec(v___y_2108_);
lean_dec_ref(v___y_2107_);
lean_dec(v___y_2106_);
lean_dec_ref(v___y_2105_);
lean_dec(v___y_2104_);
lean_dec_ref(v___y_2103_);
return v_res_2112_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__3(lean_object* v_00_u03b1_2113_, lean_object* v_type_2114_, lean_object* v_maxFVars_x3f_2115_, lean_object* v_k_2116_, uint8_t v_cleanupAnnotations_2117_, uint8_t v_whnfType_2118_, lean_object* v___y_2119_, lean_object* v___y_2120_, lean_object* v___y_2121_, lean_object* v___y_2122_, lean_object* v___y_2123_, lean_object* v___y_2124_){
_start:
{
lean_object* v___x_2126_; 
v___x_2126_ = lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__3___redArg(v_type_2114_, v_maxFVars_x3f_2115_, v_k_2116_, v_cleanupAnnotations_2117_, v_whnfType_2118_, v___y_2119_, v___y_2120_, v___y_2121_, v___y_2122_, v___y_2123_, v___y_2124_);
return v___x_2126_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__3___boxed(lean_object* v_00_u03b1_2127_, lean_object* v_type_2128_, lean_object* v_maxFVars_x3f_2129_, lean_object* v_k_2130_, lean_object* v_cleanupAnnotations_2131_, lean_object* v_whnfType_2132_, lean_object* v___y_2133_, lean_object* v___y_2134_, lean_object* v___y_2135_, lean_object* v___y_2136_, lean_object* v___y_2137_, lean_object* v___y_2138_, lean_object* v___y_2139_){
_start:
{
uint8_t v_cleanupAnnotations_boxed_2140_; uint8_t v_whnfType_boxed_2141_; lean_object* v_res_2142_; 
v_cleanupAnnotations_boxed_2140_ = lean_unbox(v_cleanupAnnotations_2131_);
v_whnfType_boxed_2141_ = lean_unbox(v_whnfType_2132_);
v_res_2142_ = lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__3(v_00_u03b1_2127_, v_type_2128_, v_maxFVars_x3f_2129_, v_k_2130_, v_cleanupAnnotations_boxed_2140_, v_whnfType_boxed_2141_, v___y_2133_, v___y_2134_, v___y_2135_, v___y_2136_, v___y_2137_, v___y_2138_);
lean_dec(v___y_2138_);
lean_dec_ref(v___y_2137_);
lean_dec(v___y_2136_);
lean_dec_ref(v___y_2135_);
lean_dec(v___y_2134_);
lean_dec_ref(v___y_2133_);
return v_res_2142_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__5___redArg(lean_object* v_a_2143_, lean_object* v___y_2144_, lean_object* v___y_2145_, lean_object* v___y_2146_, lean_object* v___y_2147_, lean_object* v___y_2148_, lean_object* v___y_2149_){
_start:
{
lean_object* v___x_2151_; 
v___x_2151_ = l_Lean_Elab_Term_withoutErrToSorryImp___redArg(v_a_2143_, v___y_2144_, v___y_2145_, v___y_2146_, v___y_2147_, v___y_2148_, v___y_2149_);
return v___x_2151_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__5___redArg___boxed(lean_object* v_a_2152_, lean_object* v___y_2153_, lean_object* v___y_2154_, lean_object* v___y_2155_, lean_object* v___y_2156_, lean_object* v___y_2157_, lean_object* v___y_2158_, lean_object* v___y_2159_){
_start:
{
lean_object* v_res_2160_; 
v_res_2160_ = lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__5___redArg(v_a_2152_, v___y_2153_, v___y_2154_, v___y_2155_, v___y_2156_, v___y_2157_, v___y_2158_);
lean_dec(v___y_2158_);
lean_dec_ref(v___y_2157_);
lean_dec(v___y_2156_);
lean_dec_ref(v___y_2155_);
lean_dec(v___y_2154_);
lean_dec_ref(v___y_2153_);
return v_res_2160_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__5(lean_object* v_00_u03b1_2161_, lean_object* v_a_2162_, lean_object* v___y_2163_, lean_object* v___y_2164_, lean_object* v___y_2165_, lean_object* v___y_2166_, lean_object* v___y_2167_, lean_object* v___y_2168_){
_start:
{
lean_object* v___x_2170_; 
v___x_2170_ = l_Lean_Elab_Term_withoutErrToSorryImp___redArg(v_a_2162_, v___y_2163_, v___y_2164_, v___y_2165_, v___y_2166_, v___y_2167_, v___y_2168_);
return v___x_2170_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__5___boxed(lean_object* v_00_u03b1_2171_, lean_object* v_a_2172_, lean_object* v___y_2173_, lean_object* v___y_2174_, lean_object* v___y_2175_, lean_object* v___y_2176_, lean_object* v___y_2177_, lean_object* v___y_2178_, lean_object* v___y_2179_){
_start:
{
lean_object* v_res_2180_; 
v_res_2180_ = lp_mathlib_Lean_Elab_Term_withoutErrToSorry___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__5(v_00_u03b1_2171_, v_a_2172_, v___y_2173_, v___y_2174_, v___y_2175_, v___y_2176_, v___y_2177_, v___y_2178_);
lean_dec(v___y_2178_);
lean_dec_ref(v___y_2177_);
lean_dec(v___y_2176_);
lean_dec_ref(v___y_2175_);
lean_dec(v___y_2174_);
lean_dec_ref(v___y_2173_);
return v_res_2180_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__6___redArg(lean_object* v_k_2181_, uint8_t v_allowLevelAssignments_2182_, lean_object* v___y_2183_, lean_object* v___y_2184_, lean_object* v___y_2185_, lean_object* v___y_2186_, lean_object* v___y_2187_, lean_object* v___y_2188_){
_start:
{
lean_object* v___f_2190_; lean_object* v___x_2191_; 
lean_inc(v___y_2184_);
lean_inc_ref(v___y_2183_);
v___f_2190_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_withNewBinderInfos___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__2___redArg___lam__0___boxed), 8, 3);
lean_closure_set(v___f_2190_, 0, v_k_2181_);
lean_closure_set(v___f_2190_, 1, v___y_2183_);
lean_closure_set(v___f_2190_, 2, v___y_2184_);
v___x_2191_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withNewMCtxDepthImp(lean_box(0), v_allowLevelAssignments_2182_, v___f_2190_, v___y_2185_, v___y_2186_, v___y_2187_, v___y_2188_);
if (lean_obj_tag(v___x_2191_) == 0)
{
return v___x_2191_;
}
else
{
lean_object* v_a_2192_; lean_object* v___x_2194_; uint8_t v_isShared_2195_; uint8_t v_isSharedCheck_2199_; 
v_a_2192_ = lean_ctor_get(v___x_2191_, 0);
v_isSharedCheck_2199_ = !lean_is_exclusive(v___x_2191_);
if (v_isSharedCheck_2199_ == 0)
{
v___x_2194_ = v___x_2191_;
v_isShared_2195_ = v_isSharedCheck_2199_;
goto v_resetjp_2193_;
}
else
{
lean_inc(v_a_2192_);
lean_dec(v___x_2191_);
v___x_2194_ = lean_box(0);
v_isShared_2195_ = v_isSharedCheck_2199_;
goto v_resetjp_2193_;
}
v_resetjp_2193_:
{
lean_object* v___x_2197_; 
if (v_isShared_2195_ == 0)
{
v___x_2197_ = v___x_2194_;
goto v_reusejp_2196_;
}
else
{
lean_object* v_reuseFailAlloc_2198_; 
v_reuseFailAlloc_2198_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2198_, 0, v_a_2192_);
v___x_2197_ = v_reuseFailAlloc_2198_;
goto v_reusejp_2196_;
}
v_reusejp_2196_:
{
return v___x_2197_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__6___redArg___boxed(lean_object* v_k_2200_, lean_object* v_allowLevelAssignments_2201_, lean_object* v___y_2202_, lean_object* v___y_2203_, lean_object* v___y_2204_, lean_object* v___y_2205_, lean_object* v___y_2206_, lean_object* v___y_2207_, lean_object* v___y_2208_){
_start:
{
uint8_t v_allowLevelAssignments_boxed_2209_; lean_object* v_res_2210_; 
v_allowLevelAssignments_boxed_2209_ = lean_unbox(v_allowLevelAssignments_2201_);
v_res_2210_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__6___redArg(v_k_2200_, v_allowLevelAssignments_boxed_2209_, v___y_2202_, v___y_2203_, v___y_2204_, v___y_2205_, v___y_2206_, v___y_2207_);
lean_dec(v___y_2207_);
lean_dec_ref(v___y_2206_);
lean_dec(v___y_2205_);
lean_dec_ref(v___y_2204_);
lean_dec(v___y_2203_);
lean_dec_ref(v___y_2202_);
return v_res_2210_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__6(lean_object* v_00_u03b1_2211_, lean_object* v_k_2212_, uint8_t v_allowLevelAssignments_2213_, lean_object* v___y_2214_, lean_object* v___y_2215_, lean_object* v___y_2216_, lean_object* v___y_2217_, lean_object* v___y_2218_, lean_object* v___y_2219_){
_start:
{
lean_object* v___x_2221_; 
v___x_2221_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__6___redArg(v_k_2212_, v_allowLevelAssignments_2213_, v___y_2214_, v___y_2215_, v___y_2216_, v___y_2217_, v___y_2218_, v___y_2219_);
return v___x_2221_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__6___boxed(lean_object* v_00_u03b1_2222_, lean_object* v_k_2223_, lean_object* v_allowLevelAssignments_2224_, lean_object* v___y_2225_, lean_object* v___y_2226_, lean_object* v___y_2227_, lean_object* v___y_2228_, lean_object* v___y_2229_, lean_object* v___y_2230_, lean_object* v___y_2231_){
_start:
{
uint8_t v_allowLevelAssignments_boxed_2232_; lean_object* v_res_2233_; 
v_allowLevelAssignments_boxed_2232_ = lean_unbox(v_allowLevelAssignments_2224_);
v_res_2233_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__6(v_00_u03b1_2222_, v_k_2223_, v_allowLevelAssignments_boxed_2232_, v___y_2225_, v___y_2226_, v___y_2227_, v___y_2228_, v___y_2229_, v___y_2230_);
lean_dec(v___y_2230_);
lean_dec_ref(v___y_2229_);
lean_dec(v___y_2228_);
lean_dec_ref(v___y_2227_);
lean_dec(v___y_2226_);
lean_dec_ref(v___y_2225_);
return v_res_2233_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go___lam__0(lean_object* v_args_2234_, lean_object* v___x_2235_, uint8_t v___x_2236_, uint8_t v___x_2237_, uint8_t v___x_2238_, lean_object* v___y_2239_, lean_object* v___y_2240_, lean_object* v___y_2241_, lean_object* v___y_2242_, lean_object* v___y_2243_, lean_object* v___y_2244_){
_start:
{
lean_object* v___x_2246_; 
v___x_2246_ = l_Lean_Meta_mkLambdaFVars(v_args_2234_, v___x_2235_, v___x_2236_, v___x_2237_, v___x_2236_, v___x_2237_, v___x_2238_, v___y_2241_, v___y_2242_, v___y_2243_, v___y_2244_);
return v___x_2246_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go___lam__0___boxed(lean_object* v_args_2247_, lean_object* v___x_2248_, lean_object* v___x_2249_, lean_object* v___x_2250_, lean_object* v___x_2251_, lean_object* v___y_2252_, lean_object* v___y_2253_, lean_object* v___y_2254_, lean_object* v___y_2255_, lean_object* v___y_2256_, lean_object* v___y_2257_, lean_object* v___y_2258_){
_start:
{
uint8_t v___x_11174__boxed_2259_; uint8_t v___x_11175__boxed_2260_; uint8_t v___x_11176__boxed_2261_; lean_object* v_res_2262_; 
v___x_11174__boxed_2259_ = lean_unbox(v___x_2249_);
v___x_11175__boxed_2260_ = lean_unbox(v___x_2250_);
v___x_11176__boxed_2261_ = lean_unbox(v___x_2251_);
v_res_2262_ = lp_mathlib___private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go___lam__0(v_args_2247_, v___x_2248_, v___x_11174__boxed_2259_, v___x_11175__boxed_2260_, v___x_11176__boxed_2261_, v___y_2252_, v___y_2253_, v___y_2254_, v___y_2255_, v___y_2256_, v___y_2257_);
lean_dec(v___y_2257_);
lean_dec_ref(v___y_2256_);
lean_dec(v___y_2255_);
lean_dec_ref(v___y_2254_);
lean_dec(v___y_2253_);
lean_dec_ref(v___y_2252_);
lean_dec_ref(v_args_2247_);
return v_res_2262_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__1_spec__1___redArg(uint8_t v___x_2263_, lean_object* v_as_2264_, size_t v_i_2265_, size_t v_stop_2266_, lean_object* v_b_2267_, lean_object* v___y_2268_, lean_object* v___y_2269_, lean_object* v___y_2270_, lean_object* v___y_2271_){
_start:
{
uint8_t v___x_2273_; 
v___x_2273_ = lean_usize_dec_eq(v_i_2265_, v_stop_2266_);
if (v___x_2273_ == 0)
{
lean_object* v___x_2274_; lean_object* v___x_2275_; 
v___x_2274_ = lean_array_uget_borrowed(v_as_2264_, v_i_2265_);
lean_inc(v___y_2271_);
lean_inc_ref(v___y_2270_);
lean_inc(v___y_2269_);
lean_inc_ref(v___y_2268_);
lean_inc(v___x_2274_);
v___x_2275_ = lean_infer_type(v___x_2274_, v___y_2268_, v___y_2269_, v___y_2270_, v___y_2271_);
if (lean_obj_tag(v___x_2275_) == 0)
{
lean_object* v_a_2276_; lean_object* v___x_2277_; 
v_a_2276_ = lean_ctor_get(v___x_2275_, 0);
lean_inc(v_a_2276_);
lean_dec_ref_known(v___x_2275_, 1);
v___x_2277_ = l_Lean_Meta_isClass_x3f(v_a_2276_, v___y_2268_, v___y_2269_, v___y_2270_, v___y_2271_);
if (lean_obj_tag(v___x_2277_) == 0)
{
lean_object* v_a_2278_; lean_object* v_a_2280_; 
v_a_2278_ = lean_ctor_get(v___x_2277_, 0);
lean_inc(v_a_2278_);
lean_dec_ref_known(v___x_2277_, 1);
if (lean_obj_tag(v_a_2278_) == 0)
{
v_a_2280_ = v_b_2267_;
goto v___jp_2279_;
}
else
{
lean_dec_ref_known(v_a_2278_, 1);
if (v___x_2263_ == 0)
{
v_a_2280_ = v_b_2267_;
goto v___jp_2279_;
}
else
{
lean_object* v___x_2284_; uint8_t v___x_2285_; lean_object* v___x_2286_; lean_object* v___x_2287_; lean_object* v___x_2288_; 
v___x_2284_ = l_Lean_Expr_fvarId_x21(v___x_2274_);
v___x_2285_ = 3;
v___x_2286_ = lean_box(v___x_2285_);
v___x_2287_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2287_, 0, v___x_2284_);
lean_ctor_set(v___x_2287_, 1, v___x_2286_);
v___x_2288_ = lean_array_push(v_b_2267_, v___x_2287_);
v_a_2280_ = v___x_2288_;
goto v___jp_2279_;
}
}
v___jp_2279_:
{
size_t v___x_2281_; size_t v___x_2282_; 
v___x_2281_ = ((size_t)1ULL);
v___x_2282_ = lean_usize_add(v_i_2265_, v___x_2281_);
v_i_2265_ = v___x_2282_;
v_b_2267_ = v_a_2280_;
goto _start;
}
}
else
{
lean_object* v_a_2289_; lean_object* v___x_2291_; uint8_t v_isShared_2292_; uint8_t v_isSharedCheck_2296_; 
lean_dec_ref(v_b_2267_);
v_a_2289_ = lean_ctor_get(v___x_2277_, 0);
v_isSharedCheck_2296_ = !lean_is_exclusive(v___x_2277_);
if (v_isSharedCheck_2296_ == 0)
{
v___x_2291_ = v___x_2277_;
v_isShared_2292_ = v_isSharedCheck_2296_;
goto v_resetjp_2290_;
}
else
{
lean_inc(v_a_2289_);
lean_dec(v___x_2277_);
v___x_2291_ = lean_box(0);
v_isShared_2292_ = v_isSharedCheck_2296_;
goto v_resetjp_2290_;
}
v_resetjp_2290_:
{
lean_object* v___x_2294_; 
if (v_isShared_2292_ == 0)
{
v___x_2294_ = v___x_2291_;
goto v_reusejp_2293_;
}
else
{
lean_object* v_reuseFailAlloc_2295_; 
v_reuseFailAlloc_2295_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2295_, 0, v_a_2289_);
v___x_2294_ = v_reuseFailAlloc_2295_;
goto v_reusejp_2293_;
}
v_reusejp_2293_:
{
return v___x_2294_;
}
}
}
}
else
{
lean_object* v_a_2297_; lean_object* v___x_2299_; uint8_t v_isShared_2300_; uint8_t v_isSharedCheck_2304_; 
lean_dec_ref(v_b_2267_);
v_a_2297_ = lean_ctor_get(v___x_2275_, 0);
v_isSharedCheck_2304_ = !lean_is_exclusive(v___x_2275_);
if (v_isSharedCheck_2304_ == 0)
{
v___x_2299_ = v___x_2275_;
v_isShared_2300_ = v_isSharedCheck_2304_;
goto v_resetjp_2298_;
}
else
{
lean_inc(v_a_2297_);
lean_dec(v___x_2275_);
v___x_2299_ = lean_box(0);
v_isShared_2300_ = v_isSharedCheck_2304_;
goto v_resetjp_2298_;
}
v_resetjp_2298_:
{
lean_object* v___x_2302_; 
if (v_isShared_2300_ == 0)
{
v___x_2302_ = v___x_2299_;
goto v_reusejp_2301_;
}
else
{
lean_object* v_reuseFailAlloc_2303_; 
v_reuseFailAlloc_2303_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2303_, 0, v_a_2297_);
v___x_2302_ = v_reuseFailAlloc_2303_;
goto v_reusejp_2301_;
}
v_reusejp_2301_:
{
return v___x_2302_;
}
}
}
}
else
{
lean_object* v___x_2305_; 
v___x_2305_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2305_, 0, v_b_2267_);
return v___x_2305_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__1_spec__1___redArg___boxed(lean_object* v___x_2306_, lean_object* v_as_2307_, lean_object* v_i_2308_, lean_object* v_stop_2309_, lean_object* v_b_2310_, lean_object* v___y_2311_, lean_object* v___y_2312_, lean_object* v___y_2313_, lean_object* v___y_2314_, lean_object* v___y_2315_){
_start:
{
uint8_t v___x_11208__boxed_2316_; size_t v_i_boxed_2317_; size_t v_stop_boxed_2318_; lean_object* v_res_2319_; 
v___x_11208__boxed_2316_ = lean_unbox(v___x_2306_);
v_i_boxed_2317_ = lean_unbox_usize(v_i_2308_);
lean_dec(v_i_2308_);
v_stop_boxed_2318_ = lean_unbox_usize(v_stop_2309_);
lean_dec(v_stop_2309_);
v_res_2319_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__1_spec__1___redArg(v___x_11208__boxed_2316_, v_as_2307_, v_i_boxed_2317_, v_stop_boxed_2318_, v_b_2310_, v___y_2311_, v___y_2312_, v___y_2313_, v___y_2314_);
lean_dec(v___y_2314_);
lean_dec_ref(v___y_2313_);
lean_dec(v___y_2312_);
lean_dec_ref(v___y_2311_);
lean_dec_ref(v_as_2307_);
return v_res_2319_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_filterMapM___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__1(uint8_t v___x_2322_, lean_object* v_as_2323_, lean_object* v_start_2324_, lean_object* v_stop_2325_, lean_object* v___y_2326_, lean_object* v___y_2327_, lean_object* v___y_2328_, lean_object* v___y_2329_, lean_object* v___y_2330_, lean_object* v___y_2331_){
_start:
{
lean_object* v___x_2333_; uint8_t v___x_2334_; 
v___x_2333_ = ((lean_object*)(lp_mathlib_Array_filterMapM___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__1___closed__0));
v___x_2334_ = lean_nat_dec_lt(v_start_2324_, v_stop_2325_);
if (v___x_2334_ == 0)
{
lean_object* v___x_2335_; 
v___x_2335_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2335_, 0, v___x_2333_);
return v___x_2335_;
}
else
{
lean_object* v___x_2336_; uint8_t v___x_2337_; 
v___x_2336_ = lean_array_get_size(v_as_2323_);
v___x_2337_ = lean_nat_dec_le(v_stop_2325_, v___x_2336_);
if (v___x_2337_ == 0)
{
uint8_t v___x_2338_; 
v___x_2338_ = lean_nat_dec_lt(v_start_2324_, v___x_2336_);
if (v___x_2338_ == 0)
{
lean_object* v___x_2339_; 
v___x_2339_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2339_, 0, v___x_2333_);
return v___x_2339_;
}
else
{
size_t v___x_2340_; size_t v___x_2341_; lean_object* v___x_2342_; 
v___x_2340_ = lean_usize_of_nat(v_start_2324_);
v___x_2341_ = lean_usize_of_nat(v___x_2336_);
v___x_2342_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__1_spec__1___redArg(v___x_2322_, v_as_2323_, v___x_2340_, v___x_2341_, v___x_2333_, v___y_2328_, v___y_2329_, v___y_2330_, v___y_2331_);
return v___x_2342_;
}
}
else
{
size_t v___x_2343_; size_t v___x_2344_; lean_object* v___x_2345_; 
v___x_2343_ = lean_usize_of_nat(v_start_2324_);
v___x_2344_ = lean_usize_of_nat(v_stop_2325_);
v___x_2345_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__1_spec__1___redArg(v___x_2322_, v_as_2323_, v___x_2343_, v___x_2344_, v___x_2333_, v___y_2328_, v___y_2329_, v___y_2330_, v___y_2331_);
return v___x_2345_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Array_filterMapM___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__1___boxed(lean_object* v___x_2346_, lean_object* v_as_2347_, lean_object* v_start_2348_, lean_object* v_stop_2349_, lean_object* v___y_2350_, lean_object* v___y_2351_, lean_object* v___y_2352_, lean_object* v___y_2353_, lean_object* v___y_2354_, lean_object* v___y_2355_, lean_object* v___y_2356_){
_start:
{
uint8_t v___x_11296__boxed_2357_; lean_object* v_res_2358_; 
v___x_11296__boxed_2357_ = lean_unbox(v___x_2346_);
v_res_2358_ = lp_mathlib_Array_filterMapM___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__1(v___x_11296__boxed_2357_, v_as_2347_, v_start_2348_, v_stop_2349_, v___y_2350_, v___y_2351_, v___y_2352_, v___y_2353_, v___y_2354_, v___y_2355_);
lean_dec(v___y_2355_);
lean_dec_ref(v___y_2354_);
lean_dec(v___y_2353_);
lean_dec_ref(v___y_2352_);
lean_dec(v___y_2351_);
lean_dec_ref(v___y_2350_);
lean_dec(v_stop_2349_);
lean_dec(v_start_2348_);
lean_dec_ref(v_as_2347_);
return v_res_2358_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go___lam__1(uint8_t v___x_2359_, lean_object* v_expr_2360_, uint8_t v___x_2361_, lean_object* v_args_2362_, lean_object* v_x_2363_, lean_object* v___y_2364_, lean_object* v___y_2365_, lean_object* v___y_2366_, lean_object* v___y_2367_, lean_object* v___y_2368_, lean_object* v___y_2369_){
_start:
{
lean_object* v___x_2371_; lean_object* v___x_2372_; lean_object* v___x_2373_; 
v___x_2371_ = lean_unsigned_to_nat(0u);
v___x_2372_ = lean_array_get_size(v_args_2362_);
v___x_2373_ = lp_mathlib_Array_filterMapM___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__1(v___x_2359_, v_args_2362_, v___x_2371_, v___x_2372_, v___y_2364_, v___y_2365_, v___y_2366_, v___y_2367_, v___y_2368_, v___y_2369_);
if (lean_obj_tag(v___x_2373_) == 0)
{
lean_object* v_a_2374_; lean_object* v___x_2375_; uint8_t v___x_2376_; uint8_t v___x_2377_; lean_object* v___x_2378_; lean_object* v___x_2379_; lean_object* v___x_2380_; lean_object* v___f_2381_; lean_object* v___x_2382_; 
v_a_2374_ = lean_ctor_get(v___x_2373_, 0);
lean_inc(v_a_2374_);
lean_dec_ref_known(v___x_2373_, 1);
lean_inc_ref(v_args_2362_);
v___x_2375_ = l_Lean_Expr_beta(v_expr_2360_, v_args_2362_);
v___x_2376_ = 0;
v___x_2377_ = 1;
v___x_2378_ = lean_box(v___x_2376_);
v___x_2379_ = lean_box(v___x_2361_);
v___x_2380_ = lean_box(v___x_2377_);
v___f_2381_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go___lam__0___boxed), 12, 5);
lean_closure_set(v___f_2381_, 0, v_args_2362_);
lean_closure_set(v___f_2381_, 1, v___x_2375_);
lean_closure_set(v___f_2381_, 2, v___x_2378_);
lean_closure_set(v___f_2381_, 3, v___x_2379_);
lean_closure_set(v___f_2381_, 4, v___x_2380_);
v___x_2382_ = lp_mathlib_Lean_Meta_withNewBinderInfos___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__2___redArg(v_a_2374_, v___f_2381_, v___y_2364_, v___y_2365_, v___y_2366_, v___y_2367_, v___y_2368_, v___y_2369_);
lean_dec(v_a_2374_);
return v___x_2382_;
}
else
{
lean_object* v_a_2383_; lean_object* v___x_2385_; uint8_t v_isShared_2386_; uint8_t v_isSharedCheck_2390_; 
lean_dec_ref(v_args_2362_);
lean_dec_ref(v_expr_2360_);
v_a_2383_ = lean_ctor_get(v___x_2373_, 0);
v_isSharedCheck_2390_ = !lean_is_exclusive(v___x_2373_);
if (v_isSharedCheck_2390_ == 0)
{
v___x_2385_ = v___x_2373_;
v_isShared_2386_ = v_isSharedCheck_2390_;
goto v_resetjp_2384_;
}
else
{
lean_inc(v_a_2383_);
lean_dec(v___x_2373_);
v___x_2385_ = lean_box(0);
v_isShared_2386_ = v_isSharedCheck_2390_;
goto v_resetjp_2384_;
}
v_resetjp_2384_:
{
lean_object* v___x_2388_; 
if (v_isShared_2386_ == 0)
{
v___x_2388_ = v___x_2385_;
goto v_reusejp_2387_;
}
else
{
lean_object* v_reuseFailAlloc_2389_; 
v_reuseFailAlloc_2389_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2389_, 0, v_a_2383_);
v___x_2388_ = v_reuseFailAlloc_2389_;
goto v_reusejp_2387_;
}
v_reusejp_2387_:
{
return v___x_2388_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go___lam__1___boxed(lean_object* v___x_2391_, lean_object* v_expr_2392_, lean_object* v___x_2393_, lean_object* v_args_2394_, lean_object* v_x_2395_, lean_object* v___y_2396_, lean_object* v___y_2397_, lean_object* v___y_2398_, lean_object* v___y_2399_, lean_object* v___y_2400_, lean_object* v___y_2401_, lean_object* v___y_2402_){
_start:
{
uint8_t v___x_11348__boxed_2403_; uint8_t v___x_11349__boxed_2404_; lean_object* v_res_2405_; 
v___x_11348__boxed_2403_ = lean_unbox(v___x_2391_);
v___x_11349__boxed_2404_ = lean_unbox(v___x_2393_);
v_res_2405_ = lp_mathlib___private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go___lam__1(v___x_11348__boxed_2403_, v_expr_2392_, v___x_11349__boxed_2404_, v_args_2394_, v_x_2395_, v___y_2396_, v___y_2397_, v___y_2398_, v___y_2399_, v___y_2400_, v___y_2401_);
lean_dec(v___y_2401_);
lean_dec_ref(v___y_2400_);
lean_dec(v___y_2399_);
lean_dec_ref(v___y_2398_);
lean_dec(v___y_2397_);
lean_dec_ref(v___y_2396_);
lean_dec_ref(v_x_2395_);
return v_res_2405_;
}
}
static lean_object* _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__4_spec__5_spec__9___closed__0(void){
_start:
{
lean_object* v___x_2406_; lean_object* v___x_2407_; 
v___x_2406_ = lean_box(1);
v___x_2407_ = l_Lean_MessageData_ofFormat(v___x_2406_);
return v___x_2407_;
}
}
static lean_object* _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__4_spec__5_spec__9___closed__3(void){
_start:
{
lean_object* v___x_2411_; lean_object* v___x_2412_; 
v___x_2411_ = ((lean_object*)(lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__4_spec__5_spec__9___closed__2));
v___x_2412_ = l_Lean_MessageData_ofFormat(v___x_2411_);
return v___x_2412_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__4_spec__5_spec__9(lean_object* v_x_2413_, lean_object* v_x_2414_){
_start:
{
if (lean_obj_tag(v_x_2414_) == 0)
{
return v_x_2413_;
}
else
{
lean_object* v_head_2415_; lean_object* v_tail_2416_; lean_object* v___x_2418_; uint8_t v_isShared_2419_; uint8_t v_isSharedCheck_2438_; 
v_head_2415_ = lean_ctor_get(v_x_2414_, 0);
v_tail_2416_ = lean_ctor_get(v_x_2414_, 1);
v_isSharedCheck_2438_ = !lean_is_exclusive(v_x_2414_);
if (v_isSharedCheck_2438_ == 0)
{
v___x_2418_ = v_x_2414_;
v_isShared_2419_ = v_isSharedCheck_2438_;
goto v_resetjp_2417_;
}
else
{
lean_inc(v_tail_2416_);
lean_inc(v_head_2415_);
lean_dec(v_x_2414_);
v___x_2418_ = lean_box(0);
v_isShared_2419_ = v_isSharedCheck_2438_;
goto v_resetjp_2417_;
}
v_resetjp_2417_:
{
lean_object* v_before_2420_; lean_object* v___x_2422_; uint8_t v_isShared_2423_; uint8_t v_isSharedCheck_2436_; 
v_before_2420_ = lean_ctor_get(v_head_2415_, 0);
v_isSharedCheck_2436_ = !lean_is_exclusive(v_head_2415_);
if (v_isSharedCheck_2436_ == 0)
{
lean_object* v_unused_2437_; 
v_unused_2437_ = lean_ctor_get(v_head_2415_, 1);
lean_dec(v_unused_2437_);
v___x_2422_ = v_head_2415_;
v_isShared_2423_ = v_isSharedCheck_2436_;
goto v_resetjp_2421_;
}
else
{
lean_inc(v_before_2420_);
lean_dec(v_head_2415_);
v___x_2422_ = lean_box(0);
v_isShared_2423_ = v_isSharedCheck_2436_;
goto v_resetjp_2421_;
}
v_resetjp_2421_:
{
lean_object* v___x_2424_; lean_object* v___x_2426_; 
v___x_2424_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__4_spec__5_spec__9___closed__0, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__4_spec__5_spec__9___closed__0_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__4_spec__5_spec__9___closed__0);
if (v_isShared_2423_ == 0)
{
lean_ctor_set_tag(v___x_2422_, 7);
lean_ctor_set(v___x_2422_, 1, v___x_2424_);
lean_ctor_set(v___x_2422_, 0, v_x_2413_);
v___x_2426_ = v___x_2422_;
goto v_reusejp_2425_;
}
else
{
lean_object* v_reuseFailAlloc_2435_; 
v_reuseFailAlloc_2435_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2435_, 0, v_x_2413_);
lean_ctor_set(v_reuseFailAlloc_2435_, 1, v___x_2424_);
v___x_2426_ = v_reuseFailAlloc_2435_;
goto v_reusejp_2425_;
}
v_reusejp_2425_:
{
lean_object* v___x_2427_; lean_object* v___x_2429_; 
v___x_2427_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__4_spec__5_spec__9___closed__3, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__4_spec__5_spec__9___closed__3_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__4_spec__5_spec__9___closed__3);
if (v_isShared_2419_ == 0)
{
lean_ctor_set_tag(v___x_2418_, 7);
lean_ctor_set(v___x_2418_, 1, v___x_2427_);
lean_ctor_set(v___x_2418_, 0, v___x_2426_);
v___x_2429_ = v___x_2418_;
goto v_reusejp_2428_;
}
else
{
lean_object* v_reuseFailAlloc_2434_; 
v_reuseFailAlloc_2434_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2434_, 0, v___x_2426_);
lean_ctor_set(v_reuseFailAlloc_2434_, 1, v___x_2427_);
v___x_2429_ = v_reuseFailAlloc_2434_;
goto v_reusejp_2428_;
}
v_reusejp_2428_:
{
lean_object* v___x_2430_; lean_object* v___x_2431_; lean_object* v___x_2432_; 
v___x_2430_ = l_Lean_MessageData_ofSyntax(v_before_2420_);
v___x_2431_ = l_Lean_indentD(v___x_2430_);
v___x_2432_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2432_, 0, v___x_2429_);
lean_ctor_set(v___x_2432_, 1, v___x_2431_);
v_x_2413_ = v___x_2432_;
v_x_2414_ = v_tail_2416_;
goto _start;
}
}
}
}
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__4_spec__5_spec__8(lean_object* v_opts_2439_, lean_object* v_opt_2440_){
_start:
{
lean_object* v_name_2441_; lean_object* v_defValue_2442_; lean_object* v_map_2443_; lean_object* v___x_2444_; 
v_name_2441_ = lean_ctor_get(v_opt_2440_, 0);
v_defValue_2442_ = lean_ctor_get(v_opt_2440_, 1);
v_map_2443_ = lean_ctor_get(v_opts_2439_, 0);
v___x_2444_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_2443_, v_name_2441_);
if (lean_obj_tag(v___x_2444_) == 0)
{
uint8_t v___x_2445_; 
v___x_2445_ = lean_unbox(v_defValue_2442_);
return v___x_2445_;
}
else
{
lean_object* v_val_2446_; 
v_val_2446_ = lean_ctor_get(v___x_2444_, 0);
lean_inc(v_val_2446_);
lean_dec_ref_known(v___x_2444_, 1);
if (lean_obj_tag(v_val_2446_) == 1)
{
uint8_t v_v_2447_; 
v_v_2447_ = lean_ctor_get_uint8(v_val_2446_, 0);
lean_dec_ref_known(v_val_2446_, 0);
return v_v_2447_;
}
else
{
uint8_t v___x_2448_; 
lean_dec(v_val_2446_);
v___x_2448_ = lean_unbox(v_defValue_2442_);
return v___x_2448_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__4_spec__5_spec__8___boxed(lean_object* v_opts_2449_, lean_object* v_opt_2450_){
_start:
{
uint8_t v_res_2451_; lean_object* v_r_2452_; 
v_res_2451_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__4_spec__5_spec__8(v_opts_2449_, v_opt_2450_);
lean_dec_ref(v_opt_2450_);
lean_dec_ref(v_opts_2449_);
v_r_2452_ = lean_box(v_res_2451_);
return v_r_2452_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__4_spec__5___redArg___closed__2(void){
_start:
{
lean_object* v___x_2456_; lean_object* v___x_2457_; 
v___x_2456_ = ((lean_object*)(lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__4_spec__5___redArg___closed__1));
v___x_2457_ = l_Lean_MessageData_ofFormat(v___x_2456_);
return v___x_2457_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__4_spec__5___redArg(lean_object* v_msgData_2458_, lean_object* v_macroStack_2459_, lean_object* v___y_2460_){
_start:
{
lean_object* v_options_2462_; lean_object* v___x_2463_; uint8_t v___x_2464_; 
v_options_2462_ = lean_ctor_get(v___y_2460_, 2);
v___x_2463_ = l_Lean_Elab_pp_macroStack;
v___x_2464_ = lp_mathlib_Lean_Option_get___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__4_spec__5_spec__8(v_options_2462_, v___x_2463_);
if (v___x_2464_ == 0)
{
lean_object* v___x_2465_; 
lean_dec(v_macroStack_2459_);
v___x_2465_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2465_, 0, v_msgData_2458_);
return v___x_2465_;
}
else
{
if (lean_obj_tag(v_macroStack_2459_) == 0)
{
lean_object* v___x_2466_; 
v___x_2466_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2466_, 0, v_msgData_2458_);
return v___x_2466_;
}
else
{
lean_object* v_head_2467_; lean_object* v_after_2468_; lean_object* v___x_2470_; uint8_t v_isShared_2471_; uint8_t v_isSharedCheck_2483_; 
v_head_2467_ = lean_ctor_get(v_macroStack_2459_, 0);
lean_inc(v_head_2467_);
v_after_2468_ = lean_ctor_get(v_head_2467_, 1);
v_isSharedCheck_2483_ = !lean_is_exclusive(v_head_2467_);
if (v_isSharedCheck_2483_ == 0)
{
lean_object* v_unused_2484_; 
v_unused_2484_ = lean_ctor_get(v_head_2467_, 0);
lean_dec(v_unused_2484_);
v___x_2470_ = v_head_2467_;
v_isShared_2471_ = v_isSharedCheck_2483_;
goto v_resetjp_2469_;
}
else
{
lean_inc(v_after_2468_);
lean_dec(v_head_2467_);
v___x_2470_ = lean_box(0);
v_isShared_2471_ = v_isSharedCheck_2483_;
goto v_resetjp_2469_;
}
v_resetjp_2469_:
{
lean_object* v___x_2472_; lean_object* v___x_2474_; 
v___x_2472_ = lean_obj_once(&lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__4_spec__5_spec__9___closed__0, &lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__4_spec__5_spec__9___closed__0_once, _init_lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__4_spec__5_spec__9___closed__0);
if (v_isShared_2471_ == 0)
{
lean_ctor_set_tag(v___x_2470_, 7);
lean_ctor_set(v___x_2470_, 1, v___x_2472_);
lean_ctor_set(v___x_2470_, 0, v_msgData_2458_);
v___x_2474_ = v___x_2470_;
goto v_reusejp_2473_;
}
else
{
lean_object* v_reuseFailAlloc_2482_; 
v_reuseFailAlloc_2482_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2482_, 0, v_msgData_2458_);
lean_ctor_set(v_reuseFailAlloc_2482_, 1, v___x_2472_);
v___x_2474_ = v_reuseFailAlloc_2482_;
goto v_reusejp_2473_;
}
v_reusejp_2473_:
{
lean_object* v___x_2475_; lean_object* v___x_2476_; lean_object* v___x_2477_; lean_object* v___x_2478_; lean_object* v_msgData_2479_; lean_object* v___x_2480_; lean_object* v___x_2481_; 
v___x_2475_ = lean_obj_once(&lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__4_spec__5___redArg___closed__2, &lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__4_spec__5___redArg___closed__2_once, _init_lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__4_spec__5___redArg___closed__2);
v___x_2476_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2476_, 0, v___x_2474_);
lean_ctor_set(v___x_2476_, 1, v___x_2475_);
v___x_2477_ = l_Lean_MessageData_ofSyntax(v_after_2468_);
v___x_2478_ = l_Lean_indentD(v___x_2477_);
v_msgData_2479_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_msgData_2479_, 0, v___x_2476_);
lean_ctor_set(v_msgData_2479_, 1, v___x_2478_);
v___x_2480_ = lp_mathlib_List_foldl___at___00Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__4_spec__5_spec__9(v_msgData_2479_, v_macroStack_2459_);
v___x_2481_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2481_, 0, v___x_2480_);
return v___x_2481_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__4_spec__5___redArg___boxed(lean_object* v_msgData_2485_, lean_object* v_macroStack_2486_, lean_object* v___y_2487_, lean_object* v___y_2488_){
_start:
{
lean_object* v_res_2489_; 
v_res_2489_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__4_spec__5___redArg(v_msgData_2485_, v_macroStack_2486_, v___y_2487_);
lean_dec_ref(v___y_2487_);
return v_res_2489_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__4___redArg(lean_object* v_msg_2490_, lean_object* v___y_2491_, lean_object* v___y_2492_, lean_object* v___y_2493_, lean_object* v___y_2494_, lean_object* v___y_2495_, lean_object* v___y_2496_){
_start:
{
lean_object* v_ref_2498_; lean_object* v___x_2499_; lean_object* v_a_2500_; lean_object* v_macroStack_2501_; lean_object* v___x_2502_; lean_object* v___x_2503_; lean_object* v_a_2504_; lean_object* v___x_2506_; uint8_t v_isShared_2507_; uint8_t v_isSharedCheck_2512_; 
v_ref_2498_ = lean_ctor_get(v___y_2495_, 5);
v___x_2499_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_throwErrorAt___at___00Lean_Meta_synthSubsingletonInst_spec__4_spec__4_spec__6(v_msg_2490_, v___y_2493_, v___y_2494_, v___y_2495_, v___y_2496_);
v_a_2500_ = lean_ctor_get(v___x_2499_, 0);
lean_inc(v_a_2500_);
lean_dec_ref(v___x_2499_);
v_macroStack_2501_ = lean_ctor_get(v___y_2491_, 1);
v___x_2502_ = l_Lean_Elab_getBetterRef(v_ref_2498_, v_macroStack_2501_);
lean_inc(v_macroStack_2501_);
v___x_2503_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__4_spec__5___redArg(v_a_2500_, v_macroStack_2501_, v___y_2495_);
v_a_2504_ = lean_ctor_get(v___x_2503_, 0);
v_isSharedCheck_2512_ = !lean_is_exclusive(v___x_2503_);
if (v_isSharedCheck_2512_ == 0)
{
v___x_2506_ = v___x_2503_;
v_isShared_2507_ = v_isSharedCheck_2512_;
goto v_resetjp_2505_;
}
else
{
lean_inc(v_a_2504_);
lean_dec(v___x_2503_);
v___x_2506_ = lean_box(0);
v_isShared_2507_ = v_isSharedCheck_2512_;
goto v_resetjp_2505_;
}
v_resetjp_2505_:
{
lean_object* v___x_2508_; lean_object* v___x_2510_; 
v___x_2508_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2508_, 0, v___x_2502_);
lean_ctor_set(v___x_2508_, 1, v_a_2504_);
if (v_isShared_2507_ == 0)
{
lean_ctor_set_tag(v___x_2506_, 1);
lean_ctor_set(v___x_2506_, 0, v___x_2508_);
v___x_2510_ = v___x_2506_;
goto v_reusejp_2509_;
}
else
{
lean_object* v_reuseFailAlloc_2511_; 
v_reuseFailAlloc_2511_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2511_, 0, v___x_2508_);
v___x_2510_ = v_reuseFailAlloc_2511_;
goto v_reusejp_2509_;
}
v_reusejp_2509_:
{
return v___x_2510_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__4___redArg___boxed(lean_object* v_msg_2513_, lean_object* v___y_2514_, lean_object* v___y_2515_, lean_object* v___y_2516_, lean_object* v___y_2517_, lean_object* v___y_2518_, lean_object* v___y_2519_, lean_object* v___y_2520_){
_start:
{
lean_object* v_res_2521_; 
v_res_2521_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__4___redArg(v_msg_2513_, v___y_2514_, v___y_2515_, v___y_2516_, v___y_2517_, v___y_2518_, v___y_2519_);
lean_dec(v___y_2519_);
lean_dec_ref(v___y_2518_);
lean_dec(v___y_2517_);
lean_dec_ref(v___y_2516_);
lean_dec(v___y_2515_);
lean_dec_ref(v___y_2514_);
return v_res_2521_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go___lam__2___closed__2(void){
_start:
{
lean_object* v___x_2525_; lean_object* v___x_2526_; 
v___x_2525_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go___lam__2___closed__1));
v___x_2526_ = l_Lean_stringToMessageData(v___x_2525_);
return v___x_2526_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go___lam__2(lean_object* v_head_2527_, lean_object* v___x_2528_, uint8_t v___x_2529_, lean_object* v___y_2530_, lean_object* v___y_2531_, lean_object* v___y_2532_, lean_object* v___y_2533_, lean_object* v___y_2534_, lean_object* v___y_2535_){
_start:
{
lean_object* v___x_2537_; 
v___x_2537_ = l_Lean_Elab_Term_elabTerm(v_head_2527_, v___x_2528_, v___x_2529_, v___x_2529_, v___y_2530_, v___y_2531_, v___y_2532_, v___y_2533_, v___y_2534_, v___y_2535_);
if (lean_obj_tag(v___x_2537_) == 0)
{
lean_object* v_a_2538_; uint8_t v___x_2539_; lean_object* v___x_2540_; 
v_a_2538_ = lean_ctor_get(v___x_2537_, 0);
lean_inc(v_a_2538_);
lean_dec_ref_known(v___x_2537_, 1);
v___x_2539_ = 1;
v___x_2540_ = l_Lean_Elab_Term_synthesizeSyntheticMVars(v___x_2539_, v___x_2529_, v___y_2530_, v___y_2531_, v___y_2532_, v___y_2533_, v___y_2534_, v___y_2535_);
if (lean_obj_tag(v___x_2540_) == 0)
{
lean_object* v___x_2541_; lean_object* v_a_2542_; lean_object* v___x_2544_; uint8_t v_isShared_2545_; uint8_t v_isSharedCheck_2656_; 
lean_dec_ref_known(v___x_2540_, 1);
v___x_2541_ = lp_mathlib_Lean_instantiateMVars___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__0___redArg(v_a_2538_, v___y_2533_);
v_a_2542_ = lean_ctor_get(v___x_2541_, 0);
v_isSharedCheck_2656_ = !lean_is_exclusive(v___x_2541_);
if (v_isSharedCheck_2656_ == 0)
{
v___x_2544_ = v___x_2541_;
v_isShared_2545_ = v_isSharedCheck_2656_;
goto v_resetjp_2543_;
}
else
{
lean_inc(v_a_2542_);
lean_dec(v___x_2541_);
v___x_2544_ = lean_box(0);
v_isShared_2545_ = v_isSharedCheck_2656_;
goto v_resetjp_2543_;
}
v_resetjp_2543_:
{
lean_object* v___y_2547_; lean_object* v___y_2548_; lean_object* v___y_2549_; lean_object* v___y_2550_; lean_object* v___y_2551_; lean_object* v___y_2552_; lean_object* v___x_2613_; 
lean_inc(v___y_2535_);
lean_inc_ref(v___y_2534_);
lean_inc(v___y_2533_);
lean_inc_ref(v___y_2532_);
lean_inc(v_a_2542_);
v___x_2613_ = lean_infer_type(v_a_2542_, v___y_2532_, v___y_2533_, v___y_2534_, v___y_2535_);
if (lean_obj_tag(v___x_2613_) == 0)
{
lean_object* v_a_2614_; lean_object* v___x_2615_; 
v_a_2614_ = lean_ctor_get(v___x_2613_, 0);
lean_inc(v_a_2614_);
lean_dec_ref_known(v___x_2613_, 1);
v___x_2615_ = l_Lean_Meta_isClass_x3f(v_a_2614_, v___y_2532_, v___y_2533_, v___y_2534_, v___y_2535_);
if (lean_obj_tag(v___x_2615_) == 0)
{
lean_object* v_a_2616_; 
v_a_2616_ = lean_ctor_get(v___x_2615_, 0);
lean_inc(v_a_2616_);
lean_dec_ref_known(v___x_2615_, 1);
if (lean_obj_tag(v_a_2616_) == 0)
{
lean_object* v___x_2617_; 
lean_del_object(v___x_2544_);
lean_inc(v___y_2535_);
lean_inc_ref(v___y_2534_);
lean_inc(v___y_2533_);
lean_inc_ref(v___y_2532_);
v___x_2617_ = lean_infer_type(v_a_2542_, v___y_2532_, v___y_2533_, v___y_2534_, v___y_2535_);
if (lean_obj_tag(v___x_2617_) == 0)
{
lean_object* v_a_2618_; lean_object* v___x_2619_; lean_object* v___x_2620_; lean_object* v___x_2621_; lean_object* v___x_2622_; lean_object* v___x_2623_; lean_object* v_a_2624_; lean_object* v___x_2626_; uint8_t v_isShared_2627_; uint8_t v_isSharedCheck_2631_; 
v_a_2618_ = lean_ctor_get(v___x_2617_, 0);
lean_inc(v_a_2618_);
lean_dec_ref_known(v___x_2617_, 1);
v___x_2619_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go___lam__2___closed__2, &lp_mathlib___private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go___lam__2___closed__2_once, _init_lp_mathlib___private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go___lam__2___closed__2);
v___x_2620_ = l_Lean_MessageData_ofExpr(v_a_2618_);
v___x_2621_ = l_Lean_indentD(v___x_2620_);
v___x_2622_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2622_, 0, v___x_2619_);
lean_ctor_set(v___x_2622_, 1, v___x_2621_);
v___x_2623_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__4___redArg(v___x_2622_, v___y_2530_, v___y_2531_, v___y_2532_, v___y_2533_, v___y_2534_, v___y_2535_);
lean_dec(v___y_2535_);
lean_dec_ref(v___y_2534_);
lean_dec(v___y_2533_);
lean_dec_ref(v___y_2532_);
v_a_2624_ = lean_ctor_get(v___x_2623_, 0);
v_isSharedCheck_2631_ = !lean_is_exclusive(v___x_2623_);
if (v_isSharedCheck_2631_ == 0)
{
v___x_2626_ = v___x_2623_;
v_isShared_2627_ = v_isSharedCheck_2631_;
goto v_resetjp_2625_;
}
else
{
lean_inc(v_a_2624_);
lean_dec(v___x_2623_);
v___x_2626_ = lean_box(0);
v_isShared_2627_ = v_isSharedCheck_2631_;
goto v_resetjp_2625_;
}
v_resetjp_2625_:
{
lean_object* v___x_2629_; 
if (v_isShared_2627_ == 0)
{
v___x_2629_ = v___x_2626_;
goto v_reusejp_2628_;
}
else
{
lean_object* v_reuseFailAlloc_2630_; 
v_reuseFailAlloc_2630_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2630_, 0, v_a_2624_);
v___x_2629_ = v_reuseFailAlloc_2630_;
goto v_reusejp_2628_;
}
v_reusejp_2628_:
{
return v___x_2629_;
}
}
}
else
{
lean_object* v_a_2632_; lean_object* v___x_2634_; uint8_t v_isShared_2635_; uint8_t v_isSharedCheck_2639_; 
lean_dec(v___y_2535_);
lean_dec_ref(v___y_2534_);
lean_dec(v___y_2533_);
lean_dec_ref(v___y_2532_);
v_a_2632_ = lean_ctor_get(v___x_2617_, 0);
v_isSharedCheck_2639_ = !lean_is_exclusive(v___x_2617_);
if (v_isSharedCheck_2639_ == 0)
{
v___x_2634_ = v___x_2617_;
v_isShared_2635_ = v_isSharedCheck_2639_;
goto v_resetjp_2633_;
}
else
{
lean_inc(v_a_2632_);
lean_dec(v___x_2617_);
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
else
{
lean_dec_ref_known(v_a_2616_, 1);
v___y_2547_ = v___y_2530_;
v___y_2548_ = v___y_2531_;
v___y_2549_ = v___y_2532_;
v___y_2550_ = v___y_2533_;
v___y_2551_ = v___y_2534_;
v___y_2552_ = v___y_2535_;
goto v___jp_2546_;
}
}
else
{
lean_object* v_a_2640_; lean_object* v___x_2642_; uint8_t v_isShared_2643_; uint8_t v_isSharedCheck_2647_; 
lean_del_object(v___x_2544_);
lean_dec(v_a_2542_);
lean_dec(v___y_2535_);
lean_dec_ref(v___y_2534_);
lean_dec(v___y_2533_);
lean_dec_ref(v___y_2532_);
v_a_2640_ = lean_ctor_get(v___x_2615_, 0);
v_isSharedCheck_2647_ = !lean_is_exclusive(v___x_2615_);
if (v_isSharedCheck_2647_ == 0)
{
v___x_2642_ = v___x_2615_;
v_isShared_2643_ = v_isSharedCheck_2647_;
goto v_resetjp_2641_;
}
else
{
lean_inc(v_a_2640_);
lean_dec(v___x_2615_);
v___x_2642_ = lean_box(0);
v_isShared_2643_ = v_isSharedCheck_2647_;
goto v_resetjp_2641_;
}
v_resetjp_2641_:
{
lean_object* v___x_2645_; 
if (v_isShared_2643_ == 0)
{
v___x_2645_ = v___x_2642_;
goto v_reusejp_2644_;
}
else
{
lean_object* v_reuseFailAlloc_2646_; 
v_reuseFailAlloc_2646_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2646_, 0, v_a_2640_);
v___x_2645_ = v_reuseFailAlloc_2646_;
goto v_reusejp_2644_;
}
v_reusejp_2644_:
{
return v___x_2645_;
}
}
}
}
else
{
lean_object* v_a_2648_; lean_object* v___x_2650_; uint8_t v_isShared_2651_; uint8_t v_isSharedCheck_2655_; 
lean_del_object(v___x_2544_);
lean_dec(v_a_2542_);
lean_dec(v___y_2535_);
lean_dec_ref(v___y_2534_);
lean_dec(v___y_2533_);
lean_dec_ref(v___y_2532_);
v_a_2648_ = lean_ctor_get(v___x_2613_, 0);
v_isSharedCheck_2655_ = !lean_is_exclusive(v___x_2613_);
if (v_isSharedCheck_2655_ == 0)
{
v___x_2650_ = v___x_2613_;
v_isShared_2651_ = v_isSharedCheck_2655_;
goto v_resetjp_2649_;
}
else
{
lean_inc(v_a_2648_);
lean_dec(v___x_2613_);
v___x_2650_ = lean_box(0);
v_isShared_2651_ = v_isSharedCheck_2655_;
goto v_resetjp_2649_;
}
v_resetjp_2649_:
{
lean_object* v___x_2653_; 
if (v_isShared_2651_ == 0)
{
v___x_2653_ = v___x_2650_;
goto v_reusejp_2652_;
}
else
{
lean_object* v_reuseFailAlloc_2654_; 
v_reuseFailAlloc_2654_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2654_, 0, v_a_2648_);
v___x_2653_ = v_reuseFailAlloc_2654_;
goto v_reusejp_2652_;
}
v_reusejp_2652_:
{
return v___x_2653_;
}
}
}
v___jp_2546_:
{
uint8_t v___x_2553_; 
v___x_2553_ = l_Lean_Expr_hasMVar(v_a_2542_);
if (v___x_2553_ == 0)
{
lean_object* v___x_2554_; lean_object* v___x_2555_; lean_object* v___x_2557_; 
lean_dec(v___y_2552_);
lean_dec_ref(v___y_2551_);
lean_dec(v___y_2550_);
lean_dec_ref(v___y_2549_);
v___x_2554_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go___lam__2___closed__0));
v___x_2555_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_2555_, 0, v___x_2554_);
lean_ctor_set(v___x_2555_, 1, v___x_2554_);
lean_ctor_set(v___x_2555_, 2, v_a_2542_);
if (v_isShared_2545_ == 0)
{
lean_ctor_set(v___x_2544_, 0, v___x_2555_);
v___x_2557_ = v___x_2544_;
goto v_reusejp_2556_;
}
else
{
lean_object* v_reuseFailAlloc_2558_; 
v_reuseFailAlloc_2558_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2558_, 0, v___x_2555_);
v___x_2557_ = v_reuseFailAlloc_2558_;
goto v_reusejp_2556_;
}
v_reusejp_2556_:
{
return v___x_2557_;
}
}
else
{
lean_object* v___x_2559_; 
lean_del_object(v___x_2544_);
v___x_2559_ = l_Lean_Meta_abstractMVars(v_a_2542_, v___x_2529_, v___y_2549_, v___y_2550_, v___y_2551_, v___y_2552_);
if (lean_obj_tag(v___x_2559_) == 0)
{
lean_object* v_a_2560_; lean_object* v___x_2562_; uint8_t v_isShared_2563_; uint8_t v_isSharedCheck_2612_; 
v_a_2560_ = lean_ctor_get(v___x_2559_, 0);
v_isSharedCheck_2612_ = !lean_is_exclusive(v___x_2559_);
if (v_isSharedCheck_2612_ == 0)
{
v___x_2562_ = v___x_2559_;
v_isShared_2563_ = v_isSharedCheck_2612_;
goto v_resetjp_2561_;
}
else
{
lean_inc(v_a_2560_);
lean_dec(v___x_2559_);
v___x_2562_ = lean_box(0);
v_isShared_2563_ = v_isSharedCheck_2612_;
goto v_resetjp_2561_;
}
v_resetjp_2561_:
{
lean_object* v_paramNames_2564_; lean_object* v_mvars_2565_; lean_object* v_expr_2566_; lean_object* v___x_2567_; 
v_paramNames_2564_ = lean_ctor_get(v_a_2560_, 0);
lean_inc_ref(v_paramNames_2564_);
v_mvars_2565_ = lean_ctor_get(v_a_2560_, 1);
lean_inc_ref(v_mvars_2565_);
v_expr_2566_ = lean_ctor_get(v_a_2560_, 2);
lean_inc(v___y_2552_);
lean_inc_ref(v___y_2551_);
lean_inc(v___y_2550_);
lean_inc_ref(v___y_2549_);
lean_inc_ref(v_expr_2566_);
v___x_2567_ = lean_infer_type(v_expr_2566_, v___y_2549_, v___y_2550_, v___y_2551_, v___y_2552_);
if (lean_obj_tag(v___x_2567_) == 0)
{
lean_object* v_a_2568_; lean_object* v___x_2569_; lean_object* v___x_2570_; lean_object* v___f_2571_; lean_object* v___x_2572_; lean_object* v___x_2574_; uint8_t v_isShared_2575_; uint8_t v_isSharedCheck_2600_; 
v_a_2568_ = lean_ctor_get(v___x_2567_, 0);
lean_inc(v_a_2568_);
lean_dec_ref_known(v___x_2567_, 1);
v___x_2569_ = lean_box(v___x_2553_);
v___x_2570_ = lean_box(v___x_2529_);
lean_inc_ref(v_expr_2566_);
v___f_2571_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go___lam__1___boxed), 12, 3);
lean_closure_set(v___f_2571_, 0, v___x_2569_);
lean_closure_set(v___f_2571_, 1, v_expr_2566_);
lean_closure_set(v___f_2571_, 2, v___x_2570_);
v___x_2572_ = l_Lean_Meta_AbstractMVarsResult_numMVars(v_a_2560_);
v_isSharedCheck_2600_ = !lean_is_exclusive(v_a_2560_);
if (v_isSharedCheck_2600_ == 0)
{
lean_object* v_unused_2601_; lean_object* v_unused_2602_; lean_object* v_unused_2603_; 
v_unused_2601_ = lean_ctor_get(v_a_2560_, 2);
lean_dec(v_unused_2601_);
v_unused_2602_ = lean_ctor_get(v_a_2560_, 1);
lean_dec(v_unused_2602_);
v_unused_2603_ = lean_ctor_get(v_a_2560_, 0);
lean_dec(v_unused_2603_);
v___x_2574_ = v_a_2560_;
v_isShared_2575_ = v_isSharedCheck_2600_;
goto v_resetjp_2573_;
}
else
{
lean_dec(v_a_2560_);
v___x_2574_ = lean_box(0);
v_isShared_2575_ = v_isSharedCheck_2600_;
goto v_resetjp_2573_;
}
v_resetjp_2573_:
{
lean_object* v___x_2577_; 
if (v_isShared_2563_ == 0)
{
lean_ctor_set_tag(v___x_2562_, 1);
lean_ctor_set(v___x_2562_, 0, v___x_2572_);
v___x_2577_ = v___x_2562_;
goto v_reusejp_2576_;
}
else
{
lean_object* v_reuseFailAlloc_2599_; 
v_reuseFailAlloc_2599_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2599_, 0, v___x_2572_);
v___x_2577_ = v_reuseFailAlloc_2599_;
goto v_reusejp_2576_;
}
v_reusejp_2576_:
{
uint8_t v___x_2578_; lean_object* v___x_2579_; 
v___x_2578_ = 0;
v___x_2579_ = lp_mathlib_Lean_Meta_forallBoundedTelescope___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__3___redArg(v_a_2568_, v___x_2577_, v___f_2571_, v___x_2578_, v___x_2578_, v___y_2547_, v___y_2548_, v___y_2549_, v___y_2550_, v___y_2551_, v___y_2552_);
lean_dec(v___y_2552_);
lean_dec_ref(v___y_2551_);
lean_dec(v___y_2550_);
lean_dec_ref(v___y_2549_);
if (lean_obj_tag(v___x_2579_) == 0)
{
lean_object* v_a_2580_; lean_object* v___x_2582_; uint8_t v_isShared_2583_; uint8_t v_isSharedCheck_2590_; 
v_a_2580_ = lean_ctor_get(v___x_2579_, 0);
v_isSharedCheck_2590_ = !lean_is_exclusive(v___x_2579_);
if (v_isSharedCheck_2590_ == 0)
{
v___x_2582_ = v___x_2579_;
v_isShared_2583_ = v_isSharedCheck_2590_;
goto v_resetjp_2581_;
}
else
{
lean_inc(v_a_2580_);
lean_dec(v___x_2579_);
v___x_2582_ = lean_box(0);
v_isShared_2583_ = v_isSharedCheck_2590_;
goto v_resetjp_2581_;
}
v_resetjp_2581_:
{
lean_object* v___x_2585_; 
if (v_isShared_2575_ == 0)
{
lean_ctor_set(v___x_2574_, 2, v_a_2580_);
v___x_2585_ = v___x_2574_;
goto v_reusejp_2584_;
}
else
{
lean_object* v_reuseFailAlloc_2589_; 
v_reuseFailAlloc_2589_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_2589_, 0, v_paramNames_2564_);
lean_ctor_set(v_reuseFailAlloc_2589_, 1, v_mvars_2565_);
lean_ctor_set(v_reuseFailAlloc_2589_, 2, v_a_2580_);
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
v_reuseFailAlloc_2588_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2588_, 0, v___x_2585_);
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
else
{
lean_object* v_a_2591_; lean_object* v___x_2593_; uint8_t v_isShared_2594_; uint8_t v_isSharedCheck_2598_; 
lean_del_object(v___x_2574_);
lean_dec_ref(v_mvars_2565_);
lean_dec_ref(v_paramNames_2564_);
v_a_2591_ = lean_ctor_get(v___x_2579_, 0);
v_isSharedCheck_2598_ = !lean_is_exclusive(v___x_2579_);
if (v_isSharedCheck_2598_ == 0)
{
v___x_2593_ = v___x_2579_;
v_isShared_2594_ = v_isSharedCheck_2598_;
goto v_resetjp_2592_;
}
else
{
lean_inc(v_a_2591_);
lean_dec(v___x_2579_);
v___x_2593_ = lean_box(0);
v_isShared_2594_ = v_isSharedCheck_2598_;
goto v_resetjp_2592_;
}
v_resetjp_2592_:
{
lean_object* v___x_2596_; 
if (v_isShared_2594_ == 0)
{
v___x_2596_ = v___x_2593_;
goto v_reusejp_2595_;
}
else
{
lean_object* v_reuseFailAlloc_2597_; 
v_reuseFailAlloc_2597_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2597_, 0, v_a_2591_);
v___x_2596_ = v_reuseFailAlloc_2597_;
goto v_reusejp_2595_;
}
v_reusejp_2595_:
{
return v___x_2596_;
}
}
}
}
}
}
else
{
lean_object* v_a_2604_; lean_object* v___x_2606_; uint8_t v_isShared_2607_; uint8_t v_isSharedCheck_2611_; 
lean_dec_ref(v_mvars_2565_);
lean_dec_ref(v_paramNames_2564_);
lean_del_object(v___x_2562_);
lean_dec(v_a_2560_);
lean_dec(v___y_2552_);
lean_dec_ref(v___y_2551_);
lean_dec(v___y_2550_);
lean_dec_ref(v___y_2549_);
v_a_2604_ = lean_ctor_get(v___x_2567_, 0);
v_isSharedCheck_2611_ = !lean_is_exclusive(v___x_2567_);
if (v_isSharedCheck_2611_ == 0)
{
v___x_2606_ = v___x_2567_;
v_isShared_2607_ = v_isSharedCheck_2611_;
goto v_resetjp_2605_;
}
else
{
lean_inc(v_a_2604_);
lean_dec(v___x_2567_);
v___x_2606_ = lean_box(0);
v_isShared_2607_ = v_isSharedCheck_2611_;
goto v_resetjp_2605_;
}
v_resetjp_2605_:
{
lean_object* v___x_2609_; 
if (v_isShared_2607_ == 0)
{
v___x_2609_ = v___x_2606_;
goto v_reusejp_2608_;
}
else
{
lean_object* v_reuseFailAlloc_2610_; 
v_reuseFailAlloc_2610_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2610_, 0, v_a_2604_);
v___x_2609_ = v_reuseFailAlloc_2610_;
goto v_reusejp_2608_;
}
v_reusejp_2608_:
{
return v___x_2609_;
}
}
}
}
}
else
{
lean_dec(v___y_2552_);
lean_dec_ref(v___y_2551_);
lean_dec(v___y_2550_);
lean_dec_ref(v___y_2549_);
return v___x_2559_;
}
}
}
}
}
else
{
lean_object* v_a_2657_; lean_object* v___x_2659_; uint8_t v_isShared_2660_; uint8_t v_isSharedCheck_2664_; 
lean_dec(v_a_2538_);
lean_dec(v___y_2535_);
lean_dec_ref(v___y_2534_);
lean_dec(v___y_2533_);
lean_dec_ref(v___y_2532_);
v_a_2657_ = lean_ctor_get(v___x_2540_, 0);
v_isSharedCheck_2664_ = !lean_is_exclusive(v___x_2540_);
if (v_isSharedCheck_2664_ == 0)
{
v___x_2659_ = v___x_2540_;
v_isShared_2660_ = v_isSharedCheck_2664_;
goto v_resetjp_2658_;
}
else
{
lean_inc(v_a_2657_);
lean_dec(v___x_2540_);
v___x_2659_ = lean_box(0);
v_isShared_2660_ = v_isSharedCheck_2664_;
goto v_resetjp_2658_;
}
v_resetjp_2658_:
{
lean_object* v___x_2662_; 
if (v_isShared_2660_ == 0)
{
v___x_2662_ = v___x_2659_;
goto v_reusejp_2661_;
}
else
{
lean_object* v_reuseFailAlloc_2663_; 
v_reuseFailAlloc_2663_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2663_, 0, v_a_2657_);
v___x_2662_ = v_reuseFailAlloc_2663_;
goto v_reusejp_2661_;
}
v_reusejp_2661_:
{
return v___x_2662_;
}
}
}
}
else
{
lean_object* v_a_2665_; lean_object* v___x_2667_; uint8_t v_isShared_2668_; uint8_t v_isSharedCheck_2672_; 
lean_dec(v___y_2535_);
lean_dec_ref(v___y_2534_);
lean_dec(v___y_2533_);
lean_dec_ref(v___y_2532_);
v_a_2665_ = lean_ctor_get(v___x_2537_, 0);
v_isSharedCheck_2672_ = !lean_is_exclusive(v___x_2537_);
if (v_isSharedCheck_2672_ == 0)
{
v___x_2667_ = v___x_2537_;
v_isShared_2668_ = v_isSharedCheck_2672_;
goto v_resetjp_2666_;
}
else
{
lean_inc(v_a_2665_);
lean_dec(v___x_2537_);
v___x_2667_ = lean_box(0);
v_isShared_2668_ = v_isSharedCheck_2672_;
goto v_resetjp_2666_;
}
v_resetjp_2666_:
{
lean_object* v___x_2670_; 
if (v_isShared_2668_ == 0)
{
v___x_2670_ = v___x_2667_;
goto v_reusejp_2669_;
}
else
{
lean_object* v_reuseFailAlloc_2671_; 
v_reuseFailAlloc_2671_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2671_, 0, v_a_2665_);
v___x_2670_ = v_reuseFailAlloc_2671_;
goto v_reusejp_2669_;
}
v_reusejp_2669_:
{
return v___x_2670_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go___lam__2___boxed(lean_object* v_head_2673_, lean_object* v___x_2674_, lean_object* v___x_2675_, lean_object* v___y_2676_, lean_object* v___y_2677_, lean_object* v___y_2678_, lean_object* v___y_2679_, lean_object* v___y_2680_, lean_object* v___y_2681_, lean_object* v___y_2682_){
_start:
{
uint8_t v___x_11623__boxed_2683_; lean_object* v_res_2684_; 
v___x_11623__boxed_2683_ = lean_unbox(v___x_2675_);
v_res_2684_ = lp_mathlib___private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go___lam__2(v_head_2673_, v___x_2674_, v___x_11623__boxed_2683_, v___y_2676_, v___y_2677_, v___y_2678_, v___y_2679_, v___y_2680_, v___y_2681_);
lean_dec(v___y_2677_);
lean_dec_ref(v___y_2676_);
return v_res_2684_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go___lam__3(lean_object* v_head_2685_, lean_object* v___f_2686_, lean_object* v___y_2687_, lean_object* v___y_2688_, lean_object* v___y_2689_, lean_object* v___y_2690_, lean_object* v___y_2691_, lean_object* v___y_2692_){
_start:
{
lean_object* v_fileName_2694_; lean_object* v_fileMap_2695_; lean_object* v_options_2696_; lean_object* v_currRecDepth_2697_; lean_object* v_maxRecDepth_2698_; lean_object* v_ref_2699_; lean_object* v_currNamespace_2700_; lean_object* v_openDecls_2701_; lean_object* v_initHeartbeats_2702_; lean_object* v_maxHeartbeats_2703_; lean_object* v_quotContext_2704_; lean_object* v_currMacroScope_2705_; uint8_t v_diag_2706_; lean_object* v_cancelTk_x3f_2707_; uint8_t v_suppressElabErrors_2708_; lean_object* v_inheritedTraceOptions_2709_; lean_object* v___x_2711_; uint8_t v_isShared_2712_; uint8_t v_isSharedCheck_2718_; 
v_fileName_2694_ = lean_ctor_get(v___y_2691_, 0);
v_fileMap_2695_ = lean_ctor_get(v___y_2691_, 1);
v_options_2696_ = lean_ctor_get(v___y_2691_, 2);
v_currRecDepth_2697_ = lean_ctor_get(v___y_2691_, 3);
v_maxRecDepth_2698_ = lean_ctor_get(v___y_2691_, 4);
v_ref_2699_ = lean_ctor_get(v___y_2691_, 5);
v_currNamespace_2700_ = lean_ctor_get(v___y_2691_, 6);
v_openDecls_2701_ = lean_ctor_get(v___y_2691_, 7);
v_initHeartbeats_2702_ = lean_ctor_get(v___y_2691_, 8);
v_maxHeartbeats_2703_ = lean_ctor_get(v___y_2691_, 9);
v_quotContext_2704_ = lean_ctor_get(v___y_2691_, 10);
v_currMacroScope_2705_ = lean_ctor_get(v___y_2691_, 11);
v_diag_2706_ = lean_ctor_get_uint8(v___y_2691_, sizeof(void*)*14);
v_cancelTk_x3f_2707_ = lean_ctor_get(v___y_2691_, 12);
v_suppressElabErrors_2708_ = lean_ctor_get_uint8(v___y_2691_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_2709_ = lean_ctor_get(v___y_2691_, 13);
v_isSharedCheck_2718_ = !lean_is_exclusive(v___y_2691_);
if (v_isSharedCheck_2718_ == 0)
{
v___x_2711_ = v___y_2691_;
v_isShared_2712_ = v_isSharedCheck_2718_;
goto v_resetjp_2710_;
}
else
{
lean_inc(v_inheritedTraceOptions_2709_);
lean_inc(v_cancelTk_x3f_2707_);
lean_inc(v_currMacroScope_2705_);
lean_inc(v_quotContext_2704_);
lean_inc(v_maxHeartbeats_2703_);
lean_inc(v_initHeartbeats_2702_);
lean_inc(v_openDecls_2701_);
lean_inc(v_currNamespace_2700_);
lean_inc(v_ref_2699_);
lean_inc(v_maxRecDepth_2698_);
lean_inc(v_currRecDepth_2697_);
lean_inc(v_options_2696_);
lean_inc(v_fileMap_2695_);
lean_inc(v_fileName_2694_);
lean_dec(v___y_2691_);
v___x_2711_ = lean_box(0);
v_isShared_2712_ = v_isSharedCheck_2718_;
goto v_resetjp_2710_;
}
v_resetjp_2710_:
{
lean_object* v_ref_2713_; lean_object* v___x_2715_; 
v_ref_2713_ = l_Lean_replaceRef(v_head_2685_, v_ref_2699_);
lean_dec(v_ref_2699_);
if (v_isShared_2712_ == 0)
{
lean_ctor_set(v___x_2711_, 5, v_ref_2713_);
v___x_2715_ = v___x_2711_;
goto v_reusejp_2714_;
}
else
{
lean_object* v_reuseFailAlloc_2717_; 
v_reuseFailAlloc_2717_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v_reuseFailAlloc_2717_, 0, v_fileName_2694_);
lean_ctor_set(v_reuseFailAlloc_2717_, 1, v_fileMap_2695_);
lean_ctor_set(v_reuseFailAlloc_2717_, 2, v_options_2696_);
lean_ctor_set(v_reuseFailAlloc_2717_, 3, v_currRecDepth_2697_);
lean_ctor_set(v_reuseFailAlloc_2717_, 4, v_maxRecDepth_2698_);
lean_ctor_set(v_reuseFailAlloc_2717_, 5, v_ref_2713_);
lean_ctor_set(v_reuseFailAlloc_2717_, 6, v_currNamespace_2700_);
lean_ctor_set(v_reuseFailAlloc_2717_, 7, v_openDecls_2701_);
lean_ctor_set(v_reuseFailAlloc_2717_, 8, v_initHeartbeats_2702_);
lean_ctor_set(v_reuseFailAlloc_2717_, 9, v_maxHeartbeats_2703_);
lean_ctor_set(v_reuseFailAlloc_2717_, 10, v_quotContext_2704_);
lean_ctor_set(v_reuseFailAlloc_2717_, 11, v_currMacroScope_2705_);
lean_ctor_set(v_reuseFailAlloc_2717_, 12, v_cancelTk_x3f_2707_);
lean_ctor_set(v_reuseFailAlloc_2717_, 13, v_inheritedTraceOptions_2709_);
lean_ctor_set_uint8(v_reuseFailAlloc_2717_, sizeof(void*)*14, v_diag_2706_);
lean_ctor_set_uint8(v_reuseFailAlloc_2717_, sizeof(void*)*14 + 1, v_suppressElabErrors_2708_);
v___x_2715_ = v_reuseFailAlloc_2717_;
goto v_reusejp_2714_;
}
v_reusejp_2714_:
{
lean_object* v___x_2716_; 
v___x_2716_ = l_Lean_Elab_Term_withoutErrToSorryImp___redArg(v___f_2686_, v___y_2687_, v___y_2688_, v___y_2689_, v___y_2690_, v___x_2715_, v___y_2692_);
lean_dec_ref(v___x_2715_);
return v___x_2716_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go___lam__3___boxed(lean_object* v_head_2719_, lean_object* v___f_2720_, lean_object* v___y_2721_, lean_object* v___y_2722_, lean_object* v___y_2723_, lean_object* v___y_2724_, lean_object* v___y_2725_, lean_object* v___y_2726_, lean_object* v___y_2727_){
_start:
{
lean_object* v_res_2728_; 
v_res_2728_ = lp_mathlib___private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go___lam__3(v_head_2719_, v___f_2720_, v___y_2721_, v___y_2722_, v___y_2723_, v___y_2724_, v___y_2725_, v___y_2726_);
lean_dec(v___y_2726_);
lean_dec(v___y_2724_);
lean_dec_ref(v___y_2723_);
lean_dec(v___y_2722_);
lean_dec_ref(v___y_2721_);
lean_dec(v_head_2719_);
return v_res_2728_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go(lean_object* v_instTerms_2729_, lean_object* v_insts_2730_, lean_object* v_a_2731_, lean_object* v_a_2732_, lean_object* v_a_2733_, lean_object* v_a_2734_, lean_object* v_a_2735_, lean_object* v_a_2736_){
_start:
{
if (lean_obj_tag(v_instTerms_2729_) == 0)
{
lean_object* v___x_2738_; 
v___x_2738_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2738_, 0, v_insts_2730_);
return v___x_2738_;
}
else
{
lean_object* v_head_2739_; lean_object* v_tail_2740_; lean_object* v___x_2742_; uint8_t v_isShared_2743_; uint8_t v_isSharedCheck_2766_; 
v_head_2739_ = lean_ctor_get(v_instTerms_2729_, 0);
v_tail_2740_ = lean_ctor_get(v_instTerms_2729_, 1);
v_isSharedCheck_2766_ = !lean_is_exclusive(v_instTerms_2729_);
if (v_isSharedCheck_2766_ == 0)
{
v___x_2742_ = v_instTerms_2729_;
v_isShared_2743_ = v_isSharedCheck_2766_;
goto v_resetjp_2741_;
}
else
{
lean_inc(v_tail_2740_);
lean_inc(v_head_2739_);
lean_dec(v_instTerms_2729_);
v___x_2742_ = lean_box(0);
v_isShared_2743_ = v_isSharedCheck_2766_;
goto v_resetjp_2741_;
}
v_resetjp_2741_:
{
lean_object* v___x_2744_; uint8_t v___x_2745_; lean_object* v___x_2746_; lean_object* v___f_2747_; lean_object* v___f_2748_; lean_object* v___x_2749_; uint8_t v___x_2750_; lean_object* v___x_2751_; 
v___x_2744_ = lean_box(0);
v___x_2745_ = 1;
v___x_2746_ = lean_box(v___x_2745_);
lean_inc_n(v_head_2739_, 2);
v___f_2747_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go___lam__2___boxed), 10, 3);
lean_closure_set(v___f_2747_, 0, v_head_2739_);
lean_closure_set(v___f_2747_, 1, v___x_2744_);
lean_closure_set(v___f_2747_, 2, v___x_2746_);
v___f_2748_ = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go___lam__3___boxed), 9, 2);
lean_closure_set(v___f_2748_, 0, v_head_2739_);
lean_closure_set(v___f_2748_, 1, v___f_2747_);
v___x_2749_ = lean_alloc_closure((void*)(l_Lean_Elab_Term_withoutModifyingElabMetaStateWithInfo___boxed), 9, 2);
lean_closure_set(v___x_2749_, 0, lean_box(0));
lean_closure_set(v___x_2749_, 1, v___f_2748_);
v___x_2750_ = 0;
v___x_2751_ = lp_mathlib_Lean_Meta_withNewMCtxDepth___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__6___redArg(v___x_2749_, v___x_2750_, v_a_2731_, v_a_2732_, v_a_2733_, v_a_2734_, v_a_2735_, v_a_2736_);
if (lean_obj_tag(v___x_2751_) == 0)
{
lean_object* v_a_2752_; lean_object* v___x_2754_; 
v_a_2752_ = lean_ctor_get(v___x_2751_, 0);
lean_inc(v_a_2752_);
lean_dec_ref_known(v___x_2751_, 1);
if (v_isShared_2743_ == 0)
{
lean_ctor_set_tag(v___x_2742_, 0);
lean_ctor_set(v___x_2742_, 1, v_a_2752_);
v___x_2754_ = v___x_2742_;
goto v_reusejp_2753_;
}
else
{
lean_object* v_reuseFailAlloc_2757_; 
v_reuseFailAlloc_2757_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2757_, 0, v_head_2739_);
lean_ctor_set(v_reuseFailAlloc_2757_, 1, v_a_2752_);
v___x_2754_ = v_reuseFailAlloc_2757_;
goto v_reusejp_2753_;
}
v_reusejp_2753_:
{
lean_object* v___x_2755_; 
v___x_2755_ = lean_array_push(v_insts_2730_, v___x_2754_);
v_instTerms_2729_ = v_tail_2740_;
v_insts_2730_ = v___x_2755_;
goto _start;
}
}
else
{
lean_object* v_a_2758_; lean_object* v___x_2760_; uint8_t v_isShared_2761_; uint8_t v_isSharedCheck_2765_; 
lean_del_object(v___x_2742_);
lean_dec(v_tail_2740_);
lean_dec(v_head_2739_);
lean_dec_ref(v_insts_2730_);
v_a_2758_ = lean_ctor_get(v___x_2751_, 0);
v_isSharedCheck_2765_ = !lean_is_exclusive(v___x_2751_);
if (v_isSharedCheck_2765_ == 0)
{
v___x_2760_ = v___x_2751_;
v_isShared_2761_ = v_isSharedCheck_2765_;
goto v_resetjp_2759_;
}
else
{
lean_inc(v_a_2758_);
lean_dec(v___x_2751_);
v___x_2760_ = lean_box(0);
v_isShared_2761_ = v_isSharedCheck_2765_;
goto v_resetjp_2759_;
}
v_resetjp_2759_:
{
lean_object* v___x_2763_; 
if (v_isShared_2761_ == 0)
{
v___x_2763_ = v___x_2760_;
goto v_reusejp_2762_;
}
else
{
lean_object* v_reuseFailAlloc_2764_; 
v_reuseFailAlloc_2764_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2764_, 0, v_a_2758_);
v___x_2763_ = v_reuseFailAlloc_2764_;
goto v_reusejp_2762_;
}
v_reusejp_2762_:
{
return v___x_2763_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go___boxed(lean_object* v_instTerms_2767_, lean_object* v_insts_2768_, lean_object* v_a_2769_, lean_object* v_a_2770_, lean_object* v_a_2771_, lean_object* v_a_2772_, lean_object* v_a_2773_, lean_object* v_a_2774_, lean_object* v_a_2775_){
_start:
{
lean_object* v_res_2776_; 
v_res_2776_ = lp_mathlib___private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go(v_instTerms_2767_, v_insts_2768_, v_a_2769_, v_a_2770_, v_a_2771_, v_a_2772_, v_a_2773_, v_a_2774_);
lean_dec(v_a_2774_);
lean_dec_ref(v_a_2773_);
lean_dec(v_a_2772_);
lean_dec_ref(v_a_2771_);
lean_dec(v_a_2770_);
lean_dec_ref(v_a_2769_);
return v_res_2776_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__4(lean_object* v_00_u03b1_2777_, lean_object* v_msg_2778_, lean_object* v___y_2779_, lean_object* v___y_2780_, lean_object* v___y_2781_, lean_object* v___y_2782_, lean_object* v___y_2783_, lean_object* v___y_2784_){
_start:
{
lean_object* v___x_2786_; 
v___x_2786_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__4___redArg(v_msg_2778_, v___y_2779_, v___y_2780_, v___y_2781_, v___y_2782_, v___y_2783_, v___y_2784_);
return v___x_2786_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__4___boxed(lean_object* v_00_u03b1_2787_, lean_object* v_msg_2788_, lean_object* v___y_2789_, lean_object* v___y_2790_, lean_object* v___y_2791_, lean_object* v___y_2792_, lean_object* v___y_2793_, lean_object* v___y_2794_, lean_object* v___y_2795_){
_start:
{
lean_object* v_res_2796_; 
v_res_2796_ = lp_mathlib_Lean_throwError___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__4(v_00_u03b1_2787_, v_msg_2788_, v___y_2789_, v___y_2790_, v___y_2791_, v___y_2792_, v___y_2793_, v___y_2794_);
lean_dec(v___y_2794_);
lean_dec_ref(v___y_2793_);
lean_dec(v___y_2792_);
lean_dec_ref(v___y_2791_);
lean_dec(v___y_2790_);
lean_dec_ref(v___y_2789_);
return v_res_2796_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__1_spec__1(uint8_t v___x_2797_, lean_object* v_as_2798_, size_t v_i_2799_, size_t v_stop_2800_, lean_object* v_b_2801_, lean_object* v___y_2802_, lean_object* v___y_2803_, lean_object* v___y_2804_, lean_object* v___y_2805_, lean_object* v___y_2806_, lean_object* v___y_2807_){
_start:
{
lean_object* v___x_2809_; 
v___x_2809_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__1_spec__1___redArg(v___x_2797_, v_as_2798_, v_i_2799_, v_stop_2800_, v_b_2801_, v___y_2804_, v___y_2805_, v___y_2806_, v___y_2807_);
return v___x_2809_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__1_spec__1___boxed(lean_object* v___x_2810_, lean_object* v_as_2811_, lean_object* v_i_2812_, lean_object* v_stop_2813_, lean_object* v_b_2814_, lean_object* v___y_2815_, lean_object* v___y_2816_, lean_object* v___y_2817_, lean_object* v___y_2818_, lean_object* v___y_2819_, lean_object* v___y_2820_, lean_object* v___y_2821_){
_start:
{
uint8_t v___x_12054__boxed_2822_; size_t v_i_boxed_2823_; size_t v_stop_boxed_2824_; lean_object* v_res_2825_; 
v___x_12054__boxed_2822_ = lean_unbox(v___x_2810_);
v_i_boxed_2823_ = lean_unbox_usize(v_i_2812_);
lean_dec(v_i_2812_);
v_stop_boxed_2824_ = lean_unbox_usize(v_stop_2813_);
lean_dec(v_stop_2813_);
v_res_2825_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Array_filterMapM___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__1_spec__1(v___x_12054__boxed_2822_, v_as_2811_, v_i_boxed_2823_, v_stop_boxed_2824_, v_b_2814_, v___y_2815_, v___y_2816_, v___y_2817_, v___y_2818_, v___y_2819_, v___y_2820_);
lean_dec(v___y_2820_);
lean_dec_ref(v___y_2819_);
lean_dec(v___y_2818_);
lean_dec_ref(v___y_2817_);
lean_dec(v___y_2816_);
lean_dec_ref(v___y_2815_);
lean_dec_ref(v_as_2811_);
return v_res_2825_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__4_spec__5(lean_object* v_msgData_2826_, lean_object* v_macroStack_2827_, lean_object* v___y_2828_, lean_object* v___y_2829_, lean_object* v___y_2830_, lean_object* v___y_2831_, lean_object* v___y_2832_, lean_object* v___y_2833_){
_start:
{
lean_object* v___x_2835_; 
v___x_2835_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__4_spec__5___redArg(v_msgData_2826_, v_macroStack_2827_, v___y_2832_);
return v___x_2835_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__4_spec__5___boxed(lean_object* v_msgData_2836_, lean_object* v_macroStack_2837_, lean_object* v___y_2838_, lean_object* v___y_2839_, lean_object* v___y_2840_, lean_object* v___y_2841_, lean_object* v___y_2842_, lean_object* v___y_2843_, lean_object* v___y_2844_){
_start:
{
lean_object* v_res_2845_; 
v_res_2845_ = lp_mathlib_Lean_Elab_addMacroStack___at___00Lean_throwError___at___00__private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go_spec__4_spec__5(v_msgData_2836_, v_macroStack_2837_, v___y_2838_, v___y_2839_, v___y_2840_, v___y_2841_, v___y_2842_, v___y_2843_);
lean_dec(v___y_2843_);
lean_dec_ref(v___y_2842_);
lean_dec(v___y_2841_);
lean_dec_ref(v___y_2840_);
lean_dec(v___y_2839_);
lean_dec_ref(v___y_2838_);
return v_res_2845_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_elabSubsingletonInsts(lean_object* v_instTerms_x3f_2848_, lean_object* v_a_2849_, lean_object* v_a_2850_, lean_object* v_a_2851_, lean_object* v_a_2852_, lean_object* v_a_2853_, lean_object* v_a_2854_){
_start:
{
if (lean_obj_tag(v_instTerms_x3f_2848_) == 1)
{
lean_object* v_val_2856_; lean_object* v___x_2857_; lean_object* v___x_2858_; lean_object* v___x_2859_; 
v_val_2856_ = lean_ctor_get(v_instTerms_x3f_2848_, 0);
lean_inc(v_val_2856_);
lean_dec_ref_known(v_instTerms_x3f_2848_, 1);
v___x_2857_ = lean_array_to_list(v_val_2856_);
v___x_2858_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_elabSubsingletonInsts___closed__0));
v___x_2859_ = lp_mathlib___private_Mathlib_Tactic_Subsingleton_0__Mathlib_Tactic_elabSubsingletonInsts_go(v___x_2857_, v___x_2858_, v_a_2849_, v_a_2850_, v_a_2851_, v_a_2852_, v_a_2853_, v_a_2854_);
return v___x_2859_;
}
else
{
lean_object* v___x_2860_; lean_object* v___x_2861_; 
lean_dec(v_instTerms_x3f_2848_);
v___x_2860_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_elabSubsingletonInsts___closed__0));
v___x_2861_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2861_, 0, v___x_2860_);
return v___x_2861_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_elabSubsingletonInsts___boxed(lean_object* v_instTerms_x3f_2862_, lean_object* v_a_2863_, lean_object* v_a_2864_, lean_object* v_a_2865_, lean_object* v_a_2866_, lean_object* v_a_2867_, lean_object* v_a_2868_, lean_object* v_a_2869_){
_start:
{
lean_object* v_res_2870_; 
v_res_2870_ = lp_mathlib_Mathlib_Tactic_elabSubsingletonInsts(v_instTerms_x3f_2862_, v_a_2863_, v_a_2864_, v_a_2865_, v_a_2866_, v_a_2867_, v_a_2868_);
lean_dec(v_a_2868_);
lean_dec_ref(v_a_2867_);
lean_dec(v_a_2866_);
lean_dec_ref(v_a_2865_);
lean_dec(v_a_2864_);
lean_dec_ref(v_a_2863_);
return v_res_2870_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_2871_; lean_object* v___x_2872_; lean_object* v___x_2873_; 
v___x_2871_ = lean_box(0);
v___x_2872_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_2873_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2873_, 0, v___x_2872_);
lean_ctor_set(v___x_2873_, 1, v___x_2871_);
return v___x_2873_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1_spec__0___redArg(){
_start:
{
lean_object* v___x_2875_; lean_object* v___x_2876_; 
v___x_2875_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1_spec__0___redArg___closed__0);
v___x_2876_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2876_, 0, v___x_2875_);
return v___x_2876_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1_spec__0___redArg___boxed(lean_object* v___y_2877_){
_start:
{
lean_object* v_res_2878_; 
v_res_2878_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1_spec__0___redArg();
return v_res_2878_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1_spec__0(lean_object* v_00_u03b1_2879_, lean_object* v___y_2880_, lean_object* v___y_2881_, lean_object* v___y_2882_, lean_object* v___y_2883_, lean_object* v___y_2884_, lean_object* v___y_2885_, lean_object* v___y_2886_, lean_object* v___y_2887_){
_start:
{
lean_object* v___x_2889_; 
v___x_2889_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1_spec__0___redArg();
return v___x_2889_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1_spec__0___boxed(lean_object* v_00_u03b1_2890_, lean_object* v___y_2891_, lean_object* v___y_2892_, lean_object* v___y_2893_, lean_object* v___y_2894_, lean_object* v___y_2895_, lean_object* v___y_2896_, lean_object* v___y_2897_, lean_object* v___y_2898_, lean_object* v___y_2899_){
_start:
{
lean_object* v_res_2900_; 
v_res_2900_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1_spec__0(v_00_u03b1_2890_, v___y_2891_, v___y_2892_, v___y_2893_, v___y_2894_, v___y_2895_, v___y_2896_, v___y_2897_, v___y_2898_);
lean_dec(v___y_2898_);
lean_dec_ref(v___y_2897_);
lean_dec(v___y_2896_);
lean_dec_ref(v___y_2895_);
lean_dec(v___y_2894_);
lean_dec_ref(v___y_2893_);
lean_dec(v___y_2892_);
lean_dec_ref(v___y_2891_);
return v_res_2900_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__0(lean_object* v_tac_2907_, lean_object* v___y_2908_, lean_object* v___y_2909_, lean_object* v___y_2910_, lean_object* v___y_2911_){
_start:
{
lean_object* v_ref_2913_; lean_object* v___x_2914_; lean_object* v___x_2915_; lean_object* v___x_2916_; lean_object* v___x_2917_; lean_object* v___x_2918_; lean_object* v___x_2919_; uint8_t v___x_2920_; lean_object* v___x_2921_; lean_object* v___x_2922_; 
v_ref_2913_ = lean_ctor_get(v___y_2910_, 5);
v___x_2914_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__0___closed__1));
v___x_2915_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2915_, 0, v___x_2914_);
lean_ctor_set(v___x_2915_, 1, v_tac_2907_);
v___x_2916_ = lean_box(0);
v___x_2917_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_2917_, 0, v___x_2915_);
lean_ctor_set(v___x_2917_, 1, v___x_2916_);
lean_ctor_set(v___x_2917_, 2, v___x_2916_);
lean_ctor_set(v___x_2917_, 3, v___x_2916_);
lean_ctor_set(v___x_2917_, 4, v___x_2916_);
lean_ctor_set(v___x_2917_, 5, v___x_2916_);
lean_inc_n(v_ref_2913_, 2);
v___x_2918_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2918_, 0, v_ref_2913_);
v___x_2919_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__0___closed__2));
v___x_2920_ = 4;
v___x_2921_ = l_Lean_MessageData_nil;
v___x_2922_ = l_Lean_Meta_Tactic_TryThis_addSuggestion(v_ref_2913_, v___x_2917_, v___x_2918_, v___x_2919_, v___x_2916_, v___x_2920_, v___x_2921_, v___y_2910_, v___y_2911_);
if (lean_obj_tag(v___x_2922_) == 0)
{
lean_object* v___x_2924_; uint8_t v_isShared_2925_; uint8_t v_isSharedCheck_2930_; 
v_isSharedCheck_2930_ = !lean_is_exclusive(v___x_2922_);
if (v_isSharedCheck_2930_ == 0)
{
lean_object* v_unused_2931_; 
v_unused_2931_ = lean_ctor_get(v___x_2922_, 0);
lean_dec(v_unused_2931_);
v___x_2924_ = v___x_2922_;
v_isShared_2925_ = v_isSharedCheck_2930_;
goto v_resetjp_2923_;
}
else
{
lean_dec(v___x_2922_);
v___x_2924_ = lean_box(0);
v_isShared_2925_ = v_isSharedCheck_2930_;
goto v_resetjp_2923_;
}
v_resetjp_2923_:
{
lean_object* v___x_2926_; lean_object* v___x_2928_; 
v___x_2926_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__0___closed__3));
if (v_isShared_2925_ == 0)
{
lean_ctor_set(v___x_2924_, 0, v___x_2926_);
v___x_2928_ = v___x_2924_;
goto v_reusejp_2927_;
}
else
{
lean_object* v_reuseFailAlloc_2929_; 
v_reuseFailAlloc_2929_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2929_, 0, v___x_2926_);
v___x_2928_ = v_reuseFailAlloc_2929_;
goto v_reusejp_2927_;
}
v_reusejp_2927_:
{
return v___x_2928_;
}
}
}
else
{
lean_object* v_a_2932_; lean_object* v___x_2934_; uint8_t v_isShared_2935_; uint8_t v_isSharedCheck_2939_; 
v_a_2932_ = lean_ctor_get(v___x_2922_, 0);
v_isSharedCheck_2939_ = !lean_is_exclusive(v___x_2922_);
if (v_isSharedCheck_2939_ == 0)
{
v___x_2934_ = v___x_2922_;
v_isShared_2935_ = v_isSharedCheck_2939_;
goto v_resetjp_2933_;
}
else
{
lean_inc(v_a_2932_);
lean_dec(v___x_2922_);
v___x_2934_ = lean_box(0);
v_isShared_2935_ = v_isSharedCheck_2939_;
goto v_resetjp_2933_;
}
v_resetjp_2933_:
{
lean_object* v___x_2937_; 
if (v_isShared_2935_ == 0)
{
v___x_2937_ = v___x_2934_;
goto v_reusejp_2936_;
}
else
{
lean_object* v_reuseFailAlloc_2938_; 
v_reuseFailAlloc_2938_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2938_, 0, v_a_2932_);
v___x_2937_ = v_reuseFailAlloc_2938_;
goto v_reusejp_2936_;
}
v_reusejp_2936_:
{
return v___x_2937_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__0___boxed(lean_object* v_tac_2940_, lean_object* v___y_2941_, lean_object* v___y_2942_, lean_object* v___y_2943_, lean_object* v___y_2944_, lean_object* v___y_2945_){
_start:
{
lean_object* v_res_2946_; 
v_res_2946_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__0(v_tac_2940_, v___y_2941_, v___y_2942_, v___y_2943_, v___y_2944_);
lean_dec(v___y_2944_);
lean_dec_ref(v___y_2943_);
lean_dec(v___y_2942_);
lean_dec_ref(v___y_2941_);
return v_res_2946_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__1___closed__9(void){
_start:
{
lean_object* v___x_2957_; 
v___x_2957_ = l_Array_mkArray0(lean_box(0));
return v___x_2957_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__1(lean_object* v_a_2962_, lean_object* v___x_2963_, lean_object* v___f_2964_, uint8_t v_recover_2965_, uint8_t v___x_2966_, lean_object* v___y_2967_, lean_object* v___y_2968_, lean_object* v___y_2969_, lean_object* v___y_2970_, lean_object* v___y_2971_, lean_object* v___y_2972_, lean_object* v___y_2973_, lean_object* v___y_2974_){
_start:
{
lean_object* v___x_2979_; 
v___x_2979_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_2968_, v___y_2971_, v___y_2972_, v___y_2973_, v___y_2974_);
if (lean_obj_tag(v___x_2979_) == 0)
{
lean_object* v_a_2980_; lean_object* v___x_2981_; 
v_a_2980_ = lean_ctor_get(v___x_2979_, 0);
lean_inc(v_a_2980_);
lean_dec_ref_known(v___x_2979_, 1);
v___x_2981_ = l_Lean_MVarId_intros(v_a_2980_, v___y_2971_, v___y_2972_, v___y_2973_, v___y_2974_);
if (lean_obj_tag(v___x_2981_) == 0)
{
lean_object* v_a_2982_; lean_object* v___x_2984_; uint8_t v_isShared_2985_; uint8_t v_isSharedCheck_3093_; 
v_a_2982_ = lean_ctor_get(v___x_2981_, 0);
v_isSharedCheck_3093_ = !lean_is_exclusive(v___x_2981_);
if (v_isSharedCheck_3093_ == 0)
{
v___x_2984_ = v___x_2981_;
v_isShared_2985_ = v_isSharedCheck_3093_;
goto v_resetjp_2983_;
}
else
{
lean_inc(v_a_2982_);
lean_dec(v___x_2981_);
v___x_2984_ = lean_box(0);
v_isShared_2985_ = v_isSharedCheck_3093_;
goto v_resetjp_2983_;
}
v_resetjp_2983_:
{
lean_object* v_fst_2986_; lean_object* v_snd_2987_; lean_object* v___x_2989_; uint8_t v_isShared_2990_; uint8_t v_isSharedCheck_3092_; 
v_fst_2986_ = lean_ctor_get(v_a_2982_, 0);
v_snd_2987_ = lean_ctor_get(v_a_2982_, 1);
v_isSharedCheck_3092_ = !lean_is_exclusive(v_a_2982_);
if (v_isSharedCheck_3092_ == 0)
{
v___x_2989_ = v_a_2982_;
v_isShared_2990_ = v_isSharedCheck_3092_;
goto v_resetjp_2988_;
}
else
{
lean_inc(v_snd_2987_);
lean_inc(v_fst_2986_);
lean_dec(v_a_2982_);
v___x_2989_ = lean_box(0);
v_isShared_2990_ = v_isSharedCheck_3092_;
goto v_resetjp_2988_;
}
v_resetjp_2988_:
{
lean_object* v___x_2991_; 
lean_inc(v_snd_2987_);
v___x_2991_ = lp_mathlib_Lean_MVarId_subsingleton(v_snd_2987_, v_a_2962_, v___y_2971_, v___y_2972_, v___y_2973_, v___y_2974_);
if (lean_obj_tag(v___x_2991_) == 0)
{
lean_dec_ref_known(v___x_2991_, 1);
lean_del_object(v___x_2989_);
lean_dec(v_snd_2987_);
lean_dec(v_fst_2986_);
lean_del_object(v___x_2984_);
lean_dec_ref(v___f_2964_);
lean_dec_ref(v___x_2963_);
goto v___jp_2976_;
}
else
{
lean_object* v_a_2992_; lean_object* v___y_2994_; uint8_t v___y_2995_; lean_object* v_a_3003_; lean_object* v___y_3007_; uint8_t v___y_3025_; lean_object* v___y_3026_; lean_object* v___y_3075_; lean_object* v___y_3076_; uint8_t v___y_3077_; uint8_t v___y_3078_; uint8_t v___y_3082_; uint8_t v___x_3090_; 
v_a_2992_ = lean_ctor_get(v___x_2991_, 0);
lean_inc(v_a_2992_);
v___x_3090_ = l_Lean_Exception_isInterrupt(v_a_2992_);
if (v___x_3090_ == 0)
{
uint8_t v___x_3091_; 
lean_inc(v_a_2992_);
v___x_3091_ = l_Lean_Exception_isRuntime(v_a_2992_);
v___y_3082_ = v___x_3091_;
goto v___jp_3081_;
}
else
{
v___y_3082_ = v___x_3090_;
goto v___jp_3081_;
}
v___jp_2993_:
{
if (v___y_2995_ == 0)
{
lean_object* v___x_2997_; 
lean_dec_ref(v___y_2994_);
if (v_isShared_2985_ == 0)
{
lean_ctor_set_tag(v___x_2984_, 1);
lean_ctor_set(v___x_2984_, 0, v_a_2992_);
v___x_2997_ = v___x_2984_;
goto v_reusejp_2996_;
}
else
{
lean_object* v_reuseFailAlloc_2998_; 
v_reuseFailAlloc_2998_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2998_, 0, v_a_2992_);
v___x_2997_ = v_reuseFailAlloc_2998_;
goto v_reusejp_2996_;
}
v_reusejp_2996_:
{
return v___x_2997_;
}
}
else
{
lean_object* v___x_3000_; 
lean_dec(v_a_2992_);
if (v_isShared_2985_ == 0)
{
lean_ctor_set_tag(v___x_2984_, 1);
lean_ctor_set(v___x_2984_, 0, v___y_2994_);
v___x_3000_ = v___x_2984_;
goto v_reusejp_2999_;
}
else
{
lean_object* v_reuseFailAlloc_3001_; 
v_reuseFailAlloc_3001_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3001_, 0, v___y_2994_);
v___x_3000_ = v_reuseFailAlloc_3001_;
goto v_reusejp_2999_;
}
v_reusejp_2999_:
{
return v___x_3000_;
}
}
}
v___jp_3002_:
{
uint8_t v___x_3004_; 
v___x_3004_ = l_Lean_Exception_isInterrupt(v_a_3003_);
if (v___x_3004_ == 0)
{
uint8_t v___x_3005_; 
lean_inc_ref(v_a_3003_);
v___x_3005_ = l_Lean_Exception_isRuntime(v_a_3003_);
v___y_2994_ = v_a_3003_;
v___y_2995_ = v___x_3005_;
goto v___jp_2993_;
}
else
{
v___y_2994_ = v_a_3003_;
v___y_2995_ = v___x_3004_;
goto v___jp_2993_;
}
}
v___jp_3006_:
{
if (lean_obj_tag(v___y_3007_) == 0)
{
lean_object* v_a_3008_; lean_object* v___x_3010_; uint8_t v_isShared_3011_; uint8_t v_isSharedCheck_3022_; 
lean_del_object(v___x_2984_);
v_a_3008_ = lean_ctor_get(v___y_3007_, 0);
v_isSharedCheck_3022_ = !lean_is_exclusive(v___y_3007_);
if (v_isSharedCheck_3022_ == 0)
{
v___x_3010_ = v___y_3007_;
v_isShared_3011_ = v_isSharedCheck_3022_;
goto v_resetjp_3009_;
}
else
{
lean_inc(v_a_3008_);
lean_dec(v___y_3007_);
v___x_3010_ = lean_box(0);
v_isShared_3011_ = v_isSharedCheck_3022_;
goto v_resetjp_3009_;
}
v_resetjp_3009_:
{
if (lean_obj_tag(v_a_3008_) == 0)
{
lean_object* v_a_3012_; 
lean_del_object(v___x_3010_);
lean_dec(v_a_2992_);
v_a_3012_ = lean_ctor_get(v_a_3008_, 0);
lean_inc(v_a_3012_);
lean_dec_ref_known(v_a_3008_, 1);
if (lean_obj_tag(v_a_3012_) == 1)
{
lean_object* v_val_3013_; lean_object* v___x_3014_; lean_object* v___x_3016_; 
v_val_3013_ = lean_ctor_get(v_a_3012_, 0);
lean_inc(v_val_3013_);
lean_dec_ref_known(v_a_3012_, 1);
v___x_3014_ = lean_box(0);
if (v_isShared_2990_ == 0)
{
lean_ctor_set_tag(v___x_2989_, 1);
lean_ctor_set(v___x_2989_, 1, v___x_3014_);
lean_ctor_set(v___x_2989_, 0, v_val_3013_);
v___x_3016_ = v___x_2989_;
goto v_reusejp_3015_;
}
else
{
lean_object* v_reuseFailAlloc_3018_; 
v_reuseFailAlloc_3018_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3018_, 0, v_val_3013_);
lean_ctor_set(v_reuseFailAlloc_3018_, 1, v___x_3014_);
v___x_3016_ = v_reuseFailAlloc_3018_;
goto v_reusejp_3015_;
}
v_reusejp_3015_:
{
lean_object* v___x_3017_; 
v___x_3017_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v___x_3016_, v___y_2968_, v___y_2971_, v___y_2972_, v___y_2973_, v___y_2974_);
lean_dec(v___y_2974_);
lean_dec_ref(v___y_2973_);
lean_dec(v___y_2972_);
lean_dec_ref(v___y_2971_);
return v___x_3017_;
}
}
else
{
lean_dec(v_a_3012_);
lean_del_object(v___x_2989_);
goto v___jp_2976_;
}
}
else
{
lean_object* v___x_3020_; 
lean_dec_ref_known(v_a_3008_, 1);
lean_del_object(v___x_2989_);
lean_dec(v___y_2974_);
lean_dec_ref(v___y_2973_);
lean_dec(v___y_2972_);
lean_dec_ref(v___y_2971_);
if (v_isShared_3011_ == 0)
{
lean_ctor_set_tag(v___x_3010_, 1);
lean_ctor_set(v___x_3010_, 0, v_a_2992_);
v___x_3020_ = v___x_3010_;
goto v_reusejp_3019_;
}
else
{
lean_object* v_reuseFailAlloc_3021_; 
v_reuseFailAlloc_3021_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3021_, 0, v_a_2992_);
v___x_3020_ = v_reuseFailAlloc_3021_;
goto v_reusejp_3019_;
}
v_reusejp_3019_:
{
return v___x_3020_;
}
}
}
}
else
{
lean_object* v_a_3023_; 
lean_del_object(v___x_2989_);
lean_dec(v___y_2974_);
lean_dec_ref(v___y_2973_);
lean_dec(v___y_2972_);
lean_dec_ref(v___y_2971_);
v_a_3023_ = lean_ctor_get(v___y_3007_, 0);
lean_inc(v_a_3023_);
lean_dec_ref_known(v___y_3007_, 1);
v_a_3003_ = v_a_3023_;
goto v___jp_3002_;
}
}
v___jp_3024_:
{
if (lean_obj_tag(v___y_3026_) == 0)
{
lean_object* v___x_3027_; lean_object* v___x_3028_; uint8_t v___x_3029_; 
lean_dec_ref_known(v___y_3026_, 1);
v___x_3027_ = lean_array_get_size(v_fst_2986_);
lean_dec(v_fst_2986_);
v___x_3028_ = lean_unsigned_to_nat(0u);
v___x_3029_ = lean_nat_dec_eq(v___x_3027_, v___x_3028_);
if (v___x_3029_ == 0)
{
lean_object* v_ref_3030_; lean_object* v___x_3031_; lean_object* v___x_3032_; lean_object* v___x_3033_; lean_object* v___x_3034_; lean_object* v___x_3035_; lean_object* v___x_3036_; lean_object* v___x_3037_; lean_object* v___x_3038_; lean_object* v___x_3039_; lean_object* v___x_3040_; lean_object* v___x_3041_; lean_object* v___x_3042_; lean_object* v___x_3043_; lean_object* v___x_3044_; lean_object* v___x_3045_; lean_object* v___x_3046_; lean_object* v___x_3047_; lean_object* v___x_3048_; lean_object* v___x_3049_; lean_object* v___x_3050_; lean_object* v___x_3051_; lean_object* v___x_3052_; lean_object* v___x_3053_; lean_object* v___x_3054_; lean_object* v___x_3055_; lean_object* v___x_3056_; lean_object* v___x_3057_; lean_object* v___x_3058_; lean_object* v___x_3059_; lean_object* v___x_3060_; lean_object* v___x_3061_; lean_object* v___x_3062_; 
v_ref_3030_ = lean_ctor_get(v___y_2973_, 5);
v___x_3031_ = l_Lean_SourceInfo_fromRef(v_ref_3030_, v___y_3025_);
v___x_3032_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__1___closed__0));
v___x_3033_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__1___closed__1));
v___x_3034_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__1___closed__2));
lean_inc_ref_n(v___x_2963_, 4);
v___x_3035_ = l_Lean_Name_mkStr4(v___x_3032_, v___x_3033_, v___x_2963_, v___x_3034_);
v___x_3036_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__1___closed__3));
lean_inc_n(v___x_3031_, 11);
v___x_3037_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3037_, 0, v___x_3031_);
lean_ctor_set(v___x_3037_, 1, v___x_3036_);
v___x_3038_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__1___closed__4));
v___x_3039_ = l_Lean_Name_mkStr4(v___x_3032_, v___x_3033_, v___x_2963_, v___x_3038_);
v___x_3040_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__1___closed__5));
v___x_3041_ = l_Lean_Name_mkStr4(v___x_3032_, v___x_3033_, v___x_2963_, v___x_3040_);
v___x_3042_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__1___closed__7));
v___x_3043_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__1___closed__8));
v___x_3044_ = l_Lean_Name_mkStr4(v___x_3032_, v___x_3033_, v___x_2963_, v___x_3043_);
v___x_3045_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3045_, 0, v___x_3031_);
lean_ctor_set(v___x_3045_, 1, v___x_3043_);
v___x_3046_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__1___closed__9, &lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__1___closed__9_once, _init_lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__1___closed__9);
v___x_3047_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_3047_, 0, v___x_3031_);
lean_ctor_set(v___x_3047_, 1, v___x_3042_);
lean_ctor_set(v___x_3047_, 2, v___x_3046_);
v___x_3048_ = l_Lean_Syntax_node2(v___x_3031_, v___x_3044_, v___x_3045_, v___x_3047_);
v___x_3049_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__1___closed__10));
v___x_3050_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3050_, 0, v___x_3031_);
lean_ctor_set(v___x_3050_, 1, v___x_3049_);
v___x_3051_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__1___closed__11));
v___x_3052_ = l_Lean_Name_mkStr4(v___x_3032_, v___x_3033_, v___x_2963_, v___x_3051_);
v___x_3053_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__1___closed__12));
v___x_3054_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3054_, 0, v___x_3031_);
lean_ctor_set(v___x_3054_, 1, v___x_3053_);
v___x_3055_ = l_Lean_Syntax_node1(v___x_3031_, v___x_3052_, v___x_3054_);
v___x_3056_ = l_Lean_Syntax_node3(v___x_3031_, v___x_3042_, v___x_3048_, v___x_3050_, v___x_3055_);
v___x_3057_ = l_Lean_Syntax_node1(v___x_3031_, v___x_3041_, v___x_3056_);
v___x_3058_ = l_Lean_Syntax_node1(v___x_3031_, v___x_3039_, v___x_3057_);
v___x_3059_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__1___closed__13));
v___x_3060_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3060_, 0, v___x_3031_);
lean_ctor_set(v___x_3060_, 1, v___x_3059_);
v___x_3061_ = l_Lean_Syntax_node3(v___x_3031_, v___x_3035_, v___x_3037_, v___x_3058_, v___x_3060_);
lean_inc(v___y_2974_);
lean_inc_ref(v___y_2973_);
lean_inc(v___y_2972_);
lean_inc_ref(v___y_2971_);
v___x_3062_ = lean_apply_6(v___f_2964_, v___x_3061_, v___y_2971_, v___y_2972_, v___y_2973_, v___y_2974_, lean_box(0));
v___y_3007_ = v___x_3062_;
goto v___jp_3006_;
}
else
{
lean_object* v_ref_3063_; lean_object* v___x_3064_; lean_object* v___x_3065_; lean_object* v___x_3066_; lean_object* v___x_3067_; lean_object* v___x_3068_; lean_object* v___x_3069_; lean_object* v___x_3070_; lean_object* v___x_3071_; lean_object* v___x_3072_; 
v_ref_3063_ = lean_ctor_get(v___y_2973_, 5);
v___x_3064_ = l_Lean_SourceInfo_fromRef(v_ref_3063_, v___y_3025_);
v___x_3065_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__1___closed__0));
v___x_3066_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__1___closed__1));
v___x_3067_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__1___closed__11));
v___x_3068_ = l_Lean_Name_mkStr4(v___x_3065_, v___x_3066_, v___x_2963_, v___x_3067_);
v___x_3069_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__1___closed__12));
lean_inc(v___x_3064_);
v___x_3070_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_3070_, 0, v___x_3064_);
lean_ctor_set(v___x_3070_, 1, v___x_3069_);
v___x_3071_ = l_Lean_Syntax_node1(v___x_3064_, v___x_3068_, v___x_3070_);
lean_inc(v___y_2974_);
lean_inc_ref(v___y_2973_);
lean_inc(v___y_2972_);
lean_inc_ref(v___y_2971_);
v___x_3072_ = lean_apply_6(v___f_2964_, v___x_3071_, v___y_2971_, v___y_2972_, v___y_2973_, v___y_2974_, lean_box(0));
v___y_3007_ = v___x_3072_;
goto v___jp_3006_;
}
}
else
{
lean_object* v_a_3073_; 
lean_del_object(v___x_2989_);
lean_dec(v_fst_2986_);
lean_dec(v___y_2974_);
lean_dec_ref(v___y_2973_);
lean_dec(v___y_2972_);
lean_dec_ref(v___y_2971_);
lean_dec_ref(v___f_2964_);
lean_dec_ref(v___x_2963_);
v_a_3073_ = lean_ctor_get(v___y_3026_, 0);
lean_inc(v_a_3073_);
lean_dec_ref_known(v___y_3026_, 1);
v_a_3003_ = v_a_3073_;
goto v___jp_3002_;
}
}
v___jp_3074_:
{
if (v___y_3078_ == 0)
{
lean_object* v___x_3079_; 
lean_dec_ref(v___y_3076_);
v___x_3079_ = l_Lean_Meta_SavedState_restore___redArg(v___y_3075_, v___y_2972_, v___y_2974_);
lean_dec_ref(v___y_3075_);
if (lean_obj_tag(v___x_3079_) == 0)
{
lean_object* v___x_3080_; 
lean_dec_ref_known(v___x_3079_, 1);
v___x_3080_ = l_Lean_MVarId_hrefl(v_snd_2987_, v___y_2971_, v___y_2972_, v___y_2973_, v___y_2974_);
v___y_3025_ = v___y_3077_;
v___y_3026_ = v___x_3080_;
goto v___jp_3024_;
}
else
{
lean_dec(v_snd_2987_);
v___y_3025_ = v___y_3077_;
v___y_3026_ = v___x_3079_;
goto v___jp_3024_;
}
}
else
{
lean_dec_ref(v___y_3075_);
lean_dec(v_snd_2987_);
v___y_3025_ = v___y_3077_;
v___y_3026_ = v___y_3076_;
goto v___jp_3024_;
}
}
v___jp_3081_:
{
if (v___y_3082_ == 0)
{
if (v_recover_2965_ == 0)
{
lean_dec(v_a_2992_);
lean_del_object(v___x_2989_);
lean_dec(v_snd_2987_);
lean_dec(v_fst_2986_);
lean_del_object(v___x_2984_);
lean_dec(v___y_2974_);
lean_dec_ref(v___y_2973_);
lean_dec(v___y_2972_);
lean_dec_ref(v___y_2971_);
lean_dec_ref(v___f_2964_);
lean_dec_ref(v___x_2963_);
return v___x_2991_;
}
else
{
lean_object* v___x_3083_; 
lean_dec_ref_known(v___x_2991_, 1);
v___x_3083_ = l_Lean_Meta_saveState___redArg(v___y_2972_, v___y_2974_);
if (lean_obj_tag(v___x_3083_) == 0)
{
lean_object* v_a_3084_; lean_object* v___x_3085_; 
v_a_3084_ = lean_ctor_get(v___x_3083_, 0);
lean_inc(v_a_3084_);
lean_dec_ref_known(v___x_3083_, 1);
lean_inc(v_snd_2987_);
v___x_3085_ = l_Lean_MVarId_refl(v_snd_2987_, v___x_2966_, v___y_2971_, v___y_2972_, v___y_2973_, v___y_2974_);
if (lean_obj_tag(v___x_3085_) == 0)
{
lean_dec(v_a_3084_);
lean_dec(v_snd_2987_);
v___y_3025_ = v___y_3082_;
v___y_3026_ = v___x_3085_;
goto v___jp_3024_;
}
else
{
lean_object* v_a_3086_; uint8_t v___x_3087_; 
v_a_3086_ = lean_ctor_get(v___x_3085_, 0);
lean_inc(v_a_3086_);
v___x_3087_ = l_Lean_Exception_isInterrupt(v_a_3086_);
if (v___x_3087_ == 0)
{
uint8_t v___x_3088_; 
v___x_3088_ = l_Lean_Exception_isRuntime(v_a_3086_);
v___y_3075_ = v_a_3084_;
v___y_3076_ = v___x_3085_;
v___y_3077_ = v___y_3082_;
v___y_3078_ = v___x_3088_;
goto v___jp_3074_;
}
else
{
lean_dec(v_a_3086_);
v___y_3075_ = v_a_3084_;
v___y_3076_ = v___x_3085_;
v___y_3077_ = v___y_3082_;
v___y_3078_ = v___x_3087_;
goto v___jp_3074_;
}
}
}
else
{
lean_object* v_a_3089_; 
lean_del_object(v___x_2989_);
lean_dec(v_snd_2987_);
lean_dec(v_fst_2986_);
lean_dec(v___y_2974_);
lean_dec_ref(v___y_2973_);
lean_dec(v___y_2972_);
lean_dec_ref(v___y_2971_);
lean_dec_ref(v___f_2964_);
lean_dec_ref(v___x_2963_);
v_a_3089_ = lean_ctor_get(v___x_3083_, 0);
lean_inc(v_a_3089_);
lean_dec_ref_known(v___x_3083_, 1);
v_a_3003_ = v_a_3089_;
goto v___jp_3002_;
}
}
}
else
{
lean_dec(v_a_2992_);
lean_del_object(v___x_2989_);
lean_dec(v_snd_2987_);
lean_dec(v_fst_2986_);
lean_del_object(v___x_2984_);
lean_dec(v___y_2974_);
lean_dec_ref(v___y_2973_);
lean_dec(v___y_2972_);
lean_dec_ref(v___y_2971_);
lean_dec_ref(v___f_2964_);
lean_dec_ref(v___x_2963_);
return v___x_2991_;
}
}
}
}
}
}
else
{
lean_object* v_a_3094_; lean_object* v___x_3096_; uint8_t v_isShared_3097_; uint8_t v_isSharedCheck_3101_; 
lean_dec(v___y_2974_);
lean_dec_ref(v___y_2973_);
lean_dec(v___y_2972_);
lean_dec_ref(v___y_2971_);
lean_dec_ref(v___f_2964_);
lean_dec_ref(v___x_2963_);
lean_dec_ref(v_a_2962_);
v_a_3094_ = lean_ctor_get(v___x_2981_, 0);
v_isSharedCheck_3101_ = !lean_is_exclusive(v___x_2981_);
if (v_isSharedCheck_3101_ == 0)
{
v___x_3096_ = v___x_2981_;
v_isShared_3097_ = v_isSharedCheck_3101_;
goto v_resetjp_3095_;
}
else
{
lean_inc(v_a_3094_);
lean_dec(v___x_2981_);
v___x_3096_ = lean_box(0);
v_isShared_3097_ = v_isSharedCheck_3101_;
goto v_resetjp_3095_;
}
v_resetjp_3095_:
{
lean_object* v___x_3099_; 
if (v_isShared_3097_ == 0)
{
v___x_3099_ = v___x_3096_;
goto v_reusejp_3098_;
}
else
{
lean_object* v_reuseFailAlloc_3100_; 
v_reuseFailAlloc_3100_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3100_, 0, v_a_3094_);
v___x_3099_ = v_reuseFailAlloc_3100_;
goto v_reusejp_3098_;
}
v_reusejp_3098_:
{
return v___x_3099_;
}
}
}
}
else
{
lean_object* v_a_3102_; lean_object* v___x_3104_; uint8_t v_isShared_3105_; uint8_t v_isSharedCheck_3109_; 
lean_dec(v___y_2974_);
lean_dec_ref(v___y_2973_);
lean_dec(v___y_2972_);
lean_dec_ref(v___y_2971_);
lean_dec_ref(v___f_2964_);
lean_dec_ref(v___x_2963_);
lean_dec_ref(v_a_2962_);
v_a_3102_ = lean_ctor_get(v___x_2979_, 0);
v_isSharedCheck_3109_ = !lean_is_exclusive(v___x_2979_);
if (v_isSharedCheck_3109_ == 0)
{
v___x_3104_ = v___x_2979_;
v_isShared_3105_ = v_isSharedCheck_3109_;
goto v_resetjp_3103_;
}
else
{
lean_inc(v_a_3102_);
lean_dec(v___x_2979_);
v___x_3104_ = lean_box(0);
v_isShared_3105_ = v_isSharedCheck_3109_;
goto v_resetjp_3103_;
}
v_resetjp_3103_:
{
lean_object* v___x_3107_; 
if (v_isShared_3105_ == 0)
{
v___x_3107_ = v___x_3104_;
goto v_reusejp_3106_;
}
else
{
lean_object* v_reuseFailAlloc_3108_; 
v_reuseFailAlloc_3108_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3108_, 0, v_a_3102_);
v___x_3107_ = v_reuseFailAlloc_3108_;
goto v_reusejp_3106_;
}
v_reusejp_3106_:
{
return v___x_3107_;
}
}
}
v___jp_2976_:
{
lean_object* v___x_2977_; lean_object* v___x_2978_; 
v___x_2977_ = lean_box(0);
v___x_2978_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v___x_2977_, v___y_2968_, v___y_2971_, v___y_2972_, v___y_2973_, v___y_2974_);
lean_dec(v___y_2974_);
lean_dec_ref(v___y_2973_);
lean_dec(v___y_2972_);
lean_dec_ref(v___y_2971_);
return v___x_2978_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__1___boxed(lean_object* v_a_3110_, lean_object* v___x_3111_, lean_object* v___f_3112_, lean_object* v_recover_3113_, lean_object* v___x_3114_, lean_object* v___y_3115_, lean_object* v___y_3116_, lean_object* v___y_3117_, lean_object* v___y_3118_, lean_object* v___y_3119_, lean_object* v___y_3120_, lean_object* v___y_3121_, lean_object* v___y_3122_, lean_object* v___y_3123_){
_start:
{
uint8_t v_recover_boxed_3124_; uint8_t v___x_12347__boxed_3125_; lean_object* v_res_3126_; 
v_recover_boxed_3124_ = lean_unbox(v_recover_3113_);
v___x_12347__boxed_3125_ = lean_unbox(v___x_3114_);
v_res_3126_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__1(v_a_3110_, v___x_3111_, v___f_3112_, v_recover_boxed_3124_, v___x_12347__boxed_3125_, v___y_3115_, v___y_3116_, v___y_3117_, v___y_3118_, v___y_3119_, v___y_3120_, v___y_3121_, v___y_3122_);
lean_dec(v___y_3118_);
lean_dec_ref(v___y_3117_);
lean_dec(v___y_3116_);
lean_dec_ref(v___y_3115_);
return v_res_3126_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__2(lean_object* v_instTerms_x3f_3127_, lean_object* v___x_3128_, lean_object* v___f_3129_, uint8_t v___x_3130_, lean_object* v___y_3131_, lean_object* v___y_3132_, lean_object* v___y_3133_, lean_object* v___y_3134_, lean_object* v___y_3135_, lean_object* v___y_3136_, lean_object* v___y_3137_, lean_object* v___y_3138_){
_start:
{
lean_object* v___x_3140_; 
v___x_3140_ = lp_mathlib_Mathlib_Tactic_elabSubsingletonInsts(v_instTerms_x3f_3127_, v___y_3133_, v___y_3134_, v___y_3135_, v___y_3136_, v___y_3137_, v___y_3138_);
if (lean_obj_tag(v___x_3140_) == 0)
{
lean_object* v_a_3141_; uint8_t v_recover_3142_; lean_object* v___x_3143_; lean_object* v___x_3144_; lean_object* v___f_3145_; lean_object* v___x_3146_; 
v_a_3141_ = lean_ctor_get(v___x_3140_, 0);
lean_inc(v_a_3141_);
lean_dec_ref_known(v___x_3140_, 1);
v_recover_3142_ = lean_ctor_get_uint8(v___y_3131_, sizeof(void*)*1);
v___x_3143_ = lean_box(v_recover_3142_);
v___x_3144_ = lean_box(v___x_3130_);
v___f_3145_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__1___boxed), 14, 5);
lean_closure_set(v___f_3145_, 0, v_a_3141_);
lean_closure_set(v___f_3145_, 1, v___x_3128_);
lean_closure_set(v___f_3145_, 2, v___f_3129_);
lean_closure_set(v___f_3145_, 3, v___x_3143_);
lean_closure_set(v___f_3145_, 4, v___x_3144_);
v___x_3146_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_3145_, v___y_3131_, v___y_3132_, v___y_3133_, v___y_3134_, v___y_3135_, v___y_3136_, v___y_3137_, v___y_3138_);
return v___x_3146_;
}
else
{
lean_object* v_a_3147_; lean_object* v___x_3149_; uint8_t v_isShared_3150_; uint8_t v_isSharedCheck_3154_; 
lean_dec_ref(v___f_3129_);
lean_dec_ref(v___x_3128_);
v_a_3147_ = lean_ctor_get(v___x_3140_, 0);
v_isSharedCheck_3154_ = !lean_is_exclusive(v___x_3140_);
if (v_isSharedCheck_3154_ == 0)
{
v___x_3149_ = v___x_3140_;
v_isShared_3150_ = v_isSharedCheck_3154_;
goto v_resetjp_3148_;
}
else
{
lean_inc(v_a_3147_);
lean_dec(v___x_3140_);
v___x_3149_ = lean_box(0);
v_isShared_3150_ = v_isSharedCheck_3154_;
goto v_resetjp_3148_;
}
v_resetjp_3148_:
{
lean_object* v___x_3152_; 
if (v_isShared_3150_ == 0)
{
v___x_3152_ = v___x_3149_;
goto v_reusejp_3151_;
}
else
{
lean_object* v_reuseFailAlloc_3153_; 
v_reuseFailAlloc_3153_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_3153_, 0, v_a_3147_);
v___x_3152_ = v_reuseFailAlloc_3153_;
goto v_reusejp_3151_;
}
v_reusejp_3151_:
{
return v___x_3152_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__2___boxed(lean_object* v_instTerms_x3f_3155_, lean_object* v___x_3156_, lean_object* v___f_3157_, lean_object* v___x_3158_, lean_object* v___y_3159_, lean_object* v___y_3160_, lean_object* v___y_3161_, lean_object* v___y_3162_, lean_object* v___y_3163_, lean_object* v___y_3164_, lean_object* v___y_3165_, lean_object* v___y_3166_, lean_object* v___y_3167_){
_start:
{
uint8_t v___x_12670__boxed_3168_; lean_object* v_res_3169_; 
v___x_12670__boxed_3168_ = lean_unbox(v___x_3158_);
v_res_3169_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__2(v_instTerms_x3f_3155_, v___x_3156_, v___f_3157_, v___x_12670__boxed_3168_, v___y_3159_, v___y_3160_, v___y_3161_, v___y_3162_, v___y_3163_, v___y_3164_, v___y_3165_, v___y_3166_);
lean_dec(v___y_3166_);
lean_dec_ref(v___y_3165_);
lean_dec(v___y_3164_);
lean_dec_ref(v___y_3163_);
lean_dec(v___y_3162_);
lean_dec_ref(v___y_3161_);
lean_dec(v___y_3160_);
lean_dec_ref(v___y_3159_);
return v_res_3169_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1_spec__2(uint8_t v___x_3170_, uint8_t v___x_3171_, lean_object* v_as_3172_, size_t v_i_3173_, size_t v_stop_3174_, lean_object* v_b_3175_){
_start:
{
lean_object* v___y_3177_; uint8_t v___x_3181_; 
v___x_3181_ = lean_usize_dec_eq(v_i_3173_, v_stop_3174_);
if (v___x_3181_ == 0)
{
lean_object* v_fst_3182_; uint8_t v___x_3183_; 
v_fst_3182_ = lean_ctor_get(v_b_3175_, 0);
v___x_3183_ = lean_unbox(v_fst_3182_);
if (v___x_3183_ == 0)
{
lean_object* v_snd_3184_; lean_object* v___x_3186_; uint8_t v_isShared_3187_; uint8_t v_isSharedCheck_3192_; 
v_snd_3184_ = lean_ctor_get(v_b_3175_, 1);
v_isSharedCheck_3192_ = !lean_is_exclusive(v_b_3175_);
if (v_isSharedCheck_3192_ == 0)
{
lean_object* v_unused_3193_; 
v_unused_3193_ = lean_ctor_get(v_b_3175_, 0);
lean_dec(v_unused_3193_);
v___x_3186_ = v_b_3175_;
v_isShared_3187_ = v_isSharedCheck_3192_;
goto v_resetjp_3185_;
}
else
{
lean_inc(v_snd_3184_);
lean_dec(v_b_3175_);
v___x_3186_ = lean_box(0);
v_isShared_3187_ = v_isSharedCheck_3192_;
goto v_resetjp_3185_;
}
v_resetjp_3185_:
{
lean_object* v___x_3188_; lean_object* v___x_3190_; 
v___x_3188_ = lean_box(v___x_3170_);
if (v_isShared_3187_ == 0)
{
lean_ctor_set(v___x_3186_, 0, v___x_3188_);
v___x_3190_ = v___x_3186_;
goto v_reusejp_3189_;
}
else
{
lean_object* v_reuseFailAlloc_3191_; 
v_reuseFailAlloc_3191_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3191_, 0, v___x_3188_);
lean_ctor_set(v_reuseFailAlloc_3191_, 1, v_snd_3184_);
v___x_3190_ = v_reuseFailAlloc_3191_;
goto v_reusejp_3189_;
}
v_reusejp_3189_:
{
v___y_3177_ = v___x_3190_;
goto v___jp_3176_;
}
}
}
else
{
lean_object* v_snd_3194_; lean_object* v___x_3196_; uint8_t v_isShared_3197_; uint8_t v_isSharedCheck_3204_; 
v_snd_3194_ = lean_ctor_get(v_b_3175_, 1);
v_isSharedCheck_3204_ = !lean_is_exclusive(v_b_3175_);
if (v_isSharedCheck_3204_ == 0)
{
lean_object* v_unused_3205_; 
v_unused_3205_ = lean_ctor_get(v_b_3175_, 0);
lean_dec(v_unused_3205_);
v___x_3196_ = v_b_3175_;
v_isShared_3197_ = v_isSharedCheck_3204_;
goto v_resetjp_3195_;
}
else
{
lean_inc(v_snd_3194_);
lean_dec(v_b_3175_);
v___x_3196_ = lean_box(0);
v_isShared_3197_ = v_isSharedCheck_3204_;
goto v_resetjp_3195_;
}
v_resetjp_3195_:
{
lean_object* v___x_3198_; lean_object* v___x_3199_; lean_object* v___x_3200_; lean_object* v___x_3202_; 
v___x_3198_ = lean_array_uget_borrowed(v_as_3172_, v_i_3173_);
lean_inc(v___x_3198_);
v___x_3199_ = lean_array_push(v_snd_3194_, v___x_3198_);
v___x_3200_ = lean_box(v___x_3171_);
if (v_isShared_3197_ == 0)
{
lean_ctor_set(v___x_3196_, 1, v___x_3199_);
lean_ctor_set(v___x_3196_, 0, v___x_3200_);
v___x_3202_ = v___x_3196_;
goto v_reusejp_3201_;
}
else
{
lean_object* v_reuseFailAlloc_3203_; 
v_reuseFailAlloc_3203_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_3203_, 0, v___x_3200_);
lean_ctor_set(v_reuseFailAlloc_3203_, 1, v___x_3199_);
v___x_3202_ = v_reuseFailAlloc_3203_;
goto v_reusejp_3201_;
}
v_reusejp_3201_:
{
v___y_3177_ = v___x_3202_;
goto v___jp_3176_;
}
}
}
}
else
{
return v_b_3175_;
}
v___jp_3176_:
{
size_t v___x_3178_; size_t v___x_3179_; 
v___x_3178_ = ((size_t)1ULL);
v___x_3179_ = lean_usize_add(v_i_3173_, v___x_3178_);
v_i_3173_ = v___x_3179_;
v_b_3175_ = v___y_3177_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1_spec__2___boxed(lean_object* v___x_3206_, lean_object* v___x_3207_, lean_object* v_as_3208_, lean_object* v_i_3209_, lean_object* v_stop_3210_, lean_object* v_b_3211_){
_start:
{
uint8_t v___x_12732__boxed_3212_; uint8_t v___x_12733__boxed_3213_; size_t v_i_boxed_3214_; size_t v_stop_boxed_3215_; lean_object* v_res_3216_; 
v___x_12732__boxed_3212_ = lean_unbox(v___x_3206_);
v___x_12733__boxed_3213_ = lean_unbox(v___x_3207_);
v_i_boxed_3214_ = lean_unbox_usize(v_i_3209_);
lean_dec(v_i_3209_);
v_stop_boxed_3215_ = lean_unbox_usize(v_stop_3210_);
lean_dec(v_stop_3210_);
v_res_3216_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1_spec__2(v___x_12732__boxed_3212_, v___x_12733__boxed_3213_, v_as_3208_, v_i_boxed_3214_, v_stop_boxed_3215_, v_b_3211_);
lean_dec_ref(v_as_3208_);
return v_res_3216_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1_spec__1(size_t v_sz_3217_, size_t v_i_3218_, lean_object* v_bs_3219_){
_start:
{
uint8_t v___x_3220_; 
v___x_3220_ = lean_usize_dec_lt(v_i_3218_, v_sz_3217_);
if (v___x_3220_ == 0)
{
lean_object* v___x_3221_; 
v___x_3221_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3221_, 0, v_bs_3219_);
return v___x_3221_;
}
else
{
lean_object* v_v_3222_; lean_object* v___x_3223_; lean_object* v_bs_x27_3224_; size_t v___x_3225_; size_t v___x_3226_; lean_object* v___x_3227_; 
v_v_3222_ = lean_array_uget(v_bs_3219_, v_i_3218_);
v___x_3223_ = lean_unsigned_to_nat(0u);
v_bs_x27_3224_ = lean_array_uset(v_bs_3219_, v_i_3218_, v___x_3223_);
v___x_3225_ = ((size_t)1ULL);
v___x_3226_ = lean_usize_add(v_i_3218_, v___x_3225_);
v___x_3227_ = lean_array_uset(v_bs_x27_3224_, v_i_3218_, v_v_3222_);
v_i_3218_ = v___x_3226_;
v_bs_3219_ = v___x_3227_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1_spec__1___boxed(lean_object* v_sz_3229_, lean_object* v_i_3230_, lean_object* v_bs_3231_){
_start:
{
size_t v_sz_boxed_3232_; size_t v_i_boxed_3233_; lean_object* v_res_3234_; 
v_sz_boxed_3232_ = lean_unbox_usize(v_sz_3229_);
lean_dec(v_sz_3229_);
v_i_boxed_3233_ = lean_unbox_usize(v_i_3230_);
lean_dec(v_i_3230_);
v_res_3234_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1_spec__1(v_sz_boxed_3232_, v_i_boxed_3233_, v_bs_3231_);
return v_res_3234_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1(lean_object* v_x_3238_, lean_object* v_a_3239_, lean_object* v_a_3240_, lean_object* v_a_3241_, lean_object* v_a_3242_, lean_object* v_a_3243_, lean_object* v_a_3244_, lean_object* v_a_3245_, lean_object* v_a_3246_){
_start:
{
lean_object* v___x_3248_; lean_object* v___x_3249_; uint8_t v___x_3250_; 
v___x_3248_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__1));
v___x_3249_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_subsingletonStx___closed__3));
lean_inc(v_x_3238_);
v___x_3250_ = l_Lean_Syntax_isOfKind(v_x_3238_, v___x_3249_);
if (v___x_3250_ == 0)
{
lean_object* v___x_3251_; 
lean_dec(v_x_3238_);
v___x_3251_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1_spec__0___redArg();
return v___x_3251_;
}
else
{
lean_object* v___f_3252_; lean_object* v_instTerms_x3f_3254_; lean_object* v___y_3255_; lean_object* v___y_3256_; lean_object* v___y_3257_; lean_object* v___y_3258_; lean_object* v___y_3259_; lean_object* v___y_3260_; lean_object* v___y_3261_; lean_object* v___y_3262_; lean_object* v___y_3267_; lean_object* v___x_3272_; lean_object* v___x_3273_; uint8_t v___x_3274_; 
v___f_3252_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___closed__0));
v___x_3272_ = lean_unsigned_to_nat(1u);
v___x_3273_ = l_Lean_Syntax_getArg(v_x_3238_, v___x_3272_);
lean_dec(v_x_3238_);
v___x_3274_ = l_Lean_Syntax_isNone(v___x_3273_);
if (v___x_3274_ == 0)
{
lean_object* v___x_3275_; uint8_t v___x_3276_; 
v___x_3275_ = lean_unsigned_to_nat(3u);
lean_inc(v___x_3273_);
v___x_3276_ = l_Lean_Syntax_matchesNull(v___x_3273_, v___x_3275_);
if (v___x_3276_ == 0)
{
lean_object* v___x_3277_; 
lean_dec(v___x_3273_);
v___x_3277_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1_spec__0___redArg();
return v___x_3277_;
}
else
{
lean_object* v___x_3278_; lean_object* v___x_3279_; lean_object* v___x_3280_; lean_object* v___x_3281_; lean_object* v___x_3282_; uint8_t v___x_3283_; 
v___x_3278_ = l_Lean_Syntax_getArg(v___x_3273_, v___x_3272_);
lean_dec(v___x_3273_);
v___x_3279_ = l_Lean_Syntax_getArgs(v___x_3278_);
lean_dec(v___x_3278_);
v___x_3280_ = lean_unsigned_to_nat(0u);
v___x_3281_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___closed__1));
v___x_3282_ = lean_array_get_size(v___x_3279_);
v___x_3283_ = lean_nat_dec_lt(v___x_3280_, v___x_3282_);
if (v___x_3283_ == 0)
{
lean_dec_ref(v___x_3279_);
v___y_3267_ = v___x_3281_;
goto v___jp_3266_;
}
else
{
lean_object* v___x_3284_; lean_object* v___x_3285_; uint8_t v___x_3286_; 
v___x_3284_ = lean_box(v___x_3276_);
v___x_3285_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3285_, 0, v___x_3284_);
lean_ctor_set(v___x_3285_, 1, v___x_3281_);
v___x_3286_ = lean_nat_dec_le(v___x_3282_, v___x_3282_);
if (v___x_3286_ == 0)
{
if (v___x_3283_ == 0)
{
lean_dec_ref_known(v___x_3285_, 2);
lean_dec_ref(v___x_3279_);
v___y_3267_ = v___x_3281_;
goto v___jp_3266_;
}
else
{
size_t v___x_3287_; size_t v___x_3288_; lean_object* v___x_3289_; lean_object* v_snd_3290_; 
v___x_3287_ = ((size_t)0ULL);
v___x_3288_ = lean_usize_of_nat(v___x_3282_);
v___x_3289_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1_spec__2(v___x_3276_, v___x_3274_, v___x_3279_, v___x_3287_, v___x_3288_, v___x_3285_);
lean_dec_ref(v___x_3279_);
v_snd_3290_ = lean_ctor_get(v___x_3289_, 1);
lean_inc(v_snd_3290_);
lean_dec_ref(v___x_3289_);
v___y_3267_ = v_snd_3290_;
goto v___jp_3266_;
}
}
else
{
size_t v___x_3291_; size_t v___x_3292_; lean_object* v___x_3293_; lean_object* v_snd_3294_; 
v___x_3291_ = ((size_t)0ULL);
v___x_3292_ = lean_usize_of_nat(v___x_3282_);
v___x_3293_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1_spec__2(v___x_3276_, v___x_3274_, v___x_3279_, v___x_3291_, v___x_3292_, v___x_3285_);
lean_dec_ref(v___x_3279_);
v_snd_3294_ = lean_ctor_get(v___x_3293_, 1);
lean_inc(v_snd_3294_);
lean_dec_ref(v___x_3293_);
v___y_3267_ = v_snd_3294_;
goto v___jp_3266_;
}
}
}
}
else
{
lean_object* v___x_3295_; 
lean_dec(v___x_3273_);
v___x_3295_ = lean_box(0);
v_instTerms_x3f_3254_ = v___x_3295_;
v___y_3255_ = v_a_3239_;
v___y_3256_ = v_a_3240_;
v___y_3257_ = v_a_3241_;
v___y_3258_ = v_a_3242_;
v___y_3259_ = v_a_3243_;
v___y_3260_ = v_a_3244_;
v___y_3261_ = v_a_3245_;
v___y_3262_ = v_a_3246_;
goto v___jp_3253_;
}
v___jp_3253_:
{
lean_object* v___x_3263_; lean_object* v___f_3264_; lean_object* v___x_3265_; 
v___x_3263_ = lean_box(v___x_3250_);
v___f_3264_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___lam__2___boxed), 13, 4);
lean_closure_set(v___f_3264_, 0, v_instTerms_x3f_3254_);
lean_closure_set(v___f_3264_, 1, v___x_3248_);
lean_closure_set(v___f_3264_, 2, v___f_3252_);
lean_closure_set(v___f_3264_, 3, v___x_3263_);
v___x_3265_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_3264_, v___y_3255_, v___y_3256_, v___y_3257_, v___y_3258_, v___y_3259_, v___y_3260_, v___y_3261_, v___y_3262_);
return v___x_3265_;
}
v___jp_3266_:
{
size_t v_sz_3268_; size_t v___x_3269_; lean_object* v___x_3270_; 
v_sz_3268_ = lean_array_size(v___y_3267_);
v___x_3269_ = ((size_t)0ULL);
v___x_3270_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1_spec__1(v_sz_3268_, v___x_3269_, v___y_3267_);
if (lean_obj_tag(v___x_3270_) == 0)
{
lean_object* v___x_3271_; 
v___x_3271_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1_spec__0___redArg();
return v___x_3271_;
}
else
{
v_instTerms_x3f_3254_ = v___x_3270_;
v___y_3255_ = v_a_3239_;
v___y_3256_ = v_a_3240_;
v___y_3257_ = v_a_3241_;
v___y_3258_ = v_a_3242_;
v___y_3259_ = v_a_3243_;
v___y_3260_ = v_a_3244_;
v___y_3261_ = v_a_3245_;
v___y_3262_ = v_a_3246_;
goto v___jp_3253_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1___boxed(lean_object* v_x_3296_, lean_object* v_a_3297_, lean_object* v_a_3298_, lean_object* v_a_3299_, lean_object* v_a_3300_, lean_object* v_a_3301_, lean_object* v_a_3302_, lean_object* v_a_3303_, lean_object* v_a_3304_, lean_object* v_a_3305_){
_start:
{
lean_object* v_res_3306_; 
v_res_3306_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Subsingleton______elabRules__Mathlib__Tactic__subsingletonStx__1(v_x_3296_, v_a_3297_, v_a_3298_, v_a_3299_, v_a_3300_, v_a_3301_, v_a_3302_, v_a_3303_, v_a_3304_);
lean_dec(v_a_3304_);
lean_dec_ref(v_a_3303_);
lean_dec(v_a_3302_);
lean_dec_ref(v_a_3301_);
lean_dec(v_a_3300_);
lean_dec_ref(v_a_3299_);
lean_dec(v_a_3298_);
lean_dec_ref(v_a_3297_);
return v_res_3306_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Basic_Logic_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Subsingleton(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Basic_Logic_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Tactic_Refl(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Subsingleton(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Tactic_Refl(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Meta_Tactic_Refl(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Basic_Logic_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Subsingleton(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Tactic_Refl(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Basic_Logic_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Subsingleton(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Subsingleton(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Subsingleton(builtin);
}
#ifdef __cplusplus
}
#endif
