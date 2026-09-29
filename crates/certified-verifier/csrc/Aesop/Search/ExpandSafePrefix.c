// Lean compiler output
// Module: Aesop.Search.ExpandSafePrefix
// Imports: public import Init public meta import Init public import Aesop.Search.Expansion import Aesop.Exception
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
lean_object* l_Lean_stringToMessageData(lean_object*);
extern lean_object* l_Lean_instMonadExceptOfExceptionCoreM;
lean_object* l_StateRefT_x27_instMonadExceptOf___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_StateRefT_x27_instMonadExceptOf___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instMonadExceptOf___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instMonadExceptOf___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_aesop_Aesop_BaseM_instMonadStats;
lean_object* lp_aesop_Aesop_instMonadStatsStateRefT_x27___redArg(lean_object*);
lean_object* lp_aesop_Aesop_instMonadStatsReaderT___redArg(lean_object*);
extern lean_object* l_Lean_Core_instMonadQuotationCoreM;
lean_object* l_StateRefT_x27_lift___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_StateRefT_x27_instMonadFunctor___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_instMonadQuotationOfMonadFunctorOfMonadLift___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instMonadLift___lam__0___boxed(lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instMonadFunctor___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_SearchM_instMonad(lean_object*, lean_object*);
lean_object* l_StateRefT_x27_instMonad___redArg(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
extern lean_object* lp_aesop_Aesop_treeImpl;
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
uint8_t lp_aesop_Aesop_GoalState_isProven(uint8_t);
lean_object* lp_aesop_Aesop_Goal_safeRapps(lean_object*);
lean_object* lp_aesop_Aesop_SearchM_instMonadLiftTreeM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Meta_instAddMessageContextMetaM;
lean_object* l_Lean_instAddMessageContextOfMonadLift___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(lean_object*, lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_Lean_throwError___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_runFirstSafeRule___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_registerInternalExceptionId(lean_object*);
lean_object* lp_aesop_Aesop_normalizeGoalIfNecessary___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lp_aesop_Aesop_Goal_hasSafeRapp(lean_object*);
extern lean_object* lp_aesop_Aesop_TraceOption_steps;
lean_object* lp_aesop_Aesop_TraceOption_isEnabled___redArg(lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Core_instMonadTraceCoreM;
lean_object* l_Lean_instMonadTraceOfMonadLift___redArg(lean_object*, lean_object*);
lean_object* l_Lean_addTrace___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_instBEqInternalExceptionId_beq(lean_object*, lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* lp_aesop_Aesop_getRootGoal(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_st_mk_ref(lean_object*);
lean_object* lp_aesop_Aesop_SearchM_instMonadRef(lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_initFn___closed__0_00___x40_Aesop_Search_ExpandSafePrefix_1752386596____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Aesop"};
static const lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_initFn___closed__0_00___x40_Aesop_Search_ExpandSafePrefix_1752386596____hygCtx___hyg_5_ = (const lean_object*)&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_initFn___closed__0_00___x40_Aesop_Search_ExpandSafePrefix_1752386596____hygCtx___hyg_5__value;
static const lean_string_object lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_initFn___closed__1_00___x40_Aesop_Search_ExpandSafePrefix_1752386596____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = "safeExpansionFailedException"};
static const lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_initFn___closed__1_00___x40_Aesop_Search_ExpandSafePrefix_1752386596____hygCtx___hyg_5_ = (const lean_object*)&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_initFn___closed__1_00___x40_Aesop_Search_ExpandSafePrefix_1752386596____hygCtx___hyg_5__value;
static const lean_ctor_object lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_initFn___closed__2_00___x40_Aesop_Search_ExpandSafePrefix_1752386596____hygCtx___hyg_5__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_initFn___closed__0_00___x40_Aesop_Search_ExpandSafePrefix_1752386596____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(97, 226, 101, 135, 78, 117, 164, 248)}};
static const lean_ctor_object lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_initFn___closed__2_00___x40_Aesop_Search_ExpandSafePrefix_1752386596____hygCtx___hyg_5__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_initFn___closed__2_00___x40_Aesop_Search_ExpandSafePrefix_1752386596____hygCtx___hyg_5__value_aux_0),((lean_object*)&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_initFn___closed__1_00___x40_Aesop_Search_ExpandSafePrefix_1752386596____hygCtx___hyg_5__value),LEAN_SCALAR_PTR_LITERAL(101, 66, 149, 18, 248, 145, 121, 101)}};
static const lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_initFn___closed__2_00___x40_Aesop_Search_ExpandSafePrefix_1752386596____hygCtx___hyg_5_ = (const lean_object*)&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_initFn___closed__2_00___x40_Aesop_Search_ExpandSafePrefix_1752386596____hygCtx___hyg_5__value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_initFn_00___x40_Aesop_Search_ExpandSafePrefix_1752386596____hygCtx___hyg_5_();
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_initFn_00___x40_Aesop_Search_ExpandSafePrefix_1752386596____hygCtx___hyg_5____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_safeExpansionFailedExceptionId;
static lean_once_cell_t lp_aesop_Aesop_safeExpansionFailedException___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_safeExpansionFailedException___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Aesop_safeExpansionFailedException;
LEAN_EXPORT uint8_t lp_aesop_Aesop_isSafeExpansionFailedException(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_isSafeExpansionFailedException___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_Goal_isSafeExpanded(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_Goal_isSafeExpanded___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_liftSearchM___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_liftSearchM___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_liftSearchM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_liftSearchM___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandFirstPrefixRapp___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandFirstPrefixRapp___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__4;
static lean_once_cell_t lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__3;
static lean_once_cell_t lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__5;
static lean_once_cell_t lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__7;
static lean_once_cell_t lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__6;
static lean_once_cell_t lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__8;
static lean_once_cell_t lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__10;
static lean_once_cell_t lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__9;
static lean_once_cell_t lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__11_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__11;
static lean_once_cell_t lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__13;
static lean_once_cell_t lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__12;
static lean_once_cell_t lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__14;
static lean_once_cell_t lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__16;
static lean_once_cell_t lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__15_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__15;
static lean_once_cell_t lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__17;
static lean_once_cell_t lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__19;
static lean_once_cell_t lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__18;
static lean_once_cell_t lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__20;
static lean_once_cell_t lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__22_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__22;
static lean_once_cell_t lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__21;
static lean_once_cell_t lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__23;
static const lean_closure_object lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_StateRefT_x27_lift___boxed, .m_arity = 6, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__1 = (const lean_object*)&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__1_value;
static const lean_closure_object lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_StateRefT_x27_instMonadFunctor___aux__1___boxed, .m_arity = 7, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__24 = (const lean_object*)&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__24_value;
static lean_once_cell_t lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__26_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__26;
static const lean_closure_object lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_ReaderT_instMonadLift___lam__0___boxed, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__0_value;
static const lean_closure_object lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_ReaderT_instMonadFunctor___lam__0, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__25 = (const lean_object*)&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__25_value;
static lean_once_cell_t lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__27_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__27;
static lean_once_cell_t lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__28_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__28;
static lean_once_cell_t lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__29_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__29;
static lean_once_cell_t lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__30_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__30;
static lean_once_cell_t lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__31_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__31;
static lean_once_cell_t lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__32_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__32;
static const lean_closure_object lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_SearchM_instMonadLiftTreeM___lam__0___boxed, .m_arity = 11, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__2 = (const lean_object*)&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__2_value;
static lean_once_cell_t lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__33_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__33;
static lean_once_cell_t lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__34_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__34;
static lean_once_cell_t lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__35_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__35;
static lean_once_cell_t lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__36_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__36;
static lean_once_cell_t lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__37_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__37;
static const lean_string_object lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = "aesop: internal error: goal "};
static const lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__38 = (const lean_object*)&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__38_value;
static lean_once_cell_t lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__39_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__39;
static const lean_string_object lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = " has multiple safe rapps"};
static const lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__40 = (const lean_object*)&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__40_value;
static lean_once_cell_t lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__41_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__41;
static lean_once_cell_t lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__42_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__42;
static lean_once_cell_t lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__43_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__43;
static lean_once_cell_t lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__44_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__44;
static lean_once_cell_t lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__45_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__45;
static lean_once_cell_t lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__46_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__46;
static lean_once_cell_t lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__47_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__47;
static lean_once_cell_t lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__48_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__48;
static lean_once_cell_t lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__49_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__49;
static lean_once_cell_t lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__50_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__50;
static lean_once_cell_t lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__51_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__51;
static lean_once_cell_t lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__52_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__52;
static const lean_string_object lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__53_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = "Applying safe rules to goal "};
static const lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__53 = (const lean_object*)&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__53_value;
static lean_once_cell_t lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__54_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__54;
static const lean_string_object lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__55_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "."};
static const lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__55 = (const lean_object*)&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__55_value;
static lean_once_cell_t lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__56_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__56;
static const lean_string_object lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__57_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 38, .m_capacity = 38, .m_length = 37, .m_data = "Skipping safe rule expansion of goal "};
static const lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__57 = (const lean_object*)&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__57_value;
static lean_once_cell_t lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__58_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__58;
static const lean_string_object lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__59_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 45, .m_capacity = 45, .m_length = 44, .m_data = " since safe rules have already been applied."};
static const lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__59 = (const lean_object*)&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__59_value;
static lean_once_cell_t lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__60_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__60;
static const lean_string_object lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__61_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = " since it is already proven."};
static const lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__61 = (const lean_object*)&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__61_value;
static lean_once_cell_t lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__62_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__62;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixMVarCluster___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixMVarCluster___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixMVarCluster___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandFirstPrefixRapp___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandFirstPrefixRapp___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixMVarCluster___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandFirstPrefixRapp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandFirstPrefixRapp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixMVarCluster(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixMVarCluster___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_expandSafePrefix___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 41, .m_capacity = 41, .m_length = 40, .m_data = "Expanding safe subtree of the root goal."};
static const lean_object* lp_aesop_Aesop_expandSafePrefix___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_expandSafePrefix___redArg___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_expandSafePrefix___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_expandSafePrefix___redArg___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_expandSafePrefix___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_expandSafePrefix___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_expandSafePrefix(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_expandSafePrefix___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_initFn_00___x40_Aesop_Search_ExpandSafePrefix_1752386596____hygCtx___hyg_5_(){
_start:
{
lean_object* v___x_7_; lean_object* v___x_8_; 
v___x_7_ = ((lean_object*)(lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_initFn___closed__2_00___x40_Aesop_Search_ExpandSafePrefix_1752386596____hygCtx___hyg_5_));
v___x_8_ = l_Lean_registerInternalExceptionId(v___x_7_);
return v___x_8_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_initFn_00___x40_Aesop_Search_ExpandSafePrefix_1752386596____hygCtx___hyg_5____boxed(lean_object* v_a_9_){
_start:
{
lean_object* v_res_10_; 
v_res_10_ = lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_initFn_00___x40_Aesop_Search_ExpandSafePrefix_1752386596____hygCtx___hyg_5_();
return v_res_10_;
}
}
static lean_object* _init_lp_aesop_Aesop_safeExpansionFailedException___closed__0(void){
_start:
{
lean_object* v___x_11_; lean_object* v___x_12_; lean_object* v___x_13_; 
v___x_11_ = lean_box(0);
v___x_12_ = lp_aesop_Aesop_safeExpansionFailedExceptionId;
v___x_13_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_13_, 0, v___x_12_);
lean_ctor_set(v___x_13_, 1, v___x_11_);
return v___x_13_;
}
}
static lean_object* _init_lp_aesop_Aesop_safeExpansionFailedException(void){
_start:
{
lean_object* v___x_14_; 
v___x_14_ = lean_obj_once(&lp_aesop_Aesop_safeExpansionFailedException___closed__0, &lp_aesop_Aesop_safeExpansionFailedException___closed__0_once, _init_lp_aesop_Aesop_safeExpansionFailedException___closed__0);
return v___x_14_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_isSafeExpansionFailedException(lean_object* v_x_15_){
_start:
{
if (lean_obj_tag(v_x_15_) == 1)
{
lean_object* v_id_16_; lean_object* v___x_17_; uint8_t v___x_18_; 
v_id_16_ = lean_ctor_get(v_x_15_, 0);
v___x_17_ = lp_aesop_Aesop_safeExpansionFailedExceptionId;
v___x_18_ = l_Lean_instBEqInternalExceptionId_beq(v_id_16_, v___x_17_);
return v___x_18_;
}
else
{
uint8_t v___x_19_; 
v___x_19_ = 0;
return v___x_19_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_isSafeExpansionFailedException___boxed(lean_object* v_x_20_){
_start:
{
uint8_t v_res_21_; lean_object* v_r_22_; 
v_res_21_ = lp_aesop_Aesop_isSafeExpansionFailedException(v_x_20_);
lean_dec_ref(v_x_20_);
v_r_22_ = lean_box(v_res_21_);
return v_r_22_;
}
}
LEAN_EXPORT uint8_t lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_Goal_isSafeExpanded(lean_object* v_g_23_){
_start:
{
lean_object* v___x_25_; lean_object* v_elimGoal_26_; lean_object* v___x_27_; uint8_t v_unsafeRulesSelected_28_; 
v___x_25_ = lp_aesop_Aesop_treeImpl;
v_elimGoal_26_ = lean_ctor_get(v___x_25_, 1);
lean_inc_ref(v_elimGoal_26_);
lean_inc(v_g_23_);
v___x_27_ = lean_apply_1(v_elimGoal_26_, v_g_23_);
v_unsafeRulesSelected_28_ = lean_ctor_get_uint8(v___x_27_, sizeof(void*)*14 + 11);
lean_dec_ref(v___x_27_);
if (v_unsafeRulesSelected_28_ == 0)
{
uint8_t v___x_29_; 
v___x_29_ = lp_aesop_Aesop_Goal_hasSafeRapp(v_g_23_);
return v___x_29_;
}
else
{
lean_dec(v_g_23_);
return v_unsafeRulesSelected_28_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_Goal_isSafeExpanded___boxed(lean_object* v_g_30_, lean_object* v_a_31_){
_start:
{
uint8_t v_res_32_; lean_object* v_r_33_; 
v_res_32_ = lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_Goal_isSafeExpanded(v_g_30_);
v_r_33_ = lean_box(v_res_32_);
return v_r_33_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_liftSearchM___redArg(lean_object* v_x_34_, lean_object* v_a_35_, lean_object* v_a_36_, lean_object* v_a_37_, lean_object* v_a_38_, lean_object* v_a_39_, lean_object* v_a_40_, lean_object* v_a_41_, lean_object* v_a_42_){
_start:
{
lean_object* v___x_44_; 
lean_inc(v_a_42_);
lean_inc_ref(v_a_41_);
lean_inc(v_a_40_);
lean_inc_ref(v_a_39_);
lean_inc(v_a_38_);
lean_inc(v_a_37_);
lean_inc(v_a_36_);
lean_inc_ref(v_a_35_);
v___x_44_ = lean_apply_9(v_x_34_, v_a_35_, v_a_36_, v_a_37_, v_a_38_, v_a_39_, v_a_40_, v_a_41_, v_a_42_, lean_box(0));
return v___x_44_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_liftSearchM___redArg___boxed(lean_object* v_x_45_, lean_object* v_a_46_, lean_object* v_a_47_, lean_object* v_a_48_, lean_object* v_a_49_, lean_object* v_a_50_, lean_object* v_a_51_, lean_object* v_a_52_, lean_object* v_a_53_, lean_object* v_a_54_){
_start:
{
lean_object* v_res_55_; 
v_res_55_ = lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_liftSearchM___redArg(v_x_45_, v_a_46_, v_a_47_, v_a_48_, v_a_49_, v_a_50_, v_a_51_, v_a_52_, v_a_53_);
lean_dec(v_a_53_);
lean_dec_ref(v_a_52_);
lean_dec(v_a_51_);
lean_dec_ref(v_a_50_);
lean_dec(v_a_49_);
lean_dec(v_a_48_);
lean_dec(v_a_47_);
lean_dec_ref(v_a_46_);
return v_res_55_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_liftSearchM(lean_object* v_Q_56_, lean_object* v_inst_57_, lean_object* v_00_u03b1_58_, lean_object* v_x_59_, lean_object* v_a_60_, lean_object* v_a_61_, lean_object* v_a_62_, lean_object* v_a_63_, lean_object* v_a_64_, lean_object* v_a_65_, lean_object* v_a_66_, lean_object* v_a_67_, lean_object* v_a_68_){
_start:
{
lean_object* v___x_70_; 
lean_inc(v_a_68_);
lean_inc_ref(v_a_67_);
lean_inc(v_a_66_);
lean_inc_ref(v_a_65_);
lean_inc(v_a_64_);
lean_inc(v_a_63_);
lean_inc(v_a_62_);
lean_inc_ref(v_a_61_);
v___x_70_ = lean_apply_9(v_x_59_, v_a_61_, v_a_62_, v_a_63_, v_a_64_, v_a_65_, v_a_66_, v_a_67_, v_a_68_, lean_box(0));
return v___x_70_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_liftSearchM___boxed(lean_object* v_Q_71_, lean_object* v_inst_72_, lean_object* v_00_u03b1_73_, lean_object* v_x_74_, lean_object* v_a_75_, lean_object* v_a_76_, lean_object* v_a_77_, lean_object* v_a_78_, lean_object* v_a_79_, lean_object* v_a_80_, lean_object* v_a_81_, lean_object* v_a_82_, lean_object* v_a_83_, lean_object* v_a_84_){
_start:
{
lean_object* v_res_85_; 
v_res_85_ = lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_liftSearchM(v_Q_71_, v_inst_72_, v_00_u03b1_73_, v_x_74_, v_a_75_, v_a_76_, v_a_77_, v_a_78_, v_a_79_, v_a_80_, v_a_81_, v_a_82_, v_a_83_);
lean_dec(v_a_83_);
lean_dec_ref(v_a_82_);
lean_dec(v_a_81_);
lean_dec_ref(v_a_80_);
lean_dec(v_a_79_);
lean_dec(v_a_78_);
lean_dec(v_a_77_);
lean_dec_ref(v_a_76_);
lean_dec(v_a_75_);
lean_dec_ref(v_inst_72_);
return v_res_85_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandFirstPrefixRapp___redArg___lam__0___boxed(lean_object* v_inst_86_, lean_object* v_x_87_, lean_object* v___y_88_, lean_object* v___y_89_, lean_object* v___y_90_, lean_object* v___y_91_, lean_object* v___y_92_, lean_object* v___y_93_, lean_object* v___y_94_, lean_object* v___y_95_, lean_object* v___y_96_, lean_object* v___y_97_, lean_object* v___y_98_){
_start:
{
lean_object* v_res_99_; 
v_res_99_ = lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandFirstPrefixRapp___redArg___lam__0(v_inst_86_, v_x_87_, v___y_88_, v___y_89_, v___y_90_, v___y_91_, v___y_92_, v___y_93_, v___y_94_, v___y_95_, v___y_96_, v___y_97_);
lean_dec(v___y_97_);
lean_dec_ref(v___y_96_);
lean_dec(v___y_95_);
lean_dec_ref(v___y_94_);
lean_dec(v___y_93_);
lean_dec(v___y_92_);
lean_dec(v___y_91_);
lean_dec_ref(v___y_90_);
lean_dec(v___y_89_);
lean_dec(v___y_88_);
return v_res_99_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandFirstPrefixRapp___redArg(lean_object* v_inst_100_, lean_object* v_rref_101_, lean_object* v_a_102_, lean_object* v_a_103_, lean_object* v_a_104_, lean_object* v_a_105_, lean_object* v_a_106_, lean_object* v_a_107_, lean_object* v_a_108_, lean_object* v_a_109_, lean_object* v_a_110_){
_start:
{
lean_object* v___x_112_; lean_object* v___x_113_; lean_object* v___x_114_; lean_object* v___x_115_; lean_object* v___x_116_; lean_object* v_elimRapp_117_; lean_object* v___x_118_; lean_object* v_children_119_; lean_object* v___x_120_; lean_object* v___x_121_; lean_object* v___x_122_; uint8_t v___x_123_; 
v___x_112_ = lp_aesop_Aesop_SearchM_instMonad(lean_box(0), v_inst_100_);
v___x_113_ = l_StateRefT_x27_instMonad___redArg(v___x_112_);
v___x_114_ = lean_st_ref_get(v_a_104_);
lean_dec(v___x_114_);
v___x_115_ = lean_st_ref_get(v_rref_101_);
v___x_116_ = lp_aesop_Aesop_treeImpl;
v_elimRapp_117_ = lean_ctor_get(v___x_116_, 3);
lean_inc_ref(v_elimRapp_117_);
v___x_118_ = lean_apply_1(v_elimRapp_117_, v___x_115_);
v_children_119_ = lean_ctor_get(v___x_118_, 2);
lean_inc_ref(v_children_119_);
lean_dec_ref(v___x_118_);
v___x_120_ = lean_unsigned_to_nat(0u);
v___x_121_ = lean_array_get_size(v_children_119_);
v___x_122_ = lean_box(0);
v___x_123_ = lean_nat_dec_lt(v___x_120_, v___x_121_);
if (v___x_123_ == 0)
{
lean_object* v___x_124_; 
lean_dec_ref(v_children_119_);
lean_dec_ref(v___x_113_);
lean_dec_ref(v_inst_100_);
v___x_124_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_124_, 0, v___x_122_);
return v___x_124_;
}
else
{
lean_object* v___f_125_; uint8_t v___x_126_; 
v___f_125_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandFirstPrefixRapp___redArg___lam__0___boxed), 13, 1);
lean_closure_set(v___f_125_, 0, v_inst_100_);
v___x_126_ = lean_nat_dec_le(v___x_121_, v___x_121_);
if (v___x_126_ == 0)
{
if (v___x_123_ == 0)
{
lean_object* v___x_127_; 
lean_dec_ref(v___f_125_);
lean_dec_ref(v_children_119_);
lean_dec_ref(v___x_113_);
v___x_127_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_127_, 0, v___x_122_);
return v___x_127_;
}
else
{
size_t v___x_128_; size_t v___x_129_; lean_object* v___x_47272__overap_130_; lean_object* v___x_131_; 
v___x_128_ = ((size_t)0ULL);
v___x_129_ = lean_usize_of_nat(v___x_121_);
v___x_47272__overap_130_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_113_, v___f_125_, v_children_119_, v___x_128_, v___x_129_, v___x_122_);
lean_inc(v_a_110_);
lean_inc_ref(v_a_109_);
lean_inc(v_a_108_);
lean_inc_ref(v_a_107_);
lean_inc(v_a_106_);
lean_inc(v_a_105_);
lean_inc(v_a_104_);
lean_inc_ref(v_a_103_);
lean_inc(v_a_102_);
v___x_131_ = lean_apply_10(v___x_47272__overap_130_, v_a_102_, v_a_103_, v_a_104_, v_a_105_, v_a_106_, v_a_107_, v_a_108_, v_a_109_, v_a_110_, lean_box(0));
return v___x_131_;
}
}
else
{
size_t v___x_132_; size_t v___x_133_; lean_object* v___x_47276__overap_134_; lean_object* v___x_135_; 
v___x_132_ = ((size_t)0ULL);
v___x_133_ = lean_usize_of_nat(v___x_121_);
v___x_47276__overap_134_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_113_, v___f_125_, v_children_119_, v___x_132_, v___x_133_, v___x_122_);
lean_inc(v_a_110_);
lean_inc_ref(v_a_109_);
lean_inc(v_a_108_);
lean_inc_ref(v_a_107_);
lean_inc(v_a_106_);
lean_inc(v_a_105_);
lean_inc(v_a_104_);
lean_inc_ref(v_a_103_);
lean_inc(v_a_102_);
v___x_135_ = lean_apply_10(v___x_47276__overap_134_, v_a_102_, v_a_103_, v_a_104_, v_a_105_, v_a_106_, v_a_107_, v_a_108_, v_a_109_, v_a_110_, lean_box(0));
return v___x_135_;
}
}
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__4(void){
_start:
{
lean_object* v___x_136_; lean_object* v___f_137_; 
v___x_136_ = l_Lean_instMonadExceptOfExceptionCoreM;
v___f_137_ = lean_alloc_closure((void*)(l_StateRefT_x27_instMonadExceptOf___redArg___lam__2), 5, 1);
lean_closure_set(v___f_137_, 0, v___x_136_);
return v___f_137_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__3(void){
_start:
{
lean_object* v___x_138_; lean_object* v___f_139_; 
v___x_138_ = l_Lean_instMonadExceptOfExceptionCoreM;
v___f_139_ = lean_alloc_closure((void*)(l_StateRefT_x27_instMonadExceptOf___redArg___lam__0___boxed), 4, 1);
lean_closure_set(v___f_139_, 0, v___x_138_);
return v___f_139_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__5(void){
_start:
{
lean_object* v___f_140_; lean_object* v___f_141_; lean_object* v___x_142_; 
v___f_140_ = lean_obj_once(&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__4, &lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__4_once, _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__4);
v___f_141_ = lean_obj_once(&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__3, &lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__3_once, _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__3);
v___x_142_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_142_, 0, v___f_141_);
lean_ctor_set(v___x_142_, 1, v___f_140_);
return v___x_142_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__7(void){
_start:
{
lean_object* v___x_143_; lean_object* v___f_144_; 
v___x_143_ = lean_obj_once(&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__5, &lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__5_once, _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__5);
v___f_144_ = lean_alloc_closure((void*)(l_ReaderT_instMonadExceptOf___redArg___lam__2), 5, 1);
lean_closure_set(v___f_144_, 0, v___x_143_);
return v___f_144_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__6(void){
_start:
{
lean_object* v___x_145_; lean_object* v___f_146_; 
v___x_145_ = lean_obj_once(&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__5, &lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__5_once, _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__5);
v___f_146_ = lean_alloc_closure((void*)(l_ReaderT_instMonadExceptOf___redArg___lam__0___boxed), 4, 1);
lean_closure_set(v___f_146_, 0, v___x_145_);
return v___f_146_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__8(void){
_start:
{
lean_object* v___f_147_; lean_object* v___f_148_; lean_object* v___x_149_; 
v___f_147_ = lean_obj_once(&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__7, &lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__7_once, _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__7);
v___f_148_ = lean_obj_once(&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__6, &lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__6_once, _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__6);
v___x_149_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_149_, 0, v___f_148_);
lean_ctor_set(v___x_149_, 1, v___f_147_);
return v___x_149_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__10(void){
_start:
{
lean_object* v___x_150_; lean_object* v___f_151_; 
v___x_150_ = lean_obj_once(&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__8, &lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__8_once, _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__8);
v___f_151_ = lean_alloc_closure((void*)(l_StateRefT_x27_instMonadExceptOf___redArg___lam__2), 5, 1);
lean_closure_set(v___f_151_, 0, v___x_150_);
return v___f_151_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__9(void){
_start:
{
lean_object* v___x_152_; lean_object* v___f_153_; 
v___x_152_ = lean_obj_once(&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__8, &lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__8_once, _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__8);
v___f_153_ = lean_alloc_closure((void*)(l_StateRefT_x27_instMonadExceptOf___redArg___lam__0___boxed), 4, 1);
lean_closure_set(v___f_153_, 0, v___x_152_);
return v___f_153_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__11(void){
_start:
{
lean_object* v___f_154_; lean_object* v___f_155_; lean_object* v___x_156_; 
v___f_154_ = lean_obj_once(&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__10, &lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__10_once, _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__10);
v___f_155_ = lean_obj_once(&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__9, &lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__9_once, _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__9);
v___x_156_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_156_, 0, v___f_155_);
lean_ctor_set(v___x_156_, 1, v___f_154_);
return v___x_156_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__13(void){
_start:
{
lean_object* v___x_157_; lean_object* v___f_158_; 
v___x_157_ = lean_obj_once(&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__11, &lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__11_once, _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__11);
v___f_158_ = lean_alloc_closure((void*)(l_StateRefT_x27_instMonadExceptOf___redArg___lam__2), 5, 1);
lean_closure_set(v___f_158_, 0, v___x_157_);
return v___f_158_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__12(void){
_start:
{
lean_object* v___x_159_; lean_object* v___f_160_; 
v___x_159_ = lean_obj_once(&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__11, &lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__11_once, _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__11);
v___f_160_ = lean_alloc_closure((void*)(l_StateRefT_x27_instMonadExceptOf___redArg___lam__0___boxed), 4, 1);
lean_closure_set(v___f_160_, 0, v___x_159_);
return v___f_160_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__14(void){
_start:
{
lean_object* v___f_161_; lean_object* v___f_162_; lean_object* v___x_163_; 
v___f_161_ = lean_obj_once(&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__13, &lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__13_once, _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__13);
v___f_162_ = lean_obj_once(&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__12, &lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__12_once, _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__12);
v___x_163_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_163_, 0, v___f_162_);
lean_ctor_set(v___x_163_, 1, v___f_161_);
return v___x_163_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__16(void){
_start:
{
lean_object* v___x_164_; lean_object* v___f_165_; 
v___x_164_ = lean_obj_once(&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__14, &lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__14_once, _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__14);
v___f_165_ = lean_alloc_closure((void*)(l_StateRefT_x27_instMonadExceptOf___redArg___lam__2), 5, 1);
lean_closure_set(v___f_165_, 0, v___x_164_);
return v___f_165_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__15(void){
_start:
{
lean_object* v___x_166_; lean_object* v___f_167_; 
v___x_166_ = lean_obj_once(&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__14, &lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__14_once, _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__14);
v___f_167_ = lean_alloc_closure((void*)(l_StateRefT_x27_instMonadExceptOf___redArg___lam__0___boxed), 4, 1);
lean_closure_set(v___f_167_, 0, v___x_166_);
return v___f_167_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__17(void){
_start:
{
lean_object* v___f_168_; lean_object* v___f_169_; lean_object* v___x_170_; 
v___f_168_ = lean_obj_once(&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__16, &lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__16_once, _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__16);
v___f_169_ = lean_obj_once(&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__15, &lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__15_once, _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__15);
v___x_170_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_170_, 0, v___f_169_);
lean_ctor_set(v___x_170_, 1, v___f_168_);
return v___x_170_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__19(void){
_start:
{
lean_object* v___x_171_; lean_object* v___f_172_; 
v___x_171_ = lean_obj_once(&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__17, &lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__17_once, _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__17);
v___f_172_ = lean_alloc_closure((void*)(l_ReaderT_instMonadExceptOf___redArg___lam__2), 5, 1);
lean_closure_set(v___f_172_, 0, v___x_171_);
return v___f_172_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__18(void){
_start:
{
lean_object* v___x_173_; lean_object* v___f_174_; 
v___x_173_ = lean_obj_once(&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__17, &lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__17_once, _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__17);
v___f_174_ = lean_alloc_closure((void*)(l_ReaderT_instMonadExceptOf___redArg___lam__0___boxed), 4, 1);
lean_closure_set(v___f_174_, 0, v___x_173_);
return v___f_174_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__20(void){
_start:
{
lean_object* v___f_175_; lean_object* v___f_176_; lean_object* v___x_177_; 
v___f_175_ = lean_obj_once(&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__19, &lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__19_once, _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__19);
v___f_176_ = lean_obj_once(&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__18, &lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__18_once, _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__18);
v___x_177_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_177_, 0, v___f_176_);
lean_ctor_set(v___x_177_, 1, v___f_175_);
return v___x_177_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__22(void){
_start:
{
lean_object* v___x_178_; lean_object* v___f_179_; 
v___x_178_ = lean_obj_once(&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__20, &lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__20_once, _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__20);
v___f_179_ = lean_alloc_closure((void*)(l_StateRefT_x27_instMonadExceptOf___redArg___lam__2), 5, 1);
lean_closure_set(v___f_179_, 0, v___x_178_);
return v___f_179_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__21(void){
_start:
{
lean_object* v___x_180_; lean_object* v___f_181_; 
v___x_180_ = lean_obj_once(&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__20, &lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__20_once, _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__20);
v___f_181_ = lean_alloc_closure((void*)(l_StateRefT_x27_instMonadExceptOf___redArg___lam__0___boxed), 4, 1);
lean_closure_set(v___f_181_, 0, v___x_180_);
return v___f_181_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__23(void){
_start:
{
lean_object* v___f_182_; lean_object* v___f_183_; lean_object* v___x_184_; 
v___f_182_ = lean_obj_once(&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__22, &lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__22_once, _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__22);
v___f_183_ = lean_obj_once(&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__21, &lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__21_once, _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__21);
v___x_184_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_184_, 0, v___f_183_);
lean_ctor_set(v___x_184_, 1, v___f_182_);
return v___x_184_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__26(void){
_start:
{
lean_object* v___x_187_; lean_object* v___x_188_; lean_object* v___x_189_; lean_object* v___x_190_; 
v___x_187_ = l_Lean_Core_instMonadQuotationCoreM;
v___x_188_ = ((lean_object*)(lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__1));
v___x_189_ = ((lean_object*)(lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__24));
v___x_190_ = l_Lean_instMonadQuotationOfMonadFunctorOfMonadLift___redArg(v___x_189_, v___x_188_, v___x_187_);
return v___x_190_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__27(void){
_start:
{
lean_object* v___x_193_; lean_object* v___f_194_; lean_object* v___f_195_; lean_object* v___x_196_; 
v___x_193_ = lean_obj_once(&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__26, &lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__26_once, _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__26);
v___f_194_ = ((lean_object*)(lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__0));
v___f_195_ = ((lean_object*)(lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__25));
v___x_196_ = l_Lean_instMonadQuotationOfMonadFunctorOfMonadLift___redArg(v___f_195_, v___f_194_, v___x_193_);
return v___x_196_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__28(void){
_start:
{
lean_object* v___x_197_; lean_object* v___x_198_; lean_object* v___x_199_; lean_object* v___x_200_; 
v___x_197_ = lean_obj_once(&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__27, &lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__27_once, _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__27);
v___x_198_ = ((lean_object*)(lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__1));
v___x_199_ = ((lean_object*)(lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__24));
v___x_200_ = l_Lean_instMonadQuotationOfMonadFunctorOfMonadLift___redArg(v___x_199_, v___x_198_, v___x_197_);
return v___x_200_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__29(void){
_start:
{
lean_object* v___x_201_; lean_object* v___x_202_; lean_object* v___x_203_; lean_object* v___x_204_; 
v___x_201_ = lean_obj_once(&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__28, &lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__28_once, _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__28);
v___x_202_ = ((lean_object*)(lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__1));
v___x_203_ = ((lean_object*)(lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__24));
v___x_204_ = l_Lean_instMonadQuotationOfMonadFunctorOfMonadLift___redArg(v___x_203_, v___x_202_, v___x_201_);
return v___x_204_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__30(void){
_start:
{
lean_object* v___x_205_; lean_object* v___x_206_; lean_object* v___x_207_; lean_object* v___x_208_; 
v___x_205_ = lean_obj_once(&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__29, &lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__29_once, _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__29);
v___x_206_ = ((lean_object*)(lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__1));
v___x_207_ = ((lean_object*)(lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__24));
v___x_208_ = l_Lean_instMonadQuotationOfMonadFunctorOfMonadLift___redArg(v___x_207_, v___x_206_, v___x_205_);
return v___x_208_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__31(void){
_start:
{
lean_object* v___x_209_; lean_object* v___f_210_; lean_object* v___f_211_; lean_object* v___x_212_; 
v___x_209_ = lean_obj_once(&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__30, &lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__30_once, _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__30);
v___f_210_ = ((lean_object*)(lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__0));
v___f_211_ = ((lean_object*)(lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__25));
v___x_212_ = l_Lean_instMonadQuotationOfMonadFunctorOfMonadLift___redArg(v___f_211_, v___f_210_, v___x_209_);
return v___x_212_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__32(void){
_start:
{
lean_object* v___x_213_; lean_object* v___x_214_; lean_object* v___x_215_; lean_object* v___x_216_; 
v___x_213_ = lean_obj_once(&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__31, &lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__31_once, _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__31);
v___x_214_ = ((lean_object*)(lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__1));
v___x_215_ = ((lean_object*)(lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__24));
v___x_216_ = l_Lean_instMonadQuotationOfMonadFunctorOfMonadLift___redArg(v___x_215_, v___x_214_, v___x_213_);
return v___x_216_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__33(void){
_start:
{
lean_object* v___x_218_; lean_object* v___x_219_; lean_object* v___f_220_; 
v___x_218_ = ((lean_object*)(lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__1));
v___x_219_ = l_Lean_Meta_instAddMessageContextMetaM;
v___f_220_ = lean_alloc_closure((void*)(l_Lean_instAddMessageContextOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(v___f_220_, 0, v___x_219_);
lean_closure_set(v___f_220_, 1, v___x_218_);
return v___f_220_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__34(void){
_start:
{
lean_object* v___x_221_; lean_object* v___f_222_; lean_object* v___f_223_; 
v___x_221_ = ((lean_object*)(lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__1));
v___f_222_ = lean_obj_once(&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__33, &lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__33_once, _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__33);
v___f_223_ = lean_alloc_closure((void*)(l_Lean_instAddMessageContextOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(v___f_223_, 0, v___f_222_);
lean_closure_set(v___f_223_, 1, v___x_221_);
return v___f_223_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__35(void){
_start:
{
lean_object* v___f_224_; lean_object* v___f_225_; lean_object* v___f_226_; 
v___f_224_ = ((lean_object*)(lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__0));
v___f_225_ = lean_obj_once(&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__34, &lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__34_once, _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__34);
v___f_226_ = lean_alloc_closure((void*)(l_Lean_instAddMessageContextOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(v___f_226_, 0, v___f_225_);
lean_closure_set(v___f_226_, 1, v___f_224_);
return v___f_226_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__36(void){
_start:
{
lean_object* v___f_227_; lean_object* v___f_228_; lean_object* v___f_229_; 
v___f_227_ = ((lean_object*)(lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__2));
v___f_228_ = lean_obj_once(&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__35, &lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__35_once, _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__35);
v___f_229_ = lean_alloc_closure((void*)(l_Lean_instAddMessageContextOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(v___f_229_, 0, v___f_228_);
lean_closure_set(v___f_229_, 1, v___f_227_);
return v___f_229_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__37(void){
_start:
{
lean_object* v___x_230_; lean_object* v___f_231_; lean_object* v___f_232_; 
v___x_230_ = ((lean_object*)(lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__1));
v___f_231_ = lean_obj_once(&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__36, &lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__36_once, _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__36);
v___f_232_ = lean_alloc_closure((void*)(l_Lean_instAddMessageContextOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(v___f_232_, 0, v___f_231_);
lean_closure_set(v___f_232_, 1, v___x_230_);
return v___f_232_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__39(void){
_start:
{
lean_object* v___x_234_; lean_object* v___x_235_; 
v___x_234_ = ((lean_object*)(lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__38));
v___x_235_ = l_Lean_stringToMessageData(v___x_234_);
return v___x_235_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__41(void){
_start:
{
lean_object* v___x_237_; lean_object* v___x_238_; 
v___x_237_ = ((lean_object*)(lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__40));
v___x_238_ = l_Lean_stringToMessageData(v___x_237_);
return v___x_238_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__42(void){
_start:
{
lean_object* v___x_239_; lean_object* v___x_240_; 
v___x_239_ = lp_aesop_Aesop_BaseM_instMonadStats;
v___x_240_ = lp_aesop_Aesop_instMonadStatsStateRefT_x27___redArg(v___x_239_);
return v___x_240_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__43(void){
_start:
{
lean_object* v___x_241_; lean_object* v___x_242_; 
v___x_241_ = lean_obj_once(&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__42, &lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__42_once, _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__42);
v___x_242_ = lp_aesop_Aesop_instMonadStatsStateRefT_x27___redArg(v___x_241_);
return v___x_242_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__44(void){
_start:
{
lean_object* v___x_243_; lean_object* v___x_244_; 
v___x_243_ = lean_obj_once(&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__43, &lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__43_once, _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__43);
v___x_244_ = lp_aesop_Aesop_instMonadStatsReaderT___redArg(v___x_243_);
return v___x_244_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__45(void){
_start:
{
lean_object* v___x_245_; lean_object* v___x_246_; 
v___x_245_ = lean_obj_once(&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__44, &lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__44_once, _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__44);
v___x_246_ = lp_aesop_Aesop_instMonadStatsStateRefT_x27___redArg(v___x_245_);
return v___x_246_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__46(void){
_start:
{
lean_object* v___x_247_; lean_object* v___x_248_; lean_object* v___x_249_; 
v___x_247_ = l_Lean_Core_instMonadTraceCoreM;
v___x_248_ = ((lean_object*)(lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__1));
v___x_249_ = l_Lean_instMonadTraceOfMonadLift___redArg(v___x_248_, v___x_247_);
return v___x_249_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__47(void){
_start:
{
lean_object* v___x_250_; lean_object* v___f_251_; lean_object* v___x_252_; 
v___x_250_ = lean_obj_once(&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__46, &lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__46_once, _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__46);
v___f_251_ = ((lean_object*)(lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__0));
v___x_252_ = l_Lean_instMonadTraceOfMonadLift___redArg(v___f_251_, v___x_250_);
return v___x_252_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__48(void){
_start:
{
lean_object* v___x_253_; lean_object* v___x_254_; lean_object* v___x_255_; 
v___x_253_ = lean_obj_once(&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__47, &lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__47_once, _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__47);
v___x_254_ = ((lean_object*)(lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__1));
v___x_255_ = l_Lean_instMonadTraceOfMonadLift___redArg(v___x_254_, v___x_253_);
return v___x_255_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__49(void){
_start:
{
lean_object* v___x_256_; lean_object* v___x_257_; lean_object* v___x_258_; 
v___x_256_ = lean_obj_once(&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__48, &lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__48_once, _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__48);
v___x_257_ = ((lean_object*)(lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__1));
v___x_258_ = l_Lean_instMonadTraceOfMonadLift___redArg(v___x_257_, v___x_256_);
return v___x_258_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__50(void){
_start:
{
lean_object* v___x_259_; lean_object* v___f_260_; lean_object* v___x_261_; 
v___x_259_ = lean_obj_once(&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__49, &lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__49_once, _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__49);
v___f_260_ = ((lean_object*)(lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__0));
v___x_261_ = l_Lean_instMonadTraceOfMonadLift___redArg(v___f_260_, v___x_259_);
return v___x_261_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__51(void){
_start:
{
lean_object* v___x_262_; lean_object* v___f_263_; lean_object* v___x_264_; 
v___x_262_ = lean_obj_once(&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__50, &lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__50_once, _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__50);
v___f_263_ = ((lean_object*)(lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__2));
v___x_264_ = l_Lean_instMonadTraceOfMonadLift___redArg(v___f_263_, v___x_262_);
return v___x_264_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__52(void){
_start:
{
lean_object* v___x_265_; lean_object* v___x_266_; lean_object* v___x_267_; 
v___x_265_ = lean_obj_once(&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__51, &lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__51_once, _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__51);
v___x_266_ = ((lean_object*)(lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__1));
v___x_267_ = l_Lean_instMonadTraceOfMonadLift___redArg(v___x_266_, v___x_265_);
return v___x_267_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__54(void){
_start:
{
lean_object* v___x_269_; lean_object* v___x_270_; 
v___x_269_ = ((lean_object*)(lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__53));
v___x_270_ = l_Lean_stringToMessageData(v___x_269_);
return v___x_270_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__56(void){
_start:
{
lean_object* v___x_272_; lean_object* v___x_273_; 
v___x_272_ = ((lean_object*)(lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__55));
v___x_273_ = l_Lean_stringToMessageData(v___x_272_);
return v___x_273_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__58(void){
_start:
{
lean_object* v___x_275_; lean_object* v___x_276_; 
v___x_275_ = ((lean_object*)(lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__57));
v___x_276_ = l_Lean_stringToMessageData(v___x_275_);
return v___x_276_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__60(void){
_start:
{
lean_object* v___x_278_; lean_object* v___x_279_; 
v___x_278_ = ((lean_object*)(lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__59));
v___x_279_ = l_Lean_stringToMessageData(v___x_278_);
return v___x_279_;
}
}
static lean_object* _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__62(void){
_start:
{
lean_object* v___x_281_; lean_object* v___x_282_; 
v___x_281_ = ((lean_object*)(lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__61));
v___x_282_ = l_Lean_stringToMessageData(v___x_281_);
return v___x_282_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg(lean_object* v_inst_283_, lean_object* v_gref_284_, lean_object* v_a_285_, lean_object* v_a_286_, lean_object* v_a_287_, lean_object* v_a_288_, lean_object* v_a_289_, lean_object* v_a_290_, lean_object* v_a_291_, lean_object* v_a_292_, lean_object* v_a_293_){
_start:
{
lean_object* v___y_299_; lean_object* v___y_300_; lean_object* v___y_301_; lean_object* v___y_302_; lean_object* v___y_303_; lean_object* v___y_304_; lean_object* v___y_305_; lean_object* v___y_306_; lean_object* v___y_307_; lean_object* v___y_308_; lean_object* v___y_309_; lean_object* v___x_312_; lean_object* v___x_313_; lean_object* v___x_314_; lean_object* v___x_315_; lean_object* v___x_316_; lean_object* v_elimGoal_317_; lean_object* v___x_318_; lean_object* v_id_319_; uint8_t v_state_320_; lean_object* v___y_322_; lean_object* v___y_323_; lean_object* v___y_324_; lean_object* v___y_325_; lean_object* v___y_326_; lean_object* v___y_327_; lean_object* v___y_328_; lean_object* v___y_329_; lean_object* v___y_330_; lean_object* v___y_365_; lean_object* v___y_366_; lean_object* v___y_367_; lean_object* v___y_368_; lean_object* v___y_369_; lean_object* v___y_370_; lean_object* v___y_371_; lean_object* v___y_372_; lean_object* v___y_373_; uint8_t v___y_374_; lean_object* v___y_392_; lean_object* v___y_393_; lean_object* v___y_394_; lean_object* v___y_395_; lean_object* v___y_396_; lean_object* v___y_397_; lean_object* v___y_398_; lean_object* v___y_399_; lean_object* v___y_400_; uint8_t v___x_428_; 
v___x_312_ = lp_aesop_Aesop_SearchM_instMonad(lean_box(0), v_inst_283_);
v___x_313_ = l_StateRefT_x27_instMonad___redArg(v___x_312_);
v___x_314_ = lean_st_ref_get(v_a_287_);
lean_dec(v___x_314_);
v___x_315_ = lean_st_ref_get(v_gref_284_);
v___x_316_ = lp_aesop_Aesop_treeImpl;
v_elimGoal_317_ = lean_ctor_get(v___x_316_, 1);
lean_inc_ref(v_elimGoal_317_);
lean_inc(v___x_315_);
v___x_318_ = lean_apply_1(v_elimGoal_317_, v___x_315_);
v_id_319_ = lean_ctor_get(v___x_318_, 0);
lean_inc(v_id_319_);
v_state_320_ = lean_ctor_get_uint8(v___x_318_, sizeof(void*)*14 + 8);
lean_dec_ref(v___x_318_);
v___x_428_ = lp_aesop_Aesop_GoalState_isProven(v_state_320_);
if (v___x_428_ == 0)
{
lean_object* v___x_429_; uint8_t v___x_430_; 
v___x_429_ = lean_st_ref_get(v_a_287_);
lean_dec(v___x_429_);
v___x_430_ = lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_Goal_isSafeExpanded(v___x_315_);
if (v___x_430_ == 0)
{
goto v___jp_431_;
}
else
{
if (v___x_428_ == 0)
{
lean_object* v___x_461_; lean_object* v_toMonadOptions_462_; lean_object* v___x_463_; lean_object* v___x_45432__overap_464_; lean_object* v___x_465_; 
v___x_461_ = lean_obj_once(&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__45, &lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__45_once, _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__45);
v_toMonadOptions_462_ = lean_ctor_get(v___x_461_, 0);
v___x_463_ = lp_aesop_Aesop_TraceOption_steps;
lean_inc(v_toMonadOptions_462_);
lean_inc_ref(v___x_313_);
v___x_45432__overap_464_ = lp_aesop_Aesop_TraceOption_isEnabled___redArg(v___x_313_, v_toMonadOptions_462_, v___x_463_);
lean_inc(v_a_293_);
lean_inc_ref(v_a_292_);
lean_inc(v_a_291_);
lean_inc_ref(v_a_290_);
lean_inc(v_a_289_);
lean_inc(v_a_288_);
lean_inc(v_a_287_);
lean_inc_ref(v_a_286_);
lean_inc(v_a_285_);
v___x_465_ = lean_apply_10(v___x_45432__overap_464_, v_a_285_, v_a_286_, v_a_287_, v_a_288_, v_a_289_, v_a_290_, v_a_291_, v_a_292_, v_a_293_, lean_box(0));
if (lean_obj_tag(v___x_465_) == 0)
{
lean_object* v_a_466_; uint8_t v___x_467_; 
v_a_466_ = lean_ctor_get(v___x_465_, 0);
lean_inc(v_a_466_);
lean_dec_ref_known(v___x_465_, 1);
v___x_467_ = lean_unbox(v_a_466_);
lean_dec(v_a_466_);
if (v___x_467_ == 0)
{
lean_dec(v_id_319_);
v___y_322_ = v_a_285_;
v___y_323_ = v_a_286_;
v___y_324_ = v_a_287_;
v___y_325_ = v_a_288_;
v___y_326_ = v_a_289_;
v___y_327_ = v_a_290_;
v___y_328_ = v_a_291_;
v___y_329_ = v_a_292_;
v___y_330_ = v_a_293_;
goto v___jp_321_;
}
else
{
lean_object* v___x_468_; lean_object* v___x_469_; lean_object* v_toMonadRef_470_; lean_object* v_traceClass_471_; lean_object* v___f_472_; lean_object* v___x_473_; lean_object* v___x_474_; lean_object* v___x_475_; lean_object* v___x_476_; lean_object* v___x_477_; lean_object* v___x_478_; lean_object* v___x_479_; lean_object* v___x_45484__overap_480_; lean_object* v___x_481_; 
v___x_468_ = lean_obj_once(&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__52, &lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__52_once, _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__52);
v___x_469_ = lean_obj_once(&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__32, &lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__32_once, _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__32);
v_toMonadRef_470_ = lean_ctor_get(v___x_469_, 0);
v_traceClass_471_ = lean_ctor_get(v___x_463_, 0);
v___f_472_ = lean_obj_once(&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__37, &lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__37_once, _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__37);
v___x_473_ = lean_obj_once(&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__58, &lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__58_once, _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__58);
v___x_474_ = l_Nat_reprFast(v_id_319_);
v___x_475_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_475_, 0, v___x_474_);
v___x_476_ = l_Lean_MessageData_ofFormat(v___x_475_);
v___x_477_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_477_, 0, v___x_473_);
lean_ctor_set(v___x_477_, 1, v___x_476_);
v___x_478_ = lean_obj_once(&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__60, &lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__60_once, _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__60);
v___x_479_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_479_, 0, v___x_477_);
lean_ctor_set(v___x_479_, 1, v___x_478_);
lean_inc(v_traceClass_471_);
lean_inc_ref(v_toMonadRef_470_);
lean_inc_ref(v___x_313_);
v___x_45484__overap_480_ = l_Lean_addTrace___redArg(v___x_313_, v___x_468_, v_toMonadRef_470_, v___f_472_, v_traceClass_471_, v___x_479_);
lean_inc(v_a_293_);
lean_inc_ref(v_a_292_);
lean_inc(v_a_291_);
lean_inc_ref(v_a_290_);
lean_inc(v_a_289_);
lean_inc(v_a_288_);
lean_inc(v_a_287_);
lean_inc_ref(v_a_286_);
lean_inc(v_a_285_);
v___x_481_ = lean_apply_10(v___x_45484__overap_480_, v_a_285_, v_a_286_, v_a_287_, v_a_288_, v_a_289_, v_a_290_, v_a_291_, v_a_292_, v_a_293_, lean_box(0));
if (lean_obj_tag(v___x_481_) == 0)
{
lean_dec_ref_known(v___x_481_, 1);
v___y_322_ = v_a_285_;
v___y_323_ = v_a_286_;
v___y_324_ = v_a_287_;
v___y_325_ = v_a_288_;
v___y_326_ = v_a_289_;
v___y_327_ = v_a_290_;
v___y_328_ = v_a_291_;
v___y_329_ = v_a_292_;
v___y_330_ = v_a_293_;
goto v___jp_321_;
}
else
{
lean_dec_ref(v___x_313_);
lean_dec(v_gref_284_);
lean_dec_ref(v_inst_283_);
return v___x_481_;
}
}
}
else
{
lean_object* v_a_482_; lean_object* v___x_484_; uint8_t v_isShared_485_; uint8_t v_isSharedCheck_489_; 
lean_dec(v_id_319_);
lean_dec_ref(v___x_313_);
lean_dec(v_gref_284_);
lean_dec_ref(v_inst_283_);
v_a_482_ = lean_ctor_get(v___x_465_, 0);
v_isSharedCheck_489_ = !lean_is_exclusive(v___x_465_);
if (v_isSharedCheck_489_ == 0)
{
v___x_484_ = v___x_465_;
v_isShared_485_ = v_isSharedCheck_489_;
goto v_resetjp_483_;
}
else
{
lean_inc(v_a_482_);
lean_dec(v___x_465_);
v___x_484_ = lean_box(0);
v_isShared_485_ = v_isSharedCheck_489_;
goto v_resetjp_483_;
}
v_resetjp_483_:
{
lean_object* v___x_487_; 
if (v_isShared_485_ == 0)
{
v___x_487_ = v___x_484_;
goto v_reusejp_486_;
}
else
{
lean_object* v_reuseFailAlloc_488_; 
v_reuseFailAlloc_488_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_488_, 0, v_a_482_);
v___x_487_ = v_reuseFailAlloc_488_;
goto v_reusejp_486_;
}
v_reusejp_486_:
{
return v___x_487_;
}
}
}
}
else
{
goto v___jp_431_;
}
}
v___jp_431_:
{
lean_object* v___x_432_; lean_object* v_toMonadOptions_433_; lean_object* v___x_434_; lean_object* v___x_45303__overap_435_; lean_object* v___x_436_; 
v___x_432_ = lean_obj_once(&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__45, &lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__45_once, _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__45);
v_toMonadOptions_433_ = lean_ctor_get(v___x_432_, 0);
v___x_434_ = lp_aesop_Aesop_TraceOption_steps;
lean_inc(v_toMonadOptions_433_);
lean_inc_ref(v___x_313_);
v___x_45303__overap_435_ = lp_aesop_Aesop_TraceOption_isEnabled___redArg(v___x_313_, v_toMonadOptions_433_, v___x_434_);
lean_inc(v_a_293_);
lean_inc_ref(v_a_292_);
lean_inc(v_a_291_);
lean_inc_ref(v_a_290_);
lean_inc(v_a_289_);
lean_inc(v_a_288_);
lean_inc(v_a_287_);
lean_inc_ref(v_a_286_);
lean_inc(v_a_285_);
v___x_436_ = lean_apply_10(v___x_45303__overap_435_, v_a_285_, v_a_286_, v_a_287_, v_a_288_, v_a_289_, v_a_290_, v_a_291_, v_a_292_, v_a_293_, lean_box(0));
if (lean_obj_tag(v___x_436_) == 0)
{
lean_object* v_a_437_; uint8_t v___x_438_; 
v_a_437_ = lean_ctor_get(v___x_436_, 0);
lean_inc(v_a_437_);
lean_dec_ref_known(v___x_436_, 1);
v___x_438_ = lean_unbox(v_a_437_);
lean_dec(v_a_437_);
if (v___x_438_ == 0)
{
lean_dec(v_id_319_);
v___y_392_ = v_a_285_;
v___y_393_ = v_a_286_;
v___y_394_ = v_a_287_;
v___y_395_ = v_a_288_;
v___y_396_ = v_a_289_;
v___y_397_ = v_a_290_;
v___y_398_ = v_a_291_;
v___y_399_ = v_a_292_;
v___y_400_ = v_a_293_;
goto v___jp_391_;
}
else
{
lean_object* v___x_439_; lean_object* v___x_440_; lean_object* v_toMonadRef_441_; lean_object* v_traceClass_442_; lean_object* v___f_443_; lean_object* v___x_444_; lean_object* v___x_445_; lean_object* v___x_446_; lean_object* v___x_447_; lean_object* v___x_448_; lean_object* v___x_449_; lean_object* v___x_450_; lean_object* v___x_45415__overap_451_; lean_object* v___x_452_; 
v___x_439_ = lean_obj_once(&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__52, &lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__52_once, _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__52);
v___x_440_ = lean_obj_once(&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__32, &lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__32_once, _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__32);
v_toMonadRef_441_ = lean_ctor_get(v___x_440_, 0);
v_traceClass_442_ = lean_ctor_get(v___x_434_, 0);
v___f_443_ = lean_obj_once(&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__37, &lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__37_once, _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__37);
v___x_444_ = lean_obj_once(&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__54, &lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__54_once, _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__54);
v___x_445_ = l_Nat_reprFast(v_id_319_);
v___x_446_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_446_, 0, v___x_445_);
v___x_447_ = l_Lean_MessageData_ofFormat(v___x_446_);
v___x_448_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_448_, 0, v___x_444_);
lean_ctor_set(v___x_448_, 1, v___x_447_);
v___x_449_ = lean_obj_once(&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__56, &lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__56_once, _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__56);
v___x_450_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_450_, 0, v___x_448_);
lean_ctor_set(v___x_450_, 1, v___x_449_);
lean_inc(v_traceClass_442_);
lean_inc_ref(v_toMonadRef_441_);
lean_inc_ref(v___x_313_);
v___x_45415__overap_451_ = l_Lean_addTrace___redArg(v___x_313_, v___x_439_, v_toMonadRef_441_, v___f_443_, v_traceClass_442_, v___x_450_);
lean_inc(v_a_293_);
lean_inc_ref(v_a_292_);
lean_inc(v_a_291_);
lean_inc_ref(v_a_290_);
lean_inc(v_a_289_);
lean_inc(v_a_288_);
lean_inc(v_a_287_);
lean_inc_ref(v_a_286_);
lean_inc(v_a_285_);
v___x_452_ = lean_apply_10(v___x_45415__overap_451_, v_a_285_, v_a_286_, v_a_287_, v_a_288_, v_a_289_, v_a_290_, v_a_291_, v_a_292_, v_a_293_, lean_box(0));
if (lean_obj_tag(v___x_452_) == 0)
{
lean_dec_ref_known(v___x_452_, 1);
v___y_392_ = v_a_285_;
v___y_393_ = v_a_286_;
v___y_394_ = v_a_287_;
v___y_395_ = v_a_288_;
v___y_396_ = v_a_289_;
v___y_397_ = v_a_290_;
v___y_398_ = v_a_291_;
v___y_399_ = v_a_292_;
v___y_400_ = v_a_293_;
goto v___jp_391_;
}
else
{
lean_dec_ref(v___x_313_);
lean_dec(v_gref_284_);
lean_dec_ref(v_inst_283_);
return v___x_452_;
}
}
}
else
{
lean_object* v_a_453_; lean_object* v___x_455_; uint8_t v_isShared_456_; uint8_t v_isSharedCheck_460_; 
lean_dec(v_id_319_);
lean_dec_ref(v___x_313_);
lean_dec(v_gref_284_);
lean_dec_ref(v_inst_283_);
v_a_453_ = lean_ctor_get(v___x_436_, 0);
v_isSharedCheck_460_ = !lean_is_exclusive(v___x_436_);
if (v_isSharedCheck_460_ == 0)
{
v___x_455_ = v___x_436_;
v_isShared_456_ = v_isSharedCheck_460_;
goto v_resetjp_454_;
}
else
{
lean_inc(v_a_453_);
lean_dec(v___x_436_);
v___x_455_ = lean_box(0);
v_isShared_456_ = v_isSharedCheck_460_;
goto v_resetjp_454_;
}
v_resetjp_454_:
{
lean_object* v___x_458_; 
if (v_isShared_456_ == 0)
{
v___x_458_ = v___x_455_;
goto v_reusejp_457_;
}
else
{
lean_object* v_reuseFailAlloc_459_; 
v_reuseFailAlloc_459_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_459_, 0, v_a_453_);
v___x_458_ = v_reuseFailAlloc_459_;
goto v_reusejp_457_;
}
v_reusejp_457_:
{
return v___x_458_;
}
}
}
}
}
else
{
lean_object* v___x_490_; lean_object* v_toMonadOptions_491_; lean_object* v___x_492_; lean_object* v___x_45164__overap_493_; lean_object* v___x_494_; 
lean_dec(v___x_315_);
lean_dec(v_gref_284_);
lean_dec_ref(v_inst_283_);
v___x_490_ = lean_obj_once(&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__45, &lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__45_once, _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__45);
v_toMonadOptions_491_ = lean_ctor_get(v___x_490_, 0);
v___x_492_ = lp_aesop_Aesop_TraceOption_steps;
lean_inc(v_toMonadOptions_491_);
lean_inc_ref(v___x_313_);
v___x_45164__overap_493_ = lp_aesop_Aesop_TraceOption_isEnabled___redArg(v___x_313_, v_toMonadOptions_491_, v___x_492_);
lean_inc(v_a_293_);
lean_inc_ref(v_a_292_);
lean_inc(v_a_291_);
lean_inc_ref(v_a_290_);
lean_inc(v_a_289_);
lean_inc(v_a_288_);
lean_inc(v_a_287_);
lean_inc_ref(v_a_286_);
lean_inc(v_a_285_);
v___x_494_ = lean_apply_10(v___x_45164__overap_493_, v_a_285_, v_a_286_, v_a_287_, v_a_288_, v_a_289_, v_a_290_, v_a_291_, v_a_292_, v_a_293_, lean_box(0));
if (lean_obj_tag(v___x_494_) == 0)
{
lean_object* v_a_495_; uint8_t v___x_496_; 
v_a_495_ = lean_ctor_get(v___x_494_, 0);
lean_inc(v_a_495_);
lean_dec_ref_known(v___x_494_, 1);
v___x_496_ = lean_unbox(v_a_495_);
lean_dec(v_a_495_);
if (v___x_496_ == 0)
{
lean_dec(v_id_319_);
lean_dec_ref(v___x_313_);
goto v___jp_295_;
}
else
{
lean_object* v___x_497_; lean_object* v___x_498_; lean_object* v_toMonadRef_499_; lean_object* v_traceClass_500_; lean_object* v___f_501_; lean_object* v___x_502_; lean_object* v___x_503_; lean_object* v___x_504_; lean_object* v___x_505_; lean_object* v___x_506_; lean_object* v___x_507_; lean_object* v___x_508_; lean_object* v___x_45290__overap_509_; lean_object* v___x_510_; 
v___x_497_ = lean_obj_once(&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__52, &lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__52_once, _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__52);
v___x_498_ = lean_obj_once(&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__32, &lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__32_once, _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__32);
v_toMonadRef_499_ = lean_ctor_get(v___x_498_, 0);
v_traceClass_500_ = lean_ctor_get(v___x_492_, 0);
v___f_501_ = lean_obj_once(&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__37, &lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__37_once, _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__37);
v___x_502_ = lean_obj_once(&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__58, &lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__58_once, _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__58);
v___x_503_ = l_Nat_reprFast(v_id_319_);
v___x_504_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_504_, 0, v___x_503_);
v___x_505_ = l_Lean_MessageData_ofFormat(v___x_504_);
v___x_506_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_506_, 0, v___x_502_);
lean_ctor_set(v___x_506_, 1, v___x_505_);
v___x_507_ = lean_obj_once(&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__62, &lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__62_once, _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__62);
v___x_508_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_508_, 0, v___x_506_);
lean_ctor_set(v___x_508_, 1, v___x_507_);
lean_inc(v_traceClass_500_);
lean_inc_ref(v_toMonadRef_499_);
v___x_45290__overap_509_ = l_Lean_addTrace___redArg(v___x_313_, v___x_497_, v_toMonadRef_499_, v___f_501_, v_traceClass_500_, v___x_508_);
lean_inc(v_a_293_);
lean_inc_ref(v_a_292_);
lean_inc(v_a_291_);
lean_inc_ref(v_a_290_);
lean_inc(v_a_289_);
lean_inc(v_a_288_);
lean_inc(v_a_287_);
lean_inc_ref(v_a_286_);
lean_inc(v_a_285_);
v___x_510_ = lean_apply_10(v___x_45290__overap_509_, v_a_285_, v_a_286_, v_a_287_, v_a_288_, v_a_289_, v_a_290_, v_a_291_, v_a_292_, v_a_293_, lean_box(0));
if (lean_obj_tag(v___x_510_) == 0)
{
lean_dec_ref_known(v___x_510_, 1);
goto v___jp_295_;
}
else
{
return v___x_510_;
}
}
}
else
{
lean_object* v_a_511_; lean_object* v___x_513_; uint8_t v_isShared_514_; uint8_t v_isSharedCheck_518_; 
lean_dec(v_id_319_);
lean_dec_ref(v___x_313_);
v_a_511_ = lean_ctor_get(v___x_494_, 0);
v_isSharedCheck_518_ = !lean_is_exclusive(v___x_494_);
if (v_isSharedCheck_518_ == 0)
{
v___x_513_ = v___x_494_;
v_isShared_514_ = v_isSharedCheck_518_;
goto v_resetjp_512_;
}
else
{
lean_inc(v_a_511_);
lean_dec(v___x_494_);
v___x_513_ = lean_box(0);
v_isShared_514_ = v_isSharedCheck_518_;
goto v_resetjp_512_;
}
v_resetjp_512_:
{
lean_object* v___x_516_; 
if (v_isShared_514_ == 0)
{
v___x_516_ = v___x_513_;
goto v_reusejp_515_;
}
else
{
lean_object* v_reuseFailAlloc_517_; 
v_reuseFailAlloc_517_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_517_, 0, v_a_511_);
v___x_516_ = v_reuseFailAlloc_517_;
goto v_reusejp_515_;
}
v_reusejp_515_:
{
return v___x_516_;
}
}
}
}
v___jp_295_:
{
lean_object* v___x_296_; lean_object* v___x_297_; 
v___x_296_ = lean_box(0);
v___x_297_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_297_, 0, v___x_296_);
return v___x_297_;
}
v___jp_298_:
{
lean_object* v___x_310_; lean_object* v___x_311_; 
v___x_310_ = lean_array_fget(v___y_299_, v___y_300_);
lean_dec_ref(v___y_299_);
v___x_311_ = lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandFirstPrefixRapp___redArg(v_inst_283_, v___x_310_, v___y_301_, v___y_302_, v___y_303_, v___y_304_, v___y_305_, v___y_306_, v___y_307_, v___y_308_, v___y_309_);
lean_dec(v___x_310_);
return v___x_311_;
}
v___jp_321_:
{
lean_object* v___x_331_; lean_object* v___x_332_; lean_object* v_elimGoal_333_; lean_object* v___x_334_; lean_object* v_id_335_; uint8_t v_state_336_; uint8_t v___x_337_; 
v___x_331_ = lean_st_ref_get(v___y_324_);
lean_dec(v___x_331_);
v___x_332_ = lean_st_ref_get(v_gref_284_);
lean_dec(v_gref_284_);
v_elimGoal_333_ = lean_ctor_get(v___x_316_, 1);
lean_inc_ref(v_elimGoal_333_);
lean_inc(v___x_332_);
v___x_334_ = lean_apply_1(v_elimGoal_333_, v___x_332_);
v_id_335_ = lean_ctor_get(v___x_334_, 0);
lean_inc(v_id_335_);
v_state_336_ = lean_ctor_get_uint8(v___x_334_, sizeof(void*)*14 + 8);
lean_dec_ref(v___x_334_);
v___x_337_ = lp_aesop_Aesop_GoalState_isProven(v_state_336_);
if (v___x_337_ == 0)
{
lean_object* v___x_338_; lean_object* v___x_339_; lean_object* v___x_340_; lean_object* v___x_341_; uint8_t v___x_342_; 
v___x_338_ = lean_st_ref_get(v___y_324_);
lean_dec(v___x_338_);
v___x_339_ = lp_aesop_Aesop_Goal_safeRapps(v___x_332_);
v___x_340_ = lean_unsigned_to_nat(0u);
v___x_341_ = lean_array_get_size(v___x_339_);
v___x_342_ = lean_nat_dec_lt(v___x_340_, v___x_341_);
if (v___x_342_ == 0)
{
lean_object* v___x_343_; lean_object* v___x_344_; 
lean_dec_ref(v___x_339_);
lean_dec(v_id_335_);
lean_dec_ref(v___x_313_);
lean_dec_ref(v_inst_283_);
v___x_343_ = lean_box(0);
v___x_344_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_344_, 0, v___x_343_);
return v___x_344_;
}
else
{
lean_object* v___x_345_; uint8_t v___x_346_; 
v___x_345_ = lean_unsigned_to_nat(1u);
v___x_346_ = lean_nat_dec_lt(v___x_345_, v___x_341_);
if (v___x_346_ == 0)
{
lean_dec(v_id_335_);
lean_dec_ref(v___x_313_);
v___y_299_ = v___x_339_;
v___y_300_ = v___x_340_;
v___y_301_ = v___y_322_;
v___y_302_ = v___y_323_;
v___y_303_ = v___y_324_;
v___y_304_ = v___y_325_;
v___y_305_ = v___y_326_;
v___y_306_ = v___y_327_;
v___y_307_ = v___y_328_;
v___y_308_ = v___y_329_;
v___y_309_ = v___y_330_;
goto v___jp_298_;
}
else
{
lean_object* v___x_347_; lean_object* v___x_348_; lean_object* v_toMonadRef_349_; lean_object* v___f_350_; lean_object* v___x_351_; lean_object* v___x_352_; lean_object* v___x_353_; lean_object* v___x_354_; lean_object* v___x_355_; lean_object* v___x_356_; lean_object* v___x_357_; lean_object* v___x_358_; lean_object* v___x_359_; lean_object* v___x_44747__overap_360_; lean_object* v___x_361_; 
v___x_347_ = lean_obj_once(&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__23, &lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__23_once, _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__23);
v___x_348_ = lean_obj_once(&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__32, &lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__32_once, _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__32);
v_toMonadRef_349_ = lean_ctor_get(v___x_348_, 0);
v___f_350_ = lean_obj_once(&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__37, &lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__37_once, _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__37);
lean_inc_ref(v___x_313_);
v___x_351_ = l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(v___f_350_, v___x_313_);
lean_inc_ref(v_toMonadRef_349_);
v___x_352_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_352_, 0, v___x_347_);
lean_ctor_set(v___x_352_, 1, v_toMonadRef_349_);
lean_ctor_set(v___x_352_, 2, v___x_351_);
v___x_353_ = lean_obj_once(&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__39, &lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__39_once, _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__39);
v___x_354_ = l_Nat_reprFast(v_id_335_);
v___x_355_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_355_, 0, v___x_354_);
v___x_356_ = l_Lean_MessageData_ofFormat(v___x_355_);
v___x_357_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_357_, 0, v___x_353_);
lean_ctor_set(v___x_357_, 1, v___x_356_);
v___x_358_ = lean_obj_once(&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__41, &lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__41_once, _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__41);
v___x_359_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_359_, 0, v___x_357_);
lean_ctor_set(v___x_359_, 1, v___x_358_);
v___x_44747__overap_360_ = l_Lean_throwError___redArg(v___x_313_, v___x_352_, v___x_359_);
lean_inc(v___y_330_);
lean_inc_ref(v___y_329_);
lean_inc(v___y_328_);
lean_inc_ref(v___y_327_);
lean_inc(v___y_326_);
lean_inc(v___y_325_);
lean_inc(v___y_324_);
lean_inc_ref(v___y_323_);
lean_inc(v___y_322_);
v___x_361_ = lean_apply_10(v___x_44747__overap_360_, v___y_322_, v___y_323_, v___y_324_, v___y_325_, v___y_326_, v___y_327_, v___y_328_, v___y_329_, v___y_330_, lean_box(0));
if (lean_obj_tag(v___x_361_) == 0)
{
lean_dec_ref_known(v___x_361_, 1);
v___y_299_ = v___x_339_;
v___y_300_ = v___x_340_;
v___y_301_ = v___y_322_;
v___y_302_ = v___y_323_;
v___y_303_ = v___y_324_;
v___y_304_ = v___y_325_;
v___y_305_ = v___y_326_;
v___y_306_ = v___y_327_;
v___y_307_ = v___y_328_;
v___y_308_ = v___y_329_;
v___y_309_ = v___y_330_;
goto v___jp_298_;
}
else
{
lean_dec_ref(v___x_339_);
lean_dec_ref(v_inst_283_);
return v___x_361_;
}
}
}
}
else
{
lean_object* v___x_362_; lean_object* v___x_363_; 
lean_dec(v_id_335_);
lean_dec(v___x_332_);
lean_dec_ref(v___x_313_);
lean_dec_ref(v_inst_283_);
v___x_362_ = lean_box(0);
v___x_363_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_363_, 0, v___x_362_);
return v___x_363_;
}
}
v___jp_364_:
{
if (v___y_374_ == 0)
{
lean_object* v___x_375_; 
lean_inc(v_gref_284_);
lean_inc_ref(v_inst_283_);
v___x_375_ = lp_aesop_Aesop_runFirstSafeRule___redArg(v_inst_283_, v_gref_284_, v___y_370_, v___y_368_, v___y_369_, v___y_371_, v___y_365_, v___y_367_, v___y_373_, v___y_366_);
if (lean_obj_tag(v___x_375_) == 0)
{
lean_object* v___x_376_; lean_object* v___x_377_; lean_object* v___x_378_; lean_object* v___x_379_; lean_object* v___x_380_; 
lean_dec_ref_known(v___x_375_, 1);
v___x_376_ = lean_st_ref_get(v___y_368_);
lean_dec(v___x_376_);
v___x_377_ = lean_st_ref_take(v___y_372_);
v___x_378_ = lean_unsigned_to_nat(1u);
v___x_379_ = lean_nat_add(v___x_377_, v___x_378_);
lean_dec(v___x_377_);
v___x_380_ = lean_st_ref_set(v___y_372_, v___x_379_);
v___y_322_ = v___y_372_;
v___y_323_ = v___y_370_;
v___y_324_ = v___y_368_;
v___y_325_ = v___y_369_;
v___y_326_ = v___y_371_;
v___y_327_ = v___y_365_;
v___y_328_ = v___y_367_;
v___y_329_ = v___y_373_;
v___y_330_ = v___y_366_;
goto v___jp_321_;
}
else
{
lean_object* v_a_381_; lean_object* v___x_383_; uint8_t v_isShared_384_; uint8_t v_isSharedCheck_388_; 
lean_dec_ref(v___x_313_);
lean_dec(v_gref_284_);
lean_dec_ref(v_inst_283_);
v_a_381_ = lean_ctor_get(v___x_375_, 0);
v_isSharedCheck_388_ = !lean_is_exclusive(v___x_375_);
if (v_isSharedCheck_388_ == 0)
{
v___x_383_ = v___x_375_;
v_isShared_384_ = v_isSharedCheck_388_;
goto v_resetjp_382_;
}
else
{
lean_inc(v_a_381_);
lean_dec(v___x_375_);
v___x_383_ = lean_box(0);
v_isShared_384_ = v_isSharedCheck_388_;
goto v_resetjp_382_;
}
v_resetjp_382_:
{
lean_object* v___x_386_; 
if (v_isShared_384_ == 0)
{
v___x_386_ = v___x_383_;
goto v_reusejp_385_;
}
else
{
lean_object* v_reuseFailAlloc_387_; 
v_reuseFailAlloc_387_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_387_, 0, v_a_381_);
v___x_386_ = v_reuseFailAlloc_387_;
goto v_reusejp_385_;
}
v_reusejp_385_:
{
return v___x_386_;
}
}
}
}
else
{
lean_object* v___x_389_; lean_object* v___x_390_; 
lean_dec_ref(v___x_313_);
lean_dec(v_gref_284_);
lean_dec_ref(v_inst_283_);
v___x_389_ = lp_aesop_Aesop_safeExpansionFailedException;
v___x_390_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_390_, 0, v___x_389_);
return v___x_390_;
}
}
v___jp_391_:
{
lean_object* v___x_401_; 
v___x_401_ = lp_aesop_Aesop_normalizeGoalIfNecessary___redArg(v_gref_284_, v_inst_283_, v___y_393_, v___y_394_, v___y_395_, v___y_396_, v___y_397_, v___y_398_, v___y_399_, v___y_400_);
if (lean_obj_tag(v___x_401_) == 0)
{
lean_object* v_a_402_; lean_object* v___x_404_; uint8_t v_isShared_405_; uint8_t v_isSharedCheck_419_; 
v_a_402_ = lean_ctor_get(v___x_401_, 0);
v_isSharedCheck_419_ = !lean_is_exclusive(v___x_401_);
if (v_isSharedCheck_419_ == 0)
{
v___x_404_ = v___x_401_;
v_isShared_405_ = v_isSharedCheck_419_;
goto v_resetjp_403_;
}
else
{
lean_inc(v_a_402_);
lean_dec(v___x_401_);
v___x_404_ = lean_box(0);
v_isShared_405_ = v_isSharedCheck_419_;
goto v_resetjp_403_;
}
v_resetjp_403_:
{
uint8_t v___x_406_; 
v___x_406_ = lean_unbox(v_a_402_);
lean_dec(v_a_402_);
if (v___x_406_ == 0)
{
lean_object* v___x_407_; lean_object* v___x_408_; lean_object* v_options_409_; lean_object* v_toOptions_410_; lean_object* v_maxSafePrefixRuleApplications_411_; lean_object* v___x_412_; uint8_t v___x_413_; 
lean_del_object(v___x_404_);
v___x_407_ = lean_st_ref_get(v___y_394_);
lean_dec(v___x_407_);
v___x_408_ = lean_st_ref_get(v___y_392_);
v_options_409_ = lean_ctor_get(v___y_393_, 2);
v_toOptions_410_ = lean_ctor_get(v_options_409_, 0);
v_maxSafePrefixRuleApplications_411_ = lean_ctor_get(v_toOptions_410_, 4);
v___x_412_ = lean_unsigned_to_nat(0u);
v___x_413_ = lean_nat_dec_lt(v___x_412_, v_maxSafePrefixRuleApplications_411_);
if (v___x_413_ == 0)
{
lean_dec(v___x_408_);
v___y_365_ = v___y_397_;
v___y_366_ = v___y_400_;
v___y_367_ = v___y_398_;
v___y_368_ = v___y_394_;
v___y_369_ = v___y_395_;
v___y_370_ = v___y_393_;
v___y_371_ = v___y_396_;
v___y_372_ = v___y_392_;
v___y_373_ = v___y_399_;
v___y_374_ = v___x_413_;
goto v___jp_364_;
}
else
{
uint8_t v___x_414_; 
v___x_414_ = lean_nat_dec_lt(v_maxSafePrefixRuleApplications_411_, v___x_408_);
lean_dec(v___x_408_);
v___y_365_ = v___y_397_;
v___y_366_ = v___y_400_;
v___y_367_ = v___y_398_;
v___y_368_ = v___y_394_;
v___y_369_ = v___y_395_;
v___y_370_ = v___y_393_;
v___y_371_ = v___y_396_;
v___y_372_ = v___y_392_;
v___y_373_ = v___y_399_;
v___y_374_ = v___x_414_;
goto v___jp_364_;
}
}
else
{
lean_object* v___x_415_; lean_object* v___x_417_; 
lean_dec_ref(v___x_313_);
lean_dec(v_gref_284_);
lean_dec_ref(v_inst_283_);
v___x_415_ = lean_box(0);
if (v_isShared_405_ == 0)
{
lean_ctor_set(v___x_404_, 0, v___x_415_);
v___x_417_ = v___x_404_;
goto v_reusejp_416_;
}
else
{
lean_object* v_reuseFailAlloc_418_; 
v_reuseFailAlloc_418_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_418_, 0, v___x_415_);
v___x_417_ = v_reuseFailAlloc_418_;
goto v_reusejp_416_;
}
v_reusejp_416_:
{
return v___x_417_;
}
}
}
}
else
{
lean_object* v_a_420_; lean_object* v___x_422_; uint8_t v_isShared_423_; uint8_t v_isSharedCheck_427_; 
lean_dec_ref(v___x_313_);
lean_dec(v_gref_284_);
lean_dec_ref(v_inst_283_);
v_a_420_ = lean_ctor_get(v___x_401_, 0);
v_isSharedCheck_427_ = !lean_is_exclusive(v___x_401_);
if (v_isSharedCheck_427_ == 0)
{
v___x_422_ = v___x_401_;
v_isShared_423_ = v_isSharedCheck_427_;
goto v_resetjp_421_;
}
else
{
lean_inc(v_a_420_);
lean_dec(v___x_401_);
v___x_422_ = lean_box(0);
v_isShared_423_ = v_isSharedCheck_427_;
goto v_resetjp_421_;
}
v_resetjp_421_:
{
lean_object* v___x_425_; 
if (v_isShared_423_ == 0)
{
v___x_425_ = v___x_422_;
goto v_reusejp_424_;
}
else
{
lean_object* v_reuseFailAlloc_426_; 
v_reuseFailAlloc_426_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_426_, 0, v_a_420_);
v___x_425_ = v_reuseFailAlloc_426_;
goto v_reusejp_424_;
}
v_reusejp_424_:
{
return v___x_425_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixMVarCluster___redArg___lam__0(lean_object* v_inst_519_, lean_object* v_x_520_, lean_object* v___y_521_, lean_object* v___y_522_, lean_object* v___y_523_, lean_object* v___y_524_, lean_object* v___y_525_, lean_object* v___y_526_, lean_object* v___y_527_, lean_object* v___y_528_, lean_object* v___y_529_, lean_object* v___y_530_){
_start:
{
lean_object* v___x_532_; 
v___x_532_ = lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg(v_inst_519_, v___y_521_, v___y_522_, v___y_523_, v___y_524_, v___y_525_, v___y_526_, v___y_527_, v___y_528_, v___y_529_, v___y_530_);
return v___x_532_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixMVarCluster___redArg___lam__0___boxed(lean_object* v_inst_533_, lean_object* v_x_534_, lean_object* v___y_535_, lean_object* v___y_536_, lean_object* v___y_537_, lean_object* v___y_538_, lean_object* v___y_539_, lean_object* v___y_540_, lean_object* v___y_541_, lean_object* v___y_542_, lean_object* v___y_543_, lean_object* v___y_544_, lean_object* v___y_545_){
_start:
{
lean_object* v_res_546_; 
v_res_546_ = lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixMVarCluster___redArg___lam__0(v_inst_533_, v_x_534_, v___y_535_, v___y_536_, v___y_537_, v___y_538_, v___y_539_, v___y_540_, v___y_541_, v___y_542_, v___y_543_, v___y_544_);
lean_dec(v___y_544_);
lean_dec_ref(v___y_543_);
lean_dec(v___y_542_);
lean_dec_ref(v___y_541_);
lean_dec(v___y_540_);
lean_dec(v___y_539_);
lean_dec(v___y_538_);
lean_dec_ref(v___y_537_);
lean_dec(v___y_536_);
return v_res_546_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixMVarCluster___redArg(lean_object* v_inst_547_, lean_object* v_cref_548_, lean_object* v_a_549_, lean_object* v_a_550_, lean_object* v_a_551_, lean_object* v_a_552_, lean_object* v_a_553_, lean_object* v_a_554_, lean_object* v_a_555_, lean_object* v_a_556_, lean_object* v_a_557_){
_start:
{
lean_object* v___x_559_; lean_object* v___x_560_; lean_object* v___x_561_; lean_object* v___x_562_; lean_object* v___x_563_; lean_object* v_elimMVarCluster_564_; lean_object* v___x_565_; lean_object* v_goals_566_; lean_object* v___x_567_; lean_object* v___x_568_; lean_object* v___x_569_; uint8_t v___x_570_; 
v___x_559_ = lp_aesop_Aesop_SearchM_instMonad(lean_box(0), v_inst_547_);
v___x_560_ = l_StateRefT_x27_instMonad___redArg(v___x_559_);
v___x_561_ = lean_st_ref_get(v_a_551_);
lean_dec(v___x_561_);
v___x_562_ = lean_st_ref_get(v_cref_548_);
v___x_563_ = lp_aesop_Aesop_treeImpl;
v_elimMVarCluster_564_ = lean_ctor_get(v___x_563_, 5);
lean_inc_ref(v_elimMVarCluster_564_);
v___x_565_ = lean_apply_1(v_elimMVarCluster_564_, v___x_562_);
v_goals_566_ = lean_ctor_get(v___x_565_, 1);
lean_inc_ref(v_goals_566_);
lean_dec_ref(v___x_565_);
v___x_567_ = lean_unsigned_to_nat(0u);
v___x_568_ = lean_array_get_size(v_goals_566_);
v___x_569_ = lean_box(0);
v___x_570_ = lean_nat_dec_lt(v___x_567_, v___x_568_);
if (v___x_570_ == 0)
{
lean_object* v___x_571_; 
lean_dec_ref(v_goals_566_);
lean_dec_ref(v___x_560_);
lean_dec_ref(v_inst_547_);
v___x_571_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_571_, 0, v___x_569_);
return v___x_571_;
}
else
{
lean_object* v___f_572_; uint8_t v___x_573_; 
v___f_572_ = lean_alloc_closure((void*)(lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixMVarCluster___redArg___lam__0___boxed), 13, 1);
lean_closure_set(v___f_572_, 0, v_inst_547_);
v___x_573_ = lean_nat_dec_le(v___x_568_, v___x_568_);
if (v___x_573_ == 0)
{
if (v___x_570_ == 0)
{
lean_object* v___x_574_; 
lean_dec_ref(v___f_572_);
lean_dec_ref(v_goals_566_);
lean_dec_ref(v___x_560_);
v___x_574_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_574_, 0, v___x_569_);
return v___x_574_;
}
else
{
size_t v___x_575_; size_t v___x_576_; lean_object* v___x_49056__overap_577_; lean_object* v___x_578_; 
v___x_575_ = ((size_t)0ULL);
v___x_576_ = lean_usize_of_nat(v___x_568_);
v___x_49056__overap_577_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_560_, v___f_572_, v_goals_566_, v___x_575_, v___x_576_, v___x_569_);
lean_inc(v_a_557_);
lean_inc_ref(v_a_556_);
lean_inc(v_a_555_);
lean_inc_ref(v_a_554_);
lean_inc(v_a_553_);
lean_inc(v_a_552_);
lean_inc(v_a_551_);
lean_inc_ref(v_a_550_);
lean_inc(v_a_549_);
v___x_578_ = lean_apply_10(v___x_49056__overap_577_, v_a_549_, v_a_550_, v_a_551_, v_a_552_, v_a_553_, v_a_554_, v_a_555_, v_a_556_, v_a_557_, lean_box(0));
return v___x_578_;
}
}
else
{
size_t v___x_579_; size_t v___x_580_; lean_object* v___x_49060__overap_581_; lean_object* v___x_582_; 
v___x_579_ = ((size_t)0ULL);
v___x_580_ = lean_usize_of_nat(v___x_568_);
v___x_49060__overap_581_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_560_, v___f_572_, v_goals_566_, v___x_579_, v___x_580_, v___x_569_);
lean_inc(v_a_557_);
lean_inc_ref(v_a_556_);
lean_inc(v_a_555_);
lean_inc_ref(v_a_554_);
lean_inc(v_a_553_);
lean_inc(v_a_552_);
lean_inc(v_a_551_);
lean_inc_ref(v_a_550_);
lean_inc(v_a_549_);
v___x_582_ = lean_apply_10(v___x_49060__overap_581_, v_a_549_, v_a_550_, v_a_551_, v_a_552_, v_a_553_, v_a_554_, v_a_555_, v_a_556_, v_a_557_, lean_box(0));
return v___x_582_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandFirstPrefixRapp___redArg___lam__0(lean_object* v_inst_583_, lean_object* v_x_584_, lean_object* v___y_585_, lean_object* v___y_586_, lean_object* v___y_587_, lean_object* v___y_588_, lean_object* v___y_589_, lean_object* v___y_590_, lean_object* v___y_591_, lean_object* v___y_592_, lean_object* v___y_593_, lean_object* v___y_594_){
_start:
{
lean_object* v___x_596_; 
v___x_596_ = lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixMVarCluster___redArg(v_inst_583_, v___y_585_, v___y_586_, v___y_587_, v___y_588_, v___y_589_, v___y_590_, v___y_591_, v___y_592_, v___y_593_, v___y_594_);
return v___x_596_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandFirstPrefixRapp___redArg___boxed(lean_object* v_inst_597_, lean_object* v_rref_598_, lean_object* v_a_599_, lean_object* v_a_600_, lean_object* v_a_601_, lean_object* v_a_602_, lean_object* v_a_603_, lean_object* v_a_604_, lean_object* v_a_605_, lean_object* v_a_606_, lean_object* v_a_607_, lean_object* v_a_608_){
_start:
{
lean_object* v_res_609_; 
v_res_609_ = lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandFirstPrefixRapp___redArg(v_inst_597_, v_rref_598_, v_a_599_, v_a_600_, v_a_601_, v_a_602_, v_a_603_, v_a_604_, v_a_605_, v_a_606_, v_a_607_);
lean_dec(v_a_607_);
lean_dec_ref(v_a_606_);
lean_dec(v_a_605_);
lean_dec_ref(v_a_604_);
lean_dec(v_a_603_);
lean_dec(v_a_602_);
lean_dec(v_a_601_);
lean_dec_ref(v_a_600_);
lean_dec(v_a_599_);
lean_dec(v_rref_598_);
return v_res_609_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixMVarCluster___redArg___boxed(lean_object* v_inst_610_, lean_object* v_cref_611_, lean_object* v_a_612_, lean_object* v_a_613_, lean_object* v_a_614_, lean_object* v_a_615_, lean_object* v_a_616_, lean_object* v_a_617_, lean_object* v_a_618_, lean_object* v_a_619_, lean_object* v_a_620_, lean_object* v_a_621_){
_start:
{
lean_object* v_res_622_; 
v_res_622_ = lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixMVarCluster___redArg(v_inst_610_, v_cref_611_, v_a_612_, v_a_613_, v_a_614_, v_a_615_, v_a_616_, v_a_617_, v_a_618_, v_a_619_, v_a_620_);
lean_dec(v_a_620_);
lean_dec_ref(v_a_619_);
lean_dec(v_a_618_);
lean_dec_ref(v_a_617_);
lean_dec(v_a_616_);
lean_dec(v_a_615_);
lean_dec(v_a_614_);
lean_dec_ref(v_a_613_);
lean_dec(v_a_612_);
lean_dec(v_cref_611_);
return v_res_622_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___boxed(lean_object* v_inst_623_, lean_object* v_gref_624_, lean_object* v_a_625_, lean_object* v_a_626_, lean_object* v_a_627_, lean_object* v_a_628_, lean_object* v_a_629_, lean_object* v_a_630_, lean_object* v_a_631_, lean_object* v_a_632_, lean_object* v_a_633_, lean_object* v_a_634_){
_start:
{
lean_object* v_res_635_; 
v_res_635_ = lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg(v_inst_623_, v_gref_624_, v_a_625_, v_a_626_, v_a_627_, v_a_628_, v_a_629_, v_a_630_, v_a_631_, v_a_632_, v_a_633_);
lean_dec(v_a_633_);
lean_dec_ref(v_a_632_);
lean_dec(v_a_631_);
lean_dec_ref(v_a_630_);
lean_dec(v_a_629_);
lean_dec(v_a_628_);
lean_dec(v_a_627_);
lean_dec_ref(v_a_626_);
lean_dec(v_a_625_);
return v_res_635_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal(lean_object* v_Q_636_, lean_object* v_inst_637_, lean_object* v_gref_638_, lean_object* v_a_639_, lean_object* v_a_640_, lean_object* v_a_641_, lean_object* v_a_642_, lean_object* v_a_643_, lean_object* v_a_644_, lean_object* v_a_645_, lean_object* v_a_646_, lean_object* v_a_647_){
_start:
{
lean_object* v___x_649_; 
v___x_649_ = lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg(v_inst_637_, v_gref_638_, v_a_639_, v_a_640_, v_a_641_, v_a_642_, v_a_643_, v_a_644_, v_a_645_, v_a_646_, v_a_647_);
return v___x_649_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___boxed(lean_object* v_Q_650_, lean_object* v_inst_651_, lean_object* v_gref_652_, lean_object* v_a_653_, lean_object* v_a_654_, lean_object* v_a_655_, lean_object* v_a_656_, lean_object* v_a_657_, lean_object* v_a_658_, lean_object* v_a_659_, lean_object* v_a_660_, lean_object* v_a_661_, lean_object* v_a_662_){
_start:
{
lean_object* v_res_663_; 
v_res_663_ = lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal(v_Q_650_, v_inst_651_, v_gref_652_, v_a_653_, v_a_654_, v_a_655_, v_a_656_, v_a_657_, v_a_658_, v_a_659_, v_a_660_, v_a_661_);
lean_dec(v_a_661_);
lean_dec_ref(v_a_660_);
lean_dec(v_a_659_);
lean_dec_ref(v_a_658_);
lean_dec(v_a_657_);
lean_dec(v_a_656_);
lean_dec(v_a_655_);
lean_dec_ref(v_a_654_);
lean_dec(v_a_653_);
return v_res_663_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandFirstPrefixRapp(lean_object* v_Q_664_, lean_object* v_inst_665_, lean_object* v_rref_666_, lean_object* v_a_667_, lean_object* v_a_668_, lean_object* v_a_669_, lean_object* v_a_670_, lean_object* v_a_671_, lean_object* v_a_672_, lean_object* v_a_673_, lean_object* v_a_674_, lean_object* v_a_675_){
_start:
{
lean_object* v___x_677_; 
v___x_677_ = lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandFirstPrefixRapp___redArg(v_inst_665_, v_rref_666_, v_a_667_, v_a_668_, v_a_669_, v_a_670_, v_a_671_, v_a_672_, v_a_673_, v_a_674_, v_a_675_);
return v___x_677_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandFirstPrefixRapp___boxed(lean_object* v_Q_678_, lean_object* v_inst_679_, lean_object* v_rref_680_, lean_object* v_a_681_, lean_object* v_a_682_, lean_object* v_a_683_, lean_object* v_a_684_, lean_object* v_a_685_, lean_object* v_a_686_, lean_object* v_a_687_, lean_object* v_a_688_, lean_object* v_a_689_, lean_object* v_a_690_){
_start:
{
lean_object* v_res_691_; 
v_res_691_ = lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandFirstPrefixRapp(v_Q_678_, v_inst_679_, v_rref_680_, v_a_681_, v_a_682_, v_a_683_, v_a_684_, v_a_685_, v_a_686_, v_a_687_, v_a_688_, v_a_689_);
lean_dec(v_a_689_);
lean_dec_ref(v_a_688_);
lean_dec(v_a_687_);
lean_dec_ref(v_a_686_);
lean_dec(v_a_685_);
lean_dec(v_a_684_);
lean_dec(v_a_683_);
lean_dec_ref(v_a_682_);
lean_dec(v_a_681_);
lean_dec(v_rref_680_);
return v_res_691_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixMVarCluster(lean_object* v_Q_692_, lean_object* v_inst_693_, lean_object* v_cref_694_, lean_object* v_a_695_, lean_object* v_a_696_, lean_object* v_a_697_, lean_object* v_a_698_, lean_object* v_a_699_, lean_object* v_a_700_, lean_object* v_a_701_, lean_object* v_a_702_, lean_object* v_a_703_){
_start:
{
lean_object* v___x_705_; 
v___x_705_ = lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixMVarCluster___redArg(v_inst_693_, v_cref_694_, v_a_695_, v_a_696_, v_a_697_, v_a_698_, v_a_699_, v_a_700_, v_a_701_, v_a_702_, v_a_703_);
return v___x_705_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixMVarCluster___boxed(lean_object* v_Q_706_, lean_object* v_inst_707_, lean_object* v_cref_708_, lean_object* v_a_709_, lean_object* v_a_710_, lean_object* v_a_711_, lean_object* v_a_712_, lean_object* v_a_713_, lean_object* v_a_714_, lean_object* v_a_715_, lean_object* v_a_716_, lean_object* v_a_717_, lean_object* v_a_718_){
_start:
{
lean_object* v_res_719_; 
v_res_719_ = lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixMVarCluster(v_Q_706_, v_inst_707_, v_cref_708_, v_a_709_, v_a_710_, v_a_711_, v_a_712_, v_a_713_, v_a_714_, v_a_715_, v_a_716_, v_a_717_);
lean_dec(v_a_717_);
lean_dec_ref(v_a_716_);
lean_dec(v_a_715_);
lean_dec_ref(v_a_714_);
lean_dec(v_a_713_);
lean_dec(v_a_712_);
lean_dec(v_a_711_);
lean_dec_ref(v_a_710_);
lean_dec(v_a_709_);
lean_dec(v_cref_708_);
return v_res_719_;
}
}
static lean_object* _init_lp_aesop_Aesop_expandSafePrefix___redArg___closed__1(void){
_start:
{
lean_object* v___x_721_; lean_object* v___x_722_; 
v___x_721_ = ((lean_object*)(lp_aesop_Aesop_expandSafePrefix___redArg___closed__0));
v___x_722_ = l_Lean_stringToMessageData(v___x_721_);
return v___x_722_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_expandSafePrefix___redArg(lean_object* v_inst_723_, lean_object* v_a_724_, lean_object* v_a_725_, lean_object* v_a_726_, lean_object* v_a_727_, lean_object* v_a_728_, lean_object* v_a_729_, lean_object* v_a_730_, lean_object* v_a_731_){
_start:
{
lean_object* v___y_734_; uint8_t v___y_735_; lean_object* v_a_742_; lean_object* v___y_746_; lean_object* v___y_747_; lean_object* v___y_748_; lean_object* v___y_749_; lean_object* v___y_750_; lean_object* v___y_751_; lean_object* v___y_752_; lean_object* v___y_753_; lean_object* v___x_778_; lean_object* v___x_779_; lean_object* v_toMonadOptions_780_; lean_object* v___x_781_; lean_object* v___x_782_; lean_object* v___x_12538__overap_783_; lean_object* v___x_784_; 
v___x_778_ = lp_aesop_Aesop_SearchM_instMonad(lean_box(0), v_inst_723_);
v___x_779_ = lean_obj_once(&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__44, &lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__44_once, _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__44);
v_toMonadOptions_780_ = lean_ctor_get(v___x_779_, 0);
v___x_781_ = lp_aesop_Aesop_TraceOption_steps;
v___x_782_ = lp_aesop_Aesop_SearchM_instMonadRef(lean_box(0), v_inst_723_);
lean_inc(v_toMonadOptions_780_);
lean_inc_ref(v___x_778_);
v___x_12538__overap_783_ = lp_aesop_Aesop_TraceOption_isEnabled___redArg(v___x_778_, v_toMonadOptions_780_, v___x_781_);
lean_inc(v_a_731_);
lean_inc_ref(v_a_730_);
lean_inc(v_a_729_);
lean_inc_ref(v_a_728_);
lean_inc(v_a_727_);
lean_inc(v_a_726_);
lean_inc(v_a_725_);
lean_inc_ref(v_a_724_);
v___x_784_ = lean_apply_9(v___x_12538__overap_783_, v_a_724_, v_a_725_, v_a_726_, v_a_727_, v_a_728_, v_a_729_, v_a_730_, v_a_731_, lean_box(0));
if (lean_obj_tag(v___x_784_) == 0)
{
lean_object* v_a_785_; uint8_t v___x_786_; 
v_a_785_ = lean_ctor_get(v___x_784_, 0);
lean_inc(v_a_785_);
lean_dec_ref_known(v___x_784_, 1);
v___x_786_ = lean_unbox(v_a_785_);
lean_dec(v_a_785_);
if (v___x_786_ == 0)
{
lean_dec_ref(v___x_782_);
lean_dec_ref(v___x_778_);
v___y_746_ = v_a_724_;
v___y_747_ = v_a_725_;
v___y_748_ = v_a_726_;
v___y_749_ = v_a_727_;
v___y_750_ = v_a_728_;
v___y_751_ = v_a_729_;
v___y_752_ = v_a_730_;
v___y_753_ = v_a_731_;
goto v___jp_745_;
}
else
{
lean_object* v___x_787_; lean_object* v_traceClass_788_; lean_object* v___f_789_; lean_object* v___x_790_; lean_object* v___x_12717__overap_791_; lean_object* v___x_792_; 
v___x_787_ = lean_obj_once(&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__51, &lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__51_once, _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__51);
v_traceClass_788_ = lean_ctor_get(v___x_781_, 0);
v___f_789_ = lean_obj_once(&lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__36, &lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__36_once, _init_lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg___closed__36);
v___x_790_ = lean_obj_once(&lp_aesop_Aesop_expandSafePrefix___redArg___closed__1, &lp_aesop_Aesop_expandSafePrefix___redArg___closed__1_once, _init_lp_aesop_Aesop_expandSafePrefix___redArg___closed__1);
lean_inc(v_traceClass_788_);
v___x_12717__overap_791_ = l_Lean_addTrace___redArg(v___x_778_, v___x_787_, v___x_782_, v___f_789_, v_traceClass_788_, v___x_790_);
lean_inc(v_a_731_);
lean_inc_ref(v_a_730_);
lean_inc(v_a_729_);
lean_inc_ref(v_a_728_);
lean_inc(v_a_727_);
lean_inc(v_a_726_);
lean_inc(v_a_725_);
lean_inc_ref(v_a_724_);
v___x_792_ = lean_apply_9(v___x_12717__overap_791_, v_a_724_, v_a_725_, v_a_726_, v_a_727_, v_a_728_, v_a_729_, v_a_730_, v_a_731_, lean_box(0));
if (lean_obj_tag(v___x_792_) == 0)
{
lean_dec_ref_known(v___x_792_, 1);
v___y_746_ = v_a_724_;
v___y_747_ = v_a_725_;
v___y_748_ = v_a_726_;
v___y_749_ = v_a_727_;
v___y_750_ = v_a_728_;
v___y_751_ = v_a_729_;
v___y_752_ = v_a_730_;
v___y_753_ = v_a_731_;
goto v___jp_745_;
}
else
{
lean_object* v_a_793_; lean_object* v___x_795_; uint8_t v_isShared_796_; uint8_t v_isSharedCheck_800_; 
lean_dec_ref(v_inst_723_);
v_a_793_ = lean_ctor_get(v___x_792_, 0);
v_isSharedCheck_800_ = !lean_is_exclusive(v___x_792_);
if (v_isSharedCheck_800_ == 0)
{
v___x_795_ = v___x_792_;
v_isShared_796_ = v_isSharedCheck_800_;
goto v_resetjp_794_;
}
else
{
lean_inc(v_a_793_);
lean_dec(v___x_792_);
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
else
{
lean_dec_ref(v___x_782_);
lean_dec_ref(v___x_778_);
lean_dec_ref(v_inst_723_);
return v___x_784_;
}
v___jp_733_:
{
if (v___y_735_ == 0)
{
uint8_t v___x_736_; 
v___x_736_ = lp_aesop_Aesop_isSafeExpansionFailedException(v___y_734_);
if (v___x_736_ == 0)
{
lean_object* v___x_737_; 
v___x_737_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_737_, 0, v___y_734_);
return v___x_737_;
}
else
{
lean_object* v___x_738_; lean_object* v___x_739_; 
lean_dec_ref(v___y_734_);
v___x_738_ = lean_box(v___y_735_);
v___x_739_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_739_, 0, v___x_738_);
return v___x_739_;
}
}
else
{
lean_object* v___x_740_; 
v___x_740_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_740_, 0, v___y_734_);
return v___x_740_;
}
}
v___jp_741_:
{
uint8_t v___x_743_; 
v___x_743_ = l_Lean_Exception_isInterrupt(v_a_742_);
if (v___x_743_ == 0)
{
uint8_t v___x_744_; 
lean_inc_ref(v_a_742_);
v___x_744_ = l_Lean_Exception_isRuntime(v_a_742_);
v___y_734_ = v_a_742_;
v___y_735_ = v___x_744_;
goto v___jp_733_;
}
else
{
v___y_734_ = v_a_742_;
v___y_735_ = v___x_743_;
goto v___jp_733_;
}
}
v___jp_745_:
{
lean_object* v___x_754_; lean_object* v_iteration_755_; lean_object* v_ruleSet_756_; lean_object* v___x_757_; lean_object* v___x_758_; 
v___x_754_ = lean_st_ref_get(v___y_747_);
v_iteration_755_ = lean_ctor_get(v___x_754_, 0);
lean_inc(v_iteration_755_);
lean_dec(v___x_754_);
v_ruleSet_756_ = lean_ctor_get(v___y_746_, 0);
lean_inc_ref(v_ruleSet_756_);
v___x_757_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_757_, 0, v_iteration_755_);
lean_ctor_set(v___x_757_, 1, v_ruleSet_756_);
v___x_758_ = lp_aesop_Aesop_getRootGoal(v___x_757_, v___y_748_, v___y_749_, v___y_750_, v___y_751_, v___y_752_, v___y_753_);
lean_dec_ref_known(v___x_757_, 2);
if (lean_obj_tag(v___x_758_) == 0)
{
lean_object* v_a_759_; lean_object* v___x_760_; lean_object* v___x_761_; lean_object* v___x_762_; lean_object* v___x_763_; 
v_a_759_ = lean_ctor_get(v___x_758_, 0);
lean_inc(v_a_759_);
lean_dec_ref_known(v___x_758_, 1);
v___x_760_ = lean_st_ref_get(v___y_747_);
lean_dec(v___x_760_);
v___x_761_ = lean_unsigned_to_nat(0u);
v___x_762_ = lean_st_mk_ref(v___x_761_);
v___x_763_ = lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_expandSafePrefixGoal___redArg(v_inst_723_, v_a_759_, v___x_762_, v___y_746_, v___y_747_, v___y_748_, v___y_749_, v___y_750_, v___y_751_, v___y_752_, v___y_753_);
if (lean_obj_tag(v___x_763_) == 0)
{
lean_object* v___x_765_; uint8_t v_isShared_766_; uint8_t v_isSharedCheck_774_; 
v_isSharedCheck_774_ = !lean_is_exclusive(v___x_763_);
if (v_isSharedCheck_774_ == 0)
{
lean_object* v_unused_775_; 
v_unused_775_ = lean_ctor_get(v___x_763_, 0);
lean_dec(v_unused_775_);
v___x_765_ = v___x_763_;
v_isShared_766_ = v_isSharedCheck_774_;
goto v_resetjp_764_;
}
else
{
lean_dec(v___x_763_);
v___x_765_ = lean_box(0);
v_isShared_766_ = v_isSharedCheck_774_;
goto v_resetjp_764_;
}
v_resetjp_764_:
{
lean_object* v___x_767_; lean_object* v___x_768_; uint8_t v___x_769_; lean_object* v___x_770_; lean_object* v___x_772_; 
v___x_767_ = lean_st_ref_get(v___y_747_);
lean_dec(v___x_767_);
v___x_768_ = lean_st_ref_get(v___x_762_);
lean_dec(v___x_762_);
lean_dec(v___x_768_);
v___x_769_ = 1;
v___x_770_ = lean_box(v___x_769_);
if (v_isShared_766_ == 0)
{
lean_ctor_set(v___x_765_, 0, v___x_770_);
v___x_772_ = v___x_765_;
goto v_reusejp_771_;
}
else
{
lean_object* v_reuseFailAlloc_773_; 
v_reuseFailAlloc_773_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_773_, 0, v___x_770_);
v___x_772_ = v_reuseFailAlloc_773_;
goto v_reusejp_771_;
}
v_reusejp_771_:
{
return v___x_772_;
}
}
}
else
{
lean_object* v_a_776_; 
lean_dec(v___x_762_);
v_a_776_ = lean_ctor_get(v___x_763_, 0);
lean_inc(v_a_776_);
lean_dec_ref_known(v___x_763_, 1);
v_a_742_ = v_a_776_;
goto v___jp_741_;
}
}
else
{
lean_object* v_a_777_; 
lean_dec_ref(v_inst_723_);
v_a_777_ = lean_ctor_get(v___x_758_, 0);
lean_inc(v_a_777_);
lean_dec_ref_known(v___x_758_, 1);
v_a_742_ = v_a_777_;
goto v___jp_741_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_expandSafePrefix___redArg___boxed(lean_object* v_inst_801_, lean_object* v_a_802_, lean_object* v_a_803_, lean_object* v_a_804_, lean_object* v_a_805_, lean_object* v_a_806_, lean_object* v_a_807_, lean_object* v_a_808_, lean_object* v_a_809_, lean_object* v_a_810_){
_start:
{
lean_object* v_res_811_; 
v_res_811_ = lp_aesop_Aesop_expandSafePrefix___redArg(v_inst_801_, v_a_802_, v_a_803_, v_a_804_, v_a_805_, v_a_806_, v_a_807_, v_a_808_, v_a_809_);
lean_dec(v_a_809_);
lean_dec_ref(v_a_808_);
lean_dec(v_a_807_);
lean_dec_ref(v_a_806_);
lean_dec(v_a_805_);
lean_dec(v_a_804_);
lean_dec(v_a_803_);
lean_dec_ref(v_a_802_);
return v_res_811_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_expandSafePrefix(lean_object* v_Q_812_, lean_object* v_inst_813_, lean_object* v_a_814_, lean_object* v_a_815_, lean_object* v_a_816_, lean_object* v_a_817_, lean_object* v_a_818_, lean_object* v_a_819_, lean_object* v_a_820_, lean_object* v_a_821_){
_start:
{
lean_object* v___x_823_; 
v___x_823_ = lp_aesop_Aesop_expandSafePrefix___redArg(v_inst_813_, v_a_814_, v_a_815_, v_a_816_, v_a_817_, v_a_818_, v_a_819_, v_a_820_, v_a_821_);
return v___x_823_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_expandSafePrefix___boxed(lean_object* v_Q_824_, lean_object* v_inst_825_, lean_object* v_a_826_, lean_object* v_a_827_, lean_object* v_a_828_, lean_object* v_a_829_, lean_object* v_a_830_, lean_object* v_a_831_, lean_object* v_a_832_, lean_object* v_a_833_, lean_object* v_a_834_){
_start:
{
lean_object* v_res_835_; 
v_res_835_ = lp_aesop_Aesop_expandSafePrefix(v_Q_824_, v_inst_825_, v_a_826_, v_a_827_, v_a_828_, v_a_829_, v_a_830_, v_a_831_, v_a_832_, v_a_833_);
lean_dec(v_a_833_);
lean_dec_ref(v_a_832_);
lean_dec(v_a_831_);
lean_dec_ref(v_a_830_);
lean_dec(v_a_829_);
lean_dec(v_a_828_);
lean_dec(v_a_827_);
lean_dec_ref(v_a_826_);
return v_res_835_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Search_Expansion(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Exception(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Search_ExpandSafePrefix(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Search_Expansion(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Exception(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_aesop___private_Aesop_Search_ExpandSafePrefix_0__Aesop_initFn_00___x40_Aesop_Search_ExpandSafePrefix_1752386596____hygCtx___hyg_5_();
if (lean_io_result_is_error(res)) return res;
lp_aesop_Aesop_safeExpansionFailedExceptionId = lean_io_result_get_value(res);
lean_mark_persistent(lp_aesop_Aesop_safeExpansionFailedExceptionId);
lean_dec_ref(res);
lp_aesop_Aesop_safeExpansionFailedException = _init_lp_aesop_Aesop_safeExpansionFailedException();
lean_mark_persistent(lp_aesop_Aesop_safeExpansionFailedException);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Search_ExpandSafePrefix(uint8_t builtin) {
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
lean_object* initialize_aesop_Aesop_Search_Expansion(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Exception(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Search_ExpandSafePrefix(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Search_Expansion(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Exception(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Search_ExpandSafePrefix(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Search_ExpandSafePrefix(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Search_ExpandSafePrefix(builtin);
}
#ifdef __cplusplus
}
#endif
