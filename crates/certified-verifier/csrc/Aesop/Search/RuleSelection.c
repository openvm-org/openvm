// Lean compiler output
// Module: Aesop.Search.RuleSelection
// Imports: public import Init public meta import Init public import Aesop.Tree.RunMetaM public import Aesop.Search.SearchM
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
lean_object* lean_io_mono_nanos_now();
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_SearchM_instMonad(lean_object*, lean_object*);
extern lean_object* lp_aesop_Aesop_BaseM_instMonadStats;
lean_object* lp_aesop_Aesop_instMonadStatsStateRefT_x27___redArg(lean_object*);
lean_object* lp_aesop_Aesop_instMonadStatsReaderT___redArg(lean_object*);
extern lean_object* l_Lean_KVMap_instValueBool;
uint8_t lp_aesop_Aesop_Goal_isRoot(lean_object*);
extern lean_object* lp_aesop_Aesop_treeImpl;
lean_object* lp_aesop_Aesop_LocalRuleSet_applicableSafeRulesWith(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_SearchM_instMonadLiftTreeM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instMonadLift___lam__0___boxed(lean_object*, lean_object*, lean_object*);
lean_object* l_StateRefT_x27_lift___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instMonadLiftT___lam__0___boxed(lean_object*, lean_object*);
lean_object* l_instMonadLiftTOfMonadLift___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instMonadFinallyEIO___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instMonadFinallyStateRefT_x27___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_tryFinally___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_instMonadExceptOfExceptionCoreM;
lean_object* l_StateRefT_x27_instMonadExceptOf___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_StateRefT_x27_instMonadExceptOf___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instMonadExceptOf___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instMonadExceptOf___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_SearchM_instMonadRef(lean_object*, lean_object*);
extern lean_object* l_Lean_Meta_instAddMessageContextMetaM;
lean_object* l_Lean_instAddMessageContextOfMonadLift___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_Goal_runMetaMInPostNormState_x27___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
uint64_t lp_aesop_Aesop_instHashableScopeName_hash(uint8_t);
uint64_t lp_aesop_Aesop_instHashablePhaseName_hash(uint8_t);
uint64_t lean_uint64_mix_hash(uint64_t, uint64_t);
uint64_t lp_aesop_Aesop_instHashableBuilderName_hash(uint8_t);
lean_object* lean_nat_to_int(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
extern lean_object* lp_aesop_Aesop_aesop_collectStats;
lean_object* l_Lean_Option_get___redArg(lean_object*, lean_object*, lean_object*);
extern lean_object* lp_aesop_Aesop_TraceOption_stats;
lean_object* lp_aesop_Aesop_TraceOption_isEnabled___redArg(lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_KVMap_instValueString;
extern lean_object* lp_aesop_Aesop_aesop_stats_file;
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_LocalRuleSet_applicableUnsafeRulesWith(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_UnsafeQueue_initial(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_LocalRuleSet_applicableNormalizationRulesWith(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_Option_get___at___00Aesop_selectNormRules_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Aesop_selectNormRules_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Aesop_selectNormRules_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Aesop_selectNormRules_spec__2___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_selectNormRules___lam__0(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_selectNormRules___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_selectNormRules___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_selectNormRules___lam__1___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_selectNormRules_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_selectNormRules_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_selectNormRules___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_selectNormRules___lam__1___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_selectNormRules___closed__0 = (const lean_object*)&lp_aesop_Aesop_selectNormRules___closed__0_value;
static const lean_string_object lp_aesop_Aesop_selectNormRules___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_aesop_Aesop_selectNormRules___closed__1 = (const lean_object*)&lp_aesop_Aesop_selectNormRules___closed__1_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_selectNormRules(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_selectNormRules___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_selectNormRules_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_selectNormRules_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_preprocessRule___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Aesop"};
static const lean_object* lp_aesop_Aesop_preprocessRule___closed__0 = (const lean_object*)&lp_aesop_Aesop_preprocessRule___closed__0_value;
static const lean_string_object lp_aesop_Aesop_preprocessRule___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "BuiltinRule"};
static const lean_object* lp_aesop_Aesop_preprocessRule___closed__1 = (const lean_object*)&lp_aesop_Aesop_preprocessRule___closed__1_value;
static const lean_string_object lp_aesop_Aesop_preprocessRule___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "preprocess"};
static const lean_object* lp_aesop_Aesop_preprocessRule___closed__2 = (const lean_object*)&lp_aesop_Aesop_preprocessRule___closed__2_value;
static const lean_ctor_object lp_aesop_Aesop_preprocessRule___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_preprocessRule___closed__0_value),LEAN_SCALAR_PTR_LITERAL(97, 226, 101, 135, 78, 117, 164, 248)}};
static const lean_ctor_object lp_aesop_Aesop_preprocessRule___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_preprocessRule___closed__3_value_aux_0),((lean_object*)&lp_aesop_Aesop_preprocessRule___closed__1_value),LEAN_SCALAR_PTR_LITERAL(247, 148, 49, 66, 86, 168, 186, 51)}};
static const lean_ctor_object lp_aesop_Aesop_preprocessRule___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop_Aesop_preprocessRule___closed__3_value_aux_1),((lean_object*)&lp_aesop_Aesop_preprocessRule___closed__2_value),LEAN_SCALAR_PTR_LITERAL(193, 82, 210, 128, 137, 109, 160, 82)}};
static const lean_object* lp_aesop_Aesop_preprocessRule___closed__3 = (const lean_object*)&lp_aesop_Aesop_preprocessRule___closed__3_value;
static lean_once_cell_t lp_aesop_Aesop_preprocessRule___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static uint64_t lp_aesop_Aesop_preprocessRule___closed__4;
static lean_once_cell_t lp_aesop_Aesop_preprocessRule___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static uint64_t lp_aesop_Aesop_preprocessRule___closed__5;
static lean_once_cell_t lp_aesop_Aesop_preprocessRule___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static uint64_t lp_aesop_Aesop_preprocessRule___closed__6;
static lean_once_cell_t lp_aesop_Aesop_preprocessRule___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static uint64_t lp_aesop_Aesop_preprocessRule___closed__7;
static lean_once_cell_t lp_aesop_Aesop_preprocessRule___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static uint64_t lp_aesop_Aesop_preprocessRule___closed__8;
static lean_once_cell_t lp_aesop_Aesop_preprocessRule___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_preprocessRule___closed__9;
static lean_once_cell_t lp_aesop_Aesop_preprocessRule___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_preprocessRule___closed__10;
LEAN_EXPORT lean_object* lp_aesop_Aesop_preprocessRule;
LEAN_EXPORT uint8_t lp_aesop_Aesop_selectSafeRules___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_selectSafeRules___redArg___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_selectSafeRules___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_selectSafeRules___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Aesop_selectSafeRules___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_selectSafeRules___redArg___closed__0;
static lean_once_cell_t lp_aesop_Aesop_selectSafeRules___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_selectSafeRules___redArg___closed__1;
static lean_once_cell_t lp_aesop_Aesop_selectSafeRules___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_selectSafeRules___redArg___closed__2;
static const lean_closure_object lp_aesop_Aesop_selectSafeRules___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_selectSafeRules___redArg___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_selectSafeRules___redArg___closed__3 = (const lean_object*)&lp_aesop_Aesop_selectSafeRules___redArg___closed__3_value;
static const lean_closure_object lp_aesop_Aesop_selectSafeRules___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_ReaderT_instMonadLift___lam__0___boxed, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_selectSafeRules___redArg___closed__4 = (const lean_object*)&lp_aesop_Aesop_selectSafeRules___redArg___closed__4_value;
static const lean_closure_object lp_aesop_Aesop_selectSafeRules___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_StateRefT_x27_lift___boxed, .m_arity = 6, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_aesop_Aesop_selectSafeRules___redArg___closed__5 = (const lean_object*)&lp_aesop_Aesop_selectSafeRules___redArg___closed__5_value;
static const lean_closure_object lp_aesop_Aesop_selectSafeRules___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_SearchM_instMonadLiftTreeM___lam__0___boxed, .m_arity = 11, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_selectSafeRules___redArg___closed__6 = (const lean_object*)&lp_aesop_Aesop_selectSafeRules___redArg___closed__6_value;
static const lean_closure_object lp_aesop_Aesop_selectSafeRules___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instMonadLiftT___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_selectSafeRules___redArg___closed__7 = (const lean_object*)&lp_aesop_Aesop_selectSafeRules___redArg___closed__7_value;
static const lean_closure_object lp_aesop_Aesop_selectSafeRules___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instMonadLiftTOfMonadLift___redArg___lam__0, .m_arity = 4, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_aesop_Aesop_selectSafeRules___redArg___closed__7_value),((lean_object*)&lp_aesop_Aesop_selectSafeRules___redArg___closed__5_value)} };
static const lean_object* lp_aesop_Aesop_selectSafeRules___redArg___closed__8 = (const lean_object*)&lp_aesop_Aesop_selectSafeRules___redArg___closed__8_value;
static const lean_closure_object lp_aesop_Aesop_selectSafeRules___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instMonadLiftTOfMonadLift___redArg___lam__0, .m_arity = 4, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_aesop_Aesop_selectSafeRules___redArg___closed__8_value),((lean_object*)&lp_aesop_Aesop_selectSafeRules___redArg___closed__5_value)} };
static const lean_object* lp_aesop_Aesop_selectSafeRules___redArg___closed__9 = (const lean_object*)&lp_aesop_Aesop_selectSafeRules___redArg___closed__9_value;
static const lean_closure_object lp_aesop_Aesop_selectSafeRules___redArg___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instMonadLiftTOfMonadLift___redArg___lam__0, .m_arity = 4, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_aesop_Aesop_selectSafeRules___redArg___closed__9_value),((lean_object*)&lp_aesop_Aesop_selectSafeRules___redArg___closed__4_value)} };
static const lean_object* lp_aesop_Aesop_selectSafeRules___redArg___closed__10 = (const lean_object*)&lp_aesop_Aesop_selectSafeRules___redArg___closed__10_value;
static const lean_closure_object lp_aesop_Aesop_selectSafeRules___redArg___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instMonadLiftTOfMonadLift___redArg___lam__0, .m_arity = 4, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_aesop_Aesop_selectSafeRules___redArg___closed__10_value),((lean_object*)&lp_aesop_Aesop_selectSafeRules___redArg___closed__6_value)} };
static const lean_object* lp_aesop_Aesop_selectSafeRules___redArg___closed__11 = (const lean_object*)&lp_aesop_Aesop_selectSafeRules___redArg___closed__11_value;
static const lean_closure_object lp_aesop_Aesop_selectSafeRules___redArg___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instMonadFinallyEIO___aux__1___boxed, .m_arity = 6, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_aesop_Aesop_selectSafeRules___redArg___closed__12 = (const lean_object*)&lp_aesop_Aesop_selectSafeRules___redArg___closed__12_value;
static const lean_closure_object lp_aesop_Aesop_selectSafeRules___redArg___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*4, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instMonadFinallyStateRefT_x27___aux__1___boxed, .m_arity = 9, .m_num_fixed = 4, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_selectSafeRules___redArg___closed__12_value)} };
static const lean_object* lp_aesop_Aesop_selectSafeRules___redArg___closed__13 = (const lean_object*)&lp_aesop_Aesop_selectSafeRules___redArg___closed__13_value;
static const lean_closure_object lp_aesop_Aesop_selectSafeRules___redArg___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_ReaderT_tryFinally___redArg___lam__1, .m_arity = 6, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_aesop_Aesop_selectSafeRules___redArg___closed__13_value)} };
static const lean_object* lp_aesop_Aesop_selectSafeRules___redArg___closed__14 = (const lean_object*)&lp_aesop_Aesop_selectSafeRules___redArg___closed__14_value;
static const lean_closure_object lp_aesop_Aesop_selectSafeRules___redArg___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*4, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instMonadFinallyStateRefT_x27___aux__1___boxed, .m_arity = 9, .m_num_fixed = 4, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_selectSafeRules___redArg___closed__14_value)} };
static const lean_object* lp_aesop_Aesop_selectSafeRules___redArg___closed__15 = (const lean_object*)&lp_aesop_Aesop_selectSafeRules___redArg___closed__15_value;
static const lean_closure_object lp_aesop_Aesop_selectSafeRules___redArg___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_ReaderT_tryFinally___redArg___lam__1, .m_arity = 6, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_aesop_Aesop_selectSafeRules___redArg___closed__15_value)} };
static const lean_object* lp_aesop_Aesop_selectSafeRules___redArg___closed__16 = (const lean_object*)&lp_aesop_Aesop_selectSafeRules___redArg___closed__16_value;
static const lean_closure_object lp_aesop_Aesop_selectSafeRules___redArg___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*4, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instMonadFinallyStateRefT_x27___aux__1___boxed, .m_arity = 9, .m_num_fixed = 4, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_selectSafeRules___redArg___closed__16_value)} };
static const lean_object* lp_aesop_Aesop_selectSafeRules___redArg___closed__17 = (const lean_object*)&lp_aesop_Aesop_selectSafeRules___redArg___closed__17_value;
static const lean_closure_object lp_aesop_Aesop_selectSafeRules___redArg___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*4, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instMonadFinallyStateRefT_x27___aux__1___boxed, .m_arity = 9, .m_num_fixed = 4, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_selectSafeRules___redArg___closed__17_value)} };
static const lean_object* lp_aesop_Aesop_selectSafeRules___redArg___closed__18 = (const lean_object*)&lp_aesop_Aesop_selectSafeRules___redArg___closed__18_value;
static const lean_closure_object lp_aesop_Aesop_selectSafeRules___redArg___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*4, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instMonadFinallyStateRefT_x27___aux__1___boxed, .m_arity = 9, .m_num_fixed = 4, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_selectSafeRules___redArg___closed__18_value)} };
static const lean_object* lp_aesop_Aesop_selectSafeRules___redArg___closed__19 = (const lean_object*)&lp_aesop_Aesop_selectSafeRules___redArg___closed__19_value;
static const lean_closure_object lp_aesop_Aesop_selectSafeRules___redArg___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_ReaderT_tryFinally___redArg___lam__1, .m_arity = 6, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_aesop_Aesop_selectSafeRules___redArg___closed__19_value)} };
static const lean_object* lp_aesop_Aesop_selectSafeRules___redArg___closed__20 = (const lean_object*)&lp_aesop_Aesop_selectSafeRules___redArg___closed__20_value;
static lean_once_cell_t lp_aesop_Aesop_selectSafeRules___redArg___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_selectSafeRules___redArg___closed__21;
static lean_once_cell_t lp_aesop_Aesop_selectSafeRules___redArg___closed__22_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_selectSafeRules___redArg___closed__22;
static lean_once_cell_t lp_aesop_Aesop_selectSafeRules___redArg___closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_selectSafeRules___redArg___closed__23;
static lean_once_cell_t lp_aesop_Aesop_selectSafeRules___redArg___closed__24_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_selectSafeRules___redArg___closed__24;
static lean_once_cell_t lp_aesop_Aesop_selectSafeRules___redArg___closed__25_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_selectSafeRules___redArg___closed__25;
static lean_once_cell_t lp_aesop_Aesop_selectSafeRules___redArg___closed__26_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_selectSafeRules___redArg___closed__26;
static lean_once_cell_t lp_aesop_Aesop_selectSafeRules___redArg___closed__27_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_selectSafeRules___redArg___closed__27;
static lean_once_cell_t lp_aesop_Aesop_selectSafeRules___redArg___closed__28_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_selectSafeRules___redArg___closed__28;
static lean_once_cell_t lp_aesop_Aesop_selectSafeRules___redArg___closed__29_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_selectSafeRules___redArg___closed__29;
static lean_once_cell_t lp_aesop_Aesop_selectSafeRules___redArg___closed__30_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_selectSafeRules___redArg___closed__30;
static lean_once_cell_t lp_aesop_Aesop_selectSafeRules___redArg___closed__31_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_selectSafeRules___redArg___closed__31;
static lean_once_cell_t lp_aesop_Aesop_selectSafeRules___redArg___closed__32_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_selectSafeRules___redArg___closed__32;
static lean_once_cell_t lp_aesop_Aesop_selectSafeRules___redArg___closed__33_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_selectSafeRules___redArg___closed__33;
static lean_once_cell_t lp_aesop_Aesop_selectSafeRules___redArg___closed__34_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_selectSafeRules___redArg___closed__34;
static lean_once_cell_t lp_aesop_Aesop_selectSafeRules___redArg___closed__35_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_selectSafeRules___redArg___closed__35;
static lean_once_cell_t lp_aesop_Aesop_selectSafeRules___redArg___closed__36_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_selectSafeRules___redArg___closed__36;
static lean_once_cell_t lp_aesop_Aesop_selectSafeRules___redArg___closed__37_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_selectSafeRules___redArg___closed__37;
static lean_once_cell_t lp_aesop_Aesop_selectSafeRules___redArg___closed__38_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_selectSafeRules___redArg___closed__38;
static lean_once_cell_t lp_aesop_Aesop_selectSafeRules___redArg___closed__39_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_selectSafeRules___redArg___closed__39;
static lean_once_cell_t lp_aesop_Aesop_selectSafeRules___redArg___closed__40_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_selectSafeRules___redArg___closed__40;
static lean_once_cell_t lp_aesop_Aesop_selectSafeRules___redArg___closed__41_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_selectSafeRules___redArg___closed__41;
static lean_once_cell_t lp_aesop_Aesop_selectSafeRules___redArg___closed__42_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_selectSafeRules___redArg___closed__42;
static const lean_array_object lp_aesop_Aesop_selectSafeRules___redArg___closed__43_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_selectSafeRules___redArg___closed__43 = (const lean_object*)&lp_aesop_Aesop_selectSafeRules___redArg___closed__43_value;
static lean_once_cell_t lp_aesop_Aesop_selectSafeRules___redArg___closed__44_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_selectSafeRules___redArg___closed__44;
static lean_once_cell_t lp_aesop_Aesop_selectSafeRules___redArg___closed__45_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_selectSafeRules___redArg___closed__45;
LEAN_EXPORT lean_object* lp_aesop_Aesop_selectSafeRules___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_selectSafeRules___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_selectSafeRules(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_selectSafeRules___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_selectUnsafeRules___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_selectUnsafeRules___redArg___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_selectUnsafeRules___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_selectUnsafeRules___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_selectUnsafeRules___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_selectUnsafeRules___redArg___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_selectUnsafeRules___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_selectUnsafeRules___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_selectUnsafeRules___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_selectUnsafeRules___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_selectUnsafeRules(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_selectUnsafeRules___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_Option_get___at___00Aesop_selectNormRules_spec__0(lean_object* v_opts_1_, lean_object* v_opt_2_){
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
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Aesop_selectNormRules_spec__0___boxed(lean_object* v_opts_11_, lean_object* v_opt_12_){
_start:
{
uint8_t v_res_13_; lean_object* v_r_14_; 
v_res_13_ = lp_aesop_Lean_Option_get___at___00Aesop_selectNormRules_spec__0(v_opts_11_, v_opt_12_);
lean_dec_ref(v_opt_12_);
lean_dec_ref(v_opts_11_);
v_r_14_ = lean_box(v_res_13_);
return v_r_14_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Aesop_selectNormRules_spec__2(lean_object* v_opts_15_, lean_object* v_opt_16_){
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
if (lean_obj_tag(v_val_21_) == 0)
{
lean_object* v_v_22_; 
v_v_22_ = lean_ctor_get(v_val_21_, 0);
lean_inc_ref(v_v_22_);
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
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Aesop_selectNormRules_spec__2___boxed(lean_object* v_opts_23_, lean_object* v_opt_24_){
_start:
{
lean_object* v_res_25_; 
v_res_25_ = lp_aesop_Lean_Option_get___at___00Aesop_selectNormRules_spec__2(v_opts_23_, v_opt_24_);
lean_dec_ref(v_opt_24_);
lean_dec_ref(v_opts_23_);
return v_res_25_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_selectNormRules___lam__0(uint8_t v_a_26_, lean_object* v_x_27_){
_start:
{
return v_a_26_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_selectNormRules___lam__0___boxed(lean_object* v_a_28_, lean_object* v_x_29_){
_start:
{
uint8_t v_a_3700__boxed_30_; uint8_t v_res_31_; lean_object* v_r_32_; 
v_a_3700__boxed_30_ = lean_unbox(v_a_28_);
v_res_31_ = lp_aesop_Aesop_selectNormRules___lam__0(v_a_3700__boxed_30_, v_x_29_);
lean_dec_ref(v_x_29_);
v_r_32_ = lean_box(v_res_31_);
return v_r_32_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_selectNormRules___lam__1(lean_object* v_x_33_){
_start:
{
uint8_t v___x_34_; 
v___x_34_ = 1;
return v___x_34_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_selectNormRules___lam__1___boxed(lean_object* v_x_35_){
_start:
{
uint8_t v_res_36_; lean_object* v_r_37_; 
v_res_36_ = lp_aesop_Aesop_selectNormRules___lam__1(v_x_35_);
lean_dec_ref(v_x_35_);
v_r_37_ = lean_box(v_res_36_);
return v_r_37_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_selectNormRules_spec__1___redArg(lean_object* v_opt_38_, lean_object* v___y_39_){
_start:
{
lean_object* v_options_41_; lean_object* v_option_42_; uint8_t v___x_43_; lean_object* v___x_44_; lean_object* v___x_45_; 
v_options_41_ = lean_ctor_get(v___y_39_, 2);
v_option_42_ = lean_ctor_get(v_opt_38_, 1);
v___x_43_ = lp_aesop_Lean_Option_get___at___00Aesop_selectNormRules_spec__0(v_options_41_, v_option_42_);
v___x_44_ = lean_box(v___x_43_);
v___x_45_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_45_, 0, v___x_44_);
return v___x_45_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_selectNormRules_spec__1___redArg___boxed(lean_object* v_opt_46_, lean_object* v___y_47_, lean_object* v___y_48_){
_start:
{
lean_object* v_res_49_; 
v_res_49_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_selectNormRules_spec__1___redArg(v_opt_46_, v___y_47_);
lean_dec_ref(v___y_47_);
lean_dec_ref(v_opt_46_);
return v_res_49_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_selectNormRules(lean_object* v_rs_52_, lean_object* v_fms_53_, lean_object* v_goal_54_, lean_object* v_a_55_, lean_object* v_a_56_, lean_object* v_a_57_, lean_object* v_a_58_, lean_object* v_a_59_){
_start:
{
uint8_t v_a_62_; lean_object* v_options_106_; lean_object* v___x_107_; uint8_t v___x_108_; 
v_options_106_ = lean_ctor_get(v_a_58_, 2);
v___x_107_ = lp_aesop_Aesop_aesop_collectStats;
v___x_108_ = lp_aesop_Lean_Option_get___at___00Aesop_selectNormRules_spec__0(v_options_106_, v___x_107_);
if (v___x_108_ == 0)
{
lean_object* v___f_109_; lean_object* v___y_111_; lean_object* v___x_116_; lean_object* v___x_117_; lean_object* v_a_118_; uint8_t v___x_119_; 
v___f_109_ = ((lean_object*)(lp_aesop_Aesop_selectNormRules___closed__0));
v___x_116_ = lp_aesop_Aesop_TraceOption_stats;
v___x_117_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_selectNormRules_spec__1___redArg(v___x_116_, v_a_58_);
v_a_118_ = lean_ctor_get(v___x_117_, 0);
lean_inc(v_a_118_);
v___x_119_ = lean_unbox(v_a_118_);
lean_dec(v_a_118_);
if (v___x_119_ == 0)
{
lean_object* v___x_120_; lean_object* v___x_121_; lean_object* v___x_122_; uint8_t v___x_123_; 
lean_dec_ref(v___x_117_);
v___x_120_ = lp_aesop_Aesop_aesop_stats_file;
v___x_121_ = lp_aesop_Lean_Option_get___at___00Aesop_selectNormRules_spec__2(v_options_106_, v___x_120_);
v___x_122_ = ((lean_object*)(lp_aesop_Aesop_selectNormRules___closed__1));
v___x_123_ = lean_string_dec_eq(v___x_121_, v___x_122_);
lean_dec_ref(v___x_121_);
if (v___x_123_ == 0)
{
uint8_t v___x_124_; 
v___x_124_ = 1;
v_a_62_ = v___x_124_;
goto v___jp_61_;
}
else
{
lean_object* v___x_125_; 
v___x_125_ = lp_aesop_Aesop_LocalRuleSet_applicableNormalizationRulesWith(v_rs_52_, v_fms_53_, v_goal_54_, v___f_109_, v_a_55_, v_a_56_, v_a_57_, v_a_58_, v_a_59_);
return v___x_125_;
}
}
else
{
v___y_111_ = v___x_117_;
goto v___jp_110_;
}
v___jp_110_:
{
lean_object* v_a_112_; uint8_t v___x_113_; 
v_a_112_ = lean_ctor_get(v___y_111_, 0);
lean_inc(v_a_112_);
lean_dec_ref(v___y_111_);
v___x_113_ = lean_unbox(v_a_112_);
if (v___x_113_ == 0)
{
lean_object* v___x_114_; 
lean_dec(v_a_112_);
v___x_114_ = lp_aesop_Aesop_LocalRuleSet_applicableNormalizationRulesWith(v_rs_52_, v_fms_53_, v_goal_54_, v___f_109_, v_a_55_, v_a_56_, v_a_57_, v_a_58_, v_a_59_);
return v___x_114_;
}
else
{
uint8_t v___x_115_; 
v___x_115_ = lean_unbox(v_a_112_);
lean_dec(v_a_112_);
v_a_62_ = v___x_115_;
goto v___jp_61_;
}
}
}
else
{
v_a_62_ = v___x_108_;
goto v___jp_61_;
}
v___jp_61_:
{
lean_object* v___x_63_; lean_object* v___x_64_; lean_object* v___f_65_; lean_object* v___x_66_; 
v___x_63_ = lean_io_mono_nanos_now();
v___x_64_ = lean_box(v_a_62_);
v___f_65_ = lean_alloc_closure((void*)(lp_aesop_Aesop_selectNormRules___lam__0___boxed), 2, 1);
lean_closure_set(v___f_65_, 0, v___x_64_);
v___x_66_ = lp_aesop_Aesop_LocalRuleSet_applicableNormalizationRulesWith(v_rs_52_, v_fms_53_, v_goal_54_, v___f_65_, v_a_55_, v_a_56_, v_a_57_, v_a_58_, v_a_59_);
if (lean_obj_tag(v___x_66_) == 0)
{
lean_object* v_a_67_; lean_object* v___x_69_; uint8_t v_isShared_70_; uint8_t v_isSharedCheck_105_; 
v_a_67_ = lean_ctor_get(v___x_66_, 0);
v_isSharedCheck_105_ = !lean_is_exclusive(v___x_66_);
if (v_isSharedCheck_105_ == 0)
{
v___x_69_ = v___x_66_;
v_isShared_70_ = v_isSharedCheck_105_;
goto v_resetjp_68_;
}
else
{
lean_inc(v_a_67_);
lean_dec(v___x_66_);
v___x_69_ = lean_box(0);
v_isShared_70_ = v_isSharedCheck_105_;
goto v_resetjp_68_;
}
v_resetjp_68_:
{
lean_object* v___x_71_; lean_object* v___x_72_; lean_object* v_stats_73_; lean_object* v_rulePatternCache_74_; lean_object* v___x_76_; uint8_t v_isShared_77_; uint8_t v_isSharedCheck_104_; 
v___x_71_ = lean_io_mono_nanos_now();
v___x_72_ = lean_st_ref_take(v_a_55_);
v_stats_73_ = lean_ctor_get(v___x_72_, 1);
v_rulePatternCache_74_ = lean_ctor_get(v___x_72_, 0);
v_isSharedCheck_104_ = !lean_is_exclusive(v___x_72_);
if (v_isSharedCheck_104_ == 0)
{
v___x_76_ = v___x_72_;
v_isShared_77_ = v_isSharedCheck_104_;
goto v_resetjp_75_;
}
else
{
lean_inc(v_stats_73_);
lean_inc(v_rulePatternCache_74_);
lean_dec(v___x_72_);
v___x_76_ = lean_box(0);
v_isShared_77_ = v_isSharedCheck_104_;
goto v_resetjp_75_;
}
v_resetjp_75_:
{
lean_object* v_total_78_; lean_object* v_configParsing_79_; lean_object* v_ruleSetConstruction_80_; lean_object* v_search_81_; lean_object* v_ruleSelection_82_; lean_object* v_script_83_; lean_object* v_forwardState_84_; lean_object* v_scriptGenerated_85_; lean_object* v_ruleStats_86_; lean_object* v_goalStats_87_; lean_object* v___x_89_; uint8_t v_isShared_90_; uint8_t v_isSharedCheck_103_; 
v_total_78_ = lean_ctor_get(v_stats_73_, 0);
v_configParsing_79_ = lean_ctor_get(v_stats_73_, 1);
v_ruleSetConstruction_80_ = lean_ctor_get(v_stats_73_, 2);
v_search_81_ = lean_ctor_get(v_stats_73_, 3);
v_ruleSelection_82_ = lean_ctor_get(v_stats_73_, 4);
v_script_83_ = lean_ctor_get(v_stats_73_, 5);
v_forwardState_84_ = lean_ctor_get(v_stats_73_, 6);
v_scriptGenerated_85_ = lean_ctor_get(v_stats_73_, 7);
v_ruleStats_86_ = lean_ctor_get(v_stats_73_, 8);
v_goalStats_87_ = lean_ctor_get(v_stats_73_, 9);
v_isSharedCheck_103_ = !lean_is_exclusive(v_stats_73_);
if (v_isSharedCheck_103_ == 0)
{
v___x_89_ = v_stats_73_;
v_isShared_90_ = v_isSharedCheck_103_;
goto v_resetjp_88_;
}
else
{
lean_inc(v_goalStats_87_);
lean_inc(v_ruleStats_86_);
lean_inc(v_scriptGenerated_85_);
lean_inc(v_forwardState_84_);
lean_inc(v_script_83_);
lean_inc(v_ruleSelection_82_);
lean_inc(v_search_81_);
lean_inc(v_ruleSetConstruction_80_);
lean_inc(v_configParsing_79_);
lean_inc(v_total_78_);
lean_dec(v_stats_73_);
v___x_89_ = lean_box(0);
v_isShared_90_ = v_isSharedCheck_103_;
goto v_resetjp_88_;
}
v_resetjp_88_:
{
lean_object* v___x_91_; lean_object* v___x_92_; lean_object* v___x_94_; 
v___x_91_ = lean_nat_sub(v___x_71_, v___x_63_);
lean_dec(v___x_63_);
lean_dec(v___x_71_);
v___x_92_ = lean_nat_add(v_ruleSelection_82_, v___x_91_);
lean_dec(v___x_91_);
lean_dec(v_ruleSelection_82_);
if (v_isShared_90_ == 0)
{
lean_ctor_set(v___x_89_, 4, v___x_92_);
v___x_94_ = v___x_89_;
goto v_reusejp_93_;
}
else
{
lean_object* v_reuseFailAlloc_102_; 
v_reuseFailAlloc_102_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_102_, 0, v_total_78_);
lean_ctor_set(v_reuseFailAlloc_102_, 1, v_configParsing_79_);
lean_ctor_set(v_reuseFailAlloc_102_, 2, v_ruleSetConstruction_80_);
lean_ctor_set(v_reuseFailAlloc_102_, 3, v_search_81_);
lean_ctor_set(v_reuseFailAlloc_102_, 4, v___x_92_);
lean_ctor_set(v_reuseFailAlloc_102_, 5, v_script_83_);
lean_ctor_set(v_reuseFailAlloc_102_, 6, v_forwardState_84_);
lean_ctor_set(v_reuseFailAlloc_102_, 7, v_scriptGenerated_85_);
lean_ctor_set(v_reuseFailAlloc_102_, 8, v_ruleStats_86_);
lean_ctor_set(v_reuseFailAlloc_102_, 9, v_goalStats_87_);
v___x_94_ = v_reuseFailAlloc_102_;
goto v_reusejp_93_;
}
v_reusejp_93_:
{
lean_object* v___x_96_; 
if (v_isShared_77_ == 0)
{
lean_ctor_set(v___x_76_, 1, v___x_94_);
v___x_96_ = v___x_76_;
goto v_reusejp_95_;
}
else
{
lean_object* v_reuseFailAlloc_101_; 
v_reuseFailAlloc_101_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_101_, 0, v_rulePatternCache_74_);
lean_ctor_set(v_reuseFailAlloc_101_, 1, v___x_94_);
v___x_96_ = v_reuseFailAlloc_101_;
goto v_reusejp_95_;
}
v_reusejp_95_:
{
lean_object* v___x_97_; lean_object* v___x_99_; 
v___x_97_ = lean_st_ref_set(v_a_55_, v___x_96_);
if (v_isShared_70_ == 0)
{
v___x_99_ = v___x_69_;
goto v_reusejp_98_;
}
else
{
lean_object* v_reuseFailAlloc_100_; 
v_reuseFailAlloc_100_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_100_, 0, v_a_67_);
v___x_99_ = v_reuseFailAlloc_100_;
goto v_reusejp_98_;
}
v_reusejp_98_:
{
return v___x_99_;
}
}
}
}
}
}
}
else
{
lean_dec(v___x_63_);
return v___x_66_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_selectNormRules___boxed(lean_object* v_rs_126_, lean_object* v_fms_127_, lean_object* v_goal_128_, lean_object* v_a_129_, lean_object* v_a_130_, lean_object* v_a_131_, lean_object* v_a_132_, lean_object* v_a_133_, lean_object* v_a_134_){
_start:
{
lean_object* v_res_135_; 
v_res_135_ = lp_aesop_Aesop_selectNormRules(v_rs_126_, v_fms_127_, v_goal_128_, v_a_129_, v_a_130_, v_a_131_, v_a_132_, v_a_133_);
lean_dec(v_a_133_);
lean_dec_ref(v_a_132_);
lean_dec(v_a_131_);
lean_dec_ref(v_a_130_);
lean_dec(v_a_129_);
lean_dec_ref(v_fms_127_);
return v_res_135_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_selectNormRules_spec__1(lean_object* v_opt_136_, lean_object* v___y_137_, lean_object* v___y_138_, lean_object* v___y_139_, lean_object* v___y_140_, lean_object* v___y_141_){
_start:
{
lean_object* v___x_143_; 
v___x_143_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_selectNormRules_spec__1___redArg(v_opt_136_, v___y_140_);
return v___x_143_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_selectNormRules_spec__1___boxed(lean_object* v_opt_144_, lean_object* v___y_145_, lean_object* v___y_146_, lean_object* v___y_147_, lean_object* v___y_148_, lean_object* v___y_149_, lean_object* v___y_150_){
_start:
{
lean_object* v_res_151_; 
v_res_151_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_selectNormRules_spec__1(v_opt_144_, v___y_145_, v___y_146_, v___y_147_, v___y_148_, v___y_149_);
lean_dec(v___y_149_);
lean_dec_ref(v___y_148_);
lean_dec(v___y_147_);
lean_dec_ref(v___y_146_);
lean_dec(v___y_145_);
lean_dec_ref(v_opt_144_);
return v_res_151_;
}
}
static uint64_t _init_lp_aesop_Aesop_preprocessRule___closed__4(void){
_start:
{
uint8_t v___x_159_; uint64_t v___x_160_; 
v___x_159_ = 6;
v___x_160_ = lp_aesop_Aesop_instHashableBuilderName_hash(v___x_159_);
return v___x_160_;
}
}
static uint64_t _init_lp_aesop_Aesop_preprocessRule___closed__5(void){
_start:
{
uint8_t v___x_161_; uint64_t v___x_162_; 
v___x_161_ = 1;
v___x_162_ = lp_aesop_Aesop_instHashablePhaseName_hash(v___x_161_);
return v___x_162_;
}
}
static uint64_t _init_lp_aesop_Aesop_preprocessRule___closed__6(void){
_start:
{
uint8_t v___x_163_; uint64_t v___x_164_; 
v___x_163_ = 0;
v___x_164_ = lp_aesop_Aesop_instHashableScopeName_hash(v___x_163_);
return v___x_164_;
}
}
static uint64_t _init_lp_aesop_Aesop_preprocessRule___closed__7(void){
_start:
{
uint64_t v___x_165_; uint64_t v___x_166_; uint64_t v___x_167_; 
v___x_165_ = lean_uint64_once(&lp_aesop_Aesop_preprocessRule___closed__6, &lp_aesop_Aesop_preprocessRule___closed__6_once, _init_lp_aesop_Aesop_preprocessRule___closed__6);
v___x_166_ = lean_uint64_once(&lp_aesop_Aesop_preprocessRule___closed__5, &lp_aesop_Aesop_preprocessRule___closed__5_once, _init_lp_aesop_Aesop_preprocessRule___closed__5);
v___x_167_ = lean_uint64_mix_hash(v___x_166_, v___x_165_);
return v___x_167_;
}
}
static uint64_t _init_lp_aesop_Aesop_preprocessRule___closed__8(void){
_start:
{
uint64_t v___x_168_; uint64_t v___x_169_; uint64_t v___x_170_; 
v___x_168_ = lean_uint64_once(&lp_aesop_Aesop_preprocessRule___closed__7, &lp_aesop_Aesop_preprocessRule___closed__7_once, _init_lp_aesop_Aesop_preprocessRule___closed__7);
v___x_169_ = lean_uint64_once(&lp_aesop_Aesop_preprocessRule___closed__4, &lp_aesop_Aesop_preprocessRule___closed__4_once, _init_lp_aesop_Aesop_preprocessRule___closed__4);
v___x_170_ = lean_uint64_mix_hash(v___x_169_, v___x_168_);
return v___x_170_;
}
}
static lean_object* _init_lp_aesop_Aesop_preprocessRule___closed__9(void){
_start:
{
lean_object* v___x_171_; lean_object* v___x_172_; 
v___x_171_ = lean_unsigned_to_nat(0u);
v___x_172_ = lean_nat_to_int(v___x_171_);
return v___x_172_;
}
}
static lean_object* _init_lp_aesop_Aesop_preprocessRule___closed__10(void){
_start:
{
uint8_t v___x_173_; lean_object* v___x_174_; lean_object* v___x_175_; 
v___x_173_ = 0;
v___x_174_ = lean_obj_once(&lp_aesop_Aesop_preprocessRule___closed__9, &lp_aesop_Aesop_preprocessRule___closed__9_once, _init_lp_aesop_Aesop_preprocessRule___closed__9);
v___x_175_ = lean_alloc_ctor(0, 1, 1);
lean_ctor_set(v___x_175_, 0, v___x_174_);
lean_ctor_set_uint8(v___x_175_, sizeof(void*)*1, v___x_173_);
return v___x_175_;
}
}
static lean_object* _init_lp_aesop_Aesop_preprocessRule(void){
_start:
{
lean_object* v___x_176_; uint8_t v___x_177_; uint8_t v___x_178_; uint8_t v___x_179_; uint64_t v___y_181_; 
v___x_176_ = ((lean_object*)(lp_aesop_Aesop_preprocessRule___closed__3));
v___x_177_ = 6;
v___x_178_ = 1;
v___x_179_ = 0;
if (lean_obj_tag(v___x_176_) == 0)
{
uint64_t v___x_190_; 
v___x_190_ = 1723ULL;
v___y_181_ = v___x_190_;
goto v___jp_180_;
}
else
{
uint64_t v_hash_191_; 
v_hash_191_ = lean_ctor_get_uint64(v___x_176_, sizeof(void*)*2);
v___y_181_ = v_hash_191_;
goto v___jp_180_;
}
v___jp_180_:
{
uint64_t v___x_182_; uint64_t v___x_183_; lean_object* v___x_184_; lean_object* v___x_185_; lean_object* v___x_186_; lean_object* v___x_187_; lean_object* v___x_188_; lean_object* v___x_189_; 
v___x_182_ = lean_uint64_once(&lp_aesop_Aesop_preprocessRule___closed__8, &lp_aesop_Aesop_preprocessRule___closed__8_once, _init_lp_aesop_Aesop_preprocessRule___closed__8);
v___x_183_ = lean_uint64_mix_hash(v___y_181_, v___x_182_);
v___x_184_ = lean_alloc_ctor(0, 1, 11);
lean_ctor_set(v___x_184_, 0, v___x_176_);
lean_ctor_set_uint8(v___x_184_, sizeof(void*)*1 + 8, v___x_177_);
lean_ctor_set_uint8(v___x_184_, sizeof(void*)*1 + 9, v___x_178_);
lean_ctor_set_uint8(v___x_184_, sizeof(void*)*1 + 10, v___x_179_);
lean_ctor_set_uint64(v___x_184_, sizeof(void*)*1, v___x_183_);
v___x_185_ = lean_box(0);
v___x_186_ = lean_box(0);
v___x_187_ = lean_obj_once(&lp_aesop_Aesop_preprocessRule___closed__10, &lp_aesop_Aesop_preprocessRule___closed__10_once, _init_lp_aesop_Aesop_preprocessRule___closed__10);
v___x_188_ = lean_box(9);
v___x_189_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_189_, 0, v___x_184_);
lean_ctor_set(v___x_189_, 1, v___x_185_);
lean_ctor_set(v___x_189_, 2, v___x_186_);
lean_ctor_set(v___x_189_, 3, v___x_187_);
lean_ctor_set(v___x_189_, 4, v___x_188_);
return v___x_189_;
}
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_selectSafeRules___redArg___lam__0(lean_object* v_x_192_){
_start:
{
uint8_t v___x_193_; 
v___x_193_ = 1;
return v___x_193_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_selectSafeRules___redArg___lam__0___boxed(lean_object* v_x_194_){
_start:
{
uint8_t v_res_195_; lean_object* v_r_196_; 
v_res_195_ = lp_aesop_Aesop_selectSafeRules___redArg___lam__0(v_x_194_);
lean_dec_ref(v_x_194_);
v_r_196_ = lean_box(v_res_195_);
return v_r_196_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_selectSafeRules___redArg___lam__1(lean_object* v_g_197_, lean_object* v_ruleSet_198_, lean_object* v___f_199_, lean_object* v_postNormGoal_200_, lean_object* v___y_201_, lean_object* v___y_202_, lean_object* v___y_203_, lean_object* v___y_204_, lean_object* v___y_205_, lean_object* v___y_206_, lean_object* v___y_207_, lean_object* v___y_208_){
_start:
{
lean_object* v___x_210_; lean_object* v___x_211_; lean_object* v_elimGoal_212_; lean_object* v___x_213_; lean_object* v_forwardRuleMatches_214_; lean_object* v___x_215_; 
v___x_210_ = lean_st_ref_get(v___y_202_);
lean_dec(v___x_210_);
v___x_211_ = lp_aesop_Aesop_treeImpl;
v_elimGoal_212_ = lean_ctor_get(v___x_211_, 1);
lean_inc_ref(v_elimGoal_212_);
v___x_213_ = lean_apply_1(v_elimGoal_212_, v_g_197_);
v_forwardRuleMatches_214_ = lean_ctor_get(v___x_213_, 9);
lean_inc_ref(v_forwardRuleMatches_214_);
lean_dec_ref(v___x_213_);
v___x_215_ = lp_aesop_Aesop_LocalRuleSet_applicableSafeRulesWith(v_ruleSet_198_, v_forwardRuleMatches_214_, v_postNormGoal_200_, v___f_199_, v___y_204_, v___y_205_, v___y_206_, v___y_207_, v___y_208_);
lean_dec_ref(v_forwardRuleMatches_214_);
return v___x_215_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_selectSafeRules___redArg___lam__1___boxed(lean_object* v_g_216_, lean_object* v_ruleSet_217_, lean_object* v___f_218_, lean_object* v_postNormGoal_219_, lean_object* v___y_220_, lean_object* v___y_221_, lean_object* v___y_222_, lean_object* v___y_223_, lean_object* v___y_224_, lean_object* v___y_225_, lean_object* v___y_226_, lean_object* v___y_227_, lean_object* v___y_228_){
_start:
{
lean_object* v_res_229_; 
v_res_229_ = lp_aesop_Aesop_selectSafeRules___redArg___lam__1(v_g_216_, v_ruleSet_217_, v___f_218_, v_postNormGoal_219_, v___y_220_, v___y_221_, v___y_222_, v___y_223_, v___y_224_, v___y_225_, v___y_226_, v___y_227_);
lean_dec(v___y_227_);
lean_dec_ref(v___y_226_);
lean_dec(v___y_225_);
lean_dec_ref(v___y_224_);
lean_dec(v___y_223_);
lean_dec(v___y_222_);
lean_dec(v___y_221_);
lean_dec_ref(v___y_220_);
return v_res_229_;
}
}
static lean_object* _init_lp_aesop_Aesop_selectSafeRules___redArg___closed__0(void){
_start:
{
lean_object* v___x_230_; lean_object* v___x_231_; 
v___x_230_ = lp_aesop_Aesop_BaseM_instMonadStats;
v___x_231_ = lp_aesop_Aesop_instMonadStatsStateRefT_x27___redArg(v___x_230_);
return v___x_231_;
}
}
static lean_object* _init_lp_aesop_Aesop_selectSafeRules___redArg___closed__1(void){
_start:
{
lean_object* v___x_232_; lean_object* v___x_233_; 
v___x_232_ = lean_obj_once(&lp_aesop_Aesop_selectSafeRules___redArg___closed__0, &lp_aesop_Aesop_selectSafeRules___redArg___closed__0_once, _init_lp_aesop_Aesop_selectSafeRules___redArg___closed__0);
v___x_233_ = lp_aesop_Aesop_instMonadStatsStateRefT_x27___redArg(v___x_232_);
return v___x_233_;
}
}
static lean_object* _init_lp_aesop_Aesop_selectSafeRules___redArg___closed__2(void){
_start:
{
lean_object* v___x_234_; lean_object* v___x_235_; 
v___x_234_ = lean_obj_once(&lp_aesop_Aesop_selectSafeRules___redArg___closed__1, &lp_aesop_Aesop_selectSafeRules___redArg___closed__1_once, _init_lp_aesop_Aesop_selectSafeRules___redArg___closed__1);
v___x_235_ = lp_aesop_Aesop_instMonadStatsReaderT___redArg(v___x_234_);
return v___x_235_;
}
}
static lean_object* _init_lp_aesop_Aesop_selectSafeRules___redArg___closed__21(void){
_start:
{
lean_object* v___x_270_; lean_object* v___f_271_; 
v___x_270_ = l_Lean_instMonadExceptOfExceptionCoreM;
v___f_271_ = lean_alloc_closure((void*)(l_StateRefT_x27_instMonadExceptOf___redArg___lam__0___boxed), 4, 1);
lean_closure_set(v___f_271_, 0, v___x_270_);
return v___f_271_;
}
}
static lean_object* _init_lp_aesop_Aesop_selectSafeRules___redArg___closed__22(void){
_start:
{
lean_object* v___x_272_; lean_object* v___f_273_; 
v___x_272_ = l_Lean_instMonadExceptOfExceptionCoreM;
v___f_273_ = lean_alloc_closure((void*)(l_StateRefT_x27_instMonadExceptOf___redArg___lam__2), 5, 1);
lean_closure_set(v___f_273_, 0, v___x_272_);
return v___f_273_;
}
}
static lean_object* _init_lp_aesop_Aesop_selectSafeRules___redArg___closed__23(void){
_start:
{
lean_object* v___f_274_; lean_object* v___f_275_; lean_object* v___x_276_; 
v___f_274_ = lean_obj_once(&lp_aesop_Aesop_selectSafeRules___redArg___closed__22, &lp_aesop_Aesop_selectSafeRules___redArg___closed__22_once, _init_lp_aesop_Aesop_selectSafeRules___redArg___closed__22);
v___f_275_ = lean_obj_once(&lp_aesop_Aesop_selectSafeRules___redArg___closed__21, &lp_aesop_Aesop_selectSafeRules___redArg___closed__21_once, _init_lp_aesop_Aesop_selectSafeRules___redArg___closed__21);
v___x_276_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_276_, 0, v___f_275_);
lean_ctor_set(v___x_276_, 1, v___f_274_);
return v___x_276_;
}
}
static lean_object* _init_lp_aesop_Aesop_selectSafeRules___redArg___closed__24(void){
_start:
{
lean_object* v___x_277_; lean_object* v___f_278_; 
v___x_277_ = lean_obj_once(&lp_aesop_Aesop_selectSafeRules___redArg___closed__23, &lp_aesop_Aesop_selectSafeRules___redArg___closed__23_once, _init_lp_aesop_Aesop_selectSafeRules___redArg___closed__23);
v___f_278_ = lean_alloc_closure((void*)(l_ReaderT_instMonadExceptOf___redArg___lam__0___boxed), 4, 1);
lean_closure_set(v___f_278_, 0, v___x_277_);
return v___f_278_;
}
}
static lean_object* _init_lp_aesop_Aesop_selectSafeRules___redArg___closed__25(void){
_start:
{
lean_object* v___x_279_; lean_object* v___f_280_; 
v___x_279_ = lean_obj_once(&lp_aesop_Aesop_selectSafeRules___redArg___closed__23, &lp_aesop_Aesop_selectSafeRules___redArg___closed__23_once, _init_lp_aesop_Aesop_selectSafeRules___redArg___closed__23);
v___f_280_ = lean_alloc_closure((void*)(l_ReaderT_instMonadExceptOf___redArg___lam__2), 5, 1);
lean_closure_set(v___f_280_, 0, v___x_279_);
return v___f_280_;
}
}
static lean_object* _init_lp_aesop_Aesop_selectSafeRules___redArg___closed__26(void){
_start:
{
lean_object* v___f_281_; lean_object* v___f_282_; lean_object* v___x_283_; 
v___f_281_ = lean_obj_once(&lp_aesop_Aesop_selectSafeRules___redArg___closed__25, &lp_aesop_Aesop_selectSafeRules___redArg___closed__25_once, _init_lp_aesop_Aesop_selectSafeRules___redArg___closed__25);
v___f_282_ = lean_obj_once(&lp_aesop_Aesop_selectSafeRules___redArg___closed__24, &lp_aesop_Aesop_selectSafeRules___redArg___closed__24_once, _init_lp_aesop_Aesop_selectSafeRules___redArg___closed__24);
v___x_283_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_283_, 0, v___f_282_);
lean_ctor_set(v___x_283_, 1, v___f_281_);
return v___x_283_;
}
}
static lean_object* _init_lp_aesop_Aesop_selectSafeRules___redArg___closed__27(void){
_start:
{
lean_object* v___x_284_; lean_object* v___f_285_; 
v___x_284_ = lean_obj_once(&lp_aesop_Aesop_selectSafeRules___redArg___closed__26, &lp_aesop_Aesop_selectSafeRules___redArg___closed__26_once, _init_lp_aesop_Aesop_selectSafeRules___redArg___closed__26);
v___f_285_ = lean_alloc_closure((void*)(l_StateRefT_x27_instMonadExceptOf___redArg___lam__0___boxed), 4, 1);
lean_closure_set(v___f_285_, 0, v___x_284_);
return v___f_285_;
}
}
static lean_object* _init_lp_aesop_Aesop_selectSafeRules___redArg___closed__28(void){
_start:
{
lean_object* v___x_286_; lean_object* v___f_287_; 
v___x_286_ = lean_obj_once(&lp_aesop_Aesop_selectSafeRules___redArg___closed__26, &lp_aesop_Aesop_selectSafeRules___redArg___closed__26_once, _init_lp_aesop_Aesop_selectSafeRules___redArg___closed__26);
v___f_287_ = lean_alloc_closure((void*)(l_StateRefT_x27_instMonadExceptOf___redArg___lam__2), 5, 1);
lean_closure_set(v___f_287_, 0, v___x_286_);
return v___f_287_;
}
}
static lean_object* _init_lp_aesop_Aesop_selectSafeRules___redArg___closed__29(void){
_start:
{
lean_object* v___f_288_; lean_object* v___f_289_; lean_object* v___x_290_; 
v___f_288_ = lean_obj_once(&lp_aesop_Aesop_selectSafeRules___redArg___closed__28, &lp_aesop_Aesop_selectSafeRules___redArg___closed__28_once, _init_lp_aesop_Aesop_selectSafeRules___redArg___closed__28);
v___f_289_ = lean_obj_once(&lp_aesop_Aesop_selectSafeRules___redArg___closed__27, &lp_aesop_Aesop_selectSafeRules___redArg___closed__27_once, _init_lp_aesop_Aesop_selectSafeRules___redArg___closed__27);
v___x_290_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_290_, 0, v___f_289_);
lean_ctor_set(v___x_290_, 1, v___f_288_);
return v___x_290_;
}
}
static lean_object* _init_lp_aesop_Aesop_selectSafeRules___redArg___closed__30(void){
_start:
{
lean_object* v___x_291_; lean_object* v___f_292_; 
v___x_291_ = lean_obj_once(&lp_aesop_Aesop_selectSafeRules___redArg___closed__29, &lp_aesop_Aesop_selectSafeRules___redArg___closed__29_once, _init_lp_aesop_Aesop_selectSafeRules___redArg___closed__29);
v___f_292_ = lean_alloc_closure((void*)(l_StateRefT_x27_instMonadExceptOf___redArg___lam__0___boxed), 4, 1);
lean_closure_set(v___f_292_, 0, v___x_291_);
return v___f_292_;
}
}
static lean_object* _init_lp_aesop_Aesop_selectSafeRules___redArg___closed__31(void){
_start:
{
lean_object* v___x_293_; lean_object* v___f_294_; 
v___x_293_ = lean_obj_once(&lp_aesop_Aesop_selectSafeRules___redArg___closed__29, &lp_aesop_Aesop_selectSafeRules___redArg___closed__29_once, _init_lp_aesop_Aesop_selectSafeRules___redArg___closed__29);
v___f_294_ = lean_alloc_closure((void*)(l_StateRefT_x27_instMonadExceptOf___redArg___lam__2), 5, 1);
lean_closure_set(v___f_294_, 0, v___x_293_);
return v___f_294_;
}
}
static lean_object* _init_lp_aesop_Aesop_selectSafeRules___redArg___closed__32(void){
_start:
{
lean_object* v___f_295_; lean_object* v___f_296_; lean_object* v___x_297_; 
v___f_295_ = lean_obj_once(&lp_aesop_Aesop_selectSafeRules___redArg___closed__31, &lp_aesop_Aesop_selectSafeRules___redArg___closed__31_once, _init_lp_aesop_Aesop_selectSafeRules___redArg___closed__31);
v___f_296_ = lean_obj_once(&lp_aesop_Aesop_selectSafeRules___redArg___closed__30, &lp_aesop_Aesop_selectSafeRules___redArg___closed__30_once, _init_lp_aesop_Aesop_selectSafeRules___redArg___closed__30);
v___x_297_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_297_, 0, v___f_296_);
lean_ctor_set(v___x_297_, 1, v___f_295_);
return v___x_297_;
}
}
static lean_object* _init_lp_aesop_Aesop_selectSafeRules___redArg___closed__33(void){
_start:
{
lean_object* v___x_298_; lean_object* v___f_299_; 
v___x_298_ = lean_obj_once(&lp_aesop_Aesop_selectSafeRules___redArg___closed__32, &lp_aesop_Aesop_selectSafeRules___redArg___closed__32_once, _init_lp_aesop_Aesop_selectSafeRules___redArg___closed__32);
v___f_299_ = lean_alloc_closure((void*)(l_StateRefT_x27_instMonadExceptOf___redArg___lam__0___boxed), 4, 1);
lean_closure_set(v___f_299_, 0, v___x_298_);
return v___f_299_;
}
}
static lean_object* _init_lp_aesop_Aesop_selectSafeRules___redArg___closed__34(void){
_start:
{
lean_object* v___x_300_; lean_object* v___f_301_; 
v___x_300_ = lean_obj_once(&lp_aesop_Aesop_selectSafeRules___redArg___closed__32, &lp_aesop_Aesop_selectSafeRules___redArg___closed__32_once, _init_lp_aesop_Aesop_selectSafeRules___redArg___closed__32);
v___f_301_ = lean_alloc_closure((void*)(l_StateRefT_x27_instMonadExceptOf___redArg___lam__2), 5, 1);
lean_closure_set(v___f_301_, 0, v___x_300_);
return v___f_301_;
}
}
static lean_object* _init_lp_aesop_Aesop_selectSafeRules___redArg___closed__35(void){
_start:
{
lean_object* v___f_302_; lean_object* v___f_303_; lean_object* v___x_304_; 
v___f_302_ = lean_obj_once(&lp_aesop_Aesop_selectSafeRules___redArg___closed__34, &lp_aesop_Aesop_selectSafeRules___redArg___closed__34_once, _init_lp_aesop_Aesop_selectSafeRules___redArg___closed__34);
v___f_303_ = lean_obj_once(&lp_aesop_Aesop_selectSafeRules___redArg___closed__33, &lp_aesop_Aesop_selectSafeRules___redArg___closed__33_once, _init_lp_aesop_Aesop_selectSafeRules___redArg___closed__33);
v___x_304_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_304_, 0, v___f_303_);
lean_ctor_set(v___x_304_, 1, v___f_302_);
return v___x_304_;
}
}
static lean_object* _init_lp_aesop_Aesop_selectSafeRules___redArg___closed__36(void){
_start:
{
lean_object* v___x_305_; lean_object* v___f_306_; 
v___x_305_ = lean_obj_once(&lp_aesop_Aesop_selectSafeRules___redArg___closed__35, &lp_aesop_Aesop_selectSafeRules___redArg___closed__35_once, _init_lp_aesop_Aesop_selectSafeRules___redArg___closed__35);
v___f_306_ = lean_alloc_closure((void*)(l_ReaderT_instMonadExceptOf___redArg___lam__0___boxed), 4, 1);
lean_closure_set(v___f_306_, 0, v___x_305_);
return v___f_306_;
}
}
static lean_object* _init_lp_aesop_Aesop_selectSafeRules___redArg___closed__37(void){
_start:
{
lean_object* v___x_307_; lean_object* v___f_308_; 
v___x_307_ = lean_obj_once(&lp_aesop_Aesop_selectSafeRules___redArg___closed__35, &lp_aesop_Aesop_selectSafeRules___redArg___closed__35_once, _init_lp_aesop_Aesop_selectSafeRules___redArg___closed__35);
v___f_308_ = lean_alloc_closure((void*)(l_ReaderT_instMonadExceptOf___redArg___lam__2), 5, 1);
lean_closure_set(v___f_308_, 0, v___x_307_);
return v___f_308_;
}
}
static lean_object* _init_lp_aesop_Aesop_selectSafeRules___redArg___closed__38(void){
_start:
{
lean_object* v___f_309_; lean_object* v___f_310_; lean_object* v___x_311_; 
v___f_309_ = lean_obj_once(&lp_aesop_Aesop_selectSafeRules___redArg___closed__37, &lp_aesop_Aesop_selectSafeRules___redArg___closed__37_once, _init_lp_aesop_Aesop_selectSafeRules___redArg___closed__37);
v___f_310_ = lean_obj_once(&lp_aesop_Aesop_selectSafeRules___redArg___closed__36, &lp_aesop_Aesop_selectSafeRules___redArg___closed__36_once, _init_lp_aesop_Aesop_selectSafeRules___redArg___closed__36);
v___x_311_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_311_, 0, v___f_310_);
lean_ctor_set(v___x_311_, 1, v___f_309_);
return v___x_311_;
}
}
static lean_object* _init_lp_aesop_Aesop_selectSafeRules___redArg___closed__39(void){
_start:
{
lean_object* v___x_312_; lean_object* v___x_313_; lean_object* v___f_314_; 
v___x_312_ = ((lean_object*)(lp_aesop_Aesop_selectSafeRules___redArg___closed__5));
v___x_313_ = l_Lean_Meta_instAddMessageContextMetaM;
v___f_314_ = lean_alloc_closure((void*)(l_Lean_instAddMessageContextOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(v___f_314_, 0, v___x_313_);
lean_closure_set(v___f_314_, 1, v___x_312_);
return v___f_314_;
}
}
static lean_object* _init_lp_aesop_Aesop_selectSafeRules___redArg___closed__40(void){
_start:
{
lean_object* v___x_315_; lean_object* v___f_316_; lean_object* v___f_317_; 
v___x_315_ = ((lean_object*)(lp_aesop_Aesop_selectSafeRules___redArg___closed__5));
v___f_316_ = lean_obj_once(&lp_aesop_Aesop_selectSafeRules___redArg___closed__39, &lp_aesop_Aesop_selectSafeRules___redArg___closed__39_once, _init_lp_aesop_Aesop_selectSafeRules___redArg___closed__39);
v___f_317_ = lean_alloc_closure((void*)(l_Lean_instAddMessageContextOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(v___f_317_, 0, v___f_316_);
lean_closure_set(v___f_317_, 1, v___x_315_);
return v___f_317_;
}
}
static lean_object* _init_lp_aesop_Aesop_selectSafeRules___redArg___closed__41(void){
_start:
{
lean_object* v___f_318_; lean_object* v___f_319_; lean_object* v___f_320_; 
v___f_318_ = ((lean_object*)(lp_aesop_Aesop_selectSafeRules___redArg___closed__4));
v___f_319_ = lean_obj_once(&lp_aesop_Aesop_selectSafeRules___redArg___closed__40, &lp_aesop_Aesop_selectSafeRules___redArg___closed__40_once, _init_lp_aesop_Aesop_selectSafeRules___redArg___closed__40);
v___f_320_ = lean_alloc_closure((void*)(l_Lean_instAddMessageContextOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(v___f_320_, 0, v___f_319_);
lean_closure_set(v___f_320_, 1, v___f_318_);
return v___f_320_;
}
}
static lean_object* _init_lp_aesop_Aesop_selectSafeRules___redArg___closed__42(void){
_start:
{
lean_object* v___f_321_; lean_object* v___f_322_; lean_object* v___f_323_; 
v___f_321_ = ((lean_object*)(lp_aesop_Aesop_selectSafeRules___redArg___closed__6));
v___f_322_ = lean_obj_once(&lp_aesop_Aesop_selectSafeRules___redArg___closed__41, &lp_aesop_Aesop_selectSafeRules___redArg___closed__41_once, _init_lp_aesop_Aesop_selectSafeRules___redArg___closed__41);
v___f_323_ = lean_alloc_closure((void*)(l_Lean_instAddMessageContextOfMonadLift___redArg___lam__0), 3, 2);
lean_closure_set(v___f_323_, 0, v___f_322_);
lean_closure_set(v___f_323_, 1, v___f_321_);
return v___f_323_;
}
}
static lean_object* _init_lp_aesop_Aesop_selectSafeRules___redArg___closed__44(void){
_start:
{
lean_object* v___x_326_; lean_object* v___x_327_; lean_object* v___x_328_; lean_object* v___x_329_; 
v___x_326_ = lean_box(0);
v___x_327_ = ((lean_object*)(lp_aesop_Aesop_selectSafeRules___redArg___closed__43));
v___x_328_ = lp_aesop_Aesop_preprocessRule;
v___x_329_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_329_, 0, v___x_328_);
lean_ctor_set(v___x_329_, 1, v___x_327_);
lean_ctor_set(v___x_329_, 2, v___x_326_);
return v___x_329_;
}
}
static lean_object* _init_lp_aesop_Aesop_selectSafeRules___redArg___closed__45(void){
_start:
{
lean_object* v___x_330_; lean_object* v___x_331_; lean_object* v___x_332_; lean_object* v___x_333_; 
v___x_330_ = lean_obj_once(&lp_aesop_Aesop_selectSafeRules___redArg___closed__44, &lp_aesop_Aesop_selectSafeRules___redArg___closed__44_once, _init_lp_aesop_Aesop_selectSafeRules___redArg___closed__44);
v___x_331_ = lean_unsigned_to_nat(1u);
v___x_332_ = lean_mk_empty_array_with_capacity(v___x_331_);
v___x_333_ = lean_array_push(v___x_332_, v___x_330_);
return v___x_333_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_selectSafeRules___redArg(lean_object* v_inst_334_, lean_object* v_g_335_, lean_object* v_a_336_, lean_object* v_a_337_, lean_object* v_a_338_, lean_object* v_a_339_, lean_object* v_a_340_, lean_object* v_a_341_, lean_object* v_a_342_, lean_object* v_a_343_){
_start:
{
lean_object* v___y_346_; lean_object* v_a_347_; lean_object* v___x_381_; lean_object* v___x_382_; lean_object* v_toMonadOptions_383_; lean_object* v___x_384_; lean_object* v_options_385_; lean_object* v___f_386_; lean_object* v___x_405_; lean_object* v___x_406_; uint8_t v___x_407_; 
v___x_381_ = lp_aesop_Aesop_SearchM_instMonad(lean_box(0), v_inst_334_);
v___x_382_ = lean_obj_once(&lp_aesop_Aesop_selectSafeRules___redArg___closed__2, &lp_aesop_Aesop_selectSafeRules___redArg___closed__2_once, _init_lp_aesop_Aesop_selectSafeRules___redArg___closed__2);
v_toMonadOptions_383_ = lean_ctor_get(v___x_382_, 0);
v___x_384_ = l_Lean_KVMap_instValueBool;
v_options_385_ = lean_ctor_get(v_a_342_, 2);
v___f_386_ = ((lean_object*)(lp_aesop_Aesop_selectSafeRules___redArg___closed__3));
v___x_405_ = lp_aesop_Aesop_aesop_collectStats;
v___x_406_ = l_Lean_Option_get___redArg(v___x_384_, v_options_385_, v___x_405_);
v___x_407_ = lean_unbox(v___x_406_);
lean_dec(v___x_406_);
if (v___x_407_ == 0)
{
lean_object* v___y_425_; lean_object* v___x_436_; lean_object* v___x_15007__overap_437_; lean_object* v___x_438_; 
v___x_436_ = lp_aesop_Aesop_TraceOption_stats;
lean_inc(v_toMonadOptions_383_);
lean_inc_ref(v___x_381_);
v___x_15007__overap_437_ = lp_aesop_Aesop_TraceOption_isEnabled___redArg(v___x_381_, v_toMonadOptions_383_, v___x_436_);
lean_inc(v_a_343_);
lean_inc_ref(v_a_342_);
lean_inc(v_a_341_);
lean_inc_ref(v_a_340_);
lean_inc(v_a_339_);
lean_inc(v_a_338_);
lean_inc(v_a_337_);
lean_inc_ref(v_a_336_);
v___x_438_ = lean_apply_9(v___x_15007__overap_437_, v_a_336_, v_a_337_, v_a_338_, v_a_339_, v_a_340_, v_a_341_, v_a_342_, v_a_343_, lean_box(0));
if (lean_obj_tag(v___x_438_) == 0)
{
lean_object* v_a_439_; uint8_t v___x_440_; 
v_a_439_ = lean_ctor_get(v___x_438_, 0);
lean_inc(v_a_439_);
v___x_440_ = lean_unbox(v_a_439_);
lean_dec(v_a_439_);
if (v___x_440_ == 0)
{
lean_object* v___x_441_; lean_object* v___x_442_; lean_object* v___x_443_; lean_object* v___x_444_; uint8_t v___x_445_; 
lean_dec_ref_known(v___x_438_, 1);
v___x_441_ = l_Lean_KVMap_instValueString;
v___x_442_ = lp_aesop_Aesop_aesop_stats_file;
v___x_443_ = l_Lean_Option_get___redArg(v___x_441_, v_options_385_, v___x_442_);
v___x_444_ = ((lean_object*)(lp_aesop_Aesop_selectNormRules___closed__1));
v___x_445_ = lean_string_dec_eq(v___x_443_, v___x_444_);
lean_dec(v___x_443_);
if (v___x_445_ == 0)
{
goto v___jp_387_;
}
else
{
goto v___jp_408_;
}
}
else
{
v___y_425_ = v___x_438_;
goto v___jp_424_;
}
}
else
{
v___y_425_ = v___x_438_;
goto v___jp_424_;
}
v___jp_408_:
{
lean_object* v___x_409_; uint8_t v___x_410_; 
v___x_409_ = lean_st_ref_get(v_a_337_);
lean_dec(v___x_409_);
lean_inc(v_g_335_);
v___x_410_ = lp_aesop_Aesop_Goal_isRoot(v_g_335_);
if (v___x_410_ == 0)
{
lean_object* v_ruleSet_411_; lean_object* v___f_412_; lean_object* v___f_413_; lean_object* v___f_414_; lean_object* v___x_415_; lean_object* v___x_416_; lean_object* v___f_417_; lean_object* v___x_418_; lean_object* v___x_419_; lean_object* v___x_14862__overap_420_; lean_object* v___x_421_; 
v_ruleSet_411_ = lean_ctor_get(v_a_336_, 0);
lean_inc_ref(v_ruleSet_411_);
lean_inc(v_g_335_);
v___f_412_ = lean_alloc_closure((void*)(lp_aesop_Aesop_selectSafeRules___redArg___lam__1___boxed), 13, 3);
lean_closure_set(v___f_412_, 0, v_g_335_);
lean_closure_set(v___f_412_, 1, v_ruleSet_411_);
lean_closure_set(v___f_412_, 2, v___f_386_);
v___f_413_ = ((lean_object*)(lp_aesop_Aesop_selectSafeRules___redArg___closed__11));
v___f_414_ = ((lean_object*)(lp_aesop_Aesop_selectSafeRules___redArg___closed__20));
v___x_415_ = lean_obj_once(&lp_aesop_Aesop_selectSafeRules___redArg___closed__38, &lp_aesop_Aesop_selectSafeRules___redArg___closed__38_once, _init_lp_aesop_Aesop_selectSafeRules___redArg___closed__38);
v___x_416_ = lp_aesop_Aesop_SearchM_instMonadRef(lean_box(0), v_inst_334_);
v___f_417_ = lean_obj_once(&lp_aesop_Aesop_selectSafeRules___redArg___closed__42, &lp_aesop_Aesop_selectSafeRules___redArg___closed__42_once, _init_lp_aesop_Aesop_selectSafeRules___redArg___closed__42);
lean_inc_ref(v___x_381_);
v___x_418_ = l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(v___f_417_, v___x_381_);
v___x_419_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_419_, 0, v___x_415_);
lean_ctor_set(v___x_419_, 1, v___x_416_);
lean_ctor_set(v___x_419_, 2, v___x_418_);
v___x_14862__overap_420_ = lp_aesop_Aesop_Goal_runMetaMInPostNormState_x27___redArg(v___x_381_, v___f_413_, v___f_414_, v___x_419_, v___f_412_, v_g_335_);
lean_inc(v_a_343_);
lean_inc_ref(v_a_342_);
lean_inc(v_a_341_);
lean_inc_ref(v_a_340_);
lean_inc(v_a_339_);
lean_inc(v_a_338_);
lean_inc(v_a_337_);
lean_inc_ref(v_a_336_);
v___x_421_ = lean_apply_9(v___x_14862__overap_420_, v_a_336_, v_a_337_, v_a_338_, v_a_339_, v_a_340_, v_a_341_, v_a_342_, v_a_343_, lean_box(0));
return v___x_421_;
}
else
{
lean_object* v___x_422_; lean_object* v___x_423_; 
lean_dec_ref(v___x_381_);
lean_dec(v_g_335_);
v___x_422_ = lean_obj_once(&lp_aesop_Aesop_selectSafeRules___redArg___closed__45, &lp_aesop_Aesop_selectSafeRules___redArg___closed__45_once, _init_lp_aesop_Aesop_selectSafeRules___redArg___closed__45);
v___x_423_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_423_, 0, v___x_422_);
return v___x_423_;
}
}
v___jp_424_:
{
if (lean_obj_tag(v___y_425_) == 0)
{
lean_object* v_a_426_; uint8_t v___x_427_; 
v_a_426_ = lean_ctor_get(v___y_425_, 0);
lean_inc(v_a_426_);
lean_dec_ref_known(v___y_425_, 1);
v___x_427_ = lean_unbox(v_a_426_);
lean_dec(v_a_426_);
if (v___x_427_ == 0)
{
goto v___jp_408_;
}
else
{
goto v___jp_387_;
}
}
else
{
lean_object* v_a_428_; lean_object* v___x_430_; uint8_t v_isShared_431_; uint8_t v_isSharedCheck_435_; 
lean_dec_ref(v___x_381_);
lean_dec(v_g_335_);
v_a_428_ = lean_ctor_get(v___y_425_, 0);
v_isSharedCheck_435_ = !lean_is_exclusive(v___y_425_);
if (v_isSharedCheck_435_ == 0)
{
v___x_430_ = v___y_425_;
v_isShared_431_ = v_isSharedCheck_435_;
goto v_resetjp_429_;
}
else
{
lean_inc(v_a_428_);
lean_dec(v___y_425_);
v___x_430_ = lean_box(0);
v_isShared_431_ = v_isSharedCheck_435_;
goto v_resetjp_429_;
}
v_resetjp_429_:
{
lean_object* v___x_433_; 
if (v_isShared_431_ == 0)
{
v___x_433_ = v___x_430_;
goto v_reusejp_432_;
}
else
{
lean_object* v_reuseFailAlloc_434_; 
v_reuseFailAlloc_434_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_434_, 0, v_a_428_);
v___x_433_ = v_reuseFailAlloc_434_;
goto v_reusejp_432_;
}
v_reusejp_432_:
{
return v___x_433_;
}
}
}
}
}
else
{
goto v___jp_387_;
}
v___jp_345_:
{
lean_object* v___x_348_; lean_object* v___x_349_; lean_object* v___x_350_; lean_object* v_stats_351_; lean_object* v_rulePatternCache_352_; lean_object* v___x_354_; uint8_t v_isShared_355_; uint8_t v_isSharedCheck_380_; 
v___x_348_ = lean_st_ref_get(v_a_337_);
lean_dec(v___x_348_);
v___x_349_ = lean_io_mono_nanos_now();
v___x_350_ = lean_st_ref_take(v_a_339_);
v_stats_351_ = lean_ctor_get(v___x_350_, 1);
v_rulePatternCache_352_ = lean_ctor_get(v___x_350_, 0);
v_isSharedCheck_380_ = !lean_is_exclusive(v___x_350_);
if (v_isSharedCheck_380_ == 0)
{
v___x_354_ = v___x_350_;
v_isShared_355_ = v_isSharedCheck_380_;
goto v_resetjp_353_;
}
else
{
lean_inc(v_stats_351_);
lean_inc(v_rulePatternCache_352_);
lean_dec(v___x_350_);
v___x_354_ = lean_box(0);
v_isShared_355_ = v_isSharedCheck_380_;
goto v_resetjp_353_;
}
v_resetjp_353_:
{
lean_object* v_total_356_; lean_object* v_configParsing_357_; lean_object* v_ruleSetConstruction_358_; lean_object* v_search_359_; lean_object* v_ruleSelection_360_; lean_object* v_script_361_; lean_object* v_forwardState_362_; lean_object* v_scriptGenerated_363_; lean_object* v_ruleStats_364_; lean_object* v_goalStats_365_; lean_object* v___x_367_; uint8_t v_isShared_368_; uint8_t v_isSharedCheck_379_; 
v_total_356_ = lean_ctor_get(v_stats_351_, 0);
v_configParsing_357_ = lean_ctor_get(v_stats_351_, 1);
v_ruleSetConstruction_358_ = lean_ctor_get(v_stats_351_, 2);
v_search_359_ = lean_ctor_get(v_stats_351_, 3);
v_ruleSelection_360_ = lean_ctor_get(v_stats_351_, 4);
v_script_361_ = lean_ctor_get(v_stats_351_, 5);
v_forwardState_362_ = lean_ctor_get(v_stats_351_, 6);
v_scriptGenerated_363_ = lean_ctor_get(v_stats_351_, 7);
v_ruleStats_364_ = lean_ctor_get(v_stats_351_, 8);
v_goalStats_365_ = lean_ctor_get(v_stats_351_, 9);
v_isSharedCheck_379_ = !lean_is_exclusive(v_stats_351_);
if (v_isSharedCheck_379_ == 0)
{
v___x_367_ = v_stats_351_;
v_isShared_368_ = v_isSharedCheck_379_;
goto v_resetjp_366_;
}
else
{
lean_inc(v_goalStats_365_);
lean_inc(v_ruleStats_364_);
lean_inc(v_scriptGenerated_363_);
lean_inc(v_forwardState_362_);
lean_inc(v_script_361_);
lean_inc(v_ruleSelection_360_);
lean_inc(v_search_359_);
lean_inc(v_ruleSetConstruction_358_);
lean_inc(v_configParsing_357_);
lean_inc(v_total_356_);
lean_dec(v_stats_351_);
v___x_367_ = lean_box(0);
v_isShared_368_ = v_isSharedCheck_379_;
goto v_resetjp_366_;
}
v_resetjp_366_:
{
lean_object* v___x_369_; lean_object* v___x_370_; lean_object* v___x_372_; 
v___x_369_ = lean_nat_sub(v___x_349_, v___y_346_);
lean_dec(v___y_346_);
lean_dec(v___x_349_);
v___x_370_ = lean_nat_add(v_ruleSelection_360_, v___x_369_);
lean_dec(v___x_369_);
lean_dec(v_ruleSelection_360_);
if (v_isShared_368_ == 0)
{
lean_ctor_set(v___x_367_, 4, v___x_370_);
v___x_372_ = v___x_367_;
goto v_reusejp_371_;
}
else
{
lean_object* v_reuseFailAlloc_378_; 
v_reuseFailAlloc_378_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_378_, 0, v_total_356_);
lean_ctor_set(v_reuseFailAlloc_378_, 1, v_configParsing_357_);
lean_ctor_set(v_reuseFailAlloc_378_, 2, v_ruleSetConstruction_358_);
lean_ctor_set(v_reuseFailAlloc_378_, 3, v_search_359_);
lean_ctor_set(v_reuseFailAlloc_378_, 4, v___x_370_);
lean_ctor_set(v_reuseFailAlloc_378_, 5, v_script_361_);
lean_ctor_set(v_reuseFailAlloc_378_, 6, v_forwardState_362_);
lean_ctor_set(v_reuseFailAlloc_378_, 7, v_scriptGenerated_363_);
lean_ctor_set(v_reuseFailAlloc_378_, 8, v_ruleStats_364_);
lean_ctor_set(v_reuseFailAlloc_378_, 9, v_goalStats_365_);
v___x_372_ = v_reuseFailAlloc_378_;
goto v_reusejp_371_;
}
v_reusejp_371_:
{
lean_object* v___x_374_; 
if (v_isShared_355_ == 0)
{
lean_ctor_set(v___x_354_, 1, v___x_372_);
v___x_374_ = v___x_354_;
goto v_reusejp_373_;
}
else
{
lean_object* v_reuseFailAlloc_377_; 
v_reuseFailAlloc_377_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_377_, 0, v_rulePatternCache_352_);
lean_ctor_set(v_reuseFailAlloc_377_, 1, v___x_372_);
v___x_374_ = v_reuseFailAlloc_377_;
goto v_reusejp_373_;
}
v_reusejp_373_:
{
lean_object* v___x_375_; lean_object* v___x_376_; 
v___x_375_ = lean_st_ref_set(v_a_339_, v___x_374_);
v___x_376_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_376_, 0, v_a_347_);
return v___x_376_;
}
}
}
}
}
v___jp_387_:
{
lean_object* v___x_388_; lean_object* v___x_389_; lean_object* v___x_390_; uint8_t v___x_391_; 
v___x_388_ = lean_st_ref_get(v_a_337_);
lean_dec(v___x_388_);
v___x_389_ = lean_io_mono_nanos_now();
v___x_390_ = lean_st_ref_get(v_a_337_);
lean_dec(v___x_390_);
lean_inc(v_g_335_);
v___x_391_ = lp_aesop_Aesop_Goal_isRoot(v_g_335_);
if (v___x_391_ == 0)
{
lean_object* v_ruleSet_392_; lean_object* v___f_393_; lean_object* v___f_394_; lean_object* v___f_395_; lean_object* v___x_396_; lean_object* v___x_397_; lean_object* v___f_398_; lean_object* v___x_399_; lean_object* v___x_400_; lean_object* v___x_14993__overap_401_; lean_object* v___x_402_; 
v_ruleSet_392_ = lean_ctor_get(v_a_336_, 0);
lean_inc_ref(v_ruleSet_392_);
lean_inc(v_g_335_);
v___f_393_ = lean_alloc_closure((void*)(lp_aesop_Aesop_selectSafeRules___redArg___lam__1___boxed), 13, 3);
lean_closure_set(v___f_393_, 0, v_g_335_);
lean_closure_set(v___f_393_, 1, v_ruleSet_392_);
lean_closure_set(v___f_393_, 2, v___f_386_);
v___f_394_ = ((lean_object*)(lp_aesop_Aesop_selectSafeRules___redArg___closed__11));
v___f_395_ = ((lean_object*)(lp_aesop_Aesop_selectSafeRules___redArg___closed__20));
v___x_396_ = lean_obj_once(&lp_aesop_Aesop_selectSafeRules___redArg___closed__38, &lp_aesop_Aesop_selectSafeRules___redArg___closed__38_once, _init_lp_aesop_Aesop_selectSafeRules___redArg___closed__38);
v___x_397_ = lp_aesop_Aesop_SearchM_instMonadRef(lean_box(0), v_inst_334_);
v___f_398_ = lean_obj_once(&lp_aesop_Aesop_selectSafeRules___redArg___closed__42, &lp_aesop_Aesop_selectSafeRules___redArg___closed__42_once, _init_lp_aesop_Aesop_selectSafeRules___redArg___closed__42);
lean_inc_ref(v___x_381_);
v___x_399_ = l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(v___f_398_, v___x_381_);
v___x_400_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_400_, 0, v___x_396_);
lean_ctor_set(v___x_400_, 1, v___x_397_);
lean_ctor_set(v___x_400_, 2, v___x_399_);
v___x_14993__overap_401_ = lp_aesop_Aesop_Goal_runMetaMInPostNormState_x27___redArg(v___x_381_, v___f_394_, v___f_395_, v___x_400_, v___f_393_, v_g_335_);
lean_inc(v_a_343_);
lean_inc_ref(v_a_342_);
lean_inc(v_a_341_);
lean_inc_ref(v_a_340_);
lean_inc(v_a_339_);
lean_inc(v_a_338_);
lean_inc(v_a_337_);
lean_inc_ref(v_a_336_);
v___x_402_ = lean_apply_9(v___x_14993__overap_401_, v_a_336_, v_a_337_, v_a_338_, v_a_339_, v_a_340_, v_a_341_, v_a_342_, v_a_343_, lean_box(0));
if (lean_obj_tag(v___x_402_) == 0)
{
lean_object* v_a_403_; 
v_a_403_ = lean_ctor_get(v___x_402_, 0);
lean_inc(v_a_403_);
lean_dec_ref_known(v___x_402_, 1);
v___y_346_ = v___x_389_;
v_a_347_ = v_a_403_;
goto v___jp_345_;
}
else
{
lean_dec(v___x_389_);
return v___x_402_;
}
}
else
{
lean_object* v___x_404_; 
lean_dec_ref(v___x_381_);
lean_dec(v_g_335_);
v___x_404_ = lean_obj_once(&lp_aesop_Aesop_selectSafeRules___redArg___closed__45, &lp_aesop_Aesop_selectSafeRules___redArg___closed__45_once, _init_lp_aesop_Aesop_selectSafeRules___redArg___closed__45);
v___y_346_ = v___x_389_;
v_a_347_ = v___x_404_;
goto v___jp_345_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_selectSafeRules___redArg___boxed(lean_object* v_inst_446_, lean_object* v_g_447_, lean_object* v_a_448_, lean_object* v_a_449_, lean_object* v_a_450_, lean_object* v_a_451_, lean_object* v_a_452_, lean_object* v_a_453_, lean_object* v_a_454_, lean_object* v_a_455_, lean_object* v_a_456_){
_start:
{
lean_object* v_res_457_; 
v_res_457_ = lp_aesop_Aesop_selectSafeRules___redArg(v_inst_446_, v_g_447_, v_a_448_, v_a_449_, v_a_450_, v_a_451_, v_a_452_, v_a_453_, v_a_454_, v_a_455_);
lean_dec(v_a_455_);
lean_dec_ref(v_a_454_);
lean_dec(v_a_453_);
lean_dec_ref(v_a_452_);
lean_dec(v_a_451_);
lean_dec(v_a_450_);
lean_dec(v_a_449_);
lean_dec_ref(v_a_448_);
lean_dec_ref(v_inst_446_);
return v_res_457_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_selectSafeRules(lean_object* v_Q_458_, lean_object* v_inst_459_, lean_object* v_g_460_, lean_object* v_a_461_, lean_object* v_a_462_, lean_object* v_a_463_, lean_object* v_a_464_, lean_object* v_a_465_, lean_object* v_a_466_, lean_object* v_a_467_, lean_object* v_a_468_){
_start:
{
lean_object* v___x_470_; 
v___x_470_ = lp_aesop_Aesop_selectSafeRules___redArg(v_inst_459_, v_g_460_, v_a_461_, v_a_462_, v_a_463_, v_a_464_, v_a_465_, v_a_466_, v_a_467_, v_a_468_);
return v___x_470_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_selectSafeRules___boxed(lean_object* v_Q_471_, lean_object* v_inst_472_, lean_object* v_g_473_, lean_object* v_a_474_, lean_object* v_a_475_, lean_object* v_a_476_, lean_object* v_a_477_, lean_object* v_a_478_, lean_object* v_a_479_, lean_object* v_a_480_, lean_object* v_a_481_, lean_object* v_a_482_){
_start:
{
lean_object* v_res_483_; 
v_res_483_ = lp_aesop_Aesop_selectSafeRules(v_Q_471_, v_inst_472_, v_g_473_, v_a_474_, v_a_475_, v_a_476_, v_a_477_, v_a_478_, v_a_479_, v_a_480_, v_a_481_);
lean_dec(v_a_481_);
lean_dec_ref(v_a_480_);
lean_dec(v_a_479_);
lean_dec_ref(v_a_478_);
lean_dec(v_a_477_);
lean_dec(v_a_476_);
lean_dec(v_a_475_);
lean_dec_ref(v_a_474_);
lean_dec_ref(v_inst_472_);
return v_res_483_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_selectUnsafeRules___redArg___lam__0(lean_object* v_x_484_){
_start:
{
uint8_t v___x_485_; 
v___x_485_ = 1;
return v___x_485_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_selectUnsafeRules___redArg___lam__0___boxed(lean_object* v_x_486_){
_start:
{
uint8_t v_res_487_; lean_object* v_r_488_; 
v_res_487_ = lp_aesop_Aesop_selectUnsafeRules___redArg___lam__0(v_x_486_);
lean_dec_ref(v_x_486_);
v_r_488_ = lean_box(v_res_487_);
return v_r_488_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_selectUnsafeRules___redArg___lam__1(lean_object* v_ruleSet_489_, lean_object* v_forwardRuleMatches_490_, lean_object* v___f_491_, lean_object* v_postNormGoal_492_, lean_object* v___y_493_, lean_object* v___y_494_, lean_object* v___y_495_, lean_object* v___y_496_, lean_object* v___y_497_, lean_object* v___y_498_, lean_object* v___y_499_, lean_object* v___y_500_){
_start:
{
lean_object* v___x_502_; lean_object* v___x_503_; 
v___x_502_ = lean_st_ref_get(v___y_494_);
lean_dec(v___x_502_);
v___x_503_ = lp_aesop_Aesop_LocalRuleSet_applicableUnsafeRulesWith(v_ruleSet_489_, v_forwardRuleMatches_490_, v_postNormGoal_492_, v___f_491_, v___y_496_, v___y_497_, v___y_498_, v___y_499_, v___y_500_);
return v___x_503_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_selectUnsafeRules___redArg___lam__1___boxed(lean_object* v_ruleSet_504_, lean_object* v_forwardRuleMatches_505_, lean_object* v___f_506_, lean_object* v_postNormGoal_507_, lean_object* v___y_508_, lean_object* v___y_509_, lean_object* v___y_510_, lean_object* v___y_511_, lean_object* v___y_512_, lean_object* v___y_513_, lean_object* v___y_514_, lean_object* v___y_515_, lean_object* v___y_516_){
_start:
{
lean_object* v_res_517_; 
v_res_517_ = lp_aesop_Aesop_selectUnsafeRules___redArg___lam__1(v_ruleSet_504_, v_forwardRuleMatches_505_, v___f_506_, v_postNormGoal_507_, v___y_508_, v___y_509_, v___y_510_, v___y_511_, v___y_512_, v___y_513_, v___y_514_, v___y_515_);
lean_dec(v___y_515_);
lean_dec_ref(v___y_514_);
lean_dec(v___y_513_);
lean_dec_ref(v___y_512_);
lean_dec(v___y_511_);
lean_dec(v___y_510_);
lean_dec(v___y_509_);
lean_dec_ref(v___y_508_);
lean_dec_ref(v_forwardRuleMatches_505_);
return v_res_517_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_selectUnsafeRules___redArg(lean_object* v_inst_519_, lean_object* v_postponedSafeRules_520_, lean_object* v_gref_521_, lean_object* v_a_522_, lean_object* v_a_523_, lean_object* v_a_524_, lean_object* v_a_525_, lean_object* v_a_526_, lean_object* v_a_527_, lean_object* v_a_528_, lean_object* v_a_529_){
_start:
{
lean_object* v___y_532_; lean_object* v_a_533_; lean_object* v___x_567_; lean_object* v___x_568_; lean_object* v___f_569_; lean_object* v___x_570_; lean_object* v___x_571_; lean_object* v___f_572_; lean_object* v___x_573_; lean_object* v___x_574_; lean_object* v_toMonadOptions_575_; lean_object* v___x_576_; lean_object* v_options_577_; lean_object* v___f_578_; lean_object* v___f_579_; lean_object* v___x_662_; lean_object* v___x_663_; uint8_t v___x_664_; 
v___x_567_ = lp_aesop_Aesop_SearchM_instMonad(lean_box(0), v_inst_519_);
v___x_568_ = lean_obj_once(&lp_aesop_Aesop_selectSafeRules___redArg___closed__2, &lp_aesop_Aesop_selectSafeRules___redArg___closed__2_once, _init_lp_aesop_Aesop_selectSafeRules___redArg___closed__2);
v___f_569_ = ((lean_object*)(lp_aesop_Aesop_selectSafeRules___redArg___closed__11));
v___x_570_ = lean_obj_once(&lp_aesop_Aesop_selectSafeRules___redArg___closed__38, &lp_aesop_Aesop_selectSafeRules___redArg___closed__38_once, _init_lp_aesop_Aesop_selectSafeRules___redArg___closed__38);
v___x_571_ = lp_aesop_Aesop_SearchM_instMonadRef(lean_box(0), v_inst_519_);
v___f_572_ = lean_obj_once(&lp_aesop_Aesop_selectSafeRules___redArg___closed__42, &lp_aesop_Aesop_selectSafeRules___redArg___closed__42_once, _init_lp_aesop_Aesop_selectSafeRules___redArg___closed__42);
lean_inc_ref(v___x_567_);
v___x_573_ = l_Lean_instAddErrorMessageContextOfAddMessageContextOfMonad___redArg(v___f_572_, v___x_567_);
v___x_574_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_574_, 0, v___x_570_);
lean_ctor_set(v___x_574_, 1, v___x_571_);
lean_ctor_set(v___x_574_, 2, v___x_573_);
v_toMonadOptions_575_ = lean_ctor_get(v___x_568_, 0);
v___x_576_ = l_Lean_KVMap_instValueBool;
v_options_577_ = lean_ctor_get(v_a_528_, 2);
v___f_578_ = ((lean_object*)(lp_aesop_Aesop_selectUnsafeRules___redArg___closed__0));
v___f_579_ = ((lean_object*)(lp_aesop_Aesop_selectSafeRules___redArg___closed__20));
v___x_662_ = lp_aesop_Aesop_aesop_collectStats;
v___x_663_ = l_Lean_Option_get___redArg(v___x_576_, v_options_577_, v___x_662_);
v___x_664_ = lean_unbox(v___x_663_);
lean_dec(v___x_663_);
if (v___x_664_ == 0)
{
lean_object* v___y_754_; lean_object* v___x_765_; lean_object* v___x_19694__overap_766_; lean_object* v___x_767_; 
v___x_765_ = lp_aesop_Aesop_TraceOption_stats;
lean_inc(v_toMonadOptions_575_);
lean_inc_ref(v___x_567_);
v___x_19694__overap_766_ = lp_aesop_Aesop_TraceOption_isEnabled___redArg(v___x_567_, v_toMonadOptions_575_, v___x_765_);
lean_inc(v_a_529_);
lean_inc_ref(v_a_528_);
lean_inc(v_a_527_);
lean_inc_ref(v_a_526_);
lean_inc(v_a_525_);
lean_inc(v_a_524_);
lean_inc(v_a_523_);
lean_inc_ref(v_a_522_);
v___x_767_ = lean_apply_9(v___x_19694__overap_766_, v_a_522_, v_a_523_, v_a_524_, v_a_525_, v_a_526_, v_a_527_, v_a_528_, v_a_529_, lean_box(0));
if (lean_obj_tag(v___x_767_) == 0)
{
lean_object* v_a_768_; uint8_t v___x_769_; 
v_a_768_ = lean_ctor_get(v___x_767_, 0);
lean_inc(v_a_768_);
v___x_769_ = lean_unbox(v_a_768_);
lean_dec(v_a_768_);
if (v___x_769_ == 0)
{
lean_object* v___x_770_; lean_object* v___x_771_; lean_object* v___x_772_; lean_object* v___x_773_; uint8_t v___x_774_; 
lean_dec_ref_known(v___x_767_, 1);
v___x_770_ = l_Lean_KVMap_instValueString;
v___x_771_ = lp_aesop_Aesop_aesop_stats_file;
v___x_772_ = l_Lean_Option_get___redArg(v___x_770_, v_options_577_, v___x_771_);
v___x_773_ = ((lean_object*)(lp_aesop_Aesop_selectNormRules___closed__1));
v___x_774_ = lean_string_dec_eq(v___x_772_, v___x_773_);
lean_dec(v___x_772_);
if (v___x_774_ == 0)
{
goto v___jp_580_;
}
else
{
goto v___jp_665_;
}
}
else
{
v___y_754_ = v___x_767_;
goto v___jp_753_;
}
}
else
{
v___y_754_ = v___x_767_;
goto v___jp_753_;
}
v___jp_665_:
{
lean_object* v___x_666_; lean_object* v___x_667_; lean_object* v___x_668_; lean_object* v_introGoal_669_; lean_object* v_elimGoal_670_; lean_object* v___x_671_; uint8_t v_unsafeRulesSelected_672_; 
v___x_666_ = lean_st_ref_get(v_a_523_);
lean_dec(v___x_666_);
v___x_667_ = lean_st_ref_get(v_gref_521_);
v___x_668_ = lp_aesop_Aesop_treeImpl;
v_introGoal_669_ = lean_ctor_get(v___x_668_, 0);
v_elimGoal_670_ = lean_ctor_get(v___x_668_, 1);
lean_inc_ref(v_elimGoal_670_);
lean_inc(v___x_667_);
v___x_671_ = lean_apply_1(v_elimGoal_670_, v___x_667_);
v_unsafeRulesSelected_672_ = lean_ctor_get_uint8(v___x_671_, sizeof(void*)*14 + 11);
if (v_unsafeRulesSelected_672_ == 0)
{
lean_object* v_id_673_; lean_object* v_parent_674_; lean_object* v_children_675_; lean_object* v_origin_676_; lean_object* v_depth_677_; uint8_t v_state_678_; uint8_t v_isIrrelevant_679_; uint8_t v_isForcedUnprovable_680_; lean_object* v_preNormGoal_681_; lean_object* v_normalizationState_682_; lean_object* v_mvars_683_; lean_object* v_forwardState_684_; lean_object* v_forwardRuleMatches_685_; double v_successProbability_686_; lean_object* v_addedInIteration_687_; lean_object* v_lastExpandedInIteration_688_; lean_object* v_unsafeQueue_689_; lean_object* v_failedRapps_690_; lean_object* v___x_692_; uint8_t v_isShared_693_; uint8_t v_isSharedCheck_750_; 
v_id_673_ = lean_ctor_get(v___x_671_, 0);
v_parent_674_ = lean_ctor_get(v___x_671_, 1);
v_children_675_ = lean_ctor_get(v___x_671_, 2);
v_origin_676_ = lean_ctor_get(v___x_671_, 3);
v_depth_677_ = lean_ctor_get(v___x_671_, 4);
v_state_678_ = lean_ctor_get_uint8(v___x_671_, sizeof(void*)*14 + 8);
v_isIrrelevant_679_ = lean_ctor_get_uint8(v___x_671_, sizeof(void*)*14 + 9);
v_isForcedUnprovable_680_ = lean_ctor_get_uint8(v___x_671_, sizeof(void*)*14 + 10);
v_preNormGoal_681_ = lean_ctor_get(v___x_671_, 5);
v_normalizationState_682_ = lean_ctor_get(v___x_671_, 6);
v_mvars_683_ = lean_ctor_get(v___x_671_, 7);
v_forwardState_684_ = lean_ctor_get(v___x_671_, 8);
v_forwardRuleMatches_685_ = lean_ctor_get(v___x_671_, 9);
v_successProbability_686_ = lean_ctor_get_float(v___x_671_, sizeof(void*)*14);
v_addedInIteration_687_ = lean_ctor_get(v___x_671_, 10);
v_lastExpandedInIteration_688_ = lean_ctor_get(v___x_671_, 11);
v_unsafeQueue_689_ = lean_ctor_get(v___x_671_, 12);
v_failedRapps_690_ = lean_ctor_get(v___x_671_, 13);
v_isSharedCheck_750_ = !lean_is_exclusive(v___x_671_);
if (v_isSharedCheck_750_ == 0)
{
v___x_692_ = v___x_671_;
v_isShared_693_ = v_isSharedCheck_750_;
goto v_resetjp_691_;
}
else
{
lean_inc(v_failedRapps_690_);
lean_inc(v_unsafeQueue_689_);
lean_inc(v_lastExpandedInIteration_688_);
lean_inc(v_addedInIteration_687_);
lean_inc(v_forwardRuleMatches_685_);
lean_inc(v_forwardState_684_);
lean_inc(v_mvars_683_);
lean_inc(v_normalizationState_682_);
lean_inc(v_preNormGoal_681_);
lean_inc(v_depth_677_);
lean_inc(v_origin_676_);
lean_inc(v_children_675_);
lean_inc(v_parent_674_);
lean_inc(v_id_673_);
lean_dec(v___x_671_);
v___x_692_ = lean_box(0);
v_isShared_693_ = v_isSharedCheck_750_;
goto v_resetjp_691_;
}
v_resetjp_691_:
{
lean_object* v_ruleSet_694_; lean_object* v___f_695_; lean_object* v___x_19443__overap_696_; lean_object* v___x_697_; 
v_ruleSet_694_ = lean_ctor_get(v_a_522_, 0);
lean_inc_ref(v_forwardRuleMatches_685_);
lean_inc_ref(v_ruleSet_694_);
v___f_695_ = lean_alloc_closure((void*)(lp_aesop_Aesop_selectUnsafeRules___redArg___lam__1___boxed), 13, 3);
lean_closure_set(v___f_695_, 0, v_ruleSet_694_);
lean_closure_set(v___f_695_, 1, v_forwardRuleMatches_685_);
lean_closure_set(v___f_695_, 2, v___f_578_);
v___x_19443__overap_696_ = lp_aesop_Aesop_Goal_runMetaMInPostNormState_x27___redArg(v___x_567_, v___f_569_, v___f_579_, v___x_574_, v___f_695_, v___x_667_);
lean_inc(v_a_529_);
lean_inc_ref(v_a_528_);
lean_inc(v_a_527_);
lean_inc_ref(v_a_526_);
lean_inc(v_a_525_);
lean_inc(v_a_524_);
lean_inc(v_a_523_);
lean_inc_ref(v_a_522_);
v___x_697_ = lean_apply_9(v___x_19443__overap_696_, v_a_522_, v_a_523_, v_a_524_, v_a_525_, v_a_526_, v_a_527_, v_a_528_, v_a_529_, lean_box(0));
if (lean_obj_tag(v___x_697_) == 0)
{
lean_object* v_a_698_; lean_object* v___x_700_; uint8_t v_isShared_701_; uint8_t v_isSharedCheck_741_; 
v_a_698_ = lean_ctor_get(v___x_697_, 0);
v_isSharedCheck_741_ = !lean_is_exclusive(v___x_697_);
if (v_isSharedCheck_741_ == 0)
{
v___x_700_ = v___x_697_;
v_isShared_701_ = v_isSharedCheck_741_;
goto v_resetjp_699_;
}
else
{
lean_inc(v_a_698_);
lean_dec(v___x_697_);
v___x_700_ = lean_box(0);
v_isShared_701_ = v_isSharedCheck_741_;
goto v_resetjp_699_;
}
v_resetjp_699_:
{
lean_object* v___x_702_; uint8_t v___x_703_; lean_object* v___x_705_; 
v___x_702_ = lean_st_ref_get(v_a_523_);
lean_dec(v___x_702_);
v___x_703_ = 1;
if (v_isShared_693_ == 0)
{
v___x_705_ = v___x_692_;
goto v_reusejp_704_;
}
else
{
lean_object* v_reuseFailAlloc_740_; 
v_reuseFailAlloc_740_ = lean_alloc_ctor(0, 14, 12);
lean_ctor_set(v_reuseFailAlloc_740_, 0, v_id_673_);
lean_ctor_set(v_reuseFailAlloc_740_, 1, v_parent_674_);
lean_ctor_set(v_reuseFailAlloc_740_, 2, v_children_675_);
lean_ctor_set(v_reuseFailAlloc_740_, 3, v_origin_676_);
lean_ctor_set(v_reuseFailAlloc_740_, 4, v_depth_677_);
lean_ctor_set(v_reuseFailAlloc_740_, 5, v_preNormGoal_681_);
lean_ctor_set(v_reuseFailAlloc_740_, 6, v_normalizationState_682_);
lean_ctor_set(v_reuseFailAlloc_740_, 7, v_mvars_683_);
lean_ctor_set(v_reuseFailAlloc_740_, 8, v_forwardState_684_);
lean_ctor_set(v_reuseFailAlloc_740_, 9, v_forwardRuleMatches_685_);
lean_ctor_set(v_reuseFailAlloc_740_, 10, v_addedInIteration_687_);
lean_ctor_set(v_reuseFailAlloc_740_, 11, v_lastExpandedInIteration_688_);
lean_ctor_set(v_reuseFailAlloc_740_, 12, v_unsafeQueue_689_);
lean_ctor_set(v_reuseFailAlloc_740_, 13, v_failedRapps_690_);
lean_ctor_set_uint8(v_reuseFailAlloc_740_, sizeof(void*)*14 + 8, v_state_678_);
lean_ctor_set_uint8(v_reuseFailAlloc_740_, sizeof(void*)*14 + 9, v_isIrrelevant_679_);
lean_ctor_set_uint8(v_reuseFailAlloc_740_, sizeof(void*)*14 + 10, v_isForcedUnprovable_680_);
lean_ctor_set_float(v_reuseFailAlloc_740_, sizeof(void*)*14, v_successProbability_686_);
v___x_705_ = v_reuseFailAlloc_740_;
goto v_reusejp_704_;
}
v_reusejp_704_:
{
lean_object* v___x_706_; lean_object* v___x_707_; lean_object* v_id_708_; lean_object* v_parent_709_; lean_object* v_children_710_; lean_object* v_origin_711_; lean_object* v_depth_712_; uint8_t v_state_713_; uint8_t v_isIrrelevant_714_; uint8_t v_isForcedUnprovable_715_; lean_object* v_preNormGoal_716_; lean_object* v_normalizationState_717_; lean_object* v_mvars_718_; lean_object* v_forwardState_719_; lean_object* v_forwardRuleMatches_720_; double v_successProbability_721_; lean_object* v_addedInIteration_722_; lean_object* v_lastExpandedInIteration_723_; uint8_t v_unsafeRulesSelected_724_; lean_object* v_failedRapps_725_; lean_object* v___x_727_; uint8_t v_isShared_728_; uint8_t v_isSharedCheck_738_; 
lean_ctor_set_uint8(v___x_705_, sizeof(void*)*14 + 11, v___x_703_);
lean_inc(v_introGoal_669_);
v___x_706_ = lean_apply_1(v_introGoal_669_, v___x_705_);
lean_inc_ref(v_elimGoal_670_);
v___x_707_ = lean_apply_1(v_elimGoal_670_, v___x_706_);
v_id_708_ = lean_ctor_get(v___x_707_, 0);
v_parent_709_ = lean_ctor_get(v___x_707_, 1);
v_children_710_ = lean_ctor_get(v___x_707_, 2);
v_origin_711_ = lean_ctor_get(v___x_707_, 3);
v_depth_712_ = lean_ctor_get(v___x_707_, 4);
v_state_713_ = lean_ctor_get_uint8(v___x_707_, sizeof(void*)*14 + 8);
v_isIrrelevant_714_ = lean_ctor_get_uint8(v___x_707_, sizeof(void*)*14 + 9);
v_isForcedUnprovable_715_ = lean_ctor_get_uint8(v___x_707_, sizeof(void*)*14 + 10);
v_preNormGoal_716_ = lean_ctor_get(v___x_707_, 5);
v_normalizationState_717_ = lean_ctor_get(v___x_707_, 6);
v_mvars_718_ = lean_ctor_get(v___x_707_, 7);
v_forwardState_719_ = lean_ctor_get(v___x_707_, 8);
v_forwardRuleMatches_720_ = lean_ctor_get(v___x_707_, 9);
v_successProbability_721_ = lean_ctor_get_float(v___x_707_, sizeof(void*)*14);
v_addedInIteration_722_ = lean_ctor_get(v___x_707_, 10);
v_lastExpandedInIteration_723_ = lean_ctor_get(v___x_707_, 11);
v_unsafeRulesSelected_724_ = lean_ctor_get_uint8(v___x_707_, sizeof(void*)*14 + 11);
v_failedRapps_725_ = lean_ctor_get(v___x_707_, 13);
v_isSharedCheck_738_ = !lean_is_exclusive(v___x_707_);
if (v_isSharedCheck_738_ == 0)
{
lean_object* v_unused_739_; 
v_unused_739_ = lean_ctor_get(v___x_707_, 12);
lean_dec(v_unused_739_);
v___x_727_ = v___x_707_;
v_isShared_728_ = v_isSharedCheck_738_;
goto v_resetjp_726_;
}
else
{
lean_inc(v_failedRapps_725_);
lean_inc(v_lastExpandedInIteration_723_);
lean_inc(v_addedInIteration_722_);
lean_inc(v_forwardRuleMatches_720_);
lean_inc(v_forwardState_719_);
lean_inc(v_mvars_718_);
lean_inc(v_normalizationState_717_);
lean_inc(v_preNormGoal_716_);
lean_inc(v_depth_712_);
lean_inc(v_origin_711_);
lean_inc(v_children_710_);
lean_inc(v_parent_709_);
lean_inc(v_id_708_);
lean_dec(v___x_707_);
v___x_727_ = lean_box(0);
v_isShared_728_ = v_isSharedCheck_738_;
goto v_resetjp_726_;
}
v_resetjp_726_:
{
lean_object* v___x_729_; lean_object* v___x_731_; 
v___x_729_ = lp_aesop_Aesop_UnsafeQueue_initial(v_postponedSafeRules_520_, v_a_698_);
lean_inc_ref(v___x_729_);
if (v_isShared_728_ == 0)
{
lean_ctor_set(v___x_727_, 12, v___x_729_);
v___x_731_ = v___x_727_;
goto v_reusejp_730_;
}
else
{
lean_object* v_reuseFailAlloc_737_; 
v_reuseFailAlloc_737_ = lean_alloc_ctor(0, 14, 12);
lean_ctor_set(v_reuseFailAlloc_737_, 0, v_id_708_);
lean_ctor_set(v_reuseFailAlloc_737_, 1, v_parent_709_);
lean_ctor_set(v_reuseFailAlloc_737_, 2, v_children_710_);
lean_ctor_set(v_reuseFailAlloc_737_, 3, v_origin_711_);
lean_ctor_set(v_reuseFailAlloc_737_, 4, v_depth_712_);
lean_ctor_set(v_reuseFailAlloc_737_, 5, v_preNormGoal_716_);
lean_ctor_set(v_reuseFailAlloc_737_, 6, v_normalizationState_717_);
lean_ctor_set(v_reuseFailAlloc_737_, 7, v_mvars_718_);
lean_ctor_set(v_reuseFailAlloc_737_, 8, v_forwardState_719_);
lean_ctor_set(v_reuseFailAlloc_737_, 9, v_forwardRuleMatches_720_);
lean_ctor_set(v_reuseFailAlloc_737_, 10, v_addedInIteration_722_);
lean_ctor_set(v_reuseFailAlloc_737_, 11, v_lastExpandedInIteration_723_);
lean_ctor_set(v_reuseFailAlloc_737_, 12, v___x_729_);
lean_ctor_set(v_reuseFailAlloc_737_, 13, v_failedRapps_725_);
lean_ctor_set_uint8(v_reuseFailAlloc_737_, sizeof(void*)*14 + 8, v_state_713_);
lean_ctor_set_uint8(v_reuseFailAlloc_737_, sizeof(void*)*14 + 9, v_isIrrelevant_714_);
lean_ctor_set_uint8(v_reuseFailAlloc_737_, sizeof(void*)*14 + 10, v_isForcedUnprovable_715_);
lean_ctor_set_float(v_reuseFailAlloc_737_, sizeof(void*)*14, v_successProbability_721_);
lean_ctor_set_uint8(v_reuseFailAlloc_737_, sizeof(void*)*14 + 11, v_unsafeRulesSelected_724_);
v___x_731_ = v_reuseFailAlloc_737_;
goto v_reusejp_730_;
}
v_reusejp_730_:
{
lean_object* v___x_732_; lean_object* v___x_733_; lean_object* v___x_735_; 
lean_inc(v_introGoal_669_);
v___x_732_ = lean_apply_1(v_introGoal_669_, v___x_731_);
v___x_733_ = lean_st_ref_set(v_gref_521_, v___x_732_);
if (v_isShared_701_ == 0)
{
lean_ctor_set(v___x_700_, 0, v___x_729_);
v___x_735_ = v___x_700_;
goto v_reusejp_734_;
}
else
{
lean_object* v_reuseFailAlloc_736_; 
v_reuseFailAlloc_736_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_736_, 0, v___x_729_);
v___x_735_ = v_reuseFailAlloc_736_;
goto v_reusejp_734_;
}
v_reusejp_734_:
{
return v___x_735_;
}
}
}
}
}
}
else
{
lean_object* v_a_742_; lean_object* v___x_744_; uint8_t v_isShared_745_; uint8_t v_isSharedCheck_749_; 
lean_del_object(v___x_692_);
lean_dec_ref(v_failedRapps_690_);
lean_dec_ref(v_unsafeQueue_689_);
lean_dec(v_lastExpandedInIteration_688_);
lean_dec(v_addedInIteration_687_);
lean_dec_ref(v_forwardRuleMatches_685_);
lean_dec_ref(v_forwardState_684_);
lean_dec_ref(v_mvars_683_);
lean_dec(v_normalizationState_682_);
lean_dec(v_preNormGoal_681_);
lean_dec(v_depth_677_);
lean_dec(v_origin_676_);
lean_dec_ref(v_children_675_);
lean_dec(v_parent_674_);
lean_dec(v_id_673_);
lean_dec_ref(v_postponedSafeRules_520_);
v_a_742_ = lean_ctor_get(v___x_697_, 0);
v_isSharedCheck_749_ = !lean_is_exclusive(v___x_697_);
if (v_isSharedCheck_749_ == 0)
{
v___x_744_ = v___x_697_;
v_isShared_745_ = v_isSharedCheck_749_;
goto v_resetjp_743_;
}
else
{
lean_inc(v_a_742_);
lean_dec(v___x_697_);
v___x_744_ = lean_box(0);
v_isShared_745_ = v_isSharedCheck_749_;
goto v_resetjp_743_;
}
v_resetjp_743_:
{
lean_object* v___x_747_; 
if (v_isShared_745_ == 0)
{
v___x_747_ = v___x_744_;
goto v_reusejp_746_;
}
else
{
lean_object* v_reuseFailAlloc_748_; 
v_reuseFailAlloc_748_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_748_, 0, v_a_742_);
v___x_747_ = v_reuseFailAlloc_748_;
goto v_reusejp_746_;
}
v_reusejp_746_:
{
return v___x_747_;
}
}
}
}
}
else
{
lean_object* v_unsafeQueue_751_; lean_object* v___x_752_; 
lean_dec(v___x_667_);
lean_dec_ref_known(v___x_574_, 3);
lean_dec_ref(v___x_567_);
lean_dec_ref(v_postponedSafeRules_520_);
v_unsafeQueue_751_ = lean_ctor_get(v___x_671_, 12);
lean_inc_ref(v_unsafeQueue_751_);
lean_dec_ref(v___x_671_);
v___x_752_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_752_, 0, v_unsafeQueue_751_);
return v___x_752_;
}
}
v___jp_753_:
{
if (lean_obj_tag(v___y_754_) == 0)
{
lean_object* v_a_755_; uint8_t v___x_756_; 
v_a_755_ = lean_ctor_get(v___y_754_, 0);
lean_inc(v_a_755_);
lean_dec_ref_known(v___y_754_, 1);
v___x_756_ = lean_unbox(v_a_755_);
lean_dec(v_a_755_);
if (v___x_756_ == 0)
{
goto v___jp_665_;
}
else
{
goto v___jp_580_;
}
}
else
{
lean_object* v_a_757_; lean_object* v___x_759_; uint8_t v_isShared_760_; uint8_t v_isSharedCheck_764_; 
lean_dec_ref_known(v___x_574_, 3);
lean_dec_ref(v___x_567_);
lean_dec_ref(v_postponedSafeRules_520_);
v_a_757_ = lean_ctor_get(v___y_754_, 0);
v_isSharedCheck_764_ = !lean_is_exclusive(v___y_754_);
if (v_isSharedCheck_764_ == 0)
{
v___x_759_ = v___y_754_;
v_isShared_760_ = v_isSharedCheck_764_;
goto v_resetjp_758_;
}
else
{
lean_inc(v_a_757_);
lean_dec(v___y_754_);
v___x_759_ = lean_box(0);
v_isShared_760_ = v_isSharedCheck_764_;
goto v_resetjp_758_;
}
v_resetjp_758_:
{
lean_object* v___x_762_; 
if (v_isShared_760_ == 0)
{
v___x_762_ = v___x_759_;
goto v_reusejp_761_;
}
else
{
lean_object* v_reuseFailAlloc_763_; 
v_reuseFailAlloc_763_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_763_, 0, v_a_757_);
v___x_762_ = v_reuseFailAlloc_763_;
goto v_reusejp_761_;
}
v_reusejp_761_:
{
return v___x_762_;
}
}
}
}
}
else
{
goto v___jp_580_;
}
v___jp_531_:
{
lean_object* v___x_534_; lean_object* v___x_535_; lean_object* v___x_536_; lean_object* v_stats_537_; lean_object* v_rulePatternCache_538_; lean_object* v___x_540_; uint8_t v_isShared_541_; uint8_t v_isSharedCheck_566_; 
v___x_534_ = lean_st_ref_get(v_a_523_);
lean_dec(v___x_534_);
v___x_535_ = lean_io_mono_nanos_now();
v___x_536_ = lean_st_ref_take(v_a_525_);
v_stats_537_ = lean_ctor_get(v___x_536_, 1);
v_rulePatternCache_538_ = lean_ctor_get(v___x_536_, 0);
v_isSharedCheck_566_ = !lean_is_exclusive(v___x_536_);
if (v_isSharedCheck_566_ == 0)
{
v___x_540_ = v___x_536_;
v_isShared_541_ = v_isSharedCheck_566_;
goto v_resetjp_539_;
}
else
{
lean_inc(v_stats_537_);
lean_inc(v_rulePatternCache_538_);
lean_dec(v___x_536_);
v___x_540_ = lean_box(0);
v_isShared_541_ = v_isSharedCheck_566_;
goto v_resetjp_539_;
}
v_resetjp_539_:
{
lean_object* v_total_542_; lean_object* v_configParsing_543_; lean_object* v_ruleSetConstruction_544_; lean_object* v_search_545_; lean_object* v_ruleSelection_546_; lean_object* v_script_547_; lean_object* v_forwardState_548_; lean_object* v_scriptGenerated_549_; lean_object* v_ruleStats_550_; lean_object* v_goalStats_551_; lean_object* v___x_553_; uint8_t v_isShared_554_; uint8_t v_isSharedCheck_565_; 
v_total_542_ = lean_ctor_get(v_stats_537_, 0);
v_configParsing_543_ = lean_ctor_get(v_stats_537_, 1);
v_ruleSetConstruction_544_ = lean_ctor_get(v_stats_537_, 2);
v_search_545_ = lean_ctor_get(v_stats_537_, 3);
v_ruleSelection_546_ = lean_ctor_get(v_stats_537_, 4);
v_script_547_ = lean_ctor_get(v_stats_537_, 5);
v_forwardState_548_ = lean_ctor_get(v_stats_537_, 6);
v_scriptGenerated_549_ = lean_ctor_get(v_stats_537_, 7);
v_ruleStats_550_ = lean_ctor_get(v_stats_537_, 8);
v_goalStats_551_ = lean_ctor_get(v_stats_537_, 9);
v_isSharedCheck_565_ = !lean_is_exclusive(v_stats_537_);
if (v_isSharedCheck_565_ == 0)
{
v___x_553_ = v_stats_537_;
v_isShared_554_ = v_isSharedCheck_565_;
goto v_resetjp_552_;
}
else
{
lean_inc(v_goalStats_551_);
lean_inc(v_ruleStats_550_);
lean_inc(v_scriptGenerated_549_);
lean_inc(v_forwardState_548_);
lean_inc(v_script_547_);
lean_inc(v_ruleSelection_546_);
lean_inc(v_search_545_);
lean_inc(v_ruleSetConstruction_544_);
lean_inc(v_configParsing_543_);
lean_inc(v_total_542_);
lean_dec(v_stats_537_);
v___x_553_ = lean_box(0);
v_isShared_554_ = v_isSharedCheck_565_;
goto v_resetjp_552_;
}
v_resetjp_552_:
{
lean_object* v___x_555_; lean_object* v___x_556_; lean_object* v___x_558_; 
v___x_555_ = lean_nat_sub(v___x_535_, v___y_532_);
lean_dec(v___y_532_);
lean_dec(v___x_535_);
v___x_556_ = lean_nat_add(v_ruleSelection_546_, v___x_555_);
lean_dec(v___x_555_);
lean_dec(v_ruleSelection_546_);
if (v_isShared_554_ == 0)
{
lean_ctor_set(v___x_553_, 4, v___x_556_);
v___x_558_ = v___x_553_;
goto v_reusejp_557_;
}
else
{
lean_object* v_reuseFailAlloc_564_; 
v_reuseFailAlloc_564_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_564_, 0, v_total_542_);
lean_ctor_set(v_reuseFailAlloc_564_, 1, v_configParsing_543_);
lean_ctor_set(v_reuseFailAlloc_564_, 2, v_ruleSetConstruction_544_);
lean_ctor_set(v_reuseFailAlloc_564_, 3, v_search_545_);
lean_ctor_set(v_reuseFailAlloc_564_, 4, v___x_556_);
lean_ctor_set(v_reuseFailAlloc_564_, 5, v_script_547_);
lean_ctor_set(v_reuseFailAlloc_564_, 6, v_forwardState_548_);
lean_ctor_set(v_reuseFailAlloc_564_, 7, v_scriptGenerated_549_);
lean_ctor_set(v_reuseFailAlloc_564_, 8, v_ruleStats_550_);
lean_ctor_set(v_reuseFailAlloc_564_, 9, v_goalStats_551_);
v___x_558_ = v_reuseFailAlloc_564_;
goto v_reusejp_557_;
}
v_reusejp_557_:
{
lean_object* v___x_560_; 
if (v_isShared_541_ == 0)
{
lean_ctor_set(v___x_540_, 1, v___x_558_);
v___x_560_ = v___x_540_;
goto v_reusejp_559_;
}
else
{
lean_object* v_reuseFailAlloc_563_; 
v_reuseFailAlloc_563_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_563_, 0, v_rulePatternCache_538_);
lean_ctor_set(v_reuseFailAlloc_563_, 1, v___x_558_);
v___x_560_ = v_reuseFailAlloc_563_;
goto v_reusejp_559_;
}
v_reusejp_559_:
{
lean_object* v___x_561_; lean_object* v___x_562_; 
v___x_561_ = lean_st_ref_set(v_a_525_, v___x_560_);
v___x_562_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_562_, 0, v_a_533_);
return v___x_562_;
}
}
}
}
}
v___jp_580_:
{
lean_object* v___x_581_; lean_object* v___x_582_; lean_object* v___x_583_; lean_object* v___x_584_; lean_object* v___x_585_; lean_object* v_introGoal_586_; lean_object* v_elimGoal_587_; lean_object* v___x_588_; uint8_t v_unsafeRulesSelected_589_; 
v___x_581_ = lean_st_ref_get(v_a_523_);
lean_dec(v___x_581_);
v___x_582_ = lean_io_mono_nanos_now();
v___x_583_ = lean_st_ref_get(v_a_523_);
lean_dec(v___x_583_);
v___x_584_ = lean_st_ref_get(v_gref_521_);
v___x_585_ = lp_aesop_Aesop_treeImpl;
v_introGoal_586_ = lean_ctor_get(v___x_585_, 0);
v_elimGoal_587_ = lean_ctor_get(v___x_585_, 1);
lean_inc_ref(v_elimGoal_587_);
lean_inc(v___x_584_);
v___x_588_ = lean_apply_1(v_elimGoal_587_, v___x_584_);
v_unsafeRulesSelected_589_ = lean_ctor_get_uint8(v___x_588_, sizeof(void*)*14 + 11);
if (v_unsafeRulesSelected_589_ == 0)
{
lean_object* v_id_590_; lean_object* v_parent_591_; lean_object* v_children_592_; lean_object* v_origin_593_; lean_object* v_depth_594_; uint8_t v_state_595_; uint8_t v_isIrrelevant_596_; uint8_t v_isForcedUnprovable_597_; lean_object* v_preNormGoal_598_; lean_object* v_normalizationState_599_; lean_object* v_mvars_600_; lean_object* v_forwardState_601_; lean_object* v_forwardRuleMatches_602_; double v_successProbability_603_; lean_object* v_addedInIteration_604_; lean_object* v_lastExpandedInIteration_605_; lean_object* v_unsafeQueue_606_; lean_object* v_failedRapps_607_; lean_object* v___x_609_; uint8_t v_isShared_610_; uint8_t v_isSharedCheck_660_; 
v_id_590_ = lean_ctor_get(v___x_588_, 0);
v_parent_591_ = lean_ctor_get(v___x_588_, 1);
v_children_592_ = lean_ctor_get(v___x_588_, 2);
v_origin_593_ = lean_ctor_get(v___x_588_, 3);
v_depth_594_ = lean_ctor_get(v___x_588_, 4);
v_state_595_ = lean_ctor_get_uint8(v___x_588_, sizeof(void*)*14 + 8);
v_isIrrelevant_596_ = lean_ctor_get_uint8(v___x_588_, sizeof(void*)*14 + 9);
v_isForcedUnprovable_597_ = lean_ctor_get_uint8(v___x_588_, sizeof(void*)*14 + 10);
v_preNormGoal_598_ = lean_ctor_get(v___x_588_, 5);
v_normalizationState_599_ = lean_ctor_get(v___x_588_, 6);
v_mvars_600_ = lean_ctor_get(v___x_588_, 7);
v_forwardState_601_ = lean_ctor_get(v___x_588_, 8);
v_forwardRuleMatches_602_ = lean_ctor_get(v___x_588_, 9);
v_successProbability_603_ = lean_ctor_get_float(v___x_588_, sizeof(void*)*14);
v_addedInIteration_604_ = lean_ctor_get(v___x_588_, 10);
v_lastExpandedInIteration_605_ = lean_ctor_get(v___x_588_, 11);
v_unsafeQueue_606_ = lean_ctor_get(v___x_588_, 12);
v_failedRapps_607_ = lean_ctor_get(v___x_588_, 13);
v_isSharedCheck_660_ = !lean_is_exclusive(v___x_588_);
if (v_isSharedCheck_660_ == 0)
{
v___x_609_ = v___x_588_;
v_isShared_610_ = v_isSharedCheck_660_;
goto v_resetjp_608_;
}
else
{
lean_inc(v_failedRapps_607_);
lean_inc(v_unsafeQueue_606_);
lean_inc(v_lastExpandedInIteration_605_);
lean_inc(v_addedInIteration_604_);
lean_inc(v_forwardRuleMatches_602_);
lean_inc(v_forwardState_601_);
lean_inc(v_mvars_600_);
lean_inc(v_normalizationState_599_);
lean_inc(v_preNormGoal_598_);
lean_inc(v_depth_594_);
lean_inc(v_origin_593_);
lean_inc(v_children_592_);
lean_inc(v_parent_591_);
lean_inc(v_id_590_);
lean_dec(v___x_588_);
v___x_609_ = lean_box(0);
v_isShared_610_ = v_isSharedCheck_660_;
goto v_resetjp_608_;
}
v_resetjp_608_:
{
lean_object* v_ruleSet_611_; lean_object* v___f_612_; lean_object* v___x_19627__overap_613_; lean_object* v___x_614_; 
v_ruleSet_611_ = lean_ctor_get(v_a_522_, 0);
lean_inc_ref(v_forwardRuleMatches_602_);
lean_inc_ref(v_ruleSet_611_);
v___f_612_ = lean_alloc_closure((void*)(lp_aesop_Aesop_selectUnsafeRules___redArg___lam__1___boxed), 13, 3);
lean_closure_set(v___f_612_, 0, v_ruleSet_611_);
lean_closure_set(v___f_612_, 1, v_forwardRuleMatches_602_);
lean_closure_set(v___f_612_, 2, v___f_578_);
v___x_19627__overap_613_ = lp_aesop_Aesop_Goal_runMetaMInPostNormState_x27___redArg(v___x_567_, v___f_569_, v___f_579_, v___x_574_, v___f_612_, v___x_584_);
lean_inc(v_a_529_);
lean_inc_ref(v_a_528_);
lean_inc(v_a_527_);
lean_inc_ref(v_a_526_);
lean_inc(v_a_525_);
lean_inc(v_a_524_);
lean_inc(v_a_523_);
lean_inc_ref(v_a_522_);
v___x_614_ = lean_apply_9(v___x_19627__overap_613_, v_a_522_, v_a_523_, v_a_524_, v_a_525_, v_a_526_, v_a_527_, v_a_528_, v_a_529_, lean_box(0));
if (lean_obj_tag(v___x_614_) == 0)
{
lean_object* v_a_615_; lean_object* v___x_616_; uint8_t v___x_617_; lean_object* v___x_619_; 
v_a_615_ = lean_ctor_get(v___x_614_, 0);
lean_inc(v_a_615_);
lean_dec_ref_known(v___x_614_, 1);
v___x_616_ = lean_st_ref_get(v_a_523_);
lean_dec(v___x_616_);
v___x_617_ = 1;
if (v_isShared_610_ == 0)
{
v___x_619_ = v___x_609_;
goto v_reusejp_618_;
}
else
{
lean_object* v_reuseFailAlloc_651_; 
v_reuseFailAlloc_651_ = lean_alloc_ctor(0, 14, 12);
lean_ctor_set(v_reuseFailAlloc_651_, 0, v_id_590_);
lean_ctor_set(v_reuseFailAlloc_651_, 1, v_parent_591_);
lean_ctor_set(v_reuseFailAlloc_651_, 2, v_children_592_);
lean_ctor_set(v_reuseFailAlloc_651_, 3, v_origin_593_);
lean_ctor_set(v_reuseFailAlloc_651_, 4, v_depth_594_);
lean_ctor_set(v_reuseFailAlloc_651_, 5, v_preNormGoal_598_);
lean_ctor_set(v_reuseFailAlloc_651_, 6, v_normalizationState_599_);
lean_ctor_set(v_reuseFailAlloc_651_, 7, v_mvars_600_);
lean_ctor_set(v_reuseFailAlloc_651_, 8, v_forwardState_601_);
lean_ctor_set(v_reuseFailAlloc_651_, 9, v_forwardRuleMatches_602_);
lean_ctor_set(v_reuseFailAlloc_651_, 10, v_addedInIteration_604_);
lean_ctor_set(v_reuseFailAlloc_651_, 11, v_lastExpandedInIteration_605_);
lean_ctor_set(v_reuseFailAlloc_651_, 12, v_unsafeQueue_606_);
lean_ctor_set(v_reuseFailAlloc_651_, 13, v_failedRapps_607_);
lean_ctor_set_uint8(v_reuseFailAlloc_651_, sizeof(void*)*14 + 8, v_state_595_);
lean_ctor_set_uint8(v_reuseFailAlloc_651_, sizeof(void*)*14 + 9, v_isIrrelevant_596_);
lean_ctor_set_uint8(v_reuseFailAlloc_651_, sizeof(void*)*14 + 10, v_isForcedUnprovable_597_);
lean_ctor_set_float(v_reuseFailAlloc_651_, sizeof(void*)*14, v_successProbability_603_);
v___x_619_ = v_reuseFailAlloc_651_;
goto v_reusejp_618_;
}
v_reusejp_618_:
{
lean_object* v___x_620_; lean_object* v___x_621_; lean_object* v_id_622_; lean_object* v_parent_623_; lean_object* v_children_624_; lean_object* v_origin_625_; lean_object* v_depth_626_; uint8_t v_state_627_; uint8_t v_isIrrelevant_628_; uint8_t v_isForcedUnprovable_629_; lean_object* v_preNormGoal_630_; lean_object* v_normalizationState_631_; lean_object* v_mvars_632_; lean_object* v_forwardState_633_; lean_object* v_forwardRuleMatches_634_; double v_successProbability_635_; lean_object* v_addedInIteration_636_; lean_object* v_lastExpandedInIteration_637_; uint8_t v_unsafeRulesSelected_638_; lean_object* v_failedRapps_639_; lean_object* v___x_641_; uint8_t v_isShared_642_; uint8_t v_isSharedCheck_649_; 
lean_ctor_set_uint8(v___x_619_, sizeof(void*)*14 + 11, v___x_617_);
lean_inc(v_introGoal_586_);
v___x_620_ = lean_apply_1(v_introGoal_586_, v___x_619_);
lean_inc_ref(v_elimGoal_587_);
v___x_621_ = lean_apply_1(v_elimGoal_587_, v___x_620_);
v_id_622_ = lean_ctor_get(v___x_621_, 0);
v_parent_623_ = lean_ctor_get(v___x_621_, 1);
v_children_624_ = lean_ctor_get(v___x_621_, 2);
v_origin_625_ = lean_ctor_get(v___x_621_, 3);
v_depth_626_ = lean_ctor_get(v___x_621_, 4);
v_state_627_ = lean_ctor_get_uint8(v___x_621_, sizeof(void*)*14 + 8);
v_isIrrelevant_628_ = lean_ctor_get_uint8(v___x_621_, sizeof(void*)*14 + 9);
v_isForcedUnprovable_629_ = lean_ctor_get_uint8(v___x_621_, sizeof(void*)*14 + 10);
v_preNormGoal_630_ = lean_ctor_get(v___x_621_, 5);
v_normalizationState_631_ = lean_ctor_get(v___x_621_, 6);
v_mvars_632_ = lean_ctor_get(v___x_621_, 7);
v_forwardState_633_ = lean_ctor_get(v___x_621_, 8);
v_forwardRuleMatches_634_ = lean_ctor_get(v___x_621_, 9);
v_successProbability_635_ = lean_ctor_get_float(v___x_621_, sizeof(void*)*14);
v_addedInIteration_636_ = lean_ctor_get(v___x_621_, 10);
v_lastExpandedInIteration_637_ = lean_ctor_get(v___x_621_, 11);
v_unsafeRulesSelected_638_ = lean_ctor_get_uint8(v___x_621_, sizeof(void*)*14 + 11);
v_failedRapps_639_ = lean_ctor_get(v___x_621_, 13);
v_isSharedCheck_649_ = !lean_is_exclusive(v___x_621_);
if (v_isSharedCheck_649_ == 0)
{
lean_object* v_unused_650_; 
v_unused_650_ = lean_ctor_get(v___x_621_, 12);
lean_dec(v_unused_650_);
v___x_641_ = v___x_621_;
v_isShared_642_ = v_isSharedCheck_649_;
goto v_resetjp_640_;
}
else
{
lean_inc(v_failedRapps_639_);
lean_inc(v_lastExpandedInIteration_637_);
lean_inc(v_addedInIteration_636_);
lean_inc(v_forwardRuleMatches_634_);
lean_inc(v_forwardState_633_);
lean_inc(v_mvars_632_);
lean_inc(v_normalizationState_631_);
lean_inc(v_preNormGoal_630_);
lean_inc(v_depth_626_);
lean_inc(v_origin_625_);
lean_inc(v_children_624_);
lean_inc(v_parent_623_);
lean_inc(v_id_622_);
lean_dec(v___x_621_);
v___x_641_ = lean_box(0);
v_isShared_642_ = v_isSharedCheck_649_;
goto v_resetjp_640_;
}
v_resetjp_640_:
{
lean_object* v___x_643_; lean_object* v___x_645_; 
v___x_643_ = lp_aesop_Aesop_UnsafeQueue_initial(v_postponedSafeRules_520_, v_a_615_);
lean_inc_ref(v___x_643_);
if (v_isShared_642_ == 0)
{
lean_ctor_set(v___x_641_, 12, v___x_643_);
v___x_645_ = v___x_641_;
goto v_reusejp_644_;
}
else
{
lean_object* v_reuseFailAlloc_648_; 
v_reuseFailAlloc_648_ = lean_alloc_ctor(0, 14, 12);
lean_ctor_set(v_reuseFailAlloc_648_, 0, v_id_622_);
lean_ctor_set(v_reuseFailAlloc_648_, 1, v_parent_623_);
lean_ctor_set(v_reuseFailAlloc_648_, 2, v_children_624_);
lean_ctor_set(v_reuseFailAlloc_648_, 3, v_origin_625_);
lean_ctor_set(v_reuseFailAlloc_648_, 4, v_depth_626_);
lean_ctor_set(v_reuseFailAlloc_648_, 5, v_preNormGoal_630_);
lean_ctor_set(v_reuseFailAlloc_648_, 6, v_normalizationState_631_);
lean_ctor_set(v_reuseFailAlloc_648_, 7, v_mvars_632_);
lean_ctor_set(v_reuseFailAlloc_648_, 8, v_forwardState_633_);
lean_ctor_set(v_reuseFailAlloc_648_, 9, v_forwardRuleMatches_634_);
lean_ctor_set(v_reuseFailAlloc_648_, 10, v_addedInIteration_636_);
lean_ctor_set(v_reuseFailAlloc_648_, 11, v_lastExpandedInIteration_637_);
lean_ctor_set(v_reuseFailAlloc_648_, 12, v___x_643_);
lean_ctor_set(v_reuseFailAlloc_648_, 13, v_failedRapps_639_);
lean_ctor_set_uint8(v_reuseFailAlloc_648_, sizeof(void*)*14 + 8, v_state_627_);
lean_ctor_set_uint8(v_reuseFailAlloc_648_, sizeof(void*)*14 + 9, v_isIrrelevant_628_);
lean_ctor_set_uint8(v_reuseFailAlloc_648_, sizeof(void*)*14 + 10, v_isForcedUnprovable_629_);
lean_ctor_set_float(v_reuseFailAlloc_648_, sizeof(void*)*14, v_successProbability_635_);
lean_ctor_set_uint8(v_reuseFailAlloc_648_, sizeof(void*)*14 + 11, v_unsafeRulesSelected_638_);
v___x_645_ = v_reuseFailAlloc_648_;
goto v_reusejp_644_;
}
v_reusejp_644_:
{
lean_object* v___x_646_; lean_object* v___x_647_; 
lean_inc(v_introGoal_586_);
v___x_646_ = lean_apply_1(v_introGoal_586_, v___x_645_);
v___x_647_ = lean_st_ref_set(v_gref_521_, v___x_646_);
v___y_532_ = v___x_582_;
v_a_533_ = v___x_643_;
goto v___jp_531_;
}
}
}
}
else
{
lean_object* v_a_652_; lean_object* v___x_654_; uint8_t v_isShared_655_; uint8_t v_isSharedCheck_659_; 
lean_del_object(v___x_609_);
lean_dec_ref(v_failedRapps_607_);
lean_dec_ref(v_unsafeQueue_606_);
lean_dec(v_lastExpandedInIteration_605_);
lean_dec(v_addedInIteration_604_);
lean_dec_ref(v_forwardRuleMatches_602_);
lean_dec_ref(v_forwardState_601_);
lean_dec_ref(v_mvars_600_);
lean_dec(v_normalizationState_599_);
lean_dec(v_preNormGoal_598_);
lean_dec(v_depth_594_);
lean_dec(v_origin_593_);
lean_dec_ref(v_children_592_);
lean_dec(v_parent_591_);
lean_dec(v_id_590_);
lean_dec(v___x_582_);
lean_dec_ref(v_postponedSafeRules_520_);
v_a_652_ = lean_ctor_get(v___x_614_, 0);
v_isSharedCheck_659_ = !lean_is_exclusive(v___x_614_);
if (v_isSharedCheck_659_ == 0)
{
v___x_654_ = v___x_614_;
v_isShared_655_ = v_isSharedCheck_659_;
goto v_resetjp_653_;
}
else
{
lean_inc(v_a_652_);
lean_dec(v___x_614_);
v___x_654_ = lean_box(0);
v_isShared_655_ = v_isSharedCheck_659_;
goto v_resetjp_653_;
}
v_resetjp_653_:
{
lean_object* v___x_657_; 
if (v_isShared_655_ == 0)
{
v___x_657_ = v___x_654_;
goto v_reusejp_656_;
}
else
{
lean_object* v_reuseFailAlloc_658_; 
v_reuseFailAlloc_658_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_658_, 0, v_a_652_);
v___x_657_ = v_reuseFailAlloc_658_;
goto v_reusejp_656_;
}
v_reusejp_656_:
{
return v___x_657_;
}
}
}
}
}
else
{
lean_object* v_unsafeQueue_661_; 
lean_dec(v___x_584_);
lean_dec_ref_known(v___x_574_, 3);
lean_dec_ref(v___x_567_);
lean_dec_ref(v_postponedSafeRules_520_);
v_unsafeQueue_661_ = lean_ctor_get(v___x_588_, 12);
lean_inc_ref(v_unsafeQueue_661_);
lean_dec_ref(v___x_588_);
v___y_532_ = v___x_582_;
v_a_533_ = v_unsafeQueue_661_;
goto v___jp_531_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_selectUnsafeRules___redArg___boxed(lean_object* v_inst_775_, lean_object* v_postponedSafeRules_776_, lean_object* v_gref_777_, lean_object* v_a_778_, lean_object* v_a_779_, lean_object* v_a_780_, lean_object* v_a_781_, lean_object* v_a_782_, lean_object* v_a_783_, lean_object* v_a_784_, lean_object* v_a_785_, lean_object* v_a_786_){
_start:
{
lean_object* v_res_787_; 
v_res_787_ = lp_aesop_Aesop_selectUnsafeRules___redArg(v_inst_775_, v_postponedSafeRules_776_, v_gref_777_, v_a_778_, v_a_779_, v_a_780_, v_a_781_, v_a_782_, v_a_783_, v_a_784_, v_a_785_);
lean_dec(v_a_785_);
lean_dec_ref(v_a_784_);
lean_dec(v_a_783_);
lean_dec_ref(v_a_782_);
lean_dec(v_a_781_);
lean_dec(v_a_780_);
lean_dec(v_a_779_);
lean_dec_ref(v_a_778_);
lean_dec(v_gref_777_);
lean_dec_ref(v_inst_775_);
return v_res_787_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_selectUnsafeRules(lean_object* v_Q_788_, lean_object* v_inst_789_, lean_object* v_postponedSafeRules_790_, lean_object* v_gref_791_, lean_object* v_a_792_, lean_object* v_a_793_, lean_object* v_a_794_, lean_object* v_a_795_, lean_object* v_a_796_, lean_object* v_a_797_, lean_object* v_a_798_, lean_object* v_a_799_){
_start:
{
lean_object* v___x_801_; 
v___x_801_ = lp_aesop_Aesop_selectUnsafeRules___redArg(v_inst_789_, v_postponedSafeRules_790_, v_gref_791_, v_a_792_, v_a_793_, v_a_794_, v_a_795_, v_a_796_, v_a_797_, v_a_798_, v_a_799_);
return v___x_801_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_selectUnsafeRules___boxed(lean_object* v_Q_802_, lean_object* v_inst_803_, lean_object* v_postponedSafeRules_804_, lean_object* v_gref_805_, lean_object* v_a_806_, lean_object* v_a_807_, lean_object* v_a_808_, lean_object* v_a_809_, lean_object* v_a_810_, lean_object* v_a_811_, lean_object* v_a_812_, lean_object* v_a_813_, lean_object* v_a_814_){
_start:
{
lean_object* v_res_815_; 
v_res_815_ = lp_aesop_Aesop_selectUnsafeRules(v_Q_802_, v_inst_803_, v_postponedSafeRules_804_, v_gref_805_, v_a_806_, v_a_807_, v_a_808_, v_a_809_, v_a_810_, v_a_811_, v_a_812_, v_a_813_);
lean_dec(v_a_813_);
lean_dec_ref(v_a_812_);
lean_dec(v_a_811_);
lean_dec_ref(v_a_810_);
lean_dec(v_a_809_);
lean_dec(v_a_808_);
lean_dec(v_a_807_);
lean_dec_ref(v_a_806_);
lean_dec(v_gref_805_);
lean_dec_ref(v_inst_803_);
return v_res_815_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Tree_RunMetaM(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Search_SearchM(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Search_RuleSelection(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Tree_RunMetaM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Search_SearchM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_aesop_Aesop_preprocessRule = _init_lp_aesop_Aesop_preprocessRule();
lean_mark_persistent(lp_aesop_Aesop_preprocessRule);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Search_RuleSelection(uint8_t builtin) {
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
lean_object* initialize_aesop_Aesop_Tree_RunMetaM(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Search_SearchM(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Search_RuleSelection(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Tree_RunMetaM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Search_SearchM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Search_RuleSelection(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Search_RuleSelection(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Search_RuleSelection(builtin);
}
#ifdef __cplusplus
}
#endif
