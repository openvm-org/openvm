// Lean compiler output
// Module: Aesop.BaseM
// Imports: public import Init public meta import Init public import Aesop.Stats.Basic public import Aesop.RulePattern.Cache
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
lean_object* l_ReaderT_instMonadLift___lam__0___boxed(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Core_instMonadOptionsCoreM___lam__0___boxed(lean_object*, lean_object*, lean_object*);
lean_object* l_StateRefT_x27_lift___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_saveState___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Core_liftIOCore___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instMonadLiftBaseIOEIO___lam__0___boxed(lean_object*, lean_object*, lean_object*);
lean_object* l_IO_instMonadLiftSTRealWorldBaseIO___lam__0___boxed(lean_object*, lean_object*, lean_object*);
lean_object* l_instMonadLiftT___lam__0___boxed(lean_object*, lean_object*);
lean_object* l_instMonadLiftTOfMonadLift___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_SavedState_restore___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* lean_st_mk_ref(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
extern lean_object* lp_aesop_Aesop_Stats_empty;
lean_object* l_Lean_Meta_instMonadMetaM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_instMonadMetaM___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Core_instMonadCoreM___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_aesop_Aesop_instInhabitedStats_default;
extern lean_object* lp_aesop_Aesop_instInhabitedRulePatternCache_default;
lean_object* l_instMonadEIO(lean_object*);
lean_object* l_StateRefT_x27_instMonad___redArg(lean_object*);
lean_object* l_Lean_Core_instMonadCoreM___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instFunctorOfMonad___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instFunctorOfMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_instApplicativeOfMonad___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_StateRefT_x27_get___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ReaderT_bind___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Aesop_BaseM_instInhabitedState_default___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_BaseM_instInhabitedState_default___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseM_instInhabitedState_default;
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseM_instInhabitedState;
static lean_once_cell_t lp_aesop_Aesop_instEmptyCollectionState___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instEmptyCollectionState___closed__0;
static lean_once_cell_t lp_aesop_Aesop_instEmptyCollectionState___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instEmptyCollectionState___closed__1;
static lean_once_cell_t lp_aesop_Aesop_instEmptyCollectionState___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instEmptyCollectionState___closed__2;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instEmptyCollectionState;
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseM_run___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseM_run___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseM_run(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseM_run___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__0;
static lean_once_cell_t lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__1;
static const lean_closure_object lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadCoreM___lam__0___boxed, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__2 = (const lean_object*)&lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__2_value;
static const lean_closure_object lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadCoreM___lam__1___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__3 = (const lean_object*)&lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__3_value;
static const lean_closure_object lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__4 = (const lean_object*)&lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__4_value;
static const lean_closure_object lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___lam__1___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__5 = (const lean_object*)&lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__5_value;
static const lean_closure_object lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instMonadMetaM___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__6 = (const lean_object*)&lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__6_value;
static const lean_closure_object lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instMonadMetaM___lam__1___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__7 = (const lean_object*)&lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__7_value;
static const lean_closure_object lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_ReaderT_instMonadLift___lam__0___boxed, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__8 = (const lean_object*)&lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__8_value;
static const lean_closure_object lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*3, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_StateRefT_x27_lift___boxed, .m_arity = 6, .m_num_fixed = 3, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__9 = (const lean_object*)&lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__9_value;
static const lean_closure_object lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_liftIOCore___boxed, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__10 = (const lean_object*)&lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__10_value;
static const lean_closure_object lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instMonadLiftBaseIOEIO___lam__0___boxed, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__11 = (const lean_object*)&lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__11_value;
static const lean_closure_object lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_IO_instMonadLiftSTRealWorldBaseIO___lam__0___boxed, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__12 = (const lean_object*)&lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__12_value;
static const lean_closure_object lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instMonadLiftT___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__13 = (const lean_object*)&lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__13_value;
static const lean_closure_object lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instMonadLiftTOfMonadLift___redArg___lam__0, .m_arity = 4, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__13_value),((lean_object*)&lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__12_value)} };
static const lean_object* lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__14 = (const lean_object*)&lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__14_value;
static const lean_closure_object lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instMonadLiftTOfMonadLift___redArg___lam__0, .m_arity = 4, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__14_value),((lean_object*)&lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__11_value)} };
static const lean_object* lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__15 = (const lean_object*)&lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__15_value;
static const lean_closure_object lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instMonadLiftTOfMonadLift___redArg___lam__0, .m_arity = 4, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__15_value),((lean_object*)&lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__10_value)} };
static const lean_object* lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__16 = (const lean_object*)&lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__16_value;
static const lean_closure_object lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instMonadLiftTOfMonadLift___redArg___lam__0, .m_arity = 4, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__16_value),((lean_object*)&lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__9_value)} };
static const lean_object* lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__17 = (const lean_object*)&lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__17_value;
static const lean_closure_object lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instMonadLiftTOfMonadLift___redArg___lam__0, .m_arity = 4, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__17_value),((lean_object*)&lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__8_value)} };
static const lean_object* lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__18 = (const lean_object*)&lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__18_value;
static const lean_closure_object lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*4, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_StateRefT_x27_get___boxed, .m_arity = 5, .m_num_fixed = 4, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__18_value)} };
static const lean_object* lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__19 = (const lean_object*)&lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__19_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry;
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseM_instMonadStats___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseM_instMonadStats___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseM_instMonadStats___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseM_instMonadStats___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseM_instMonadStats___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseM_instMonadStats___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_BaseM_instMonadStats___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_BaseM_instMonadStats___lam__0___boxed, .m_arity = 8, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_BaseM_instMonadStats___closed__0 = (const lean_object*)&lp_aesop_Aesop_BaseM_instMonadStats___closed__0_value;
static const lean_closure_object lp_aesop_Aesop_BaseM_instMonadStats___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_BaseM_instMonadStats___lam__1___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_BaseM_instMonadStats___closed__1 = (const lean_object*)&lp_aesop_Aesop_BaseM_instMonadStats___closed__1_value;
static const lean_closure_object lp_aesop_Aesop_BaseM_instMonadStats___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_BaseM_instMonadStats___lam__2___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_BaseM_instMonadStats___closed__2 = (const lean_object*)&lp_aesop_Aesop_BaseM_instMonadStats___closed__2_value;
static const lean_closure_object lp_aesop_Aesop_BaseM_instMonadStats___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Core_instMonadOptionsCoreM___lam__0___boxed, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_BaseM_instMonadStats___closed__3 = (const lean_object*)&lp_aesop_Aesop_BaseM_instMonadStats___closed__3_value;
static const lean_closure_object lp_aesop_Aesop_BaseM_instMonadStats___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*5, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_StateRefT_x27_lift___boxed, .m_arity = 6, .m_num_fixed = 5, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_BaseM_instMonadStats___closed__3_value)} };
static const lean_object* lp_aesop_Aesop_BaseM_instMonadStats___closed__4 = (const lean_object*)&lp_aesop_Aesop_BaseM_instMonadStats___closed__4_value;
static const lean_closure_object lp_aesop_Aesop_BaseM_instMonadStats___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_ReaderT_instMonadLift___lam__0___boxed, .m_arity = 3, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_BaseM_instMonadStats___closed__4_value)} };
static const lean_object* lp_aesop_Aesop_BaseM_instMonadStats___closed__5 = (const lean_object*)&lp_aesop_Aesop_BaseM_instMonadStats___closed__5_value;
static const lean_closure_object lp_aesop_Aesop_BaseM_instMonadStats___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*5, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_StateRefT_x27_lift___boxed, .m_arity = 6, .m_num_fixed = 5, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_BaseM_instMonadStats___closed__5_value)} };
static const lean_object* lp_aesop_Aesop_BaseM_instMonadStats___closed__6 = (const lean_object*)&lp_aesop_Aesop_BaseM_instMonadStats___closed__6_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseM_instMonadStats;
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseM_instMonadBacktrackSavedState___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseM_instMonadBacktrackSavedState___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_BaseM_instMonadBacktrackSavedState___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_BaseM_instMonadBacktrackSavedState___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_BaseM_instMonadBacktrackSavedState___closed__0 = (const lean_object*)&lp_aesop_Aesop_BaseM_instMonadBacktrackSavedState___closed__0_value;
static const lean_closure_object lp_aesop_Aesop_BaseM_instMonadBacktrackSavedState___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_saveState___boxed, .m_arity = 5, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_BaseM_instMonadBacktrackSavedState___closed__1 = (const lean_object*)&lp_aesop_Aesop_BaseM_instMonadBacktrackSavedState___closed__1_value;
static const lean_closure_object lp_aesop_Aesop_BaseM_instMonadBacktrackSavedState___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*5, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_StateRefT_x27_lift___boxed, .m_arity = 6, .m_num_fixed = 5, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_BaseM_instMonadBacktrackSavedState___closed__1_value)} };
static const lean_object* lp_aesop_Aesop_BaseM_instMonadBacktrackSavedState___closed__2 = (const lean_object*)&lp_aesop_Aesop_BaseM_instMonadBacktrackSavedState___closed__2_value;
static const lean_ctor_object lp_aesop_Aesop_BaseM_instMonadBacktrackSavedState___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop_Aesop_BaseM_instMonadBacktrackSavedState___closed__2_value),((lean_object*)&lp_aesop_Aesop_BaseM_instMonadBacktrackSavedState___closed__0_value)}};
static const lean_object* lp_aesop_Aesop_BaseM_instMonadBacktrackSavedState___closed__3 = (const lean_object*)&lp_aesop_Aesop_BaseM_instMonadBacktrackSavedState___closed__3_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_BaseM_instMonadBacktrackSavedState = (const lean_object*)&lp_aesop_Aesop_BaseM_instMonadBacktrackSavedState___closed__3_value;
static lean_object* _init_lp_aesop_Aesop_BaseM_instInhabitedState_default___closed__0(void){
_start:
{
lean_object* v___x_1_; lean_object* v___x_2_; lean_object* v___x_3_; 
v___x_1_ = lp_aesop_Aesop_instInhabitedStats_default;
v___x_2_ = lp_aesop_Aesop_instInhabitedRulePatternCache_default;
v___x_3_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_3_, 0, v___x_2_);
lean_ctor_set(v___x_3_, 1, v___x_1_);
return v___x_3_;
}
}
static lean_object* _init_lp_aesop_Aesop_BaseM_instInhabitedState_default(void){
_start:
{
lean_object* v___x_4_; 
v___x_4_ = lean_obj_once(&lp_aesop_Aesop_BaseM_instInhabitedState_default___closed__0, &lp_aesop_Aesop_BaseM_instInhabitedState_default___closed__0_once, _init_lp_aesop_Aesop_BaseM_instInhabitedState_default___closed__0);
return v___x_4_;
}
}
static lean_object* _init_lp_aesop_Aesop_BaseM_instInhabitedState(void){
_start:
{
lean_object* v___x_5_; 
v___x_5_ = lp_aesop_Aesop_BaseM_instInhabitedState_default;
return v___x_5_;
}
}
static lean_object* _init_lp_aesop_Aesop_instEmptyCollectionState___closed__0(void){
_start:
{
lean_object* v___x_6_; lean_object* v___x_7_; lean_object* v___x_8_; 
v___x_6_ = lean_box(0);
v___x_7_ = lean_unsigned_to_nat(16u);
v___x_8_ = lean_mk_array(v___x_7_, v___x_6_);
return v___x_8_;
}
}
static lean_object* _init_lp_aesop_Aesop_instEmptyCollectionState___closed__1(void){
_start:
{
lean_object* v___x_9_; lean_object* v___x_10_; lean_object* v___x_11_; 
v___x_9_ = lean_obj_once(&lp_aesop_Aesop_instEmptyCollectionState___closed__0, &lp_aesop_Aesop_instEmptyCollectionState___closed__0_once, _init_lp_aesop_Aesop_instEmptyCollectionState___closed__0);
v___x_10_ = lean_unsigned_to_nat(0u);
v___x_11_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_11_, 0, v___x_10_);
lean_ctor_set(v___x_11_, 1, v___x_9_);
return v___x_11_;
}
}
static lean_object* _init_lp_aesop_Aesop_instEmptyCollectionState___closed__2(void){
_start:
{
lean_object* v___x_12_; lean_object* v___x_13_; lean_object* v___x_14_; 
v___x_12_ = lp_aesop_Aesop_Stats_empty;
v___x_13_ = lean_obj_once(&lp_aesop_Aesop_instEmptyCollectionState___closed__1, &lp_aesop_Aesop_instEmptyCollectionState___closed__1_once, _init_lp_aesop_Aesop_instEmptyCollectionState___closed__1);
v___x_14_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_14_, 0, v___x_13_);
lean_ctor_set(v___x_14_, 1, v___x_12_);
return v___x_14_;
}
}
static lean_object* _init_lp_aesop_Aesop_instEmptyCollectionState(void){
_start:
{
lean_object* v___x_15_; 
v___x_15_ = lean_obj_once(&lp_aesop_Aesop_instEmptyCollectionState___closed__2, &lp_aesop_Aesop_instEmptyCollectionState___closed__2_once, _init_lp_aesop_Aesop_instEmptyCollectionState___closed__2);
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseM_run___redArg(lean_object* v_x_16_, lean_object* v_stats_17_, lean_object* v_a_18_, lean_object* v_a_19_, lean_object* v_a_20_, lean_object* v_a_21_){
_start:
{
lean_object* v___x_23_; lean_object* v___x_24_; lean_object* v___x_25_; lean_object* v___x_26_; 
v___x_23_ = lean_obj_once(&lp_aesop_Aesop_instEmptyCollectionState___closed__1, &lp_aesop_Aesop_instEmptyCollectionState___closed__1_once, _init_lp_aesop_Aesop_instEmptyCollectionState___closed__1);
v___x_24_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_24_, 0, v___x_23_);
lean_ctor_set(v___x_24_, 1, v_stats_17_);
v___x_25_ = lean_st_mk_ref(v___x_24_);
lean_inc(v_a_21_);
lean_inc_ref(v_a_20_);
lean_inc(v_a_19_);
lean_inc_ref(v_a_18_);
lean_inc(v___x_25_);
v___x_26_ = lean_apply_6(v_x_16_, v___x_25_, v_a_18_, v_a_19_, v_a_20_, v_a_21_, lean_box(0));
if (lean_obj_tag(v___x_26_) == 0)
{
lean_object* v_a_27_; lean_object* v___x_29_; uint8_t v_isShared_30_; uint8_t v_isSharedCheck_44_; 
v_a_27_ = lean_ctor_get(v___x_26_, 0);
v_isSharedCheck_44_ = !lean_is_exclusive(v___x_26_);
if (v_isSharedCheck_44_ == 0)
{
v___x_29_ = v___x_26_;
v_isShared_30_ = v_isSharedCheck_44_;
goto v_resetjp_28_;
}
else
{
lean_inc(v_a_27_);
lean_dec(v___x_26_);
v___x_29_ = lean_box(0);
v_isShared_30_ = v_isSharedCheck_44_;
goto v_resetjp_28_;
}
v_resetjp_28_:
{
lean_object* v___x_31_; lean_object* v_stats_32_; lean_object* v___x_34_; uint8_t v_isShared_35_; uint8_t v_isSharedCheck_42_; 
v___x_31_ = lean_st_ref_get(v___x_25_);
lean_dec(v___x_25_);
v_stats_32_ = lean_ctor_get(v___x_31_, 1);
v_isSharedCheck_42_ = !lean_is_exclusive(v___x_31_);
if (v_isSharedCheck_42_ == 0)
{
lean_object* v_unused_43_; 
v_unused_43_ = lean_ctor_get(v___x_31_, 0);
lean_dec(v_unused_43_);
v___x_34_ = v___x_31_;
v_isShared_35_ = v_isSharedCheck_42_;
goto v_resetjp_33_;
}
else
{
lean_inc(v_stats_32_);
lean_dec(v___x_31_);
v___x_34_ = lean_box(0);
v_isShared_35_ = v_isSharedCheck_42_;
goto v_resetjp_33_;
}
v_resetjp_33_:
{
lean_object* v___x_37_; 
if (v_isShared_35_ == 0)
{
lean_ctor_set(v___x_34_, 0, v_a_27_);
v___x_37_ = v___x_34_;
goto v_reusejp_36_;
}
else
{
lean_object* v_reuseFailAlloc_41_; 
v_reuseFailAlloc_41_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_41_, 0, v_a_27_);
lean_ctor_set(v_reuseFailAlloc_41_, 1, v_stats_32_);
v___x_37_ = v_reuseFailAlloc_41_;
goto v_reusejp_36_;
}
v_reusejp_36_:
{
lean_object* v___x_39_; 
if (v_isShared_30_ == 0)
{
lean_ctor_set(v___x_29_, 0, v___x_37_);
v___x_39_ = v___x_29_;
goto v_reusejp_38_;
}
else
{
lean_object* v_reuseFailAlloc_40_; 
v_reuseFailAlloc_40_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_40_, 0, v___x_37_);
v___x_39_ = v_reuseFailAlloc_40_;
goto v_reusejp_38_;
}
v_reusejp_38_:
{
return v___x_39_;
}
}
}
}
}
else
{
lean_object* v_a_45_; lean_object* v___x_47_; uint8_t v_isShared_48_; uint8_t v_isSharedCheck_52_; 
lean_dec(v___x_25_);
v_a_45_ = lean_ctor_get(v___x_26_, 0);
v_isSharedCheck_52_ = !lean_is_exclusive(v___x_26_);
if (v_isSharedCheck_52_ == 0)
{
v___x_47_ = v___x_26_;
v_isShared_48_ = v_isSharedCheck_52_;
goto v_resetjp_46_;
}
else
{
lean_inc(v_a_45_);
lean_dec(v___x_26_);
v___x_47_ = lean_box(0);
v_isShared_48_ = v_isSharedCheck_52_;
goto v_resetjp_46_;
}
v_resetjp_46_:
{
lean_object* v___x_50_; 
if (v_isShared_48_ == 0)
{
v___x_50_ = v___x_47_;
goto v_reusejp_49_;
}
else
{
lean_object* v_reuseFailAlloc_51_; 
v_reuseFailAlloc_51_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_51_, 0, v_a_45_);
v___x_50_ = v_reuseFailAlloc_51_;
goto v_reusejp_49_;
}
v_reusejp_49_:
{
return v___x_50_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseM_run___redArg___boxed(lean_object* v_x_53_, lean_object* v_stats_54_, lean_object* v_a_55_, lean_object* v_a_56_, lean_object* v_a_57_, lean_object* v_a_58_, lean_object* v_a_59_){
_start:
{
lean_object* v_res_60_; 
v_res_60_ = lp_aesop_Aesop_BaseM_run___redArg(v_x_53_, v_stats_54_, v_a_55_, v_a_56_, v_a_57_, v_a_58_);
lean_dec(v_a_58_);
lean_dec_ref(v_a_57_);
lean_dec(v_a_56_);
lean_dec_ref(v_a_55_);
return v_res_60_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseM_run(lean_object* v_00_u03b1_61_, lean_object* v_x_62_, lean_object* v_stats_63_, lean_object* v_a_64_, lean_object* v_a_65_, lean_object* v_a_66_, lean_object* v_a_67_){
_start:
{
lean_object* v___x_69_; 
v___x_69_ = lp_aesop_Aesop_BaseM_run___redArg(v_x_62_, v_stats_63_, v_a_64_, v_a_65_, v_a_66_, v_a_67_);
return v___x_69_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseM_run___boxed(lean_object* v_00_u03b1_70_, lean_object* v_x_71_, lean_object* v_stats_72_, lean_object* v_a_73_, lean_object* v_a_74_, lean_object* v_a_75_, lean_object* v_a_76_, lean_object* v_a_77_){
_start:
{
lean_object* v_res_78_; 
v_res_78_ = lp_aesop_Aesop_BaseM_run(v_00_u03b1_70_, v_x_71_, v_stats_72_, v_a_73_, v_a_74_, v_a_75_, v_a_76_);
lean_dec(v_a_76_);
lean_dec_ref(v_a_75_);
lean_dec(v_a_74_);
lean_dec_ref(v_a_73_);
return v_res_78_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___lam__0(lean_object* v_____do__lift_79_, lean_object* v___y_80_, lean_object* v___y_81_, lean_object* v___y_82_, lean_object* v___y_83_, lean_object* v___y_84_){
_start:
{
lean_object* v_rulePatternCache_86_; lean_object* v___x_87_; 
v_rulePatternCache_86_ = lean_ctor_get(v_____do__lift_79_, 0);
lean_inc_ref(v_rulePatternCache_86_);
v___x_87_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_87_, 0, v_rulePatternCache_86_);
return v___x_87_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___lam__0___boxed(lean_object* v_____do__lift_88_, lean_object* v___y_89_, lean_object* v___y_90_, lean_object* v___y_91_, lean_object* v___y_92_, lean_object* v___y_93_, lean_object* v___y_94_){
_start:
{
lean_object* v_res_95_; 
v_res_95_ = lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___lam__0(v_____do__lift_88_, v___y_89_, v___y_90_, v___y_91_, v___y_92_, v___y_93_);
lean_dec(v___y_93_);
lean_dec_ref(v___y_92_);
lean_dec(v___y_91_);
lean_dec_ref(v___y_90_);
lean_dec(v___y_89_);
lean_dec_ref(v_____do__lift_88_);
return v_res_95_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___lam__1(lean_object* v_f_96_, lean_object* v___y_97_, lean_object* v___y_98_, lean_object* v___y_99_, lean_object* v___y_100_, lean_object* v___y_101_){
_start:
{
lean_object* v___x_103_; lean_object* v_rulePatternCache_104_; lean_object* v_stats_105_; lean_object* v___x_107_; uint8_t v_isShared_108_; uint8_t v_isSharedCheck_116_; 
v___x_103_ = lean_st_ref_take(v___y_97_);
v_rulePatternCache_104_ = lean_ctor_get(v___x_103_, 0);
v_stats_105_ = lean_ctor_get(v___x_103_, 1);
v_isSharedCheck_116_ = !lean_is_exclusive(v___x_103_);
if (v_isSharedCheck_116_ == 0)
{
v___x_107_ = v___x_103_;
v_isShared_108_ = v_isSharedCheck_116_;
goto v_resetjp_106_;
}
else
{
lean_inc(v_stats_105_);
lean_inc(v_rulePatternCache_104_);
lean_dec(v___x_103_);
v___x_107_ = lean_box(0);
v_isShared_108_ = v_isSharedCheck_116_;
goto v_resetjp_106_;
}
v_resetjp_106_:
{
lean_object* v___x_109_; lean_object* v___x_111_; 
v___x_109_ = lean_apply_1(v_f_96_, v_rulePatternCache_104_);
if (v_isShared_108_ == 0)
{
lean_ctor_set(v___x_107_, 0, v___x_109_);
v___x_111_ = v___x_107_;
goto v_reusejp_110_;
}
else
{
lean_object* v_reuseFailAlloc_115_; 
v_reuseFailAlloc_115_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_115_, 0, v___x_109_);
lean_ctor_set(v_reuseFailAlloc_115_, 1, v_stats_105_);
v___x_111_ = v_reuseFailAlloc_115_;
goto v_reusejp_110_;
}
v_reusejp_110_:
{
lean_object* v___x_112_; lean_object* v___x_113_; lean_object* v___x_114_; 
v___x_112_ = lean_st_ref_set(v___y_97_, v___x_111_);
v___x_113_ = lean_box(0);
v___x_114_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_114_, 0, v___x_113_);
return v___x_114_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___lam__1___boxed(lean_object* v_f_117_, lean_object* v___y_118_, lean_object* v___y_119_, lean_object* v___y_120_, lean_object* v___y_121_, lean_object* v___y_122_, lean_object* v___y_123_){
_start:
{
lean_object* v_res_124_; 
v_res_124_ = lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___lam__1(v_f_117_, v___y_118_, v___y_119_, v___y_120_, v___y_121_, v___y_122_);
lean_dec(v___y_122_);
lean_dec_ref(v___y_121_);
lean_dec(v___y_120_);
lean_dec_ref(v___y_119_);
lean_dec(v___y_118_);
return v_res_124_;
}
}
static lean_object* _init_lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__0(void){
_start:
{
lean_object* v___x_125_; 
v___x_125_ = l_instMonadEIO(lean_box(0));
return v___x_125_;
}
}
static lean_object* _init_lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__1(void){
_start:
{
lean_object* v___x_126_; lean_object* v___x_127_; 
v___x_126_ = lean_obj_once(&lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__0, &lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__0_once, _init_lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__0);
v___x_127_ = l_StateRefT_x27_instMonad___redArg(v___x_126_);
return v___x_127_;
}
}
static lean_object* _init_lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry(void){
_start:
{
lean_object* v___x_157_; lean_object* v_toApplicative_158_; lean_object* v_toFunctor_159_; lean_object* v_toSeq_160_; lean_object* v_toSeqLeft_161_; lean_object* v_toSeqRight_162_; lean_object* v___f_163_; lean_object* v___f_164_; lean_object* v___f_165_; lean_object* v___f_166_; lean_object* v___x_167_; lean_object* v___f_168_; lean_object* v___f_169_; lean_object* v___f_170_; lean_object* v___x_171_; lean_object* v___x_172_; lean_object* v___x_173_; lean_object* v_toApplicative_174_; lean_object* v___x_176_; uint8_t v_isShared_177_; uint8_t v_isSharedCheck_206_; 
v___x_157_ = lean_obj_once(&lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__1, &lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__1_once, _init_lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__1);
v_toApplicative_158_ = lean_ctor_get(v___x_157_, 0);
v_toFunctor_159_ = lean_ctor_get(v_toApplicative_158_, 0);
v_toSeq_160_ = lean_ctor_get(v_toApplicative_158_, 2);
v_toSeqLeft_161_ = lean_ctor_get(v_toApplicative_158_, 3);
v_toSeqRight_162_ = lean_ctor_get(v_toApplicative_158_, 4);
v___f_163_ = ((lean_object*)(lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__2));
v___f_164_ = ((lean_object*)(lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__3));
lean_inc_ref_n(v_toFunctor_159_, 2);
v___f_165_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_165_, 0, v_toFunctor_159_);
v___f_166_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_166_, 0, v_toFunctor_159_);
v___x_167_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_167_, 0, v___f_165_);
lean_ctor_set(v___x_167_, 1, v___f_166_);
lean_inc(v_toSeqRight_162_);
v___f_168_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_168_, 0, v_toSeqRight_162_);
lean_inc(v_toSeqLeft_161_);
v___f_169_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_169_, 0, v_toSeqLeft_161_);
lean_inc(v_toSeq_160_);
v___f_170_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_170_, 0, v_toSeq_160_);
v___x_171_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_171_, 0, v___x_167_);
lean_ctor_set(v___x_171_, 1, v___f_163_);
lean_ctor_set(v___x_171_, 2, v___f_170_);
lean_ctor_set(v___x_171_, 3, v___f_169_);
lean_ctor_set(v___x_171_, 4, v___f_168_);
v___x_172_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_172_, 0, v___x_171_);
lean_ctor_set(v___x_172_, 1, v___f_164_);
v___x_173_ = l_StateRefT_x27_instMonad___redArg(v___x_172_);
v_toApplicative_174_ = lean_ctor_get(v___x_173_, 0);
v_isSharedCheck_206_ = !lean_is_exclusive(v___x_173_);
if (v_isSharedCheck_206_ == 0)
{
lean_object* v_unused_207_; 
v_unused_207_ = lean_ctor_get(v___x_173_, 1);
lean_dec(v_unused_207_);
v___x_176_ = v___x_173_;
v_isShared_177_ = v_isSharedCheck_206_;
goto v_resetjp_175_;
}
else
{
lean_inc(v_toApplicative_174_);
lean_dec(v___x_173_);
v___x_176_ = lean_box(0);
v_isShared_177_ = v_isSharedCheck_206_;
goto v_resetjp_175_;
}
v_resetjp_175_:
{
lean_object* v_toFunctor_178_; lean_object* v_toSeq_179_; lean_object* v_toSeqLeft_180_; lean_object* v_toSeqRight_181_; lean_object* v___x_183_; uint8_t v_isShared_184_; uint8_t v_isSharedCheck_204_; 
v_toFunctor_178_ = lean_ctor_get(v_toApplicative_174_, 0);
v_toSeq_179_ = lean_ctor_get(v_toApplicative_174_, 2);
v_toSeqLeft_180_ = lean_ctor_get(v_toApplicative_174_, 3);
v_toSeqRight_181_ = lean_ctor_get(v_toApplicative_174_, 4);
v_isSharedCheck_204_ = !lean_is_exclusive(v_toApplicative_174_);
if (v_isSharedCheck_204_ == 0)
{
lean_object* v_unused_205_; 
v_unused_205_ = lean_ctor_get(v_toApplicative_174_, 1);
lean_dec(v_unused_205_);
v___x_183_ = v_toApplicative_174_;
v_isShared_184_ = v_isSharedCheck_204_;
goto v_resetjp_182_;
}
else
{
lean_inc(v_toSeqRight_181_);
lean_inc(v_toSeqLeft_180_);
lean_inc(v_toSeq_179_);
lean_inc(v_toFunctor_178_);
lean_dec(v_toApplicative_174_);
v___x_183_ = lean_box(0);
v_isShared_184_ = v_isSharedCheck_204_;
goto v_resetjp_182_;
}
v_resetjp_182_:
{
lean_object* v___f_185_; lean_object* v___f_186_; lean_object* v___f_187_; lean_object* v___f_188_; lean_object* v___f_189_; lean_object* v___f_190_; lean_object* v___x_191_; lean_object* v___f_192_; lean_object* v___f_193_; lean_object* v___f_194_; lean_object* v___x_196_; 
v___f_185_ = ((lean_object*)(lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__4));
v___f_186_ = ((lean_object*)(lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__5));
v___f_187_ = ((lean_object*)(lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__6));
v___f_188_ = ((lean_object*)(lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__7));
lean_inc_ref(v_toFunctor_178_);
v___f_189_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_189_, 0, v_toFunctor_178_);
v___f_190_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_190_, 0, v_toFunctor_178_);
v___x_191_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_191_, 0, v___f_189_);
lean_ctor_set(v___x_191_, 1, v___f_190_);
v___f_192_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_192_, 0, v_toSeqRight_181_);
v___f_193_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_193_, 0, v_toSeqLeft_180_);
v___f_194_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_194_, 0, v_toSeq_179_);
if (v_isShared_184_ == 0)
{
lean_ctor_set(v___x_183_, 4, v___f_192_);
lean_ctor_set(v___x_183_, 3, v___f_193_);
lean_ctor_set(v___x_183_, 2, v___f_194_);
lean_ctor_set(v___x_183_, 1, v___f_187_);
lean_ctor_set(v___x_183_, 0, v___x_191_);
v___x_196_ = v___x_183_;
goto v_reusejp_195_;
}
else
{
lean_object* v_reuseFailAlloc_203_; 
v_reuseFailAlloc_203_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_203_, 0, v___x_191_);
lean_ctor_set(v_reuseFailAlloc_203_, 1, v___f_187_);
lean_ctor_set(v_reuseFailAlloc_203_, 2, v___f_194_);
lean_ctor_set(v_reuseFailAlloc_203_, 3, v___f_193_);
lean_ctor_set(v_reuseFailAlloc_203_, 4, v___f_192_);
v___x_196_ = v_reuseFailAlloc_203_;
goto v_reusejp_195_;
}
v_reusejp_195_:
{
lean_object* v___x_198_; 
if (v_isShared_177_ == 0)
{
lean_ctor_set(v___x_176_, 1, v___f_188_);
lean_ctor_set(v___x_176_, 0, v___x_196_);
v___x_198_ = v___x_176_;
goto v_reusejp_197_;
}
else
{
lean_object* v_reuseFailAlloc_202_; 
v_reuseFailAlloc_202_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_202_, 0, v___x_196_);
lean_ctor_set(v_reuseFailAlloc_202_, 1, v___f_188_);
v___x_198_ = v_reuseFailAlloc_202_;
goto v_reusejp_197_;
}
v_reusejp_197_:
{
lean_object* v___x_199_; lean_object* v___x_200_; lean_object* v___x_201_; 
v___x_199_ = ((lean_object*)(lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__19));
v___x_200_ = lean_alloc_closure((void*)(l_ReaderT_bind___boxed), 8, 7);
lean_closure_set(v___x_200_, 0, lean_box(0));
lean_closure_set(v___x_200_, 1, lean_box(0));
lean_closure_set(v___x_200_, 2, v___x_198_);
lean_closure_set(v___x_200_, 3, lean_box(0));
lean_closure_set(v___x_200_, 4, lean_box(0));
lean_closure_set(v___x_200_, 5, v___x_199_);
lean_closure_set(v___x_200_, 6, v___f_185_);
v___x_201_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_201_, 0, v___x_200_);
lean_ctor_set(v___x_201_, 1, v___f_186_);
return v___x_201_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseM_instMonadStats___lam__0(lean_object* v_00_u03b1_208_, lean_object* v_f_209_, lean_object* v___y_210_, lean_object* v___y_211_, lean_object* v___y_212_, lean_object* v___y_213_, lean_object* v___y_214_){
_start:
{
lean_object* v___x_216_; lean_object* v_rulePatternCache_217_; lean_object* v_stats_218_; lean_object* v___x_220_; uint8_t v_isShared_221_; uint8_t v_isSharedCheck_230_; 
v___x_216_ = lean_st_ref_take(v___y_210_);
v_rulePatternCache_217_ = lean_ctor_get(v___x_216_, 0);
v_stats_218_ = lean_ctor_get(v___x_216_, 1);
v_isSharedCheck_230_ = !lean_is_exclusive(v___x_216_);
if (v_isSharedCheck_230_ == 0)
{
v___x_220_ = v___x_216_;
v_isShared_221_ = v_isSharedCheck_230_;
goto v_resetjp_219_;
}
else
{
lean_inc(v_stats_218_);
lean_inc(v_rulePatternCache_217_);
lean_dec(v___x_216_);
v___x_220_ = lean_box(0);
v_isShared_221_ = v_isSharedCheck_230_;
goto v_resetjp_219_;
}
v_resetjp_219_:
{
lean_object* v___x_222_; lean_object* v_fst_223_; lean_object* v_snd_224_; lean_object* v___x_226_; 
v___x_222_ = lean_apply_1(v_f_209_, v_stats_218_);
v_fst_223_ = lean_ctor_get(v___x_222_, 0);
lean_inc(v_fst_223_);
v_snd_224_ = lean_ctor_get(v___x_222_, 1);
lean_inc(v_snd_224_);
lean_dec_ref(v___x_222_);
if (v_isShared_221_ == 0)
{
lean_ctor_set(v___x_220_, 1, v_snd_224_);
v___x_226_ = v___x_220_;
goto v_reusejp_225_;
}
else
{
lean_object* v_reuseFailAlloc_229_; 
v_reuseFailAlloc_229_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_229_, 0, v_rulePatternCache_217_);
lean_ctor_set(v_reuseFailAlloc_229_, 1, v_snd_224_);
v___x_226_ = v_reuseFailAlloc_229_;
goto v_reusejp_225_;
}
v_reusejp_225_:
{
lean_object* v___x_227_; lean_object* v___x_228_; 
v___x_227_ = lean_st_ref_set(v___y_210_, v___x_226_);
v___x_228_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_228_, 0, v_fst_223_);
return v___x_228_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseM_instMonadStats___lam__0___boxed(lean_object* v_00_u03b1_231_, lean_object* v_f_232_, lean_object* v___y_233_, lean_object* v___y_234_, lean_object* v___y_235_, lean_object* v___y_236_, lean_object* v___y_237_, lean_object* v___y_238_){
_start:
{
lean_object* v_res_239_; 
v_res_239_ = lp_aesop_Aesop_BaseM_instMonadStats___lam__0(v_00_u03b1_231_, v_f_232_, v___y_233_, v___y_234_, v___y_235_, v___y_236_, v___y_237_);
lean_dec(v___y_237_);
lean_dec_ref(v___y_236_);
lean_dec(v___y_235_);
lean_dec_ref(v___y_234_);
lean_dec(v___y_233_);
return v_res_239_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseM_instMonadStats___lam__1(lean_object* v_____do__lift_240_, lean_object* v___y_241_, lean_object* v___y_242_, lean_object* v___y_243_, lean_object* v___y_244_, lean_object* v___y_245_){
_start:
{
lean_object* v_stats_247_; lean_object* v___x_248_; 
v_stats_247_ = lean_ctor_get(v_____do__lift_240_, 1);
lean_inc_ref(v_stats_247_);
v___x_248_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_248_, 0, v_stats_247_);
return v___x_248_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseM_instMonadStats___lam__1___boxed(lean_object* v_____do__lift_249_, lean_object* v___y_250_, lean_object* v___y_251_, lean_object* v___y_252_, lean_object* v___y_253_, lean_object* v___y_254_, lean_object* v___y_255_){
_start:
{
lean_object* v_res_256_; 
v_res_256_ = lp_aesop_Aesop_BaseM_instMonadStats___lam__1(v_____do__lift_249_, v___y_250_, v___y_251_, v___y_252_, v___y_253_, v___y_254_);
lean_dec(v___y_254_);
lean_dec_ref(v___y_253_);
lean_dec(v___y_252_);
lean_dec_ref(v___y_251_);
lean_dec(v___y_250_);
lean_dec_ref(v_____do__lift_249_);
return v_res_256_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseM_instMonadStats___lam__2(lean_object* v_f_257_, lean_object* v___y_258_, lean_object* v___y_259_, lean_object* v___y_260_, lean_object* v___y_261_, lean_object* v___y_262_){
_start:
{
lean_object* v___x_264_; lean_object* v_rulePatternCache_265_; lean_object* v_stats_266_; lean_object* v___x_268_; uint8_t v_isShared_269_; uint8_t v_isSharedCheck_277_; 
v___x_264_ = lean_st_ref_take(v___y_258_);
v_rulePatternCache_265_ = lean_ctor_get(v___x_264_, 0);
v_stats_266_ = lean_ctor_get(v___x_264_, 1);
v_isSharedCheck_277_ = !lean_is_exclusive(v___x_264_);
if (v_isSharedCheck_277_ == 0)
{
v___x_268_ = v___x_264_;
v_isShared_269_ = v_isSharedCheck_277_;
goto v_resetjp_267_;
}
else
{
lean_inc(v_stats_266_);
lean_inc(v_rulePatternCache_265_);
lean_dec(v___x_264_);
v___x_268_ = lean_box(0);
v_isShared_269_ = v_isSharedCheck_277_;
goto v_resetjp_267_;
}
v_resetjp_267_:
{
lean_object* v___x_270_; lean_object* v___x_272_; 
v___x_270_ = lean_apply_1(v_f_257_, v_stats_266_);
if (v_isShared_269_ == 0)
{
lean_ctor_set(v___x_268_, 1, v___x_270_);
v___x_272_ = v___x_268_;
goto v_reusejp_271_;
}
else
{
lean_object* v_reuseFailAlloc_276_; 
v_reuseFailAlloc_276_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_276_, 0, v_rulePatternCache_265_);
lean_ctor_set(v_reuseFailAlloc_276_, 1, v___x_270_);
v___x_272_ = v_reuseFailAlloc_276_;
goto v_reusejp_271_;
}
v_reusejp_271_:
{
lean_object* v___x_273_; lean_object* v___x_274_; lean_object* v___x_275_; 
v___x_273_ = lean_st_ref_set(v___y_258_, v___x_272_);
v___x_274_ = lean_box(0);
v___x_275_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_275_, 0, v___x_274_);
return v___x_275_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseM_instMonadStats___lam__2___boxed(lean_object* v_f_278_, lean_object* v___y_279_, lean_object* v___y_280_, lean_object* v___y_281_, lean_object* v___y_282_, lean_object* v___y_283_, lean_object* v___y_284_){
_start:
{
lean_object* v_res_285_; 
v_res_285_ = lp_aesop_Aesop_BaseM_instMonadStats___lam__2(v_f_278_, v___y_279_, v___y_280_, v___y_281_, v___y_282_, v___y_283_);
lean_dec(v___y_283_);
lean_dec_ref(v___y_282_);
lean_dec(v___y_281_);
lean_dec_ref(v___y_280_);
lean_dec(v___y_279_);
return v_res_285_;
}
}
static lean_object* _init_lp_aesop_Aesop_BaseM_instMonadStats(void){
_start:
{
lean_object* v___x_296_; lean_object* v_toApplicative_297_; lean_object* v_toFunctor_298_; lean_object* v_toSeq_299_; lean_object* v_toSeqLeft_300_; lean_object* v_toSeqRight_301_; lean_object* v___f_302_; lean_object* v___f_303_; lean_object* v___f_304_; lean_object* v___f_305_; lean_object* v___x_306_; lean_object* v___f_307_; lean_object* v___f_308_; lean_object* v___f_309_; lean_object* v___x_310_; lean_object* v___x_311_; lean_object* v___x_312_; lean_object* v_toApplicative_313_; lean_object* v___x_315_; uint8_t v_isShared_316_; uint8_t v_isSharedCheck_347_; 
v___x_296_ = lean_obj_once(&lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__1, &lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__1_once, _init_lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__1);
v_toApplicative_297_ = lean_ctor_get(v___x_296_, 0);
v_toFunctor_298_ = lean_ctor_get(v_toApplicative_297_, 0);
v_toSeq_299_ = lean_ctor_get(v_toApplicative_297_, 2);
v_toSeqLeft_300_ = lean_ctor_get(v_toApplicative_297_, 3);
v_toSeqRight_301_ = lean_ctor_get(v_toApplicative_297_, 4);
v___f_302_ = ((lean_object*)(lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__2));
v___f_303_ = ((lean_object*)(lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__3));
lean_inc_ref_n(v_toFunctor_298_, 2);
v___f_304_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_304_, 0, v_toFunctor_298_);
v___f_305_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_305_, 0, v_toFunctor_298_);
v___x_306_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_306_, 0, v___f_304_);
lean_ctor_set(v___x_306_, 1, v___f_305_);
lean_inc(v_toSeqRight_301_);
v___f_307_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_307_, 0, v_toSeqRight_301_);
lean_inc(v_toSeqLeft_300_);
v___f_308_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_308_, 0, v_toSeqLeft_300_);
lean_inc(v_toSeq_299_);
v___f_309_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_309_, 0, v_toSeq_299_);
v___x_310_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_310_, 0, v___x_306_);
lean_ctor_set(v___x_310_, 1, v___f_302_);
lean_ctor_set(v___x_310_, 2, v___f_309_);
lean_ctor_set(v___x_310_, 3, v___f_308_);
lean_ctor_set(v___x_310_, 4, v___f_307_);
v___x_311_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_311_, 0, v___x_310_);
lean_ctor_set(v___x_311_, 1, v___f_303_);
v___x_312_ = l_StateRefT_x27_instMonad___redArg(v___x_311_);
v_toApplicative_313_ = lean_ctor_get(v___x_312_, 0);
v_isSharedCheck_347_ = !lean_is_exclusive(v___x_312_);
if (v_isSharedCheck_347_ == 0)
{
lean_object* v_unused_348_; 
v_unused_348_ = lean_ctor_get(v___x_312_, 1);
lean_dec(v_unused_348_);
v___x_315_ = v___x_312_;
v_isShared_316_ = v_isSharedCheck_347_;
goto v_resetjp_314_;
}
else
{
lean_inc(v_toApplicative_313_);
lean_dec(v___x_312_);
v___x_315_ = lean_box(0);
v_isShared_316_ = v_isSharedCheck_347_;
goto v_resetjp_314_;
}
v_resetjp_314_:
{
lean_object* v_toFunctor_317_; lean_object* v_toSeq_318_; lean_object* v_toSeqLeft_319_; lean_object* v_toSeqRight_320_; lean_object* v___x_322_; uint8_t v_isShared_323_; uint8_t v_isSharedCheck_345_; 
v_toFunctor_317_ = lean_ctor_get(v_toApplicative_313_, 0);
v_toSeq_318_ = lean_ctor_get(v_toApplicative_313_, 2);
v_toSeqLeft_319_ = lean_ctor_get(v_toApplicative_313_, 3);
v_toSeqRight_320_ = lean_ctor_get(v_toApplicative_313_, 4);
v_isSharedCheck_345_ = !lean_is_exclusive(v_toApplicative_313_);
if (v_isSharedCheck_345_ == 0)
{
lean_object* v_unused_346_; 
v_unused_346_ = lean_ctor_get(v_toApplicative_313_, 1);
lean_dec(v_unused_346_);
v___x_322_ = v_toApplicative_313_;
v_isShared_323_ = v_isSharedCheck_345_;
goto v_resetjp_321_;
}
else
{
lean_inc(v_toSeqRight_320_);
lean_inc(v_toSeqLeft_319_);
lean_inc(v_toSeq_318_);
lean_inc(v_toFunctor_317_);
lean_dec(v_toApplicative_313_);
v___x_322_ = lean_box(0);
v_isShared_323_ = v_isSharedCheck_345_;
goto v_resetjp_321_;
}
v_resetjp_321_:
{
lean_object* v___f_324_; lean_object* v___f_325_; lean_object* v___f_326_; lean_object* v___x_327_; lean_object* v___f_328_; lean_object* v___f_329_; lean_object* v___f_330_; lean_object* v___f_331_; lean_object* v___x_332_; lean_object* v___f_333_; lean_object* v___f_334_; lean_object* v___f_335_; lean_object* v___x_337_; 
v___f_324_ = ((lean_object*)(lp_aesop_Aesop_BaseM_instMonadStats___closed__0));
v___f_325_ = ((lean_object*)(lp_aesop_Aesop_BaseM_instMonadStats___closed__1));
v___f_326_ = ((lean_object*)(lp_aesop_Aesop_BaseM_instMonadStats___closed__2));
v___x_327_ = ((lean_object*)(lp_aesop_Aesop_BaseM_instMonadStats___closed__6));
v___f_328_ = ((lean_object*)(lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__6));
v___f_329_ = ((lean_object*)(lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__7));
lean_inc_ref(v_toFunctor_317_);
v___f_330_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__0), 6, 1);
lean_closure_set(v___f_330_, 0, v_toFunctor_317_);
v___f_331_ = lean_alloc_closure((void*)(l_ReaderT_instFunctorOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_331_, 0, v_toFunctor_317_);
v___x_332_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_332_, 0, v___f_330_);
lean_ctor_set(v___x_332_, 1, v___f_331_);
v___f_333_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_333_, 0, v_toSeqRight_320_);
v___f_334_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__3), 6, 1);
lean_closure_set(v___f_334_, 0, v_toSeqLeft_319_);
v___f_335_ = lean_alloc_closure((void*)(l_ReaderT_instApplicativeOfMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_335_, 0, v_toSeq_318_);
if (v_isShared_323_ == 0)
{
lean_ctor_set(v___x_322_, 4, v___f_333_);
lean_ctor_set(v___x_322_, 3, v___f_334_);
lean_ctor_set(v___x_322_, 2, v___f_335_);
lean_ctor_set(v___x_322_, 1, v___f_328_);
lean_ctor_set(v___x_322_, 0, v___x_332_);
v___x_337_ = v___x_322_;
goto v_reusejp_336_;
}
else
{
lean_object* v_reuseFailAlloc_344_; 
v_reuseFailAlloc_344_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_344_, 0, v___x_332_);
lean_ctor_set(v_reuseFailAlloc_344_, 1, v___f_328_);
lean_ctor_set(v_reuseFailAlloc_344_, 2, v___f_335_);
lean_ctor_set(v_reuseFailAlloc_344_, 3, v___f_334_);
lean_ctor_set(v_reuseFailAlloc_344_, 4, v___f_333_);
v___x_337_ = v_reuseFailAlloc_344_;
goto v_reusejp_336_;
}
v_reusejp_336_:
{
lean_object* v___x_339_; 
if (v_isShared_316_ == 0)
{
lean_ctor_set(v___x_315_, 1, v___f_329_);
lean_ctor_set(v___x_315_, 0, v___x_337_);
v___x_339_ = v___x_315_;
goto v_reusejp_338_;
}
else
{
lean_object* v_reuseFailAlloc_343_; 
v_reuseFailAlloc_343_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_343_, 0, v___x_337_);
lean_ctor_set(v_reuseFailAlloc_343_, 1, v___f_329_);
v___x_339_ = v_reuseFailAlloc_343_;
goto v_reusejp_338_;
}
v_reusejp_338_:
{
lean_object* v___x_340_; lean_object* v___x_341_; lean_object* v___x_342_; 
v___x_340_ = ((lean_object*)(lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry___closed__19));
v___x_341_ = lean_alloc_closure((void*)(l_ReaderT_bind___boxed), 8, 7);
lean_closure_set(v___x_341_, 0, lean_box(0));
lean_closure_set(v___x_341_, 1, lean_box(0));
lean_closure_set(v___x_341_, 2, v___x_339_);
lean_closure_set(v___x_341_, 3, lean_box(0));
lean_closure_set(v___x_341_, 4, lean_box(0));
lean_closure_set(v___x_341_, 5, v___x_340_);
lean_closure_set(v___x_341_, 6, v___f_325_);
v___x_342_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_342_, 0, v___x_327_);
lean_ctor_set(v___x_342_, 1, v___f_324_);
lean_ctor_set(v___x_342_, 2, v___x_341_);
lean_ctor_set(v___x_342_, 3, v___f_326_);
return v___x_342_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseM_instMonadBacktrackSavedState___lam__0(lean_object* v_x_349_, lean_object* v___y_350_, lean_object* v___y_351_, lean_object* v___y_352_, lean_object* v___y_353_, lean_object* v___y_354_){
_start:
{
lean_object* v___x_356_; 
v___x_356_ = l_Lean_Meta_SavedState_restore___redArg(v_x_349_, v___y_352_, v___y_354_);
return v___x_356_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BaseM_instMonadBacktrackSavedState___lam__0___boxed(lean_object* v_x_357_, lean_object* v___y_358_, lean_object* v___y_359_, lean_object* v___y_360_, lean_object* v___y_361_, lean_object* v___y_362_, lean_object* v___y_363_){
_start:
{
lean_object* v_res_364_; 
v_res_364_ = lp_aesop_Aesop_BaseM_instMonadBacktrackSavedState___lam__0(v_x_357_, v___y_358_, v___y_359_, v___y_360_, v___y_361_, v___y_362_);
lean_dec(v___y_362_);
lean_dec_ref(v___y_361_);
lean_dec(v___y_360_);
lean_dec_ref(v___y_359_);
lean_dec(v___y_358_);
lean_dec_ref(v_x_357_);
return v_res_364_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Stats_Basic(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_RulePattern_Cache(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_BaseM(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Stats_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_RulePattern_Cache(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_aesop_Aesop_BaseM_instInhabitedState_default = _init_lp_aesop_Aesop_BaseM_instInhabitedState_default();
lean_mark_persistent(lp_aesop_Aesop_BaseM_instInhabitedState_default);
lp_aesop_Aesop_BaseM_instInhabitedState = _init_lp_aesop_Aesop_BaseM_instInhabitedState();
lean_mark_persistent(lp_aesop_Aesop_BaseM_instInhabitedState);
lp_aesop_Aesop_instEmptyCollectionState = _init_lp_aesop_Aesop_instEmptyCollectionState();
lean_mark_persistent(lp_aesop_Aesop_instEmptyCollectionState);
lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry = _init_lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry();
lean_mark_persistent(lp_aesop_Aesop_BaseM_instMonadHashMapCacheAdapterExprEntry);
lp_aesop_Aesop_BaseM_instMonadStats = _init_lp_aesop_Aesop_BaseM_instMonadStats();
lean_mark_persistent(lp_aesop_Aesop_BaseM_instMonadStats);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_BaseM(uint8_t builtin) {
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
lean_object* initialize_aesop_Aesop_Stats_Basic(uint8_t builtin);
lean_object* initialize_aesop_Aesop_RulePattern_Cache(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_BaseM(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Stats_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_RulePattern_Cache(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_BaseM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_BaseM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_BaseM(builtin);
}
#ifdef __cplusplus
}
#endif
