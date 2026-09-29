// Lean compiler output
// Module: Batteries.Data.MLList.Basic
// Imports: public import Init public meta import Init public import Batteries.Control.AlternativeMonad
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
lean_object* lp_batteries_AlternativeMonad_toMonad___redArg(lean_object*);
lean_object* lean_thunk_get_own(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_mk_thunk(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Option_isNone___boxed(lean_object*, lean_object*);
lean_object* l_ReaderT_instMonad___redArg(lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* l_Function_const___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_instMonadLiftT___lam__0___boxed(lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* l_ST_Prim_mkRef___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Function_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ST_Prim_Ref_get___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_StateT_instMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_StateT_instMonad___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_StateT_instMonad___redArg___lam__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_StateT_instMonad___redArg___lam__9(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_StateT_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_StateT_pure(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_StateT_bind(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Option_get_x21___boxed(lean_object*, lean_object*, lean_object*);
lean_object* l_StateRefT_x27_instMonad___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_MLListImpl_ctorIdx___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_MLListImpl_ctorIdx___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_MLListImpl_ctorIdx(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_MLListImpl_ctorIdx___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_MLListImpl_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_MLListImpl_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_MLListImpl_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_MLListImpl_nil_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_MLListImpl_nil_elim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_MLListImpl_cons_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_MLListImpl_cons_elim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_MLListImpl_thunk_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_MLListImpl_thunk_elim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_MLListImpl_squash_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_MLListImpl_squash_elim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_unconsImpl___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_unconsImpl(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_uncons_x3fImpl___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_uncons_x3fImpl___redArg___closed__0 = (const lean_object*)&lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_uncons_x3fImpl___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_uncons_x3fImpl___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_uncons_x3fImpl(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_specImpl___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_specImpl___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_specImpl___lam__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_specImpl___lam__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_specImpl___lam__4(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_specImpl___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_specImpl___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_specImpl___closed__0 = (const lean_object*)&lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_specImpl___closed__0_value;
static const lean_closure_object lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_specImpl___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_specImpl___lam__1, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_specImpl___closed__1 = (const lean_object*)&lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_specImpl___closed__1_value;
static const lean_closure_object lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_specImpl___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_specImpl___lam__2, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_specImpl___closed__2 = (const lean_object*)&lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_specImpl___closed__2_value;
static const lean_closure_object lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_specImpl___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_specImpl___lam__3, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_specImpl___closed__3 = (const lean_object*)&lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_specImpl___closed__3_value;
static const lean_closure_object lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_specImpl___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_specImpl___lam__4, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_specImpl___closed__4 = (const lean_object*)&lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_specImpl___closed__4_value;
static const lean_closure_object lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_specImpl___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_uncons_x3fImpl, .m_arity = 3, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_specImpl___closed__5 = (const lean_object*)&lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_specImpl___closed__5_value;
static const lean_ctor_object lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_specImpl___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*6 + 0, .m_other = 6, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_specImpl___closed__0_value),((lean_object*)&lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_specImpl___closed__1_value),((lean_object*)&lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_specImpl___closed__2_value),((lean_object*)&lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_specImpl___closed__3_value),((lean_object*)&lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_specImpl___closed__4_value),((lean_object*)&lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_specImpl___closed__5_value)}};
static const lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_specImpl___closed__6 = (const lean_object*)&lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_specImpl___closed__6_value;
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_specImpl(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_nil(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_cons___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_cons(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_thunk___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_thunk(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_squash___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_squash(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_uncons___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_uncons(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_uncons_x3f___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_uncons_x3f(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_instEmptyCollection(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_instInhabited(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_instInhabitedForallForallForallForallForInStepOfMonad___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_instInhabitedForallForallForallForallForInStepOfMonad___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_instInhabitedForallForallForallForallForInStepOfMonad___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_instInhabitedForallForallForallForallForInStepOfMonad(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_forIn___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_forIn___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_forIn___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_forIn(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_instForInOfMonadOfMonadLiftT___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_instForInOfMonadOfMonadLiftT___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_instForInOfMonadOfMonadLiftT(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_singletonM___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_singletonM___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_singletonM___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_singletonM(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_singleton___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_singleton(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_fix___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_fix___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_fix(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_fix_x3f_x27___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_fix_x3f_x27___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_fix_x3f_x27___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_fix_x3f_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_fix_x3f___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_fix_x3f___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_fix_x3f___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_fix_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_iterate___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_iterate___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_iterate___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_iterate(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_fixlWith___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_fixlWith___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_fixlWith___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_fixlWith___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_fixlWith(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_fixl___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_fixl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_MLList_isEmpty___redArg___lam__0(uint8_t);
LEAN_EXPORT lean_object* lp_batteries_MLList_isEmpty___redArg___lam__0___boxed(lean_object*);
static const lean_closure_object lp_batteries_MLList_isEmpty___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries_MLList_isEmpty___redArg___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_MLList_isEmpty___redArg___closed__0 = (const lean_object*)&lp_batteries_MLList_isEmpty___redArg___closed__0_value;
static const lean_closure_object lp_batteries_MLList_isEmpty___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Option_isNone___boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_batteries_MLList_isEmpty___redArg___closed__1 = (const lean_object*)&lp_batteries_MLList_isEmpty___redArg___closed__1_value;
static const lean_closure_object lp_batteries_MLList_isEmpty___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*5, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Function_comp, .m_arity = 6, .m_num_fixed = 5, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_MLList_isEmpty___redArg___closed__0_value),((lean_object*)&lp_batteries_MLList_isEmpty___redArg___closed__1_value)} };
static const lean_object* lp_batteries_MLList_isEmpty___redArg___closed__2 = (const lean_object*)&lp_batteries_MLList_isEmpty___redArg___closed__2_value;
LEAN_EXPORT lean_object* lp_batteries_MLList_isEmpty___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_isEmpty(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_ofList___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_ofList___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_ofList(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_ofListM___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_ofListM___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_ofListM___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_ofListM(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_force___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_force___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_force___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_force(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_asArray___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_asArray___redArg___lam__1(lean_object*, lean_object*);
static const lean_array_object lp_batteries_MLList_asArray___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_batteries_MLList_asArray___redArg___closed__0 = (const lean_object*)&lp_batteries_MLList_asArray___redArg___closed__0_value;
static const lean_closure_object lp_batteries_MLList_asArray___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instMonadLiftT___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_MLList_asArray___redArg___closed__1 = (const lean_object*)&lp_batteries_MLList_asArray___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_batteries_MLList_asArray___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_asArray(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_casesM___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_casesM___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_casesM___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_casesM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_cases___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_cases___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_cases___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_cases___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_cases(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_foldsM___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_foldsM___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_foldsM___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_foldsM___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_foldsM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_folds___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_folds___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_folds(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_takeAsList_go___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_takeAsList_go___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_takeAsList_go___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_takeAsList_go___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_takeAsList_go___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_takeAsList_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_takeAsList_go___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_takeAsList___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_takeAsList___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_takeAsList(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_takeAsList___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_takeAsArray_go___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_takeAsArray_go___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_takeAsArray_go___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_takeAsArray_go___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_takeAsArray_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_takeAsArray_go___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_takeAsArray___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_takeAsArray___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_takeAsArray(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_takeAsArray___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_take___redArg___lam__0(lean_object*);
static const lean_closure_object lp_batteries_MLList_take___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries_MLList_take___redArg___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_MLList_take___redArg___closed__0 = (const lean_object*)&lp_batteries_MLList_take___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_batteries_MLList_take___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_take___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_take___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_take___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_take(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_take___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_drop___redArg___lam__0(lean_object*);
static const lean_closure_object lp_batteries_MLList_drop___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries_MLList_drop___redArg___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_MLList_drop___redArg___closed__0 = (const lean_object*)&lp_batteries_MLList_drop___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_batteries_MLList_drop___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_drop___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_drop___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_drop___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_drop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_drop___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_mapM___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_mapM___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_mapM___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_mapM___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_mapM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_map___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_map___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_filterM___redArg___lam__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_filterM___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_filterM___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_batteries_MLList_filterM___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_filterM___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_filterM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_filter___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_filter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_filter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_filterMapM___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_filterMapM___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_filterMapM___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_filterMapM___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_filterMapM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_filterMap___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_filterMap___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_filterMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_takeWhileM___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_takeWhileM___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_batteries_MLList_takeWhileM___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_takeWhileM___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_takeWhileM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_takeWhile___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_takeWhile___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_takeWhile(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_append___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_append___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_append(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_join___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_join___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_join___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_join(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_enumFrom___redArg___lam__0(lean_object*);
static const lean_closure_object lp_batteries_MLList_enumFrom___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries_MLList_enumFrom___redArg___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_MLList_enumFrom___redArg___closed__0 = (const lean_object*)&lp_batteries_MLList_enumFrom___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_batteries_MLList_enumFrom___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_enumFrom___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_enumFrom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_enum___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_enum(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_range___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_range___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_range___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_range(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_fin_go___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_fin_go___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_fin_go___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_fin_go(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_fin___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_fin(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_ofArray_go___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_ofArray_go___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_ofArray_go___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_ofArray_go(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_ofArray___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_ofArray(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_chunk_go___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_chunk_go___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_chunk_go___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_chunk_go___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_chunk_go___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_chunk_go___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_chunk_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_chunk_go___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_chunk___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_chunk(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_concat___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_concat___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_concat(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_zip___redArg___lam__0(lean_object*);
static const lean_closure_object lp_batteries_MLList_zip___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries_MLList_zip___redArg___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_MLList_zip___redArg___closed__0 = (const lean_object*)&lp_batteries_MLList_zip___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_batteries_MLList_zip___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_zip___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_zip___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_zip(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_bind___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_bind___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_bind___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_bind(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_monadLift___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_monadLift(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_liftM___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_liftM___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_liftM___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_liftM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_runState___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_runState___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_runState___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_runState(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_runState_x27___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_runState_x27___redArg___lam__0___boxed(lean_object*);
static const lean_closure_object lp_batteries_MLList_runState_x27___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries_MLList_runState_x27___redArg___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_MLList_runState_x27___redArg___closed__0 = (const lean_object*)&lp_batteries_MLList_runState_x27___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_batteries_MLList_runState_x27___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_runState_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_runReader___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_runReader___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_runReader___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_runReader(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_runStateRef___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_runStateRef___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_runStateRef___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_runStateRef___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_runStateRef___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_runStateRef___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_runStateRef(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_head_x3f___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_head_x3f___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_head_x3f(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_takeUpToFirstM___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_takeUpToFirstM___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_batteries_MLList_takeUpToFirstM___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_takeUpToFirstM___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_takeUpToFirstM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_takeUpToFirst___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_takeUpToFirst(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_getLast_x3f_aux___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_getLast_x3f_aux___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_getLast_x3f_aux(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_getLast_x3f___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_getLast_x3f___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_getLast_x3f(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_getLast_x21___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_getLast_x21(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_foldM___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_foldM___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_foldM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_fold___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_fold(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_head___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_head___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_head(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_firstM___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_firstM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_first___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_first(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_instAlternativeMonadOfMonad___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_instAlternativeMonadOfMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_instAlternativeMonadOfMonad___redArg___lam__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_instAlternativeMonadOfMonad___redArg___lam__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_instAlternativeMonadOfMonad___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_instAlternativeMonadOfMonad___redArg___lam__5(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_instAlternativeMonadOfMonad___redArg___lam__5___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_instAlternativeMonadOfMonad___redArg___lam__6(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_instAlternativeMonadOfMonad___redArg___lam__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_instAlternativeMonadOfMonad___redArg___lam__8(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_instAlternativeMonadOfMonad___redArg___lam__8___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_instAlternativeMonadOfMonad___redArg___lam__9(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_instAlternativeMonadOfMonad___redArg___lam__10(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_instAlternativeMonadOfMonad___redArg___lam__11(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_batteries_MLList_instAlternativeMonadOfMonad___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries_MLList_instAlternativeMonadOfMonad___redArg___lam__2, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_MLList_instAlternativeMonadOfMonad___redArg___closed__0 = (const lean_object*)&lp_batteries_MLList_instAlternativeMonadOfMonad___redArg___closed__0_value;
static const lean_closure_object lp_batteries_MLList_instAlternativeMonadOfMonad___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries_MLList_nil, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_batteries_MLList_instAlternativeMonadOfMonad___redArg___closed__1 = (const lean_object*)&lp_batteries_MLList_instAlternativeMonadOfMonad___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_batteries_MLList_instAlternativeMonadOfMonad___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_instAlternativeMonadOfMonad(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_instMonadLiftOfMonad___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_instMonadLiftOfMonad___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_MLList_instMonadLiftOfMonad(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_MLListImpl_ctorIdx___redArg(lean_object* v_x_1_){
_start:
{
switch(lean_obj_tag(v_x_1_))
{
case 0:
{
lean_object* v___x_2_; 
v___x_2_ = lean_unsigned_to_nat(0u);
return v___x_2_;
}
case 1:
{
lean_object* v___x_3_; 
v___x_3_ = lean_unsigned_to_nat(1u);
return v___x_3_;
}
case 2:
{
lean_object* v___x_4_; 
v___x_4_ = lean_unsigned_to_nat(2u);
return v___x_4_;
}
default: 
{
lean_object* v___x_5_; 
v___x_5_ = lean_unsigned_to_nat(3u);
return v___x_5_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_MLListImpl_ctorIdx___redArg___boxed(lean_object* v_x_6_){
_start:
{
lean_object* v_res_7_; 
v_res_7_ = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_MLListImpl_ctorIdx___redArg(v_x_6_);
lean_dec(v_x_6_);
return v_res_7_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_MLListImpl_ctorIdx(lean_object* v_m_8_, lean_object* v_00_u03b1_9_, lean_object* v_x_10_){
_start:
{
lean_object* v___x_11_; 
v___x_11_ = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_MLListImpl_ctorIdx___redArg(v_x_10_);
return v___x_11_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_MLListImpl_ctorIdx___boxed(lean_object* v_m_12_, lean_object* v_00_u03b1_13_, lean_object* v_x_14_){
_start:
{
lean_object* v_res_15_; 
v_res_15_ = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_MLListImpl_ctorIdx(v_m_12_, v_00_u03b1_13_, v_x_14_);
lean_dec(v_x_14_);
return v_res_15_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_MLListImpl_ctorElim___redArg(lean_object* v_t_16_, lean_object* v_k_17_){
_start:
{
switch(lean_obj_tag(v_t_16_))
{
case 0:
{
return v_k_17_;
}
case 1:
{
lean_object* v_a_18_; lean_object* v_a_19_; lean_object* v___x_20_; 
v_a_18_ = lean_ctor_get(v_t_16_, 0);
lean_inc(v_a_18_);
v_a_19_ = lean_ctor_get(v_t_16_, 1);
lean_inc(v_a_19_);
lean_dec_ref_known(v_t_16_, 2);
v___x_20_ = lean_apply_2(v_k_17_, v_a_18_, v_a_19_);
return v___x_20_;
}
case 2:
{
lean_object* v_a_21_; lean_object* v___x_22_; 
v_a_21_ = lean_ctor_get(v_t_16_, 0);
lean_inc_ref(v_a_21_);
lean_dec_ref_known(v_t_16_, 1);
v___x_22_ = lean_apply_1(v_k_17_, v_a_21_);
return v___x_22_;
}
default: 
{
lean_object* v_a_23_; lean_object* v___x_24_; 
v_a_23_ = lean_ctor_get(v_t_16_, 0);
lean_inc(v_a_23_);
lean_dec_ref_known(v_t_16_, 1);
v___x_24_ = lean_apply_1(v_k_17_, v_a_23_);
return v___x_24_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_MLListImpl_ctorElim(lean_object* v_m_25_, lean_object* v_00_u03b1_26_, lean_object* v_motive__1_27_, lean_object* v_ctorIdx_28_, lean_object* v_t_29_, lean_object* v_h_30_, lean_object* v_k_31_){
_start:
{
lean_object* v___x_32_; 
v___x_32_ = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_MLListImpl_ctorElim___redArg(v_t_29_, v_k_31_);
return v___x_32_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_MLListImpl_ctorElim___boxed(lean_object* v_m_33_, lean_object* v_00_u03b1_34_, lean_object* v_motive__1_35_, lean_object* v_ctorIdx_36_, lean_object* v_t_37_, lean_object* v_h_38_, lean_object* v_k_39_){
_start:
{
lean_object* v_res_40_; 
v_res_40_ = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_MLListImpl_ctorElim(v_m_33_, v_00_u03b1_34_, v_motive__1_35_, v_ctorIdx_36_, v_t_37_, v_h_38_, v_k_39_);
lean_dec(v_ctorIdx_36_);
return v_res_40_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_MLListImpl_nil_elim___redArg(lean_object* v_t_41_, lean_object* v_nil_42_){
_start:
{
lean_object* v___x_43_; 
v___x_43_ = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_MLListImpl_ctorElim___redArg(v_t_41_, v_nil_42_);
return v___x_43_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_MLListImpl_nil_elim(lean_object* v_m_44_, lean_object* v_00_u03b1_45_, lean_object* v_motive__1_46_, lean_object* v_t_47_, lean_object* v_h_48_, lean_object* v_nil_49_){
_start:
{
lean_object* v___x_50_; 
v___x_50_ = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_MLListImpl_ctorElim___redArg(v_t_47_, v_nil_49_);
return v___x_50_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_MLListImpl_cons_elim___redArg(lean_object* v_t_51_, lean_object* v_cons_52_){
_start:
{
lean_object* v___x_53_; 
v___x_53_ = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_MLListImpl_ctorElim___redArg(v_t_51_, v_cons_52_);
return v___x_53_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_MLListImpl_cons_elim(lean_object* v_m_54_, lean_object* v_00_u03b1_55_, lean_object* v_motive__1_56_, lean_object* v_t_57_, lean_object* v_h_58_, lean_object* v_cons_59_){
_start:
{
lean_object* v___x_60_; 
v___x_60_ = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_MLListImpl_ctorElim___redArg(v_t_57_, v_cons_59_);
return v___x_60_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_MLListImpl_thunk_elim___redArg(lean_object* v_t_61_, lean_object* v_thunk_62_){
_start:
{
lean_object* v___x_63_; 
v___x_63_ = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_MLListImpl_ctorElim___redArg(v_t_61_, v_thunk_62_);
return v___x_63_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_MLListImpl_thunk_elim(lean_object* v_m_64_, lean_object* v_00_u03b1_65_, lean_object* v_motive__1_66_, lean_object* v_t_67_, lean_object* v_h_68_, lean_object* v_thunk_69_){
_start:
{
lean_object* v___x_70_; 
v___x_70_ = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_MLListImpl_ctorElim___redArg(v_t_67_, v_thunk_69_);
return v___x_70_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_MLListImpl_squash_elim___redArg(lean_object* v_t_71_, lean_object* v_squash_72_){
_start:
{
lean_object* v___x_73_; 
v___x_73_ = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_MLListImpl_ctorElim___redArg(v_t_71_, v_squash_72_);
return v___x_73_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_MLListImpl_squash_elim(lean_object* v_m_74_, lean_object* v_00_u03b1_75_, lean_object* v_motive__1_76_, lean_object* v_t_77_, lean_object* v_h_78_, lean_object* v_squash_79_){
_start:
{
lean_object* v___x_80_; 
v___x_80_ = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_MLListImpl_ctorElim___redArg(v_t_77_, v_squash_79_);
return v___x_80_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_unconsImpl___redArg(lean_object* v_inst_81_, lean_object* v_x_82_){
_start:
{
switch(lean_obj_tag(v_x_82_))
{
case 0:
{
lean_object* v_toApplicative_83_; lean_object* v_toPure_84_; lean_object* v___x_85_; lean_object* v___x_86_; 
v_toApplicative_83_ = lean_ctor_get(v_inst_81_, 0);
lean_inc_ref(v_toApplicative_83_);
lean_dec_ref(v_inst_81_);
v_toPure_84_ = lean_ctor_get(v_toApplicative_83_, 1);
lean_inc(v_toPure_84_);
lean_dec_ref(v_toApplicative_83_);
v___x_85_ = lean_box(0);
v___x_86_ = lean_apply_2(v_toPure_84_, lean_box(0), v___x_85_);
return v___x_86_;
}
case 1:
{
lean_object* v_toApplicative_87_; lean_object* v_toPure_88_; lean_object* v_a_89_; lean_object* v_a_90_; lean_object* v___x_92_; uint8_t v_isShared_93_; uint8_t v_isSharedCheck_99_; 
v_toApplicative_87_ = lean_ctor_get(v_inst_81_, 0);
lean_inc_ref(v_toApplicative_87_);
lean_dec_ref(v_inst_81_);
v_toPure_88_ = lean_ctor_get(v_toApplicative_87_, 1);
lean_inc(v_toPure_88_);
lean_dec_ref(v_toApplicative_87_);
v_a_89_ = lean_ctor_get(v_x_82_, 0);
v_a_90_ = lean_ctor_get(v_x_82_, 1);
v_isSharedCheck_99_ = !lean_is_exclusive(v_x_82_);
if (v_isSharedCheck_99_ == 0)
{
v___x_92_ = v_x_82_;
v_isShared_93_ = v_isSharedCheck_99_;
goto v_resetjp_91_;
}
else
{
lean_inc(v_a_90_);
lean_inc(v_a_89_);
lean_dec(v_x_82_);
v___x_92_ = lean_box(0);
v_isShared_93_ = v_isSharedCheck_99_;
goto v_resetjp_91_;
}
v_resetjp_91_:
{
lean_object* v___x_95_; 
if (v_isShared_93_ == 0)
{
lean_ctor_set_tag(v___x_92_, 0);
v___x_95_ = v___x_92_;
goto v_reusejp_94_;
}
else
{
lean_object* v_reuseFailAlloc_98_; 
v_reuseFailAlloc_98_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_98_, 0, v_a_89_);
lean_ctor_set(v_reuseFailAlloc_98_, 1, v_a_90_);
v___x_95_ = v_reuseFailAlloc_98_;
goto v_reusejp_94_;
}
v_reusejp_94_:
{
lean_object* v___x_96_; lean_object* v___x_97_; 
v___x_96_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_96_, 0, v___x_95_);
v___x_97_ = lean_apply_2(v_toPure_88_, lean_box(0), v___x_96_);
return v___x_97_;
}
}
}
case 2:
{
lean_object* v_a_100_; lean_object* v___x_101_; 
v_a_100_ = lean_ctor_get(v_x_82_, 0);
lean_inc_ref(v_a_100_);
lean_dec_ref_known(v_x_82_, 1);
v___x_101_ = lean_thunk_get_own(v_a_100_);
lean_dec_ref(v_a_100_);
v_x_82_ = v___x_101_;
goto _start;
}
default: 
{
lean_object* v_toBind_103_; lean_object* v_a_104_; lean_object* v___x_105_; lean_object* v___x_106_; lean_object* v___x_107_; lean_object* v___x_108_; 
v_toBind_103_ = lean_ctor_get(v_inst_81_, 1);
lean_inc(v_toBind_103_);
v_a_104_ = lean_ctor_get(v_x_82_, 0);
lean_inc(v_a_104_);
lean_dec_ref_known(v_x_82_, 1);
v___x_105_ = lean_box(0);
v___x_106_ = lean_apply_1(v_a_104_, v___x_105_);
v___x_107_ = lean_alloc_closure((void*)(lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_unconsImpl___redArg), 2, 1);
lean_closure_set(v___x_107_, 0, v_inst_81_);
v___x_108_ = lean_apply_4(v_toBind_103_, lean_box(0), lean_box(0), v___x_106_, v___x_107_);
return v___x_108_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_unconsImpl(lean_object* v_00_u03b1_109_, lean_object* v_m_110_, lean_object* v_inst_111_, lean_object* v_x_112_){
_start:
{
lean_object* v___x_113_; 
v___x_113_ = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_unconsImpl___redArg(v_inst_111_, v_x_112_);
return v___x_113_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_uncons_x3fImpl___redArg(lean_object* v_x_116_){
_start:
{
switch(lean_obj_tag(v_x_116_))
{
case 0:
{
lean_object* v___x_117_; 
v___x_117_ = ((lean_object*)(lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_uncons_x3fImpl___redArg___closed__0));
return v___x_117_;
}
case 1:
{
lean_object* v_a_118_; lean_object* v_a_119_; lean_object* v___x_121_; uint8_t v_isShared_122_; uint8_t v_isSharedCheck_128_; 
v_a_118_ = lean_ctor_get(v_x_116_, 0);
v_a_119_ = lean_ctor_get(v_x_116_, 1);
v_isSharedCheck_128_ = !lean_is_exclusive(v_x_116_);
if (v_isSharedCheck_128_ == 0)
{
v___x_121_ = v_x_116_;
v_isShared_122_ = v_isSharedCheck_128_;
goto v_resetjp_120_;
}
else
{
lean_inc(v_a_119_);
lean_inc(v_a_118_);
lean_dec(v_x_116_);
v___x_121_ = lean_box(0);
v_isShared_122_ = v_isSharedCheck_128_;
goto v_resetjp_120_;
}
v_resetjp_120_:
{
lean_object* v___x_124_; 
if (v_isShared_122_ == 0)
{
lean_ctor_set_tag(v___x_121_, 0);
v___x_124_ = v___x_121_;
goto v_reusejp_123_;
}
else
{
lean_object* v_reuseFailAlloc_127_; 
v_reuseFailAlloc_127_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_127_, 0, v_a_118_);
lean_ctor_set(v_reuseFailAlloc_127_, 1, v_a_119_);
v___x_124_ = v_reuseFailAlloc_127_;
goto v_reusejp_123_;
}
v_reusejp_123_:
{
lean_object* v___x_125_; lean_object* v___x_126_; 
v___x_125_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_125_, 0, v___x_124_);
v___x_126_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_126_, 0, v___x_125_);
return v___x_126_;
}
}
}
default: 
{
lean_object* v___x_129_; 
lean_dec(v_x_116_);
v___x_129_ = lean_box(0);
return v___x_129_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_uncons_x3fImpl(lean_object* v_m_130_, lean_object* v_00_u03b1_131_, lean_object* v_x_132_){
_start:
{
lean_object* v___x_133_; 
v___x_133_ = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_uncons_x3fImpl___redArg(v_x_132_);
return v___x_133_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_specImpl___lam__0(lean_object* v_00_u03b1_134_){
_start:
{
lean_object* v___x_135_; 
v___x_135_ = lean_box(0);
return v___x_135_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_specImpl___lam__1(lean_object* v_00_u03b1_136_, lean_object* v___y_137_, lean_object* v___y_138_){
_start:
{
lean_object* v___x_139_; 
v___x_139_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_139_, 0, v___y_137_);
lean_ctor_set(v___x_139_, 1, v___y_138_);
return v___x_139_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_specImpl___lam__2(lean_object* v_00_u03b1_140_, lean_object* v_f_141_){
_start:
{
lean_object* v___x_142_; lean_object* v___x_143_; 
v___x_142_ = lean_mk_thunk(v_f_141_);
v___x_143_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v___x_143_, 0, v___x_142_);
return v___x_143_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_specImpl___lam__3(lean_object* v_00_u03b1_144_, lean_object* v___y_145_){
_start:
{
lean_object* v___x_146_; 
v___x_146_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_146_, 0, v___y_145_);
return v___x_146_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_specImpl___lam__4(lean_object* v_00_u03b1_147_, lean_object* v_inst_148_, lean_object* v___y_149_){
_start:
{
lean_object* v___x_150_; 
v___x_150_ = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_unconsImpl___redArg(v_inst_148_, v___y_149_);
return v___x_150_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_specImpl(lean_object* v_m_164_){
_start:
{
lean_object* v___x_165_; 
v___x_165_ = ((lean_object*)(lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_specImpl___closed__6));
return v___x_165_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_nil(lean_object* v_m_166_, lean_object* v_00_u03b1_167_){
_start:
{
lean_object* v___x_168_; 
v___x_168_ = lean_box(0);
return v___x_168_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_cons___redArg(lean_object* v_a_169_, lean_object* v_a_170_){
_start:
{
lean_object* v___x_171_; 
v___x_171_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_171_, 0, v_a_169_);
lean_ctor_set(v___x_171_, 1, v_a_170_);
return v___x_171_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_cons(lean_object* v_00_u03b1_172_, lean_object* v_m_173_, lean_object* v_a_174_, lean_object* v_a_175_){
_start:
{
lean_object* v___x_176_; 
v___x_176_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_176_, 0, v_a_174_);
lean_ctor_set(v___x_176_, 1, v_a_175_);
return v___x_176_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_thunk___redArg(lean_object* v_a_177_){
_start:
{
lean_object* v___x_178_; lean_object* v___x_179_; 
v___x_178_ = lean_mk_thunk(v_a_177_);
v___x_179_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v___x_179_, 0, v___x_178_);
return v___x_179_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_thunk(lean_object* v_m_180_, lean_object* v_00_u03b1_181_, lean_object* v_a_182_){
_start:
{
lean_object* v___x_183_; lean_object* v___x_184_; 
v___x_183_ = lean_mk_thunk(v_a_182_);
v___x_184_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v___x_184_, 0, v___x_183_);
return v___x_184_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_squash___redArg(lean_object* v_a_185_){
_start:
{
lean_object* v___x_186_; 
v___x_186_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_186_, 0, v_a_185_);
return v___x_186_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_squash(lean_object* v_m_187_, lean_object* v_00_u03b1_188_, lean_object* v_a_189_){
_start:
{
lean_object* v___x_190_; 
v___x_190_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_190_, 0, v_a_189_);
return v___x_190_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_uncons___redArg(lean_object* v_inst_191_, lean_object* v_a_192_){
_start:
{
lean_object* v___x_193_; 
v___x_193_ = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_unconsImpl___redArg(v_inst_191_, v_a_192_);
return v___x_193_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_uncons(lean_object* v_m_194_, lean_object* v_00_u03b1_195_, lean_object* v_inst_196_, lean_object* v_a_197_){
_start:
{
lean_object* v___x_198_; 
v___x_198_ = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_unconsImpl___redArg(v_inst_196_, v_a_197_);
return v___x_198_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_uncons_x3f___redArg(lean_object* v_a_199_){
_start:
{
lean_object* v___x_200_; 
v___x_200_ = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_uncons_x3fImpl___redArg(v_a_199_);
return v___x_200_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_uncons_x3f(lean_object* v_m_201_, lean_object* v_00_u03b1_202_, lean_object* v_a_203_){
_start:
{
lean_object* v___x_204_; 
v___x_204_ = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_uncons_x3fImpl___redArg(v_a_203_);
return v___x_204_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_instEmptyCollection(lean_object* v_m_205_, lean_object* v_00_u03b1_206_){
_start:
{
lean_object* v___x_207_; 
v___x_207_ = lean_box(0);
return v___x_207_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_instInhabited(lean_object* v_m_208_, lean_object* v_00_u03b1_209_){
_start:
{
lean_object* v___x_210_; 
v___x_210_ = lean_box(0);
return v___x_210_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_instInhabitedForallForallForallForallForInStepOfMonad___redArg___lam__0(lean_object* v_toPure_211_, lean_object* v_d_212_, lean_object* v_x_213_){
_start:
{
lean_object* v___x_214_; 
v___x_214_ = lean_apply_2(v_toPure_211_, lean_box(0), v_d_212_);
return v___x_214_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_instInhabitedForallForallForallForallForInStepOfMonad___redArg___lam__0___boxed(lean_object* v_toPure_215_, lean_object* v_d_216_, lean_object* v_x_217_){
_start:
{
lean_object* v_res_218_; 
v_res_218_ = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_instInhabitedForallForallForallForallForInStepOfMonad___redArg___lam__0(v_toPure_215_, v_d_216_, v_x_217_);
lean_dec(v_x_217_);
return v_res_218_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_instInhabitedForallForallForallForallForInStepOfMonad___redArg(lean_object* v_inst_219_){
_start:
{
lean_object* v_toApplicative_220_; lean_object* v_toPure_221_; lean_object* v___f_222_; 
v_toApplicative_220_ = lean_ctor_get(v_inst_219_, 0);
lean_inc_ref(v_toApplicative_220_);
lean_dec_ref(v_inst_219_);
v_toPure_221_ = lean_ctor_get(v_toApplicative_220_, 1);
lean_inc(v_toPure_221_);
lean_dec_ref(v_toApplicative_220_);
v___f_222_ = lean_alloc_closure((void*)(lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_instInhabitedForallForallForallForallForInStepOfMonad___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_222_, 0, v_toPure_221_);
return v___f_222_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_instInhabitedForallForallForallForallForInStepOfMonad(lean_object* v_n_223_, lean_object* v_00_u03b4_224_, lean_object* v_00_u03b1_225_, lean_object* v_inst_226_){
_start:
{
lean_object* v___x_227_; 
v___x_227_ = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_instInhabitedForallForallForallForallForInStepOfMonad___redArg(v_inst_226_);
return v___x_227_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_forIn___redArg___lam__1(lean_object* v_toPure_228_, lean_object* v_init_229_, lean_object* v_inst_230_, lean_object* v_inst_231_, lean_object* v_inst_232_, lean_object* v_f_233_, lean_object* v_toBind_234_, lean_object* v_____do__lift_235_){
_start:
{
if (lean_obj_tag(v_____do__lift_235_) == 0)
{
lean_object* v___x_236_; 
lean_dec(v_toBind_234_);
lean_dec(v_f_233_);
lean_dec(v_inst_232_);
lean_dec_ref(v_inst_231_);
lean_dec_ref(v_inst_230_);
v___x_236_ = lean_apply_2(v_toPure_228_, lean_box(0), v_init_229_);
return v___x_236_;
}
else
{
lean_object* v_val_237_; lean_object* v_fst_238_; lean_object* v_snd_239_; lean_object* v___f_240_; lean_object* v___x_241_; lean_object* v___x_242_; 
v_val_237_ = lean_ctor_get(v_____do__lift_235_, 0);
lean_inc(v_val_237_);
lean_dec_ref_known(v_____do__lift_235_, 1);
v_fst_238_ = lean_ctor_get(v_val_237_, 0);
lean_inc(v_fst_238_);
v_snd_239_ = lean_ctor_get(v_val_237_, 1);
lean_inc(v_snd_239_);
lean_dec(v_val_237_);
lean_inc(v_f_233_);
v___f_240_ = lean_alloc_closure((void*)(lp_batteries_MLList_forIn___redArg___lam__0), 7, 6);
lean_closure_set(v___f_240_, 0, v_toPure_228_);
lean_closure_set(v___f_240_, 1, v_inst_230_);
lean_closure_set(v___f_240_, 2, v_inst_231_);
lean_closure_set(v___f_240_, 3, v_inst_232_);
lean_closure_set(v___f_240_, 4, v_snd_239_);
lean_closure_set(v___f_240_, 5, v_f_233_);
v___x_241_ = lean_apply_2(v_f_233_, v_fst_238_, v_init_229_);
v___x_242_ = lean_apply_4(v_toBind_234_, lean_box(0), lean_box(0), v___x_241_, v___f_240_);
return v___x_242_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_forIn___redArg(lean_object* v_inst_243_, lean_object* v_inst_244_, lean_object* v_inst_245_, lean_object* v_as_246_, lean_object* v_init_247_, lean_object* v_f_248_){
_start:
{
lean_object* v_toApplicative_249_; lean_object* v_toBind_250_; lean_object* v_toPure_251_; lean_object* v___x_252_; lean_object* v___x_253_; lean_object* v___f_254_; lean_object* v___x_255_; 
v_toApplicative_249_ = lean_ctor_get(v_inst_244_, 0);
v_toBind_250_ = lean_ctor_get(v_inst_244_, 1);
lean_inc_n(v_toBind_250_, 2);
v_toPure_251_ = lean_ctor_get(v_toApplicative_249_, 1);
lean_inc(v_toPure_251_);
lean_inc_ref(v_inst_243_);
v___x_252_ = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_unconsImpl___redArg(v_inst_243_, v_as_246_);
lean_inc(v_inst_245_);
v___x_253_ = lean_apply_2(v_inst_245_, lean_box(0), v___x_252_);
v___f_254_ = lean_alloc_closure((void*)(lp_batteries_MLList_forIn___redArg___lam__1), 8, 7);
lean_closure_set(v___f_254_, 0, v_toPure_251_);
lean_closure_set(v___f_254_, 1, v_init_247_);
lean_closure_set(v___f_254_, 2, v_inst_243_);
lean_closure_set(v___f_254_, 3, v_inst_244_);
lean_closure_set(v___f_254_, 4, v_inst_245_);
lean_closure_set(v___f_254_, 5, v_f_248_);
lean_closure_set(v___f_254_, 6, v_toBind_250_);
v___x_255_ = lean_apply_4(v_toBind_250_, lean_box(0), lean_box(0), v___x_253_, v___f_254_);
return v___x_255_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_forIn___redArg___lam__0(lean_object* v_toPure_256_, lean_object* v_inst_257_, lean_object* v_inst_258_, lean_object* v_inst_259_, lean_object* v_snd_260_, lean_object* v_f_261_, lean_object* v_____do__lift_262_){
_start:
{
if (lean_obj_tag(v_____do__lift_262_) == 0)
{
lean_object* v_a_263_; lean_object* v___x_264_; 
lean_dec(v_f_261_);
lean_dec(v_snd_260_);
lean_dec(v_inst_259_);
lean_dec_ref(v_inst_258_);
lean_dec_ref(v_inst_257_);
v_a_263_ = lean_ctor_get(v_____do__lift_262_, 0);
lean_inc(v_a_263_);
lean_dec_ref_known(v_____do__lift_262_, 1);
v___x_264_ = lean_apply_2(v_toPure_256_, lean_box(0), v_a_263_);
return v___x_264_;
}
else
{
lean_object* v_a_265_; lean_object* v___x_266_; 
lean_dec(v_toPure_256_);
v_a_265_ = lean_ctor_get(v_____do__lift_262_, 0);
lean_inc(v_a_265_);
lean_dec_ref_known(v_____do__lift_262_, 1);
v___x_266_ = lp_batteries_MLList_forIn___redArg(v_inst_257_, v_inst_258_, v_inst_259_, v_snd_260_, v_a_265_, v_f_261_);
return v___x_266_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_forIn(lean_object* v_m_267_, lean_object* v_n_268_, lean_object* v_00_u03b1_269_, lean_object* v_00_u03b4_270_, lean_object* v_inst_271_, lean_object* v_inst_272_, lean_object* v_inst_273_, lean_object* v_as_274_, lean_object* v_init_275_, lean_object* v_f_276_){
_start:
{
lean_object* v___x_277_; 
v___x_277_ = lp_batteries_MLList_forIn___redArg(v_inst_271_, v_inst_272_, v_inst_273_, v_as_274_, v_init_275_, v_f_276_);
return v___x_277_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_instForInOfMonadOfMonadLiftT___redArg___lam__0(lean_object* v_inst_278_, lean_object* v_inst_279_, lean_object* v_inst_280_, lean_object* v_00_u03b2_281_, lean_object* v___y_282_, lean_object* v___y_283_, lean_object* v___y_284_){
_start:
{
lean_object* v___x_285_; 
v___x_285_ = lp_batteries_MLList_forIn___redArg(v_inst_278_, v_inst_279_, v_inst_280_, v___y_282_, v___y_283_, v___y_284_);
return v___x_285_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_instForInOfMonadOfMonadLiftT___redArg(lean_object* v_inst_286_, lean_object* v_inst_287_, lean_object* v_inst_288_){
_start:
{
lean_object* v___f_289_; 
v___f_289_ = lean_alloc_closure((void*)(lp_batteries_MLList_instForInOfMonadOfMonadLiftT___redArg___lam__0), 7, 3);
lean_closure_set(v___f_289_, 0, v_inst_286_);
lean_closure_set(v___f_289_, 1, v_inst_287_);
lean_closure_set(v___f_289_, 2, v_inst_288_);
return v___f_289_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_instForInOfMonadOfMonadLiftT(lean_object* v_m_290_, lean_object* v_n_291_, lean_object* v_00_u03b1_292_, lean_object* v_inst_293_, lean_object* v_inst_294_, lean_object* v_inst_295_){
_start:
{
lean_object* v___f_296_; 
v___f_296_ = lean_alloc_closure((void*)(lp_batteries_MLList_instForInOfMonadOfMonadLiftT___redArg___lam__0), 7, 3);
lean_closure_set(v___f_296_, 0, v_inst_293_);
lean_closure_set(v___f_296_, 1, v_inst_294_);
lean_closure_set(v___f_296_, 2, v_inst_295_);
return v___f_296_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_singletonM___redArg___lam__0(lean_object* v_toPure_297_, lean_object* v_____do__lift_298_){
_start:
{
lean_object* v___x_299_; lean_object* v___x_300_; lean_object* v___x_301_; 
v___x_299_ = lean_box(0);
v___x_300_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_300_, 0, v_____do__lift_298_);
lean_ctor_set(v___x_300_, 1, v___x_299_);
v___x_301_ = lean_apply_2(v_toPure_297_, lean_box(0), v___x_300_);
return v___x_301_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_singletonM___redArg___lam__1(lean_object* v_toBind_302_, lean_object* v_x_303_, lean_object* v___f_304_, lean_object* v_x_305_){
_start:
{
lean_object* v___x_306_; 
v___x_306_ = lean_apply_4(v_toBind_302_, lean_box(0), lean_box(0), v_x_303_, v___f_304_);
return v___x_306_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_singletonM___redArg(lean_object* v_inst_307_, lean_object* v_x_308_){
_start:
{
lean_object* v_toApplicative_309_; lean_object* v_toBind_310_; lean_object* v_toPure_311_; lean_object* v___f_312_; lean_object* v___f_313_; lean_object* v___x_314_; 
v_toApplicative_309_ = lean_ctor_get(v_inst_307_, 0);
lean_inc_ref(v_toApplicative_309_);
v_toBind_310_ = lean_ctor_get(v_inst_307_, 1);
lean_inc(v_toBind_310_);
lean_dec_ref(v_inst_307_);
v_toPure_311_ = lean_ctor_get(v_toApplicative_309_, 1);
lean_inc(v_toPure_311_);
lean_dec_ref(v_toApplicative_309_);
v___f_312_ = lean_alloc_closure((void*)(lp_batteries_MLList_singletonM___redArg___lam__0), 2, 1);
lean_closure_set(v___f_312_, 0, v_toPure_311_);
v___f_313_ = lean_alloc_closure((void*)(lp_batteries_MLList_singletonM___redArg___lam__1), 4, 3);
lean_closure_set(v___f_313_, 0, v_toBind_310_);
lean_closure_set(v___f_313_, 1, v_x_308_);
lean_closure_set(v___f_313_, 2, v___f_312_);
v___x_314_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_314_, 0, v___f_313_);
return v___x_314_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_singletonM(lean_object* v_m_315_, lean_object* v_00_u03b1_316_, lean_object* v_inst_317_, lean_object* v_x_318_){
_start:
{
lean_object* v___x_319_; 
v___x_319_ = lp_batteries_MLList_singletonM___redArg(v_inst_317_, v_x_318_);
return v___x_319_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_singleton___redArg(lean_object* v_inst_320_, lean_object* v_x_321_){
_start:
{
lean_object* v_toApplicative_322_; lean_object* v_toPure_323_; lean_object* v___x_324_; lean_object* v___x_325_; 
v_toApplicative_322_ = lean_ctor_get(v_inst_320_, 0);
v_toPure_323_ = lean_ctor_get(v_toApplicative_322_, 1);
lean_inc(v_toPure_323_);
v___x_324_ = lean_apply_2(v_toPure_323_, lean_box(0), v_x_321_);
v___x_325_ = lp_batteries_MLList_singletonM___redArg(v_inst_320_, v___x_324_);
return v___x_325_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_singleton(lean_object* v_m_326_, lean_object* v_00_u03b1_327_, lean_object* v_inst_328_, lean_object* v_x_329_){
_start:
{
lean_object* v___x_330_; 
v___x_330_ = lp_batteries_MLList_singleton___redArg(v_inst_328_, v_x_329_);
return v___x_330_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_fix___redArg(lean_object* v_inst_331_, lean_object* v_f_332_, lean_object* v_x_333_){
_start:
{
lean_object* v_toApplicative_334_; lean_object* v_toFunctor_335_; lean_object* v___f_336_; lean_object* v___x_337_; lean_object* v___x_338_; 
v_toApplicative_334_ = lean_ctor_get(v_inst_331_, 0);
v_toFunctor_335_ = lean_ctor_get(v_toApplicative_334_, 0);
lean_inc_ref(v_toFunctor_335_);
lean_inc(v_x_333_);
v___f_336_ = lean_alloc_closure((void*)(lp_batteries_MLList_fix___redArg___lam__0), 5, 4);
lean_closure_set(v___f_336_, 0, v_toFunctor_335_);
lean_closure_set(v___f_336_, 1, v_inst_331_);
lean_closure_set(v___f_336_, 2, v_f_332_);
lean_closure_set(v___f_336_, 3, v_x_333_);
v___x_337_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_337_, 0, v___f_336_);
v___x_338_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_338_, 0, v_x_333_);
lean_ctor_set(v___x_338_, 1, v___x_337_);
return v___x_338_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_fix___redArg___lam__0(lean_object* v_toFunctor_339_, lean_object* v_inst_340_, lean_object* v_f_341_, lean_object* v_x_342_, lean_object* v_x_343_){
_start:
{
lean_object* v_map_344_; lean_object* v___x_345_; lean_object* v___x_346_; lean_object* v___x_347_; 
v_map_344_ = lean_ctor_get(v_toFunctor_339_, 0);
lean_inc(v_map_344_);
lean_dec_ref(v_toFunctor_339_);
lean_inc(v_f_341_);
v___x_345_ = lean_alloc_closure((void*)(lp_batteries_MLList_fix___redArg), 3, 2);
lean_closure_set(v___x_345_, 0, v_inst_340_);
lean_closure_set(v___x_345_, 1, v_f_341_);
v___x_346_ = lean_apply_1(v_f_341_, v_x_342_);
v___x_347_ = lean_apply_4(v_map_344_, lean_box(0), lean_box(0), v___x_345_, v___x_346_);
return v___x_347_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_fix(lean_object* v_m_348_, lean_object* v_00_u03b1_349_, lean_object* v_inst_350_, lean_object* v_f_351_, lean_object* v_x_352_){
_start:
{
lean_object* v___x_353_; 
v___x_353_ = lp_batteries_MLList_fix___redArg(v_inst_350_, v_f_351_, v_x_352_);
return v___x_353_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_fix_x3f_x27___redArg___lam__1(lean_object* v_f_354_, lean_object* v_init_355_, lean_object* v_toBind_356_, lean_object* v___f_357_, lean_object* v_x_358_){
_start:
{
lean_object* v___x_359_; lean_object* v___x_360_; 
v___x_359_ = lean_apply_1(v_f_354_, v_init_355_);
v___x_360_ = lean_apply_4(v_toBind_356_, lean_box(0), lean_box(0), v___x_359_, v___f_357_);
return v___x_360_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_fix_x3f_x27___redArg___lam__0(lean_object* v_toPure_361_, lean_object* v_inst_362_, lean_object* v_f_363_, lean_object* v_____do__lift_364_){
_start:
{
if (lean_obj_tag(v_____do__lift_364_) == 0)
{
lean_object* v___x_365_; lean_object* v___x_366_; 
lean_dec(v_f_363_);
lean_dec_ref(v_inst_362_);
v___x_365_ = lean_box(0);
v___x_366_ = lean_apply_2(v_toPure_361_, lean_box(0), v___x_365_);
return v___x_366_;
}
else
{
lean_object* v_val_367_; lean_object* v_fst_368_; lean_object* v_snd_369_; lean_object* v___x_371_; uint8_t v_isShared_372_; uint8_t v_isSharedCheck_378_; 
v_val_367_ = lean_ctor_get(v_____do__lift_364_, 0);
lean_inc(v_val_367_);
lean_dec_ref_known(v_____do__lift_364_, 1);
v_fst_368_ = lean_ctor_get(v_val_367_, 0);
v_snd_369_ = lean_ctor_get(v_val_367_, 1);
v_isSharedCheck_378_ = !lean_is_exclusive(v_val_367_);
if (v_isSharedCheck_378_ == 0)
{
v___x_371_ = v_val_367_;
v_isShared_372_ = v_isSharedCheck_378_;
goto v_resetjp_370_;
}
else
{
lean_inc(v_snd_369_);
lean_inc(v_fst_368_);
lean_dec(v_val_367_);
v___x_371_ = lean_box(0);
v_isShared_372_ = v_isSharedCheck_378_;
goto v_resetjp_370_;
}
v_resetjp_370_:
{
lean_object* v___x_373_; lean_object* v___x_375_; 
v___x_373_ = lp_batteries_MLList_fix_x3f_x27___redArg(v_inst_362_, v_f_363_, v_snd_369_);
if (v_isShared_372_ == 0)
{
lean_ctor_set_tag(v___x_371_, 1);
lean_ctor_set(v___x_371_, 1, v___x_373_);
v___x_375_ = v___x_371_;
goto v_reusejp_374_;
}
else
{
lean_object* v_reuseFailAlloc_377_; 
v_reuseFailAlloc_377_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_377_, 0, v_fst_368_);
lean_ctor_set(v_reuseFailAlloc_377_, 1, v___x_373_);
v___x_375_ = v_reuseFailAlloc_377_;
goto v_reusejp_374_;
}
v_reusejp_374_:
{
lean_object* v___x_376_; 
v___x_376_ = lean_apply_2(v_toPure_361_, lean_box(0), v___x_375_);
return v___x_376_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_fix_x3f_x27___redArg(lean_object* v_inst_379_, lean_object* v_f_380_, lean_object* v_init_381_){
_start:
{
lean_object* v_toApplicative_382_; lean_object* v_toBind_383_; lean_object* v_toPure_384_; lean_object* v___f_385_; lean_object* v___f_386_; lean_object* v___x_387_; 
v_toApplicative_382_ = lean_ctor_get(v_inst_379_, 0);
v_toBind_383_ = lean_ctor_get(v_inst_379_, 1);
lean_inc(v_toBind_383_);
v_toPure_384_ = lean_ctor_get(v_toApplicative_382_, 1);
lean_inc(v_toPure_384_);
lean_inc(v_f_380_);
v___f_385_ = lean_alloc_closure((void*)(lp_batteries_MLList_fix_x3f_x27___redArg___lam__0), 4, 3);
lean_closure_set(v___f_385_, 0, v_toPure_384_);
lean_closure_set(v___f_385_, 1, v_inst_379_);
lean_closure_set(v___f_385_, 2, v_f_380_);
v___f_386_ = lean_alloc_closure((void*)(lp_batteries_MLList_fix_x3f_x27___redArg___lam__1), 5, 4);
lean_closure_set(v___f_386_, 0, v_f_380_);
lean_closure_set(v___f_386_, 1, v_init_381_);
lean_closure_set(v___f_386_, 2, v_toBind_383_);
lean_closure_set(v___f_386_, 3, v___f_385_);
v___x_387_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_387_, 0, v___f_386_);
return v___x_387_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_fix_x3f_x27(lean_object* v_m_388_, lean_object* v_00_u03b1_389_, lean_object* v_00_u03b2_390_, lean_object* v_inst_391_, lean_object* v_f_392_, lean_object* v_init_393_){
_start:
{
lean_object* v___x_394_; 
v___x_394_ = lp_batteries_MLList_fix_x3f_x27___redArg(v_inst_391_, v_f_392_, v_init_393_);
return v___x_394_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_fix_x3f___redArg___lam__1(lean_object* v_f_395_, lean_object* v_x_396_, lean_object* v_toBind_397_, lean_object* v___f_398_, lean_object* v_x_399_){
_start:
{
lean_object* v___x_400_; lean_object* v___x_401_; 
v___x_400_ = lean_apply_1(v_f_395_, v_x_396_);
v___x_401_ = lean_apply_4(v_toBind_397_, lean_box(0), lean_box(0), v___x_400_, v___f_398_);
return v___x_401_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_fix_x3f___redArg(lean_object* v_inst_402_, lean_object* v_f_403_, lean_object* v_x_404_){
_start:
{
lean_object* v_toApplicative_405_; lean_object* v_toBind_406_; lean_object* v_toPure_407_; lean_object* v___f_408_; lean_object* v___f_409_; lean_object* v___x_410_; lean_object* v___x_411_; 
v_toApplicative_405_ = lean_ctor_get(v_inst_402_, 0);
v_toBind_406_ = lean_ctor_get(v_inst_402_, 1);
lean_inc(v_toBind_406_);
v_toPure_407_ = lean_ctor_get(v_toApplicative_405_, 1);
lean_inc(v_toPure_407_);
lean_inc(v_f_403_);
v___f_408_ = lean_alloc_closure((void*)(lp_batteries_MLList_fix_x3f___redArg___lam__0), 4, 3);
lean_closure_set(v___f_408_, 0, v_toPure_407_);
lean_closure_set(v___f_408_, 1, v_inst_402_);
lean_closure_set(v___f_408_, 2, v_f_403_);
lean_inc(v_x_404_);
v___f_409_ = lean_alloc_closure((void*)(lp_batteries_MLList_fix_x3f___redArg___lam__1), 5, 4);
lean_closure_set(v___f_409_, 0, v_f_403_);
lean_closure_set(v___f_409_, 1, v_x_404_);
lean_closure_set(v___f_409_, 2, v_toBind_406_);
lean_closure_set(v___f_409_, 3, v___f_408_);
v___x_410_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_410_, 0, v___f_409_);
v___x_411_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_411_, 0, v_x_404_);
lean_ctor_set(v___x_411_, 1, v___x_410_);
return v___x_411_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_fix_x3f___redArg___lam__0(lean_object* v_toPure_412_, lean_object* v_inst_413_, lean_object* v_f_414_, lean_object* v_____do__lift_415_){
_start:
{
if (lean_obj_tag(v_____do__lift_415_) == 0)
{
lean_object* v___x_416_; lean_object* v___x_417_; 
lean_dec(v_f_414_);
lean_dec_ref(v_inst_413_);
v___x_416_ = lean_box(0);
v___x_417_ = lean_apply_2(v_toPure_412_, lean_box(0), v___x_416_);
return v___x_417_;
}
else
{
lean_object* v_val_418_; lean_object* v___x_419_; lean_object* v___x_420_; 
v_val_418_ = lean_ctor_get(v_____do__lift_415_, 0);
lean_inc(v_val_418_);
lean_dec_ref_known(v_____do__lift_415_, 1);
v___x_419_ = lp_batteries_MLList_fix_x3f___redArg(v_inst_413_, v_f_414_, v_val_418_);
v___x_420_ = lean_apply_2(v_toPure_412_, lean_box(0), v___x_419_);
return v___x_420_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_fix_x3f(lean_object* v_m_421_, lean_object* v_00_u03b1_422_, lean_object* v_inst_423_, lean_object* v_f_424_, lean_object* v_x_425_){
_start:
{
lean_object* v___x_426_; 
v___x_426_ = lp_batteries_MLList_fix_x3f___redArg(v_inst_423_, v_f_424_, v_x_425_);
return v___x_426_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_iterate___redArg___lam__1(lean_object* v_toBind_427_, lean_object* v_f_428_, lean_object* v___f_429_, lean_object* v_x_430_){
_start:
{
lean_object* v___x_431_; 
v___x_431_ = lean_apply_4(v_toBind_427_, lean_box(0), lean_box(0), v_f_428_, v___f_429_);
return v___x_431_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_iterate___redArg(lean_object* v_inst_432_, lean_object* v_f_433_){
_start:
{
lean_object* v_toApplicative_434_; lean_object* v_toBind_435_; lean_object* v_toPure_436_; lean_object* v___f_437_; lean_object* v___f_438_; lean_object* v___x_439_; 
v_toApplicative_434_ = lean_ctor_get(v_inst_432_, 0);
v_toBind_435_ = lean_ctor_get(v_inst_432_, 1);
lean_inc(v_toBind_435_);
v_toPure_436_ = lean_ctor_get(v_toApplicative_434_, 1);
lean_inc(v_toPure_436_);
lean_inc(v_f_433_);
v___f_437_ = lean_alloc_closure((void*)(lp_batteries_MLList_iterate___redArg___lam__0), 4, 3);
lean_closure_set(v___f_437_, 0, v_inst_432_);
lean_closure_set(v___f_437_, 1, v_f_433_);
lean_closure_set(v___f_437_, 2, v_toPure_436_);
v___f_438_ = lean_alloc_closure((void*)(lp_batteries_MLList_iterate___redArg___lam__1), 4, 3);
lean_closure_set(v___f_438_, 0, v_toBind_435_);
lean_closure_set(v___f_438_, 1, v_f_433_);
lean_closure_set(v___f_438_, 2, v___f_437_);
v___x_439_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_439_, 0, v___f_438_);
return v___x_439_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_iterate___redArg___lam__0(lean_object* v_inst_440_, lean_object* v_f_441_, lean_object* v_toPure_442_, lean_object* v_____do__lift_443_){
_start:
{
lean_object* v___x_444_; lean_object* v___x_445_; lean_object* v___x_446_; 
v___x_444_ = lp_batteries_MLList_iterate___redArg(v_inst_440_, v_f_441_);
v___x_445_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_445_, 0, v_____do__lift_443_);
lean_ctor_set(v___x_445_, 1, v___x_444_);
v___x_446_ = lean_apply_2(v_toPure_442_, lean_box(0), v___x_445_);
return v___x_446_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_iterate(lean_object* v_m_447_, lean_object* v_00_u03b1_448_, lean_object* v_inst_449_, lean_object* v_f_450_){
_start:
{
lean_object* v___x_451_; 
v___x_451_ = lp_batteries_MLList_iterate___redArg(v_inst_449_, v_f_450_);
return v___x_451_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_fixlWith___redArg___lam__1(lean_object* v_f_452_, lean_object* v_s_453_, lean_object* v_toBind_454_, lean_object* v___f_455_, lean_object* v_x_456_){
_start:
{
lean_object* v___x_457_; lean_object* v___x_458_; 
v___x_457_ = lean_apply_1(v_f_452_, v_s_453_);
v___x_458_ = lean_apply_4(v_toBind_454_, lean_box(0), lean_box(0), v___x_457_, v___f_455_);
return v___x_458_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_fixlWith___redArg___lam__0(lean_object* v_inst_459_, lean_object* v_f_460_, lean_object* v_toPure_461_, lean_object* v_____x_462_){
_start:
{
lean_object* v_snd_463_; 
v_snd_463_ = lean_ctor_get(v_____x_462_, 1);
lean_inc(v_snd_463_);
if (lean_obj_tag(v_snd_463_) == 0)
{
lean_object* v_fst_464_; lean_object* v___x_465_; lean_object* v___x_466_; 
v_fst_464_ = lean_ctor_get(v_____x_462_, 0);
lean_inc(v_fst_464_);
lean_dec_ref(v_____x_462_);
v___x_465_ = lp_batteries_MLList_fixlWith___redArg(v_inst_459_, v_f_460_, v_fst_464_, v_snd_463_);
v___x_466_ = lean_apply_2(v_toPure_461_, lean_box(0), v___x_465_);
return v___x_466_;
}
else
{
lean_object* v_fst_467_; lean_object* v_head_468_; lean_object* v_tail_469_; lean_object* v___x_471_; uint8_t v_isShared_472_; uint8_t v_isSharedCheck_478_; 
v_fst_467_ = lean_ctor_get(v_____x_462_, 0);
lean_inc(v_fst_467_);
lean_dec_ref(v_____x_462_);
v_head_468_ = lean_ctor_get(v_snd_463_, 0);
v_tail_469_ = lean_ctor_get(v_snd_463_, 1);
v_isSharedCheck_478_ = !lean_is_exclusive(v_snd_463_);
if (v_isSharedCheck_478_ == 0)
{
v___x_471_ = v_snd_463_;
v_isShared_472_ = v_isSharedCheck_478_;
goto v_resetjp_470_;
}
else
{
lean_inc(v_tail_469_);
lean_inc(v_head_468_);
lean_dec(v_snd_463_);
v___x_471_ = lean_box(0);
v_isShared_472_ = v_isSharedCheck_478_;
goto v_resetjp_470_;
}
v_resetjp_470_:
{
lean_object* v___x_473_; lean_object* v___x_475_; 
v___x_473_ = lp_batteries_MLList_fixlWith___redArg(v_inst_459_, v_f_460_, v_fst_467_, v_tail_469_);
if (v_isShared_472_ == 0)
{
lean_ctor_set(v___x_471_, 1, v___x_473_);
v___x_475_ = v___x_471_;
goto v_reusejp_474_;
}
else
{
lean_object* v_reuseFailAlloc_477_; 
v_reuseFailAlloc_477_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_477_, 0, v_head_468_);
lean_ctor_set(v_reuseFailAlloc_477_, 1, v___x_473_);
v___x_475_ = v_reuseFailAlloc_477_;
goto v_reusejp_474_;
}
v_reusejp_474_:
{
lean_object* v___x_476_; 
v___x_476_ = lean_apply_2(v_toPure_461_, lean_box(0), v___x_475_);
return v___x_476_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_fixlWith___redArg(lean_object* v_inst_479_, lean_object* v_f_480_, lean_object* v_s_481_, lean_object* v_l_482_){
_start:
{
lean_object* v_toApplicative_483_; lean_object* v_toBind_484_; lean_object* v_toPure_485_; lean_object* v___f_486_; lean_object* v___f_487_; lean_object* v___f_488_; lean_object* v___x_489_; lean_object* v___x_490_; 
v_toApplicative_483_ = lean_ctor_get(v_inst_479_, 0);
v_toBind_484_ = lean_ctor_get(v_inst_479_, 1);
v_toPure_485_ = lean_ctor_get(v_toApplicative_483_, 1);
lean_inc(v_toPure_485_);
lean_inc_n(v_f_480_, 2);
lean_inc_ref(v_inst_479_);
v___f_486_ = lean_alloc_closure((void*)(lp_batteries_MLList_fixlWith___redArg___lam__0), 4, 3);
lean_closure_set(v___f_486_, 0, v_inst_479_);
lean_closure_set(v___f_486_, 1, v_f_480_);
lean_closure_set(v___f_486_, 2, v_toPure_485_);
lean_inc(v_toBind_484_);
lean_inc(v_s_481_);
v___f_487_ = lean_alloc_closure((void*)(lp_batteries_MLList_fixlWith___redArg___lam__1), 5, 4);
lean_closure_set(v___f_487_, 0, v_f_480_);
lean_closure_set(v___f_487_, 1, v_s_481_);
lean_closure_set(v___f_487_, 2, v_toBind_484_);
lean_closure_set(v___f_487_, 3, v___f_486_);
v___f_488_ = lean_alloc_closure((void*)(lp_batteries_MLList_fixlWith___redArg___lam__2), 6, 5);
lean_closure_set(v___f_488_, 0, v_l_482_);
lean_closure_set(v___f_488_, 1, v___f_487_);
lean_closure_set(v___f_488_, 2, v_inst_479_);
lean_closure_set(v___f_488_, 3, v_f_480_);
lean_closure_set(v___f_488_, 4, v_s_481_);
v___x_489_ = lean_mk_thunk(v___f_488_);
v___x_490_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v___x_490_, 0, v___x_489_);
return v___x_490_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_fixlWith___redArg___lam__2(lean_object* v_l_491_, lean_object* v___f_492_, lean_object* v_inst_493_, lean_object* v_f_494_, lean_object* v_s_495_, lean_object* v_x_496_){
_start:
{
if (lean_obj_tag(v_l_491_) == 0)
{
lean_object* v___x_497_; 
lean_dec(v_s_495_);
lean_dec(v_f_494_);
lean_dec_ref(v_inst_493_);
v___x_497_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_497_, 0, v___f_492_);
return v___x_497_;
}
else
{
lean_object* v_head_498_; lean_object* v_tail_499_; lean_object* v___x_501_; uint8_t v_isShared_502_; uint8_t v_isSharedCheck_507_; 
lean_dec(v___f_492_);
v_head_498_ = lean_ctor_get(v_l_491_, 0);
v_tail_499_ = lean_ctor_get(v_l_491_, 1);
v_isSharedCheck_507_ = !lean_is_exclusive(v_l_491_);
if (v_isSharedCheck_507_ == 0)
{
v___x_501_ = v_l_491_;
v_isShared_502_ = v_isSharedCheck_507_;
goto v_resetjp_500_;
}
else
{
lean_inc(v_tail_499_);
lean_inc(v_head_498_);
lean_dec(v_l_491_);
v___x_501_ = lean_box(0);
v_isShared_502_ = v_isSharedCheck_507_;
goto v_resetjp_500_;
}
v_resetjp_500_:
{
lean_object* v___x_503_; lean_object* v___x_505_; 
v___x_503_ = lp_batteries_MLList_fixlWith___redArg(v_inst_493_, v_f_494_, v_s_495_, v_tail_499_);
if (v_isShared_502_ == 0)
{
lean_ctor_set(v___x_501_, 1, v___x_503_);
v___x_505_ = v___x_501_;
goto v_reusejp_504_;
}
else
{
lean_object* v_reuseFailAlloc_506_; 
v_reuseFailAlloc_506_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_506_, 0, v_head_498_);
lean_ctor_set(v_reuseFailAlloc_506_, 1, v___x_503_);
v___x_505_ = v_reuseFailAlloc_506_;
goto v_reusejp_504_;
}
v_reusejp_504_:
{
return v___x_505_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_fixlWith(lean_object* v_m_508_, lean_object* v_inst_509_, lean_object* v_00_u03b1_510_, lean_object* v_00_u03b2_511_, lean_object* v_f_512_, lean_object* v_s_513_, lean_object* v_l_514_){
_start:
{
lean_object* v___x_515_; 
v___x_515_ = lp_batteries_MLList_fixlWith___redArg(v_inst_509_, v_f_512_, v_s_513_, v_l_514_);
return v___x_515_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_fixl___redArg(lean_object* v_inst_516_, lean_object* v_f_517_, lean_object* v_s_518_){
_start:
{
lean_object* v___x_519_; lean_object* v___x_520_; 
v___x_519_ = lean_box(0);
v___x_520_ = lp_batteries_MLList_fixlWith___redArg(v_inst_516_, v_f_517_, v_s_518_, v___x_519_);
return v___x_520_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_fixl(lean_object* v_m_521_, lean_object* v_inst_522_, lean_object* v_00_u03b1_523_, lean_object* v_00_u03b2_524_, lean_object* v_f_525_, lean_object* v_s_526_){
_start:
{
lean_object* v___x_527_; 
v___x_527_ = lp_batteries_MLList_fixl___redArg(v_inst_522_, v_f_525_, v_s_526_);
return v___x_527_;
}
}
LEAN_EXPORT uint8_t lp_batteries_MLList_isEmpty___redArg___lam__0(uint8_t v_down_528_){
_start:
{
return v_down_528_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_isEmpty___redArg___lam__0___boxed(lean_object* v_down_529_){
_start:
{
uint8_t v_down_boxed_530_; uint8_t v_res_531_; lean_object* v_r_532_; 
v_down_boxed_530_ = lean_unbox(v_down_529_);
v_res_531_ = lp_batteries_MLList_isEmpty___redArg___lam__0(v_down_boxed_530_);
v_r_532_ = lean_box(v_res_531_);
return v_r_532_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_isEmpty___redArg(lean_object* v_inst_538_, lean_object* v_xs_539_){
_start:
{
lean_object* v_toApplicative_540_; lean_object* v_toFunctor_541_; lean_object* v_map_542_; lean_object* v___x_543_; lean_object* v___x_544_; lean_object* v___x_545_; 
v_toApplicative_540_ = lean_ctor_get(v_inst_538_, 0);
v_toFunctor_541_ = lean_ctor_get(v_toApplicative_540_, 0);
v_map_542_ = lean_ctor_get(v_toFunctor_541_, 0);
lean_inc(v_map_542_);
v___x_543_ = ((lean_object*)(lp_batteries_MLList_isEmpty___redArg___closed__2));
v___x_544_ = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_unconsImpl___redArg(v_inst_538_, v_xs_539_);
v___x_545_ = lean_apply_4(v_map_542_, lean_box(0), lean_box(0), v___x_543_, v___x_544_);
return v___x_545_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_isEmpty(lean_object* v_m_546_, lean_object* v_00_u03b1_547_, lean_object* v_inst_548_, lean_object* v_xs_549_){
_start:
{
lean_object* v___x_550_; 
v___x_550_ = lp_batteries_MLList_isEmpty___redArg(v_inst_548_, v_xs_549_);
return v___x_550_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_ofList___redArg(lean_object* v_x_551_){
_start:
{
if (lean_obj_tag(v_x_551_) == 0)
{
lean_object* v___x_552_; 
v___x_552_ = lean_box(0);
return v___x_552_;
}
else
{
lean_object* v_head_553_; lean_object* v_tail_554_; lean_object* v___x_556_; uint8_t v_isShared_557_; uint8_t v_isSharedCheck_564_; 
v_head_553_ = lean_ctor_get(v_x_551_, 0);
v_tail_554_ = lean_ctor_get(v_x_551_, 1);
v_isSharedCheck_564_ = !lean_is_exclusive(v_x_551_);
if (v_isSharedCheck_564_ == 0)
{
v___x_556_ = v_x_551_;
v_isShared_557_ = v_isSharedCheck_564_;
goto v_resetjp_555_;
}
else
{
lean_inc(v_tail_554_);
lean_inc(v_head_553_);
lean_dec(v_x_551_);
v___x_556_ = lean_box(0);
v_isShared_557_ = v_isSharedCheck_564_;
goto v_resetjp_555_;
}
v_resetjp_555_:
{
lean_object* v___f_558_; lean_object* v___x_559_; lean_object* v___x_560_; lean_object* v___x_562_; 
v___f_558_ = lean_alloc_closure((void*)(lp_batteries_MLList_ofList___redArg___lam__0), 2, 1);
lean_closure_set(v___f_558_, 0, v_tail_554_);
v___x_559_ = lean_mk_thunk(v___f_558_);
v___x_560_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v___x_560_, 0, v___x_559_);
if (v_isShared_557_ == 0)
{
lean_ctor_set(v___x_556_, 1, v___x_560_);
v___x_562_ = v___x_556_;
goto v_reusejp_561_;
}
else
{
lean_object* v_reuseFailAlloc_563_; 
v_reuseFailAlloc_563_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_563_, 0, v_head_553_);
lean_ctor_set(v_reuseFailAlloc_563_, 1, v___x_560_);
v___x_562_ = v_reuseFailAlloc_563_;
goto v_reusejp_561_;
}
v_reusejp_561_:
{
return v___x_562_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_ofList___redArg___lam__0(lean_object* v_tail_565_, lean_object* v_x_566_){
_start:
{
lean_object* v___x_567_; 
v___x_567_ = lp_batteries_MLList_ofList___redArg(v_tail_565_);
return v___x_567_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_ofList(lean_object* v_00_u03b1_568_, lean_object* v_m_569_, lean_object* v_x_570_){
_start:
{
lean_object* v___x_571_; 
v___x_571_ = lp_batteries_MLList_ofList___redArg(v_x_570_);
return v___x_571_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_ofListM___redArg___lam__1(lean_object* v_toBind_572_, lean_object* v_head_573_, lean_object* v___f_574_, lean_object* v_x_575_){
_start:
{
lean_object* v___x_576_; 
v___x_576_ = lean_apply_4(v_toBind_572_, lean_box(0), lean_box(0), v_head_573_, v___f_574_);
return v___x_576_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_ofListM___redArg(lean_object* v_inst_577_, lean_object* v_x_578_){
_start:
{
if (lean_obj_tag(v_x_578_) == 0)
{
lean_object* v___x_579_; 
lean_dec_ref(v_inst_577_);
v___x_579_ = lean_box(0);
return v___x_579_;
}
else
{
lean_object* v_toApplicative_580_; lean_object* v_toBind_581_; lean_object* v_toPure_582_; lean_object* v_head_583_; lean_object* v_tail_584_; lean_object* v___f_585_; lean_object* v___f_586_; lean_object* v___x_587_; 
v_toApplicative_580_ = lean_ctor_get(v_inst_577_, 0);
v_toBind_581_ = lean_ctor_get(v_inst_577_, 1);
lean_inc(v_toBind_581_);
v_toPure_582_ = lean_ctor_get(v_toApplicative_580_, 1);
lean_inc(v_toPure_582_);
v_head_583_ = lean_ctor_get(v_x_578_, 0);
lean_inc(v_head_583_);
v_tail_584_ = lean_ctor_get(v_x_578_, 1);
lean_inc(v_tail_584_);
lean_dec_ref_known(v_x_578_, 2);
v___f_585_ = lean_alloc_closure((void*)(lp_batteries_MLList_ofListM___redArg___lam__0), 4, 3);
lean_closure_set(v___f_585_, 0, v_inst_577_);
lean_closure_set(v___f_585_, 1, v_tail_584_);
lean_closure_set(v___f_585_, 2, v_toPure_582_);
v___f_586_ = lean_alloc_closure((void*)(lp_batteries_MLList_ofListM___redArg___lam__1), 4, 3);
lean_closure_set(v___f_586_, 0, v_toBind_581_);
lean_closure_set(v___f_586_, 1, v_head_583_);
lean_closure_set(v___f_586_, 2, v___f_585_);
v___x_587_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_587_, 0, v___f_586_);
return v___x_587_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_ofListM___redArg___lam__0(lean_object* v_inst_588_, lean_object* v_tail_589_, lean_object* v_toPure_590_, lean_object* v_____do__lift_591_){
_start:
{
lean_object* v___x_592_; lean_object* v___x_593_; lean_object* v___x_594_; 
v___x_592_ = lp_batteries_MLList_ofListM___redArg(v_inst_588_, v_tail_589_);
v___x_593_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_593_, 0, v_____do__lift_591_);
lean_ctor_set(v___x_593_, 1, v___x_592_);
v___x_594_ = lean_apply_2(v_toPure_590_, lean_box(0), v___x_593_);
return v___x_594_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_ofListM(lean_object* v_m_595_, lean_object* v_00_u03b1_596_, lean_object* v_inst_597_, lean_object* v_x_598_){
_start:
{
lean_object* v___x_599_; 
v___x_599_ = lp_batteries_MLList_ofListM___redArg(v_inst_597_, v_x_598_);
return v___x_599_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_force___redArg___lam__0(lean_object* v_fst_600_, lean_object* v_toPure_601_, lean_object* v_____do__lift_602_){
_start:
{
lean_object* v___x_603_; lean_object* v___x_604_; 
v___x_603_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_603_, 0, v_fst_600_);
lean_ctor_set(v___x_603_, 1, v_____do__lift_602_);
v___x_604_ = lean_apply_2(v_toPure_601_, lean_box(0), v___x_603_);
return v___x_604_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_force___redArg___lam__1(lean_object* v_toPure_605_, lean_object* v_inst_606_, lean_object* v_toBind_607_, lean_object* v_____do__lift_608_){
_start:
{
if (lean_obj_tag(v_____do__lift_608_) == 0)
{
lean_object* v___x_609_; lean_object* v___x_610_; 
lean_dec(v_toBind_607_);
lean_dec_ref(v_inst_606_);
v___x_609_ = lean_box(0);
v___x_610_ = lean_apply_2(v_toPure_605_, lean_box(0), v___x_609_);
return v___x_610_;
}
else
{
lean_object* v_val_611_; lean_object* v_fst_612_; lean_object* v_snd_613_; lean_object* v___f_614_; lean_object* v___x_615_; lean_object* v___x_616_; 
v_val_611_ = lean_ctor_get(v_____do__lift_608_, 0);
lean_inc(v_val_611_);
lean_dec_ref_known(v_____do__lift_608_, 1);
v_fst_612_ = lean_ctor_get(v_val_611_, 0);
lean_inc(v_fst_612_);
v_snd_613_ = lean_ctor_get(v_val_611_, 1);
lean_inc(v_snd_613_);
lean_dec(v_val_611_);
v___f_614_ = lean_alloc_closure((void*)(lp_batteries_MLList_force___redArg___lam__0), 3, 2);
lean_closure_set(v___f_614_, 0, v_fst_612_);
lean_closure_set(v___f_614_, 1, v_toPure_605_);
v___x_615_ = lp_batteries_MLList_force___redArg(v_inst_606_, v_snd_613_);
v___x_616_ = lean_apply_4(v_toBind_607_, lean_box(0), lean_box(0), v___x_615_, v___f_614_);
return v___x_616_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_force___redArg(lean_object* v_inst_617_, lean_object* v_L_618_){
_start:
{
lean_object* v_toApplicative_619_; lean_object* v_toBind_620_; lean_object* v_toPure_621_; lean_object* v___x_622_; lean_object* v___f_623_; lean_object* v___x_624_; 
v_toApplicative_619_ = lean_ctor_get(v_inst_617_, 0);
v_toBind_620_ = lean_ctor_get(v_inst_617_, 1);
lean_inc_n(v_toBind_620_, 2);
v_toPure_621_ = lean_ctor_get(v_toApplicative_619_, 1);
lean_inc(v_toPure_621_);
lean_inc_ref(v_inst_617_);
v___x_622_ = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_unconsImpl___redArg(v_inst_617_, v_L_618_);
v___f_623_ = lean_alloc_closure((void*)(lp_batteries_MLList_force___redArg___lam__1), 4, 3);
lean_closure_set(v___f_623_, 0, v_toPure_621_);
lean_closure_set(v___f_623_, 1, v_inst_617_);
lean_closure_set(v___f_623_, 2, v_toBind_620_);
v___x_624_ = lean_apply_4(v_toBind_620_, lean_box(0), lean_box(0), v___x_622_, v___f_623_);
return v___x_624_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_force(lean_object* v_m_625_, lean_object* v_00_u03b1_626_, lean_object* v_inst_627_, lean_object* v_L_628_){
_start:
{
lean_object* v___x_629_; 
v___x_629_ = lp_batteries_MLList_force___redArg(v_inst_627_, v_L_628_);
return v___x_629_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_asArray___redArg___lam__0(lean_object* v_toPure_630_, lean_object* v_a_631_, lean_object* v_____s_632_){
_start:
{
lean_object* v_r_633_; lean_object* v___x_634_; lean_object* v___x_635_; 
v_r_633_ = lean_array_push(v_____s_632_, v_a_631_);
v___x_634_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_634_, 0, v_r_633_);
v___x_635_ = lean_apply_2(v_toPure_630_, lean_box(0), v___x_634_);
return v___x_635_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_asArray___redArg___lam__1(lean_object* v_toPure_636_, lean_object* v_____s_637_){
_start:
{
lean_object* v___x_638_; 
v___x_638_ = lean_apply_2(v_toPure_636_, lean_box(0), v_____s_637_);
return v___x_638_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_asArray___redArg(lean_object* v_inst_642_, lean_object* v_L_643_){
_start:
{
lean_object* v_toApplicative_644_; lean_object* v_toBind_645_; lean_object* v_toPure_646_; lean_object* v_r_647_; lean_object* v___f_648_; lean_object* v___f_649_; lean_object* v___f_650_; lean_object* v___x_651_; lean_object* v___x_652_; 
v_toApplicative_644_ = lean_ctor_get(v_inst_642_, 0);
v_toBind_645_ = lean_ctor_get(v_inst_642_, 1);
lean_inc(v_toBind_645_);
v_toPure_646_ = lean_ctor_get(v_toApplicative_644_, 1);
v_r_647_ = ((lean_object*)(lp_batteries_MLList_asArray___redArg___closed__0));
v___f_648_ = ((lean_object*)(lp_batteries_MLList_asArray___redArg___closed__1));
lean_inc_n(v_toPure_646_, 2);
v___f_649_ = lean_alloc_closure((void*)(lp_batteries_MLList_asArray___redArg___lam__0), 3, 1);
lean_closure_set(v___f_649_, 0, v_toPure_646_);
v___f_650_ = lean_alloc_closure((void*)(lp_batteries_MLList_asArray___redArg___lam__1), 2, 1);
lean_closure_set(v___f_650_, 0, v_toPure_646_);
lean_inc_ref(v_inst_642_);
v___x_651_ = lp_batteries_MLList_forIn___redArg(v_inst_642_, v_inst_642_, v___f_648_, v_L_643_, v_r_647_, v___f_649_);
v___x_652_ = lean_apply_4(v_toBind_645_, lean_box(0), lean_box(0), v___x_651_, v___f_650_);
return v___x_652_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_asArray(lean_object* v_m_653_, lean_object* v_00_u03b1_654_, lean_object* v_inst_655_, lean_object* v_L_656_){
_start:
{
lean_object* v___x_657_; 
v___x_657_ = lp_batteries_MLList_asArray___redArg(v_inst_655_, v_L_656_);
return v___x_657_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_casesM___redArg___lam__0(lean_object* v_hnil_658_, lean_object* v_hcons_659_, lean_object* v_____do__lift_660_){
_start:
{
if (lean_obj_tag(v_____do__lift_660_) == 0)
{
lean_object* v___x_661_; lean_object* v___x_662_; 
lean_dec(v_hcons_659_);
v___x_661_ = lean_box(0);
v___x_662_ = lean_apply_1(v_hnil_658_, v___x_661_);
return v___x_662_;
}
else
{
lean_object* v_val_663_; lean_object* v_fst_664_; lean_object* v_snd_665_; lean_object* v___x_666_; 
lean_dec(v_hnil_658_);
v_val_663_ = lean_ctor_get(v_____do__lift_660_, 0);
lean_inc(v_val_663_);
lean_dec_ref_known(v_____do__lift_660_, 1);
v_fst_664_ = lean_ctor_get(v_val_663_, 0);
lean_inc(v_fst_664_);
v_snd_665_ = lean_ctor_get(v_val_663_, 1);
lean_inc(v_snd_665_);
lean_dec(v_val_663_);
v___x_666_ = lean_apply_2(v_hcons_659_, v_fst_664_, v_snd_665_);
return v___x_666_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_casesM___redArg___lam__1(lean_object* v_inst_667_, lean_object* v_xs_668_, lean_object* v_toBind_669_, lean_object* v___f_670_, lean_object* v_x_671_){
_start:
{
lean_object* v___x_672_; lean_object* v___x_673_; 
v___x_672_ = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_unconsImpl___redArg(v_inst_667_, v_xs_668_);
v___x_673_ = lean_apply_4(v_toBind_669_, lean_box(0), lean_box(0), v___x_672_, v___f_670_);
return v___x_673_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_casesM___redArg(lean_object* v_inst_674_, lean_object* v_xs_675_, lean_object* v_hnil_676_, lean_object* v_hcons_677_){
_start:
{
lean_object* v_toBind_678_; lean_object* v___f_679_; lean_object* v___f_680_; lean_object* v___x_681_; 
v_toBind_678_ = lean_ctor_get(v_inst_674_, 1);
lean_inc(v_toBind_678_);
v___f_679_ = lean_alloc_closure((void*)(lp_batteries_MLList_casesM___redArg___lam__0), 3, 2);
lean_closure_set(v___f_679_, 0, v_hnil_676_);
lean_closure_set(v___f_679_, 1, v_hcons_677_);
v___f_680_ = lean_alloc_closure((void*)(lp_batteries_MLList_casesM___redArg___lam__1), 5, 4);
lean_closure_set(v___f_680_, 0, v_inst_674_);
lean_closure_set(v___f_680_, 1, v_xs_675_);
lean_closure_set(v___f_680_, 2, v_toBind_678_);
lean_closure_set(v___f_680_, 3, v___f_679_);
v___x_681_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_681_, 0, v___f_680_);
return v___x_681_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_casesM(lean_object* v_m_682_, lean_object* v_00_u03b1_683_, lean_object* v_00_u03b2_684_, lean_object* v_inst_685_, lean_object* v_xs_686_, lean_object* v_hnil_687_, lean_object* v_hcons_688_){
_start:
{
lean_object* v___x_689_; 
v___x_689_ = lp_batteries_MLList_casesM___redArg(v_inst_685_, v_xs_686_, v_hnil_687_, v_hcons_688_);
return v___x_689_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_cases___redArg___lam__0(lean_object* v_hnil_690_, lean_object* v_toPure_691_, lean_object* v_x_692_){
_start:
{
lean_object* v___x_693_; lean_object* v___x_694_; lean_object* v___x_695_; 
v___x_693_ = lean_box(0);
v___x_694_ = lean_apply_1(v_hnil_690_, v___x_693_);
v___x_695_ = lean_apply_2(v_toPure_691_, lean_box(0), v___x_694_);
return v___x_695_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_cases___redArg___lam__1(lean_object* v_hcons_696_, lean_object* v_toPure_697_, lean_object* v_x_698_, lean_object* v_xs_699_){
_start:
{
lean_object* v___x_700_; lean_object* v___x_701_; 
v___x_700_ = lean_apply_2(v_hcons_696_, v_x_698_, v_xs_699_);
v___x_701_ = lean_apply_2(v_toPure_697_, lean_box(0), v___x_700_);
return v___x_701_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_cases___redArg___lam__2(lean_object* v_hcons_702_, lean_object* v_fst_703_, lean_object* v_snd_704_, lean_object* v_x_705_){
_start:
{
lean_object* v___x_706_; 
v___x_706_ = lean_apply_2(v_hcons_702_, v_fst_703_, v_snd_704_);
return v___x_706_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_cases___redArg(lean_object* v_inst_707_, lean_object* v_xs_708_, lean_object* v_hnil_709_, lean_object* v_hcons_710_){
_start:
{
lean_object* v_toApplicative_711_; lean_object* v_toPure_712_; lean_object* v___x_713_; 
v_toApplicative_711_ = lean_ctor_get(v_inst_707_, 0);
v_toPure_712_ = lean_ctor_get(v_toApplicative_711_, 1);
lean_inc(v_xs_708_);
v___x_713_ = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_uncons_x3fImpl___redArg(v_xs_708_);
if (lean_obj_tag(v___x_713_) == 0)
{
lean_object* v___f_714_; lean_object* v___f_715_; lean_object* v___x_716_; 
lean_inc_n(v_toPure_712_, 2);
v___f_714_ = lean_alloc_closure((void*)(lp_batteries_MLList_cases___redArg___lam__0), 3, 2);
lean_closure_set(v___f_714_, 0, v_hnil_709_);
lean_closure_set(v___f_714_, 1, v_toPure_712_);
v___f_715_ = lean_alloc_closure((void*)(lp_batteries_MLList_cases___redArg___lam__1), 4, 2);
lean_closure_set(v___f_715_, 0, v_hcons_710_);
lean_closure_set(v___f_715_, 1, v_toPure_712_);
v___x_716_ = lp_batteries_MLList_casesM___redArg(v_inst_707_, v_xs_708_, v___f_714_, v___f_715_);
return v___x_716_;
}
else
{
lean_object* v_val_717_; lean_object* v___x_719_; uint8_t v_isShared_720_; uint8_t v_isSharedCheck_737_; 
lean_dec(v_xs_708_);
lean_dec_ref(v_inst_707_);
v_val_717_ = lean_ctor_get(v___x_713_, 0);
v_isSharedCheck_737_ = !lean_is_exclusive(v___x_713_);
if (v_isSharedCheck_737_ == 0)
{
v___x_719_ = v___x_713_;
v_isShared_720_ = v_isSharedCheck_737_;
goto v_resetjp_718_;
}
else
{
lean_inc(v_val_717_);
lean_dec(v___x_713_);
v___x_719_ = lean_box(0);
v_isShared_720_ = v_isSharedCheck_737_;
goto v_resetjp_718_;
}
v_resetjp_718_:
{
if (lean_obj_tag(v_val_717_) == 0)
{
lean_object* v___x_721_; lean_object* v___x_723_; 
lean_dec(v_hcons_710_);
v___x_721_ = lean_mk_thunk(v_hnil_709_);
if (v_isShared_720_ == 0)
{
lean_ctor_set_tag(v___x_719_, 2);
lean_ctor_set(v___x_719_, 0, v___x_721_);
v___x_723_ = v___x_719_;
goto v_reusejp_722_;
}
else
{
lean_object* v_reuseFailAlloc_724_; 
v_reuseFailAlloc_724_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v_reuseFailAlloc_724_, 0, v___x_721_);
v___x_723_ = v_reuseFailAlloc_724_;
goto v_reusejp_722_;
}
v_reusejp_722_:
{
return v___x_723_;
}
}
else
{
lean_object* v_val_725_; lean_object* v___x_727_; uint8_t v_isShared_728_; uint8_t v_isSharedCheck_736_; 
lean_del_object(v___x_719_);
lean_dec(v_hnil_709_);
v_val_725_ = lean_ctor_get(v_val_717_, 0);
v_isSharedCheck_736_ = !lean_is_exclusive(v_val_717_);
if (v_isSharedCheck_736_ == 0)
{
v___x_727_ = v_val_717_;
v_isShared_728_ = v_isSharedCheck_736_;
goto v_resetjp_726_;
}
else
{
lean_inc(v_val_725_);
lean_dec(v_val_717_);
v___x_727_ = lean_box(0);
v_isShared_728_ = v_isSharedCheck_736_;
goto v_resetjp_726_;
}
v_resetjp_726_:
{
lean_object* v_fst_729_; lean_object* v_snd_730_; lean_object* v___f_731_; lean_object* v___x_732_; lean_object* v___x_734_; 
v_fst_729_ = lean_ctor_get(v_val_725_, 0);
lean_inc(v_fst_729_);
v_snd_730_ = lean_ctor_get(v_val_725_, 1);
lean_inc(v_snd_730_);
lean_dec(v_val_725_);
v___f_731_ = lean_alloc_closure((void*)(lp_batteries_MLList_cases___redArg___lam__2), 4, 3);
lean_closure_set(v___f_731_, 0, v_hcons_710_);
lean_closure_set(v___f_731_, 1, v_fst_729_);
lean_closure_set(v___f_731_, 2, v_snd_730_);
v___x_732_ = lean_mk_thunk(v___f_731_);
if (v_isShared_728_ == 0)
{
lean_ctor_set_tag(v___x_727_, 2);
lean_ctor_set(v___x_727_, 0, v___x_732_);
v___x_734_ = v___x_727_;
goto v_reusejp_733_;
}
else
{
lean_object* v_reuseFailAlloc_735_; 
v_reuseFailAlloc_735_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v_reuseFailAlloc_735_, 0, v___x_732_);
v___x_734_ = v_reuseFailAlloc_735_;
goto v_reusejp_733_;
}
v_reusejp_733_:
{
return v___x_734_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_cases(lean_object* v_m_738_, lean_object* v_00_u03b1_739_, lean_object* v_00_u03b2_740_, lean_object* v_inst_741_, lean_object* v_xs_742_, lean_object* v_hnil_743_, lean_object* v_hcons_744_){
_start:
{
lean_object* v___x_745_; 
v___x_745_ = lp_batteries_MLList_cases___redArg(v_inst_741_, v_xs_742_, v_hnil_743_, v_hcons_744_);
return v___x_745_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_foldsM___redArg___lam__2(lean_object* v_inst_746_, lean_object* v_L_747_, lean_object* v_toBind_748_, lean_object* v___f_749_, lean_object* v_x_750_){
_start:
{
lean_object* v___x_751_; lean_object* v___x_752_; 
v___x_751_ = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_unconsImpl___redArg(v_inst_746_, v_L_747_);
v___x_752_ = lean_apply_4(v_toBind_748_, lean_box(0), lean_box(0), v___x_751_, v___f_749_);
return v___x_752_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_foldsM___redArg___lam__1(lean_object* v_toPure_753_, lean_object* v_inst_754_, lean_object* v_f_755_, lean_object* v_init_756_, lean_object* v_toBind_757_, lean_object* v_____do__lift_758_){
_start:
{
if (lean_obj_tag(v_____do__lift_758_) == 0)
{
lean_object* v___x_759_; lean_object* v___x_760_; 
lean_dec(v_toBind_757_);
lean_dec(v_init_756_);
lean_dec(v_f_755_);
lean_dec_ref(v_inst_754_);
v___x_759_ = lean_box(0);
v___x_760_ = lean_apply_2(v_toPure_753_, lean_box(0), v___x_759_);
return v___x_760_;
}
else
{
lean_object* v_val_761_; lean_object* v_fst_762_; lean_object* v_snd_763_; lean_object* v___f_764_; lean_object* v___x_765_; lean_object* v___x_766_; 
v_val_761_ = lean_ctor_get(v_____do__lift_758_, 0);
lean_inc(v_val_761_);
lean_dec_ref_known(v_____do__lift_758_, 1);
v_fst_762_ = lean_ctor_get(v_val_761_, 0);
lean_inc(v_fst_762_);
v_snd_763_ = lean_ctor_get(v_val_761_, 1);
lean_inc(v_snd_763_);
lean_dec(v_val_761_);
lean_inc(v_f_755_);
v___f_764_ = lean_alloc_closure((void*)(lp_batteries_MLList_foldsM___redArg___lam__0), 5, 4);
lean_closure_set(v___f_764_, 0, v_inst_754_);
lean_closure_set(v___f_764_, 1, v_f_755_);
lean_closure_set(v___f_764_, 2, v_snd_763_);
lean_closure_set(v___f_764_, 3, v_toPure_753_);
v___x_765_ = lean_apply_2(v_f_755_, v_init_756_, v_fst_762_);
v___x_766_ = lean_apply_4(v_toBind_757_, lean_box(0), lean_box(0), v___x_765_, v___f_764_);
return v___x_766_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_foldsM___redArg(lean_object* v_inst_767_, lean_object* v_f_768_, lean_object* v_init_769_, lean_object* v_L_770_){
_start:
{
lean_object* v_toApplicative_771_; lean_object* v_toBind_772_; lean_object* v_toPure_773_; lean_object* v___f_774_; lean_object* v___f_775_; lean_object* v___x_776_; lean_object* v___x_777_; 
v_toApplicative_771_ = lean_ctor_get(v_inst_767_, 0);
v_toBind_772_ = lean_ctor_get(v_inst_767_, 1);
lean_inc_n(v_toBind_772_, 2);
v_toPure_773_ = lean_ctor_get(v_toApplicative_771_, 1);
lean_inc(v_init_769_);
lean_inc_ref(v_inst_767_);
lean_inc(v_toPure_773_);
v___f_774_ = lean_alloc_closure((void*)(lp_batteries_MLList_foldsM___redArg___lam__1), 6, 5);
lean_closure_set(v___f_774_, 0, v_toPure_773_);
lean_closure_set(v___f_774_, 1, v_inst_767_);
lean_closure_set(v___f_774_, 2, v_f_768_);
lean_closure_set(v___f_774_, 3, v_init_769_);
lean_closure_set(v___f_774_, 4, v_toBind_772_);
v___f_775_ = lean_alloc_closure((void*)(lp_batteries_MLList_foldsM___redArg___lam__2), 5, 4);
lean_closure_set(v___f_775_, 0, v_inst_767_);
lean_closure_set(v___f_775_, 1, v_L_770_);
lean_closure_set(v___f_775_, 2, v_toBind_772_);
lean_closure_set(v___f_775_, 3, v___f_774_);
v___x_776_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_776_, 0, v___f_775_);
v___x_777_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_777_, 0, v_init_769_);
lean_ctor_set(v___x_777_, 1, v___x_776_);
return v___x_777_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_foldsM___redArg___lam__0(lean_object* v_inst_778_, lean_object* v_f_779_, lean_object* v_snd_780_, lean_object* v_toPure_781_, lean_object* v_____do__lift_782_){
_start:
{
lean_object* v___x_783_; lean_object* v___x_784_; 
v___x_783_ = lp_batteries_MLList_foldsM___redArg(v_inst_778_, v_f_779_, v_____do__lift_782_, v_snd_780_);
v___x_784_ = lean_apply_2(v_toPure_781_, lean_box(0), v___x_783_);
return v___x_784_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_foldsM(lean_object* v_m_785_, lean_object* v_00_u03b2_786_, lean_object* v_00_u03b1_787_, lean_object* v_inst_788_, lean_object* v_f_789_, lean_object* v_init_790_, lean_object* v_L_791_){
_start:
{
lean_object* v___x_792_; 
v___x_792_ = lp_batteries_MLList_foldsM___redArg(v_inst_788_, v_f_789_, v_init_790_, v_L_791_);
return v___x_792_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_folds___redArg___lam__0(lean_object* v_f_793_, lean_object* v_toPure_794_, lean_object* v_b_795_, lean_object* v_a_796_){
_start:
{
lean_object* v___x_797_; lean_object* v___x_798_; 
v___x_797_ = lean_apply_2(v_f_793_, v_b_795_, v_a_796_);
v___x_798_ = lean_apply_2(v_toPure_794_, lean_box(0), v___x_797_);
return v___x_798_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_folds___redArg(lean_object* v_inst_799_, lean_object* v_f_800_, lean_object* v_init_801_, lean_object* v_L_802_){
_start:
{
lean_object* v_toApplicative_803_; lean_object* v_toPure_804_; lean_object* v___f_805_; lean_object* v___x_806_; 
v_toApplicative_803_ = lean_ctor_get(v_inst_799_, 0);
v_toPure_804_ = lean_ctor_get(v_toApplicative_803_, 1);
lean_inc(v_toPure_804_);
v___f_805_ = lean_alloc_closure((void*)(lp_batteries_MLList_folds___redArg___lam__0), 4, 2);
lean_closure_set(v___f_805_, 0, v_f_800_);
lean_closure_set(v___f_805_, 1, v_toPure_804_);
v___x_806_ = lp_batteries_MLList_foldsM___redArg(v_inst_799_, v___f_805_, v_init_801_, v_L_802_);
return v___x_806_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_folds(lean_object* v_m_807_, lean_object* v_00_u03b2_808_, lean_object* v_00_u03b1_809_, lean_object* v_inst_810_, lean_object* v_f_811_, lean_object* v_init_812_, lean_object* v_L_813_){
_start:
{
lean_object* v___x_814_; 
v___x_814_ = lp_batteries_MLList_folds___redArg(v_inst_810_, v_f_811_, v_init_812_, v_L_813_);
return v___x_814_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_takeAsList_go___redArg___lam__0(lean_object* v_acc_815_, lean_object* v_toPure_816_, lean_object* v_00___817_){
_start:
{
lean_object* v___x_818_; lean_object* v___x_819_; 
v___x_818_ = l_List_reverse___redArg(v_acc_815_);
v___x_819_ = lean_apply_2(v_toPure_816_, lean_box(0), v___x_818_);
return v___x_819_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_takeAsList_go___redArg___lam__1___boxed(lean_object* v___f_820_, lean_object* v_acc_821_, lean_object* v_inst_822_, lean_object* v_n_823_, lean_object* v_____do__lift_824_){
_start:
{
lean_object* v_res_825_; 
v_res_825_ = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_takeAsList_go___redArg___lam__1(v___f_820_, v_acc_821_, v_inst_822_, v_n_823_, v_____do__lift_824_);
lean_dec(v_n_823_);
return v_res_825_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_takeAsList_go___redArg(lean_object* v_inst_826_, lean_object* v_r_827_, lean_object* v_acc_828_, lean_object* v_xs_829_){
_start:
{
lean_object* v_toApplicative_830_; lean_object* v_toBind_831_; lean_object* v_toPure_832_; lean_object* v___f_833_; lean_object* v_zero_834_; uint8_t v_isZero_835_; 
v_toApplicative_830_ = lean_ctor_get(v_inst_826_, 0);
v_toBind_831_ = lean_ctor_get(v_inst_826_, 1);
lean_inc(v_toBind_831_);
v_toPure_832_ = lean_ctor_get(v_toApplicative_830_, 1);
lean_inc(v_toPure_832_);
lean_inc(v_acc_828_);
v___f_833_ = lean_alloc_closure((void*)(lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_takeAsList_go___redArg___lam__0), 3, 2);
lean_closure_set(v___f_833_, 0, v_acc_828_);
lean_closure_set(v___f_833_, 1, v_toPure_832_);
v_zero_834_ = lean_unsigned_to_nat(0u);
v_isZero_835_ = lean_nat_dec_eq(v_r_827_, v_zero_834_);
if (v_isZero_835_ == 1)
{
lean_object* v___x_836_; lean_object* v___x_837_; 
lean_inc(v_toPure_832_);
lean_dec_ref(v___f_833_);
lean_dec(v_toBind_831_);
lean_dec(v_xs_829_);
lean_dec_ref(v_inst_826_);
v___x_836_ = lean_box(0);
v___x_837_ = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_takeAsList_go___redArg___lam__0(v_acc_828_, v_toPure_832_, v___x_836_);
return v___x_837_;
}
else
{
lean_object* v_one_838_; lean_object* v_n_839_; lean_object* v___f_840_; lean_object* v___x_841_; lean_object* v___x_842_; 
v_one_838_ = lean_unsigned_to_nat(1u);
v_n_839_ = lean_nat_sub(v_r_827_, v_one_838_);
lean_inc_ref(v_inst_826_);
v___f_840_ = lean_alloc_closure((void*)(lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_takeAsList_go___redArg___lam__1___boxed), 5, 4);
lean_closure_set(v___f_840_, 0, v___f_833_);
lean_closure_set(v___f_840_, 1, v_acc_828_);
lean_closure_set(v___f_840_, 2, v_inst_826_);
lean_closure_set(v___f_840_, 3, v_n_839_);
v___x_841_ = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_unconsImpl___redArg(v_inst_826_, v_xs_829_);
v___x_842_ = lean_apply_4(v_toBind_831_, lean_box(0), lean_box(0), v___x_841_, v___f_840_);
return v___x_842_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_takeAsList_go___redArg___lam__1(lean_object* v___f_843_, lean_object* v_acc_844_, lean_object* v_inst_845_, lean_object* v_n_846_, lean_object* v_____do__lift_847_){
_start:
{
if (lean_obj_tag(v_____do__lift_847_) == 0)
{
lean_object* v___x_848_; lean_object* v___x_849_; 
lean_dec_ref(v_inst_845_);
lean_dec(v_acc_844_);
v___x_848_ = lean_box(0);
v___x_849_ = lean_apply_1(v___f_843_, v___x_848_);
return v___x_849_;
}
else
{
lean_object* v_val_850_; lean_object* v_fst_851_; lean_object* v_snd_852_; lean_object* v___x_854_; uint8_t v_isShared_855_; uint8_t v_isSharedCheck_860_; 
lean_dec(v___f_843_);
v_val_850_ = lean_ctor_get(v_____do__lift_847_, 0);
lean_inc(v_val_850_);
lean_dec_ref_known(v_____do__lift_847_, 1);
v_fst_851_ = lean_ctor_get(v_val_850_, 0);
v_snd_852_ = lean_ctor_get(v_val_850_, 1);
v_isSharedCheck_860_ = !lean_is_exclusive(v_val_850_);
if (v_isSharedCheck_860_ == 0)
{
v___x_854_ = v_val_850_;
v_isShared_855_ = v_isSharedCheck_860_;
goto v_resetjp_853_;
}
else
{
lean_inc(v_snd_852_);
lean_inc(v_fst_851_);
lean_dec(v_val_850_);
v___x_854_ = lean_box(0);
v_isShared_855_ = v_isSharedCheck_860_;
goto v_resetjp_853_;
}
v_resetjp_853_:
{
lean_object* v___x_857_; 
if (v_isShared_855_ == 0)
{
lean_ctor_set_tag(v___x_854_, 1);
lean_ctor_set(v___x_854_, 1, v_acc_844_);
v___x_857_ = v___x_854_;
goto v_reusejp_856_;
}
else
{
lean_object* v_reuseFailAlloc_859_; 
v_reuseFailAlloc_859_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_859_, 0, v_fst_851_);
lean_ctor_set(v_reuseFailAlloc_859_, 1, v_acc_844_);
v___x_857_ = v_reuseFailAlloc_859_;
goto v_reusejp_856_;
}
v_reusejp_856_:
{
lean_object* v___x_858_; 
v___x_858_ = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_takeAsList_go___redArg(v_inst_845_, v_n_846_, v___x_857_, v_snd_852_);
return v___x_858_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_takeAsList_go___redArg___boxed(lean_object* v_inst_861_, lean_object* v_r_862_, lean_object* v_acc_863_, lean_object* v_xs_864_){
_start:
{
lean_object* v_res_865_; 
v_res_865_ = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_takeAsList_go___redArg(v_inst_861_, v_r_862_, v_acc_863_, v_xs_864_);
lean_dec(v_r_862_);
return v_res_865_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_takeAsList_go(lean_object* v_m_866_, lean_object* v_00_u03b1_867_, lean_object* v_inst_868_, lean_object* v_r_869_, lean_object* v_acc_870_, lean_object* v_xs_871_){
_start:
{
lean_object* v___x_872_; 
v___x_872_ = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_takeAsList_go___redArg(v_inst_868_, v_r_869_, v_acc_870_, v_xs_871_);
return v___x_872_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_takeAsList_go___boxed(lean_object* v_m_873_, lean_object* v_00_u03b1_874_, lean_object* v_inst_875_, lean_object* v_r_876_, lean_object* v_acc_877_, lean_object* v_xs_878_){
_start:
{
lean_object* v_res_879_; 
v_res_879_ = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_takeAsList_go(v_m_873_, v_00_u03b1_874_, v_inst_875_, v_r_876_, v_acc_877_, v_xs_878_);
lean_dec(v_r_876_);
return v_res_879_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_takeAsList___redArg(lean_object* v_inst_880_, lean_object* v_xs_881_, lean_object* v_n_882_){
_start:
{
lean_object* v___x_883_; lean_object* v___x_884_; 
v___x_883_ = lean_box(0);
v___x_884_ = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_takeAsList_go___redArg(v_inst_880_, v_n_882_, v___x_883_, v_xs_881_);
return v___x_884_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_takeAsList___redArg___boxed(lean_object* v_inst_885_, lean_object* v_xs_886_, lean_object* v_n_887_){
_start:
{
lean_object* v_res_888_; 
v_res_888_ = lp_batteries_MLList_takeAsList___redArg(v_inst_885_, v_xs_886_, v_n_887_);
lean_dec(v_n_887_);
return v_res_888_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_takeAsList(lean_object* v_m_889_, lean_object* v_00_u03b1_890_, lean_object* v_inst_891_, lean_object* v_xs_892_, lean_object* v_n_893_){
_start:
{
lean_object* v___x_894_; 
v___x_894_ = lp_batteries_MLList_takeAsList___redArg(v_inst_891_, v_xs_892_, v_n_893_);
return v___x_894_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_takeAsList___boxed(lean_object* v_m_895_, lean_object* v_00_u03b1_896_, lean_object* v_inst_897_, lean_object* v_xs_898_, lean_object* v_n_899_){
_start:
{
lean_object* v_res_900_; 
v_res_900_ = lp_batteries_MLList_takeAsList(v_m_895_, v_00_u03b1_896_, v_inst_897_, v_xs_898_, v_n_899_);
lean_dec(v_n_899_);
return v_res_900_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_takeAsArray_go___redArg___lam__0___boxed(lean_object* v_toPure_901_, lean_object* v_acc_902_, lean_object* v_inst_903_, lean_object* v_n_904_, lean_object* v_____do__lift_905_){
_start:
{
lean_object* v_res_906_; 
v_res_906_ = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_takeAsArray_go___redArg___lam__0(v_toPure_901_, v_acc_902_, v_inst_903_, v_n_904_, v_____do__lift_905_);
lean_dec(v_n_904_);
return v_res_906_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_takeAsArray_go___redArg(lean_object* v_inst_907_, lean_object* v_r_908_, lean_object* v_acc_909_, lean_object* v_xs_910_){
_start:
{
lean_object* v_toApplicative_911_; lean_object* v_toBind_912_; lean_object* v_toPure_913_; lean_object* v_zero_914_; uint8_t v_isZero_915_; 
v_toApplicative_911_ = lean_ctor_get(v_inst_907_, 0);
v_toBind_912_ = lean_ctor_get(v_inst_907_, 1);
lean_inc(v_toBind_912_);
v_toPure_913_ = lean_ctor_get(v_toApplicative_911_, 1);
v_zero_914_ = lean_unsigned_to_nat(0u);
v_isZero_915_ = lean_nat_dec_eq(v_r_908_, v_zero_914_);
if (v_isZero_915_ == 1)
{
lean_object* v___x_916_; 
lean_inc(v_toPure_913_);
lean_dec(v_toBind_912_);
lean_dec(v_xs_910_);
lean_dec_ref(v_inst_907_);
v___x_916_ = lean_apply_2(v_toPure_913_, lean_box(0), v_acc_909_);
return v___x_916_;
}
else
{
lean_object* v_one_917_; lean_object* v_n_918_; lean_object* v___f_919_; lean_object* v___x_920_; lean_object* v___x_921_; 
v_one_917_ = lean_unsigned_to_nat(1u);
v_n_918_ = lean_nat_sub(v_r_908_, v_one_917_);
lean_inc_ref(v_inst_907_);
lean_inc(v_toPure_913_);
v___f_919_ = lean_alloc_closure((void*)(lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_takeAsArray_go___redArg___lam__0___boxed), 5, 4);
lean_closure_set(v___f_919_, 0, v_toPure_913_);
lean_closure_set(v___f_919_, 1, v_acc_909_);
lean_closure_set(v___f_919_, 2, v_inst_907_);
lean_closure_set(v___f_919_, 3, v_n_918_);
v___x_920_ = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_unconsImpl___redArg(v_inst_907_, v_xs_910_);
v___x_921_ = lean_apply_4(v_toBind_912_, lean_box(0), lean_box(0), v___x_920_, v___f_919_);
return v___x_921_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_takeAsArray_go___redArg___lam__0(lean_object* v_toPure_922_, lean_object* v_acc_923_, lean_object* v_inst_924_, lean_object* v_n_925_, lean_object* v_____do__lift_926_){
_start:
{
if (lean_obj_tag(v_____do__lift_926_) == 0)
{
lean_object* v___x_927_; 
lean_dec_ref(v_inst_924_);
v___x_927_ = lean_apply_2(v_toPure_922_, lean_box(0), v_acc_923_);
return v___x_927_;
}
else
{
lean_object* v_val_928_; lean_object* v_fst_929_; lean_object* v_snd_930_; lean_object* v___x_931_; lean_object* v___x_932_; 
lean_dec(v_toPure_922_);
v_val_928_ = lean_ctor_get(v_____do__lift_926_, 0);
lean_inc(v_val_928_);
lean_dec_ref_known(v_____do__lift_926_, 1);
v_fst_929_ = lean_ctor_get(v_val_928_, 0);
lean_inc(v_fst_929_);
v_snd_930_ = lean_ctor_get(v_val_928_, 1);
lean_inc(v_snd_930_);
lean_dec(v_val_928_);
v___x_931_ = lean_array_push(v_acc_923_, v_fst_929_);
v___x_932_ = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_takeAsArray_go___redArg(v_inst_924_, v_n_925_, v___x_931_, v_snd_930_);
return v___x_932_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_takeAsArray_go___redArg___boxed(lean_object* v_inst_933_, lean_object* v_r_934_, lean_object* v_acc_935_, lean_object* v_xs_936_){
_start:
{
lean_object* v_res_937_; 
v_res_937_ = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_takeAsArray_go___redArg(v_inst_933_, v_r_934_, v_acc_935_, v_xs_936_);
lean_dec(v_r_934_);
return v_res_937_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_takeAsArray_go(lean_object* v_m_938_, lean_object* v_00_u03b1_939_, lean_object* v_inst_940_, lean_object* v_r_941_, lean_object* v_acc_942_, lean_object* v_xs_943_){
_start:
{
lean_object* v___x_944_; 
v___x_944_ = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_takeAsArray_go___redArg(v_inst_940_, v_r_941_, v_acc_942_, v_xs_943_);
return v___x_944_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_takeAsArray_go___boxed(lean_object* v_m_945_, lean_object* v_00_u03b1_946_, lean_object* v_inst_947_, lean_object* v_r_948_, lean_object* v_acc_949_, lean_object* v_xs_950_){
_start:
{
lean_object* v_res_951_; 
v_res_951_ = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_takeAsArray_go(v_m_945_, v_00_u03b1_946_, v_inst_947_, v_r_948_, v_acc_949_, v_xs_950_);
lean_dec(v_r_948_);
return v_res_951_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_takeAsArray___redArg(lean_object* v_inst_952_, lean_object* v_xs_953_, lean_object* v_n_954_){
_start:
{
lean_object* v___x_955_; lean_object* v___x_956_; 
v___x_955_ = ((lean_object*)(lp_batteries_MLList_asArray___redArg___closed__0));
v___x_956_ = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_takeAsArray_go___redArg(v_inst_952_, v_n_954_, v___x_955_, v_xs_953_);
return v___x_956_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_takeAsArray___redArg___boxed(lean_object* v_inst_957_, lean_object* v_xs_958_, lean_object* v_n_959_){
_start:
{
lean_object* v_res_960_; 
v_res_960_ = lp_batteries_MLList_takeAsArray___redArg(v_inst_957_, v_xs_958_, v_n_959_);
lean_dec(v_n_959_);
return v_res_960_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_takeAsArray(lean_object* v_m_961_, lean_object* v_00_u03b1_962_, lean_object* v_inst_963_, lean_object* v_xs_964_, lean_object* v_n_965_){
_start:
{
lean_object* v___x_966_; 
v___x_966_ = lp_batteries_MLList_takeAsArray___redArg(v_inst_963_, v_xs_964_, v_n_965_);
return v___x_966_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_takeAsArray___boxed(lean_object* v_m_967_, lean_object* v_00_u03b1_968_, lean_object* v_inst_969_, lean_object* v_xs_970_, lean_object* v_n_971_){
_start:
{
lean_object* v_res_972_; 
v_res_972_ = lp_batteries_MLList_takeAsArray(v_m_967_, v_00_u03b1_968_, v_inst_969_, v_xs_970_, v_n_971_);
lean_dec(v_n_971_);
return v_res_972_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_take___redArg___lam__0(lean_object* v_x_973_){
_start:
{
lean_object* v___x_974_; 
v___x_974_ = lean_box(0);
return v___x_974_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_take___redArg___lam__1___boxed(lean_object* v_inst_976_, lean_object* v_n_977_, lean_object* v_h_978_, lean_object* v_l_979_){
_start:
{
lean_object* v_res_980_; 
v_res_980_ = lp_batteries_MLList_take___redArg___lam__1(v_inst_976_, v_n_977_, v_h_978_, v_l_979_);
lean_dec(v_n_977_);
return v_res_980_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_take___redArg(lean_object* v_inst_981_, lean_object* v_xs_982_, lean_object* v_x_983_){
_start:
{
lean_object* v_zero_984_; uint8_t v_isZero_985_; 
v_zero_984_ = lean_unsigned_to_nat(0u);
v_isZero_985_ = lean_nat_dec_eq(v_x_983_, v_zero_984_);
if (v_isZero_985_ == 1)
{
lean_object* v___x_986_; 
lean_dec(v_xs_982_);
lean_dec_ref(v_inst_981_);
v___x_986_ = lean_box(0);
return v___x_986_;
}
else
{
lean_object* v___f_987_; lean_object* v_one_988_; lean_object* v_n_989_; lean_object* v___f_990_; lean_object* v___x_991_; 
v___f_987_ = ((lean_object*)(lp_batteries_MLList_take___redArg___closed__0));
v_one_988_ = lean_unsigned_to_nat(1u);
v_n_989_ = lean_nat_sub(v_x_983_, v_one_988_);
lean_inc_ref(v_inst_981_);
v___f_990_ = lean_alloc_closure((void*)(lp_batteries_MLList_take___redArg___lam__1___boxed), 4, 2);
lean_closure_set(v___f_990_, 0, v_inst_981_);
lean_closure_set(v___f_990_, 1, v_n_989_);
v___x_991_ = lp_batteries_MLList_cases___redArg(v_inst_981_, v_xs_982_, v___f_987_, v___f_990_);
return v___x_991_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_take___redArg___lam__1(lean_object* v_inst_992_, lean_object* v_n_993_, lean_object* v_h_994_, lean_object* v_l_995_){
_start:
{
lean_object* v___x_996_; lean_object* v___x_997_; 
v___x_996_ = lp_batteries_MLList_take___redArg(v_inst_992_, v_l_995_, v_n_993_);
v___x_997_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_997_, 0, v_h_994_);
lean_ctor_set(v___x_997_, 1, v___x_996_);
return v___x_997_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_take___redArg___boxed(lean_object* v_inst_998_, lean_object* v_xs_999_, lean_object* v_x_1000_){
_start:
{
lean_object* v_res_1001_; 
v_res_1001_ = lp_batteries_MLList_take___redArg(v_inst_998_, v_xs_999_, v_x_1000_);
lean_dec(v_x_1000_);
return v_res_1001_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_take(lean_object* v_m_1002_, lean_object* v_00_u03b1_1003_, lean_object* v_inst_1004_, lean_object* v_xs_1005_, lean_object* v_x_1006_){
_start:
{
lean_object* v___x_1007_; 
v___x_1007_ = lp_batteries_MLList_take___redArg(v_inst_1004_, v_xs_1005_, v_x_1006_);
return v___x_1007_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_take___boxed(lean_object* v_m_1008_, lean_object* v_00_u03b1_1009_, lean_object* v_inst_1010_, lean_object* v_xs_1011_, lean_object* v_x_1012_){
_start:
{
lean_object* v_res_1013_; 
v_res_1013_ = lp_batteries_MLList_take(v_m_1008_, v_00_u03b1_1009_, v_inst_1010_, v_xs_1011_, v_x_1012_);
lean_dec(v_x_1012_);
return v_res_1013_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_drop___redArg___lam__0(lean_object* v_x_1014_){
_start:
{
lean_object* v___x_1015_; 
v___x_1015_ = lean_box(0);
return v___x_1015_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_drop___redArg___lam__1___boxed(lean_object* v_inst_1017_, lean_object* v_n_1018_, lean_object* v_x_1019_, lean_object* v_l_1020_){
_start:
{
lean_object* v_res_1021_; 
v_res_1021_ = lp_batteries_MLList_drop___redArg___lam__1(v_inst_1017_, v_n_1018_, v_x_1019_, v_l_1020_);
lean_dec(v_x_1019_);
lean_dec(v_n_1018_);
return v_res_1021_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_drop___redArg(lean_object* v_inst_1022_, lean_object* v_xs_1023_, lean_object* v_x_1024_){
_start:
{
lean_object* v_zero_1025_; uint8_t v_isZero_1026_; 
v_zero_1025_ = lean_unsigned_to_nat(0u);
v_isZero_1026_ = lean_nat_dec_eq(v_x_1024_, v_zero_1025_);
if (v_isZero_1026_ == 1)
{
lean_dec_ref(v_inst_1022_);
return v_xs_1023_;
}
else
{
lean_object* v___f_1027_; lean_object* v_one_1028_; lean_object* v_n_1029_; lean_object* v___f_1030_; lean_object* v___x_1031_; 
v___f_1027_ = ((lean_object*)(lp_batteries_MLList_drop___redArg___closed__0));
v_one_1028_ = lean_unsigned_to_nat(1u);
v_n_1029_ = lean_nat_sub(v_x_1024_, v_one_1028_);
lean_inc_ref(v_inst_1022_);
v___f_1030_ = lean_alloc_closure((void*)(lp_batteries_MLList_drop___redArg___lam__1___boxed), 4, 2);
lean_closure_set(v___f_1030_, 0, v_inst_1022_);
lean_closure_set(v___f_1030_, 1, v_n_1029_);
v___x_1031_ = lp_batteries_MLList_cases___redArg(v_inst_1022_, v_xs_1023_, v___f_1027_, v___f_1030_);
return v___x_1031_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_drop___redArg___lam__1(lean_object* v_inst_1032_, lean_object* v_n_1033_, lean_object* v_x_1034_, lean_object* v_l_1035_){
_start:
{
lean_object* v___x_1036_; 
v___x_1036_ = lp_batteries_MLList_drop___redArg(v_inst_1032_, v_l_1035_, v_n_1033_);
return v___x_1036_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_drop___redArg___boxed(lean_object* v_inst_1037_, lean_object* v_xs_1038_, lean_object* v_x_1039_){
_start:
{
lean_object* v_res_1040_; 
v_res_1040_ = lp_batteries_MLList_drop___redArg(v_inst_1037_, v_xs_1038_, v_x_1039_);
lean_dec(v_x_1039_);
return v_res_1040_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_drop(lean_object* v_m_1041_, lean_object* v_00_u03b1_1042_, lean_object* v_inst_1043_, lean_object* v_xs_1044_, lean_object* v_x_1045_){
_start:
{
lean_object* v___x_1046_; 
v___x_1046_ = lp_batteries_MLList_drop___redArg(v_inst_1043_, v_xs_1044_, v_x_1045_);
return v___x_1046_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_drop___boxed(lean_object* v_m_1047_, lean_object* v_00_u03b1_1048_, lean_object* v_inst_1049_, lean_object* v_xs_1050_, lean_object* v_x_1051_){
_start:
{
lean_object* v_res_1052_; 
v_res_1052_ = lp_batteries_MLList_drop(v_m_1047_, v_00_u03b1_1048_, v_inst_1049_, v_xs_1050_, v_x_1051_);
lean_dec(v_x_1051_);
return v_res_1052_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_mapM___redArg___lam__0(lean_object* v_f_1053_, lean_object* v_x_1054_, lean_object* v_toBind_1055_, lean_object* v___f_1056_, lean_object* v_x_1057_){
_start:
{
lean_object* v___x_1058_; lean_object* v___x_1059_; 
v___x_1058_ = lean_apply_1(v_f_1053_, v_x_1054_);
v___x_1059_ = lean_apply_4(v_toBind_1055_, lean_box(0), lean_box(0), v___x_1058_, v___f_1056_);
return v___x_1059_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_mapM___redArg___lam__2(lean_object* v_inst_1060_, lean_object* v_f_1061_, lean_object* v_toPure_1062_, lean_object* v_toBind_1063_, lean_object* v_x_1064_, lean_object* v_xs_1065_){
_start:
{
lean_object* v___f_1066_; lean_object* v___f_1067_; lean_object* v___x_1068_; 
lean_inc(v_f_1061_);
v___f_1066_ = lean_alloc_closure((void*)(lp_batteries_MLList_mapM___redArg___lam__1), 5, 4);
lean_closure_set(v___f_1066_, 0, v_inst_1060_);
lean_closure_set(v___f_1066_, 1, v_f_1061_);
lean_closure_set(v___f_1066_, 2, v_xs_1065_);
lean_closure_set(v___f_1066_, 3, v_toPure_1062_);
v___f_1067_ = lean_alloc_closure((void*)(lp_batteries_MLList_mapM___redArg___lam__0), 5, 4);
lean_closure_set(v___f_1067_, 0, v_f_1061_);
lean_closure_set(v___f_1067_, 1, v_x_1064_);
lean_closure_set(v___f_1067_, 2, v_toBind_1063_);
lean_closure_set(v___f_1067_, 3, v___f_1066_);
v___x_1068_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1068_, 0, v___f_1067_);
return v___x_1068_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_mapM___redArg(lean_object* v_inst_1069_, lean_object* v_f_1070_, lean_object* v_xs_1071_){
_start:
{
lean_object* v_toApplicative_1072_; lean_object* v_toBind_1073_; lean_object* v_toPure_1074_; lean_object* v___f_1075_; lean_object* v___f_1076_; lean_object* v___x_1077_; 
v_toApplicative_1072_ = lean_ctor_get(v_inst_1069_, 0);
v_toBind_1073_ = lean_ctor_get(v_inst_1069_, 1);
v_toPure_1074_ = lean_ctor_get(v_toApplicative_1072_, 1);
v___f_1075_ = ((lean_object*)(lp_batteries_MLList_take___redArg___closed__0));
lean_inc(v_toBind_1073_);
lean_inc(v_toPure_1074_);
lean_inc_ref(v_inst_1069_);
v___f_1076_ = lean_alloc_closure((void*)(lp_batteries_MLList_mapM___redArg___lam__2), 6, 4);
lean_closure_set(v___f_1076_, 0, v_inst_1069_);
lean_closure_set(v___f_1076_, 1, v_f_1070_);
lean_closure_set(v___f_1076_, 2, v_toPure_1074_);
lean_closure_set(v___f_1076_, 3, v_toBind_1073_);
v___x_1077_ = lp_batteries_MLList_cases___redArg(v_inst_1069_, v_xs_1071_, v___f_1075_, v___f_1076_);
return v___x_1077_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_mapM___redArg___lam__1(lean_object* v_inst_1078_, lean_object* v_f_1079_, lean_object* v_xs_1080_, lean_object* v_toPure_1081_, lean_object* v_____do__lift_1082_){
_start:
{
lean_object* v___x_1083_; lean_object* v___x_1084_; lean_object* v___x_1085_; 
v___x_1083_ = lp_batteries_MLList_mapM___redArg(v_inst_1078_, v_f_1079_, v_xs_1080_);
v___x_1084_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1084_, 0, v_____do__lift_1082_);
lean_ctor_set(v___x_1084_, 1, v___x_1083_);
v___x_1085_ = lean_apply_2(v_toPure_1081_, lean_box(0), v___x_1084_);
return v___x_1085_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_mapM(lean_object* v_m_1086_, lean_object* v_00_u03b1_1087_, lean_object* v_00_u03b2_1088_, lean_object* v_inst_1089_, lean_object* v_f_1090_, lean_object* v_xs_1091_){
_start:
{
lean_object* v___x_1092_; 
v___x_1092_ = lp_batteries_MLList_mapM___redArg(v_inst_1089_, v_f_1090_, v_xs_1091_);
return v___x_1092_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_map___redArg___lam__0(lean_object* v_f_1093_, lean_object* v_toPure_1094_, lean_object* v_a_1095_){
_start:
{
lean_object* v___x_1096_; lean_object* v___x_1097_; 
v___x_1096_ = lean_apply_1(v_f_1093_, v_a_1095_);
v___x_1097_ = lean_apply_2(v_toPure_1094_, lean_box(0), v___x_1096_);
return v___x_1097_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_map___redArg(lean_object* v_inst_1098_, lean_object* v_f_1099_, lean_object* v_L_1100_){
_start:
{
lean_object* v_toApplicative_1101_; lean_object* v_toPure_1102_; lean_object* v___f_1103_; lean_object* v___x_1104_; 
v_toApplicative_1101_ = lean_ctor_get(v_inst_1098_, 0);
v_toPure_1102_ = lean_ctor_get(v_toApplicative_1101_, 1);
lean_inc(v_toPure_1102_);
v___f_1103_ = lean_alloc_closure((void*)(lp_batteries_MLList_map___redArg___lam__0), 3, 2);
lean_closure_set(v___f_1103_, 0, v_f_1099_);
lean_closure_set(v___f_1103_, 1, v_toPure_1102_);
v___x_1104_ = lp_batteries_MLList_mapM___redArg(v_inst_1098_, v___f_1103_, v_L_1100_);
return v___x_1104_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_map(lean_object* v_m_1105_, lean_object* v_00_u03b1_1106_, lean_object* v_00_u03b2_1107_, lean_object* v_inst_1108_, lean_object* v_f_1109_, lean_object* v_L_1110_){
_start:
{
lean_object* v___x_1111_; 
v___x_1111_ = lp_batteries_MLList_map___redArg(v_inst_1108_, v_f_1109_, v_L_1110_);
return v___x_1111_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_filterM___redArg___lam__2(lean_object* v_toPure_1112_, lean_object* v_x_1113_){
_start:
{
lean_object* v___x_1114_; lean_object* v___x_1115_; 
v___x_1114_ = lean_box(0);
v___x_1115_ = lean_apply_2(v_toPure_1112_, lean_box(0), v___x_1114_);
return v___x_1115_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_filterM___redArg(lean_object* v_inst_1116_, lean_object* v_p_1117_, lean_object* v_L_1118_){
_start:
{
lean_object* v_toApplicative_1119_; lean_object* v_toBind_1120_; lean_object* v_toPure_1121_; lean_object* v___f_1122_; lean_object* v___f_1123_; lean_object* v___x_1124_; 
v_toApplicative_1119_ = lean_ctor_get(v_inst_1116_, 0);
v_toBind_1120_ = lean_ctor_get(v_inst_1116_, 1);
v_toPure_1121_ = lean_ctor_get(v_toApplicative_1119_, 1);
lean_inc(v_toBind_1120_);
lean_inc_n(v_toPure_1121_, 2);
lean_inc_ref(v_inst_1116_);
v___f_1122_ = lean_alloc_closure((void*)(lp_batteries_MLList_filterM___redArg___lam__1), 6, 4);
lean_closure_set(v___f_1122_, 0, v_inst_1116_);
lean_closure_set(v___f_1122_, 1, v_p_1117_);
lean_closure_set(v___f_1122_, 2, v_toPure_1121_);
lean_closure_set(v___f_1122_, 3, v_toBind_1120_);
v___f_1123_ = lean_alloc_closure((void*)(lp_batteries_MLList_filterM___redArg___lam__2), 2, 1);
lean_closure_set(v___f_1123_, 0, v_toPure_1121_);
v___x_1124_ = lp_batteries_MLList_casesM___redArg(v_inst_1116_, v_L_1118_, v___f_1123_, v___f_1122_);
return v___x_1124_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_filterM___redArg___lam__0(lean_object* v_inst_1125_, lean_object* v_p_1126_, lean_object* v_xs_1127_, lean_object* v_toPure_1128_, lean_object* v_x_1129_, uint8_t v_____do__lift_1130_){
_start:
{
if (v_____do__lift_1130_ == 0)
{
lean_object* v___x_1131_; lean_object* v___x_1132_; 
lean_dec(v_x_1129_);
v___x_1131_ = lp_batteries_MLList_filterM___redArg(v_inst_1125_, v_p_1126_, v_xs_1127_);
v___x_1132_ = lean_apply_2(v_toPure_1128_, lean_box(0), v___x_1131_);
return v___x_1132_;
}
else
{
lean_object* v___x_1133_; lean_object* v___x_1134_; lean_object* v___x_1135_; 
v___x_1133_ = lp_batteries_MLList_filterM___redArg(v_inst_1125_, v_p_1126_, v_xs_1127_);
v___x_1134_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1134_, 0, v_x_1129_);
lean_ctor_set(v___x_1134_, 1, v___x_1133_);
v___x_1135_ = lean_apply_2(v_toPure_1128_, lean_box(0), v___x_1134_);
return v___x_1135_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_filterM___redArg___lam__0___boxed(lean_object* v_inst_1136_, lean_object* v_p_1137_, lean_object* v_xs_1138_, lean_object* v_toPure_1139_, lean_object* v_x_1140_, lean_object* v_____do__lift_1141_){
_start:
{
uint8_t v_____do__lift_87__boxed_1142_; lean_object* v_res_1143_; 
v_____do__lift_87__boxed_1142_ = lean_unbox(v_____do__lift_1141_);
v_res_1143_ = lp_batteries_MLList_filterM___redArg___lam__0(v_inst_1136_, v_p_1137_, v_xs_1138_, v_toPure_1139_, v_x_1140_, v_____do__lift_87__boxed_1142_);
return v_res_1143_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_filterM___redArg___lam__1(lean_object* v_inst_1144_, lean_object* v_p_1145_, lean_object* v_toPure_1146_, lean_object* v_toBind_1147_, lean_object* v_x_1148_, lean_object* v_xs_1149_){
_start:
{
lean_object* v___f_1150_; lean_object* v___x_1151_; lean_object* v___x_1152_; 
lean_inc(v_x_1148_);
lean_inc(v_p_1145_);
v___f_1150_ = lean_alloc_closure((void*)(lp_batteries_MLList_filterM___redArg___lam__0___boxed), 6, 5);
lean_closure_set(v___f_1150_, 0, v_inst_1144_);
lean_closure_set(v___f_1150_, 1, v_p_1145_);
lean_closure_set(v___f_1150_, 2, v_xs_1149_);
lean_closure_set(v___f_1150_, 3, v_toPure_1146_);
lean_closure_set(v___f_1150_, 4, v_x_1148_);
v___x_1151_ = lean_apply_1(v_p_1145_, v_x_1148_);
v___x_1152_ = lean_apply_4(v_toBind_1147_, lean_box(0), lean_box(0), v___x_1151_, v___f_1150_);
return v___x_1152_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_filterM(lean_object* v_m_1153_, lean_object* v_00_u03b1_1154_, lean_object* v_inst_1155_, lean_object* v_p_1156_, lean_object* v_L_1157_){
_start:
{
lean_object* v___x_1158_; 
v___x_1158_ = lp_batteries_MLList_filterM___redArg(v_inst_1155_, v_p_1156_, v_L_1157_);
return v___x_1158_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_filter___redArg___lam__0(lean_object* v_p_1159_, lean_object* v_toPure_1160_, lean_object* v_a_1161_){
_start:
{
lean_object* v___x_1162_; lean_object* v___x_1163_; 
v___x_1162_ = lean_apply_1(v_p_1159_, v_a_1161_);
v___x_1163_ = lean_apply_2(v_toPure_1160_, lean_box(0), v___x_1162_);
return v___x_1163_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_filter___redArg(lean_object* v_inst_1164_, lean_object* v_p_1165_, lean_object* v_L_1166_){
_start:
{
lean_object* v_toApplicative_1167_; lean_object* v_toPure_1168_; lean_object* v___f_1169_; lean_object* v___x_1170_; 
v_toApplicative_1167_ = lean_ctor_get(v_inst_1164_, 0);
v_toPure_1168_ = lean_ctor_get(v_toApplicative_1167_, 1);
lean_inc(v_toPure_1168_);
v___f_1169_ = lean_alloc_closure((void*)(lp_batteries_MLList_filter___redArg___lam__0), 3, 2);
lean_closure_set(v___f_1169_, 0, v_p_1165_);
lean_closure_set(v___f_1169_, 1, v_toPure_1168_);
v___x_1170_ = lp_batteries_MLList_filterM___redArg(v_inst_1164_, v___f_1169_, v_L_1166_);
return v___x_1170_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_filter(lean_object* v_m_1171_, lean_object* v_00_u03b1_1172_, lean_object* v_inst_1173_, lean_object* v_p_1174_, lean_object* v_L_1175_){
_start:
{
lean_object* v___x_1176_; 
v___x_1176_ = lp_batteries_MLList_filter___redArg(v_inst_1173_, v_p_1174_, v_L_1175_);
return v___x_1176_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_filterMapM___redArg(lean_object* v_inst_1177_, lean_object* v_f_1178_, lean_object* v_xs_1179_){
_start:
{
lean_object* v_toApplicative_1180_; lean_object* v_toBind_1181_; lean_object* v_toPure_1182_; lean_object* v___f_1183_; lean_object* v___f_1184_; lean_object* v___x_1185_; 
v_toApplicative_1180_ = lean_ctor_get(v_inst_1177_, 0);
v_toBind_1181_ = lean_ctor_get(v_inst_1177_, 1);
v_toPure_1182_ = lean_ctor_get(v_toApplicative_1180_, 1);
lean_inc(v_toBind_1181_);
lean_inc_n(v_toPure_1182_, 2);
lean_inc_ref(v_inst_1177_);
v___f_1183_ = lean_alloc_closure((void*)(lp_batteries_MLList_filterMapM___redArg___lam__1), 6, 4);
lean_closure_set(v___f_1183_, 0, v_inst_1177_);
lean_closure_set(v___f_1183_, 1, v_f_1178_);
lean_closure_set(v___f_1183_, 2, v_toPure_1182_);
lean_closure_set(v___f_1183_, 3, v_toBind_1181_);
v___f_1184_ = lean_alloc_closure((void*)(lp_batteries_MLList_filterM___redArg___lam__2), 2, 1);
lean_closure_set(v___f_1184_, 0, v_toPure_1182_);
v___x_1185_ = lp_batteries_MLList_casesM___redArg(v_inst_1177_, v_xs_1179_, v___f_1184_, v___f_1183_);
return v___x_1185_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_filterMapM___redArg___lam__0(lean_object* v_inst_1186_, lean_object* v_f_1187_, lean_object* v_xs_1188_, lean_object* v_toPure_1189_, lean_object* v_____do__lift_1190_){
_start:
{
if (lean_obj_tag(v_____do__lift_1190_) == 0)
{
lean_object* v___x_1191_; lean_object* v___x_1192_; 
v___x_1191_ = lp_batteries_MLList_filterMapM___redArg(v_inst_1186_, v_f_1187_, v_xs_1188_);
v___x_1192_ = lean_apply_2(v_toPure_1189_, lean_box(0), v___x_1191_);
return v___x_1192_;
}
else
{
lean_object* v_val_1193_; lean_object* v___x_1194_; lean_object* v___x_1195_; lean_object* v___x_1196_; 
v_val_1193_ = lean_ctor_get(v_____do__lift_1190_, 0);
v___x_1194_ = lp_batteries_MLList_filterMapM___redArg(v_inst_1186_, v_f_1187_, v_xs_1188_);
lean_inc(v_val_1193_);
v___x_1195_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1195_, 0, v_val_1193_);
lean_ctor_set(v___x_1195_, 1, v___x_1194_);
v___x_1196_ = lean_apply_2(v_toPure_1189_, lean_box(0), v___x_1195_);
return v___x_1196_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_filterMapM___redArg___lam__0___boxed(lean_object* v_inst_1197_, lean_object* v_f_1198_, lean_object* v_xs_1199_, lean_object* v_toPure_1200_, lean_object* v_____do__lift_1201_){
_start:
{
lean_object* v_res_1202_; 
v_res_1202_ = lp_batteries_MLList_filterMapM___redArg___lam__0(v_inst_1197_, v_f_1198_, v_xs_1199_, v_toPure_1200_, v_____do__lift_1201_);
lean_dec(v_____do__lift_1201_);
return v_res_1202_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_filterMapM___redArg___lam__1(lean_object* v_inst_1203_, lean_object* v_f_1204_, lean_object* v_toPure_1205_, lean_object* v_toBind_1206_, lean_object* v_x_1207_, lean_object* v_xs_1208_){
_start:
{
lean_object* v___f_1209_; lean_object* v___x_1210_; lean_object* v___x_1211_; 
lean_inc(v_f_1204_);
v___f_1209_ = lean_alloc_closure((void*)(lp_batteries_MLList_filterMapM___redArg___lam__0___boxed), 5, 4);
lean_closure_set(v___f_1209_, 0, v_inst_1203_);
lean_closure_set(v___f_1209_, 1, v_f_1204_);
lean_closure_set(v___f_1209_, 2, v_xs_1208_);
lean_closure_set(v___f_1209_, 3, v_toPure_1205_);
v___x_1210_ = lean_apply_1(v_f_1204_, v_x_1207_);
v___x_1211_ = lean_apply_4(v_toBind_1206_, lean_box(0), lean_box(0), v___x_1210_, v___f_1209_);
return v___x_1211_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_filterMapM(lean_object* v_m_1212_, lean_object* v_00_u03b1_1213_, lean_object* v_00_u03b2_1214_, lean_object* v_inst_1215_, lean_object* v_f_1216_, lean_object* v_xs_1217_){
_start:
{
lean_object* v___x_1218_; 
v___x_1218_ = lp_batteries_MLList_filterMapM___redArg(v_inst_1215_, v_f_1216_, v_xs_1217_);
return v___x_1218_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_filterMap___redArg___lam__0(lean_object* v_f_1219_, lean_object* v_toPure_1220_, lean_object* v_a_1221_){
_start:
{
lean_object* v___x_1222_; lean_object* v___x_1223_; 
v___x_1222_ = lean_apply_1(v_f_1219_, v_a_1221_);
v___x_1223_ = lean_apply_2(v_toPure_1220_, lean_box(0), v___x_1222_);
return v___x_1223_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_filterMap___redArg(lean_object* v_inst_1224_, lean_object* v_f_1225_, lean_object* v_xs_1226_){
_start:
{
lean_object* v_toApplicative_1227_; lean_object* v_toPure_1228_; lean_object* v___f_1229_; lean_object* v___x_1230_; 
v_toApplicative_1227_ = lean_ctor_get(v_inst_1224_, 0);
v_toPure_1228_ = lean_ctor_get(v_toApplicative_1227_, 1);
lean_inc(v_toPure_1228_);
v___f_1229_ = lean_alloc_closure((void*)(lp_batteries_MLList_filterMap___redArg___lam__0), 3, 2);
lean_closure_set(v___f_1229_, 0, v_f_1225_);
lean_closure_set(v___f_1229_, 1, v_toPure_1228_);
v___x_1230_ = lp_batteries_MLList_filterMapM___redArg(v_inst_1224_, v___f_1229_, v_xs_1226_);
return v___x_1230_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_filterMap(lean_object* v_m_1231_, lean_object* v_00_u03b1_1232_, lean_object* v_00_u03b2_1233_, lean_object* v_inst_1234_, lean_object* v_f_1235_, lean_object* v_xs_1236_){
_start:
{
lean_object* v___x_1237_; 
v___x_1237_ = lp_batteries_MLList_filterMap___redArg(v_inst_1234_, v_f_1235_, v_xs_1236_);
return v___x_1237_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_takeWhileM___redArg(lean_object* v_inst_1238_, lean_object* v_f_1239_, lean_object* v_L_1240_){
_start:
{
lean_object* v_toApplicative_1241_; lean_object* v_toBind_1242_; lean_object* v_toPure_1243_; lean_object* v___f_1244_; lean_object* v___f_1245_; lean_object* v___x_1246_; 
v_toApplicative_1241_ = lean_ctor_get(v_inst_1238_, 0);
v_toBind_1242_ = lean_ctor_get(v_inst_1238_, 1);
v_toPure_1243_ = lean_ctor_get(v_toApplicative_1241_, 1);
lean_inc(v_toBind_1242_);
lean_inc_ref(v_inst_1238_);
lean_inc_n(v_toPure_1243_, 2);
v___f_1244_ = lean_alloc_closure((void*)(lp_batteries_MLList_takeWhileM___redArg___lam__1), 6, 4);
lean_closure_set(v___f_1244_, 0, v_toPure_1243_);
lean_closure_set(v___f_1244_, 1, v_inst_1238_);
lean_closure_set(v___f_1244_, 2, v_f_1239_);
lean_closure_set(v___f_1244_, 3, v_toBind_1242_);
v___f_1245_ = lean_alloc_closure((void*)(lp_batteries_MLList_filterM___redArg___lam__2), 2, 1);
lean_closure_set(v___f_1245_, 0, v_toPure_1243_);
v___x_1246_ = lp_batteries_MLList_casesM___redArg(v_inst_1238_, v_L_1240_, v___f_1245_, v___f_1244_);
return v___x_1246_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_takeWhileM___redArg___lam__0(lean_object* v_toPure_1247_, lean_object* v_inst_1248_, lean_object* v_f_1249_, lean_object* v_xs_1250_, lean_object* v_x_1251_, uint8_t v_____do__lift_1252_){
_start:
{
if (v_____do__lift_1252_ == 0)
{
lean_object* v___x_1253_; lean_object* v___x_1254_; 
lean_dec(v_x_1251_);
lean_dec(v_xs_1250_);
lean_dec(v_f_1249_);
lean_dec_ref(v_inst_1248_);
v___x_1253_ = lean_box(0);
v___x_1254_ = lean_apply_2(v_toPure_1247_, lean_box(0), v___x_1253_);
return v___x_1254_;
}
else
{
lean_object* v___x_1255_; lean_object* v___x_1256_; lean_object* v___x_1257_; 
v___x_1255_ = lp_batteries_MLList_takeWhileM___redArg(v_inst_1248_, v_f_1249_, v_xs_1250_);
v___x_1256_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1256_, 0, v_x_1251_);
lean_ctor_set(v___x_1256_, 1, v___x_1255_);
v___x_1257_ = lean_apply_2(v_toPure_1247_, lean_box(0), v___x_1256_);
return v___x_1257_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_takeWhileM___redArg___lam__0___boxed(lean_object* v_toPure_1258_, lean_object* v_inst_1259_, lean_object* v_f_1260_, lean_object* v_xs_1261_, lean_object* v_x_1262_, lean_object* v_____do__lift_1263_){
_start:
{
uint8_t v_____do__lift_99__boxed_1264_; lean_object* v_res_1265_; 
v_____do__lift_99__boxed_1264_ = lean_unbox(v_____do__lift_1263_);
v_res_1265_ = lp_batteries_MLList_takeWhileM___redArg___lam__0(v_toPure_1258_, v_inst_1259_, v_f_1260_, v_xs_1261_, v_x_1262_, v_____do__lift_99__boxed_1264_);
return v_res_1265_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_takeWhileM___redArg___lam__1(lean_object* v_toPure_1266_, lean_object* v_inst_1267_, lean_object* v_f_1268_, lean_object* v_toBind_1269_, lean_object* v_x_1270_, lean_object* v_xs_1271_){
_start:
{
lean_object* v___f_1272_; lean_object* v___x_1273_; lean_object* v___x_1274_; 
lean_inc(v_x_1270_);
lean_inc(v_f_1268_);
v___f_1272_ = lean_alloc_closure((void*)(lp_batteries_MLList_takeWhileM___redArg___lam__0___boxed), 6, 5);
lean_closure_set(v___f_1272_, 0, v_toPure_1266_);
lean_closure_set(v___f_1272_, 1, v_inst_1267_);
lean_closure_set(v___f_1272_, 2, v_f_1268_);
lean_closure_set(v___f_1272_, 3, v_xs_1271_);
lean_closure_set(v___f_1272_, 4, v_x_1270_);
v___x_1273_ = lean_apply_1(v_f_1268_, v_x_1270_);
v___x_1274_ = lean_apply_4(v_toBind_1269_, lean_box(0), lean_box(0), v___x_1273_, v___f_1272_);
return v___x_1274_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_takeWhileM(lean_object* v_m_1275_, lean_object* v_00_u03b1_1276_, lean_object* v_inst_1277_, lean_object* v_f_1278_, lean_object* v_L_1279_){
_start:
{
lean_object* v___x_1280_; 
v___x_1280_ = lp_batteries_MLList_takeWhileM___redArg(v_inst_1277_, v_f_1278_, v_L_1279_);
return v___x_1280_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_takeWhile___redArg___lam__0(lean_object* v_f_1281_, lean_object* v_toPure_1282_, lean_object* v_a_1283_){
_start:
{
lean_object* v___x_1284_; lean_object* v___x_1285_; 
v___x_1284_ = lean_apply_1(v_f_1281_, v_a_1283_);
v___x_1285_ = lean_apply_2(v_toPure_1282_, lean_box(0), v___x_1284_);
return v___x_1285_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_takeWhile___redArg(lean_object* v_inst_1286_, lean_object* v_f_1287_, lean_object* v_L_1288_){
_start:
{
lean_object* v_toApplicative_1289_; lean_object* v_toPure_1290_; lean_object* v___f_1291_; lean_object* v___x_1292_; 
v_toApplicative_1289_ = lean_ctor_get(v_inst_1286_, 0);
v_toPure_1290_ = lean_ctor_get(v_toApplicative_1289_, 1);
lean_inc(v_toPure_1290_);
v___f_1291_ = lean_alloc_closure((void*)(lp_batteries_MLList_takeWhile___redArg___lam__0), 3, 2);
lean_closure_set(v___f_1291_, 0, v_f_1287_);
lean_closure_set(v___f_1291_, 1, v_toPure_1290_);
v___x_1292_ = lp_batteries_MLList_takeWhileM___redArg(v_inst_1286_, v___f_1291_, v_L_1288_);
return v___x_1292_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_takeWhile(lean_object* v_m_1293_, lean_object* v_00_u03b1_1294_, lean_object* v_inst_1295_, lean_object* v_f_1296_, lean_object* v_L_1297_){
_start:
{
lean_object* v___x_1298_; 
v___x_1298_ = lp_batteries_MLList_takeWhile___redArg(v_inst_1295_, v_f_1296_, v_L_1297_);
return v___x_1298_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_append___redArg___lam__0(lean_object* v_inst_1299_, lean_object* v_ys_1300_, lean_object* v_x_1301_, lean_object* v_xs_1302_){
_start:
{
lean_object* v___x_1303_; lean_object* v___x_1304_; 
v___x_1303_ = lp_batteries_MLList_append___redArg(v_inst_1299_, v_xs_1302_, v_ys_1300_);
v___x_1304_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1304_, 0, v_x_1301_);
lean_ctor_set(v___x_1304_, 1, v___x_1303_);
return v___x_1304_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_append___redArg(lean_object* v_inst_1305_, lean_object* v_xs_1306_, lean_object* v_ys_1307_){
_start:
{
lean_object* v___f_1308_; lean_object* v___x_1309_; 
lean_inc(v_ys_1307_);
lean_inc_ref(v_inst_1305_);
v___f_1308_ = lean_alloc_closure((void*)(lp_batteries_MLList_append___redArg___lam__0), 4, 2);
lean_closure_set(v___f_1308_, 0, v_inst_1305_);
lean_closure_set(v___f_1308_, 1, v_ys_1307_);
v___x_1309_ = lp_batteries_MLList_cases___redArg(v_inst_1305_, v_xs_1306_, v_ys_1307_, v___f_1308_);
return v___x_1309_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_append(lean_object* v_m_1310_, lean_object* v_00_u03b1_1311_, lean_object* v_inst_1312_, lean_object* v_xs_1313_, lean_object* v_ys_1314_){
_start:
{
lean_object* v___x_1315_; 
v___x_1315_ = lp_batteries_MLList_append___redArg(v_inst_1312_, v_xs_1313_, v_ys_1314_);
return v___x_1315_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_join___redArg___lam__0(lean_object* v_inst_1316_, lean_object* v_x_1317_, lean_object* v_xs_1318_){
_start:
{
lean_object* v___f_1319_; lean_object* v___x_1320_; 
lean_inc_ref(v_inst_1316_);
v___f_1319_ = lean_alloc_closure((void*)(lp_batteries_MLList_join___redArg___lam__1), 3, 2);
lean_closure_set(v___f_1319_, 0, v_inst_1316_);
lean_closure_set(v___f_1319_, 1, v_xs_1318_);
v___x_1320_ = lp_batteries_MLList_append___redArg(v_inst_1316_, v_x_1317_, v___f_1319_);
return v___x_1320_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_join___redArg(lean_object* v_inst_1321_, lean_object* v_xs_1322_){
_start:
{
lean_object* v___f_1323_; lean_object* v___f_1324_; lean_object* v___x_1325_; 
v___f_1323_ = ((lean_object*)(lp_batteries_MLList_take___redArg___closed__0));
lean_inc_ref(v_inst_1321_);
v___f_1324_ = lean_alloc_closure((void*)(lp_batteries_MLList_join___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1324_, 0, v_inst_1321_);
v___x_1325_ = lp_batteries_MLList_cases___redArg(v_inst_1321_, v_xs_1322_, v___f_1323_, v___f_1324_);
return v___x_1325_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_join___redArg___lam__1(lean_object* v_inst_1326_, lean_object* v_xs_1327_, lean_object* v_x_1328_){
_start:
{
lean_object* v___x_1329_; 
v___x_1329_ = lp_batteries_MLList_join___redArg(v_inst_1326_, v_xs_1327_);
return v___x_1329_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_join(lean_object* v_m_1330_, lean_object* v_00_u03b1_1331_, lean_object* v_inst_1332_, lean_object* v_xs_1333_){
_start:
{
lean_object* v___x_1334_; 
v___x_1334_ = lp_batteries_MLList_join___redArg(v_inst_1332_, v_xs_1333_);
return v___x_1334_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_enumFrom___redArg___lam__0(lean_object* v_x_1335_){
_start:
{
lean_object* v___x_1336_; 
v___x_1336_ = lean_box(0);
return v___x_1336_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_enumFrom___redArg___lam__1(lean_object* v_n_1338_, lean_object* v_inst_1339_, lean_object* v_x_1340_, lean_object* v_xs_1341_){
_start:
{
lean_object* v___x_1342_; lean_object* v___x_1343_; lean_object* v___x_1344_; lean_object* v___x_1345_; lean_object* v___x_1346_; 
lean_inc(v_n_1338_);
v___x_1342_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1342_, 0, v_n_1338_);
lean_ctor_set(v___x_1342_, 1, v_x_1340_);
v___x_1343_ = lean_unsigned_to_nat(1u);
v___x_1344_ = lean_nat_add(v_n_1338_, v___x_1343_);
lean_dec(v_n_1338_);
v___x_1345_ = lp_batteries_MLList_enumFrom___redArg(v_inst_1339_, v___x_1344_, v_xs_1341_);
v___x_1346_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1346_, 0, v___x_1342_);
lean_ctor_set(v___x_1346_, 1, v___x_1345_);
return v___x_1346_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_enumFrom___redArg(lean_object* v_inst_1347_, lean_object* v_n_1348_, lean_object* v_xs_1349_){
_start:
{
lean_object* v___f_1350_; lean_object* v___f_1351_; lean_object* v___x_1352_; 
v___f_1350_ = ((lean_object*)(lp_batteries_MLList_enumFrom___redArg___closed__0));
lean_inc_ref(v_inst_1347_);
v___f_1351_ = lean_alloc_closure((void*)(lp_batteries_MLList_enumFrom___redArg___lam__1), 4, 2);
lean_closure_set(v___f_1351_, 0, v_n_1348_);
lean_closure_set(v___f_1351_, 1, v_inst_1347_);
v___x_1352_ = lp_batteries_MLList_cases___redArg(v_inst_1347_, v_xs_1349_, v___f_1350_, v___f_1351_);
return v___x_1352_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_enumFrom(lean_object* v_m_1353_, lean_object* v_00_u03b1_1354_, lean_object* v_inst_1355_, lean_object* v_n_1356_, lean_object* v_xs_1357_){
_start:
{
lean_object* v___x_1358_; 
v___x_1358_ = lp_batteries_MLList_enumFrom___redArg(v_inst_1355_, v_n_1356_, v_xs_1357_);
return v___x_1358_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_enum___redArg(lean_object* v_inst_1359_, lean_object* v_xs_1360_){
_start:
{
lean_object* v___x_1361_; lean_object* v___x_1362_; 
v___x_1361_ = lean_unsigned_to_nat(0u);
v___x_1362_ = lp_batteries_MLList_enumFrom___redArg(v_inst_1359_, v___x_1361_, v_xs_1360_);
return v___x_1362_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_enum(lean_object* v_m_1363_, lean_object* v_00_u03b1_1364_, lean_object* v_inst_1365_, lean_object* v_xs_1366_){
_start:
{
lean_object* v___x_1367_; 
v___x_1367_ = lp_batteries_MLList_enum___redArg(v_inst_1365_, v_xs_1366_);
return v___x_1367_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_range___redArg___lam__0(lean_object* v_toPure_1368_, lean_object* v_n_1369_){
_start:
{
lean_object* v___x_1370_; lean_object* v___x_1371_; lean_object* v___x_1372_; 
v___x_1370_ = lean_unsigned_to_nat(1u);
v___x_1371_ = lean_nat_add(v_n_1369_, v___x_1370_);
v___x_1372_ = lean_apply_2(v_toPure_1368_, lean_box(0), v___x_1371_);
return v___x_1372_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_range___redArg___lam__0___boxed(lean_object* v_toPure_1373_, lean_object* v_n_1374_){
_start:
{
lean_object* v_res_1375_; 
v_res_1375_ = lp_batteries_MLList_range___redArg___lam__0(v_toPure_1373_, v_n_1374_);
lean_dec(v_n_1374_);
return v_res_1375_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_range___redArg(lean_object* v_inst_1376_){
_start:
{
lean_object* v_toApplicative_1377_; lean_object* v_toPure_1378_; lean_object* v___f_1379_; lean_object* v___x_1380_; lean_object* v___x_1381_; 
v_toApplicative_1377_ = lean_ctor_get(v_inst_1376_, 0);
v_toPure_1378_ = lean_ctor_get(v_toApplicative_1377_, 1);
lean_inc(v_toPure_1378_);
v___f_1379_ = lean_alloc_closure((void*)(lp_batteries_MLList_range___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_1379_, 0, v_toPure_1378_);
v___x_1380_ = lean_unsigned_to_nat(0u);
v___x_1381_ = lp_batteries_MLList_fix___redArg(v_inst_1376_, v___f_1379_, v___x_1380_);
return v___x_1381_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_range(lean_object* v_m_1382_, lean_object* v_inst_1383_){
_start:
{
lean_object* v___x_1384_; 
v___x_1384_ = lp_batteries_MLList_range___redArg(v_inst_1383_);
return v___x_1384_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_fin_go___redArg___lam__0___boxed(lean_object* v_i_1385_, lean_object* v_n_1386_, lean_object* v_x_1387_){
_start:
{
lean_object* v_res_1388_; 
v_res_1388_ = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_fin_go___redArg___lam__0(v_i_1385_, v_n_1386_, v_x_1387_);
lean_dec(v_i_1385_);
return v_res_1388_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_fin_go___redArg(lean_object* v_n_1389_, lean_object* v_i_1390_){
_start:
{
uint8_t v___x_1391_; 
v___x_1391_ = lean_nat_dec_lt(v_i_1390_, v_n_1389_);
if (v___x_1391_ == 0)
{
lean_object* v___x_1392_; 
lean_dec(v_i_1390_);
lean_dec(v_n_1389_);
v___x_1392_ = lean_box(0);
return v___x_1392_;
}
else
{
lean_object* v___f_1393_; lean_object* v___x_1394_; lean_object* v___x_1395_; lean_object* v___x_1396_; 
lean_inc(v_i_1390_);
v___f_1393_ = lean_alloc_closure((void*)(lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_fin_go___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_1393_, 0, v_i_1390_);
lean_closure_set(v___f_1393_, 1, v_n_1389_);
v___x_1394_ = lean_mk_thunk(v___f_1393_);
v___x_1395_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v___x_1395_, 0, v___x_1394_);
v___x_1396_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1396_, 0, v_i_1390_);
lean_ctor_set(v___x_1396_, 1, v___x_1395_);
return v___x_1396_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_fin_go___redArg___lam__0(lean_object* v_i_1397_, lean_object* v_n_1398_, lean_object* v_x_1399_){
_start:
{
lean_object* v___x_1400_; lean_object* v___x_1401_; lean_object* v___x_1402_; 
v___x_1400_ = lean_unsigned_to_nat(1u);
v___x_1401_ = lean_nat_add(v_i_1397_, v___x_1400_);
v___x_1402_ = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_fin_go___redArg(v_n_1398_, v___x_1401_);
return v___x_1402_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_fin_go(lean_object* v_m_1403_, lean_object* v_n_1404_, lean_object* v_i_1405_){
_start:
{
lean_object* v___x_1406_; 
v___x_1406_ = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_fin_go___redArg(v_n_1404_, v_i_1405_);
return v___x_1406_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_fin___redArg(lean_object* v_n_1407_){
_start:
{
lean_object* v___x_1408_; lean_object* v___x_1409_; 
v___x_1408_ = lean_unsigned_to_nat(0u);
v___x_1409_ = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_fin_go___redArg(v_n_1407_, v___x_1408_);
return v___x_1409_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_fin(lean_object* v_m_1410_, lean_object* v_n_1411_){
_start:
{
lean_object* v___x_1412_; 
v___x_1412_ = lp_batteries_MLList_fin___redArg(v_n_1411_);
return v___x_1412_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_ofArray_go___redArg___lam__0___boxed(lean_object* v_i_1413_, lean_object* v_L_1414_, lean_object* v_x_1415_){
_start:
{
lean_object* v_res_1416_; 
v_res_1416_ = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_ofArray_go___redArg___lam__0(v_i_1413_, v_L_1414_, v_x_1415_);
lean_dec(v_i_1413_);
return v_res_1416_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_ofArray_go___redArg(lean_object* v_L_1417_, lean_object* v_i_1418_){
_start:
{
lean_object* v___x_1419_; uint8_t v___x_1420_; 
v___x_1419_ = lean_array_get_size(v_L_1417_);
v___x_1420_ = lean_nat_dec_lt(v_i_1418_, v___x_1419_);
if (v___x_1420_ == 0)
{
lean_object* v___x_1421_; 
lean_dec(v_i_1418_);
lean_dec_ref(v_L_1417_);
v___x_1421_ = lean_box(0);
return v___x_1421_;
}
else
{
lean_object* v___f_1422_; lean_object* v___x_1423_; lean_object* v___x_1424_; lean_object* v___x_1425_; lean_object* v___x_1426_; 
lean_inc_ref(v_L_1417_);
lean_inc(v_i_1418_);
v___f_1422_ = lean_alloc_closure((void*)(lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_ofArray_go___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_1422_, 0, v_i_1418_);
lean_closure_set(v___f_1422_, 1, v_L_1417_);
v___x_1423_ = lean_array_fget(v_L_1417_, v_i_1418_);
lean_dec(v_i_1418_);
lean_dec_ref(v_L_1417_);
v___x_1424_ = lean_mk_thunk(v___f_1422_);
v___x_1425_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v___x_1425_, 0, v___x_1424_);
v___x_1426_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1426_, 0, v___x_1423_);
lean_ctor_set(v___x_1426_, 1, v___x_1425_);
return v___x_1426_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_ofArray_go___redArg___lam__0(lean_object* v_i_1427_, lean_object* v_L_1428_, lean_object* v_x_1429_){
_start:
{
lean_object* v___x_1430_; lean_object* v___x_1431_; lean_object* v___x_1432_; 
v___x_1430_ = lean_unsigned_to_nat(1u);
v___x_1431_ = lean_nat_add(v_i_1427_, v___x_1430_);
v___x_1432_ = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_ofArray_go___redArg(v_L_1428_, v___x_1431_);
return v___x_1432_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_ofArray_go(lean_object* v_m_1433_, lean_object* v_00_u03b1_1434_, lean_object* v_L_1435_, lean_object* v_i_1436_){
_start:
{
lean_object* v___x_1437_; 
v___x_1437_ = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_ofArray_go___redArg(v_L_1435_, v_i_1436_);
return v___x_1437_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_ofArray___redArg(lean_object* v_L_1438_){
_start:
{
lean_object* v___x_1439_; lean_object* v___x_1440_; 
v___x_1439_ = lean_unsigned_to_nat(0u);
v___x_1440_ = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_ofArray_go___redArg(v_L_1438_, v___x_1439_);
return v___x_1440_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_ofArray(lean_object* v_m_1441_, lean_object* v_00_u03b1_1442_, lean_object* v_L_1443_){
_start:
{
lean_object* v___x_1444_; 
v___x_1444_ = lp_batteries_MLList_ofArray___redArg(v_L_1443_);
return v___x_1444_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_chunk_go___redArg___lam__2(lean_object* v_inst_1445_, lean_object* v_M_1446_, lean_object* v_toBind_1447_, lean_object* v___f_1448_, lean_object* v_x_1449_){
_start:
{
lean_object* v___x_1450_; lean_object* v___x_1451_; 
v___x_1450_ = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_unconsImpl___redArg(v_inst_1445_, v_M_1446_);
v___x_1451_ = lean_apply_4(v_toBind_1447_, lean_box(0), lean_box(0), v___x_1450_, v___f_1448_);
return v___x_1451_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_chunk_go___redArg___lam__1(lean_object* v_acc_1452_, lean_object* v_toPure_1453_, lean_object* v_inst_1454_, lean_object* v_n_1455_, lean_object* v_n_1456_, lean_object* v_____do__lift_1457_){
_start:
{
if (lean_obj_tag(v_____do__lift_1457_) == 0)
{
lean_object* v___x_1458_; lean_object* v___x_1459_; lean_object* v___x_1460_; 
lean_dec(v_n_1455_);
lean_dec_ref(v_inst_1454_);
v___x_1458_ = lean_box(0);
v___x_1459_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1459_, 0, v_acc_1452_);
lean_ctor_set(v___x_1459_, 1, v___x_1458_);
v___x_1460_ = lean_apply_2(v_toPure_1453_, lean_box(0), v___x_1459_);
return v___x_1460_;
}
else
{
lean_object* v_val_1461_; lean_object* v_fst_1462_; lean_object* v_snd_1463_; lean_object* v___x_1464_; lean_object* v___x_1465_; lean_object* v___x_1466_; 
v_val_1461_ = lean_ctor_get(v_____do__lift_1457_, 0);
lean_inc(v_val_1461_);
lean_dec_ref_known(v_____do__lift_1457_, 1);
v_fst_1462_ = lean_ctor_get(v_val_1461_, 0);
lean_inc(v_fst_1462_);
v_snd_1463_ = lean_ctor_get(v_val_1461_, 1);
lean_inc(v_snd_1463_);
lean_dec(v_val_1461_);
v___x_1464_ = lean_array_push(v_acc_1452_, v_fst_1462_);
v___x_1465_ = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_chunk_go___redArg(v_inst_1454_, v_n_1455_, v_n_1456_, v___x_1464_, v_snd_1463_);
v___x_1466_ = lean_apply_2(v_toPure_1453_, lean_box(0), v___x_1465_);
return v___x_1466_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_chunk_go___redArg___lam__1___boxed(lean_object* v_acc_1467_, lean_object* v_toPure_1468_, lean_object* v_inst_1469_, lean_object* v_n_1470_, lean_object* v_n_1471_, lean_object* v_____do__lift_1472_){
_start:
{
lean_object* v_res_1473_; 
v_res_1473_ = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_chunk_go___redArg___lam__1(v_acc_1467_, v_toPure_1468_, v_inst_1469_, v_n_1470_, v_n_1471_, v_____do__lift_1472_);
lean_dec(v_n_1471_);
return v_res_1473_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_chunk_go___redArg(lean_object* v_inst_1474_, lean_object* v_n_1475_, lean_object* v_r_1476_, lean_object* v_acc_1477_, lean_object* v_M_1478_){
_start:
{
lean_object* v_toApplicative_1479_; lean_object* v_toBind_1480_; lean_object* v_toPure_1481_; lean_object* v_zero_1482_; uint8_t v_isZero_1483_; 
v_toApplicative_1479_ = lean_ctor_get(v_inst_1474_, 0);
v_toBind_1480_ = lean_ctor_get(v_inst_1474_, 1);
v_toPure_1481_ = lean_ctor_get(v_toApplicative_1479_, 1);
v_zero_1482_ = lean_unsigned_to_nat(0u);
v_isZero_1483_ = lean_nat_dec_eq(v_r_1476_, v_zero_1482_);
if (v_isZero_1483_ == 1)
{
lean_object* v___f_1484_; lean_object* v___x_1485_; lean_object* v___x_1486_; lean_object* v___x_1487_; 
v___f_1484_ = lean_alloc_closure((void*)(lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_chunk_go___redArg___lam__0), 4, 3);
lean_closure_set(v___f_1484_, 0, v_inst_1474_);
lean_closure_set(v___f_1484_, 1, v_n_1475_);
lean_closure_set(v___f_1484_, 2, v_M_1478_);
v___x_1485_ = lean_mk_thunk(v___f_1484_);
v___x_1486_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v___x_1486_, 0, v___x_1485_);
v___x_1487_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1487_, 0, v_acc_1477_);
lean_ctor_set(v___x_1487_, 1, v___x_1486_);
return v___x_1487_;
}
else
{
lean_object* v_one_1488_; lean_object* v_n_1489_; lean_object* v___f_1490_; lean_object* v___f_1491_; lean_object* v___x_1492_; 
lean_inc(v_toBind_1480_);
v_one_1488_ = lean_unsigned_to_nat(1u);
v_n_1489_ = lean_nat_sub(v_r_1476_, v_one_1488_);
lean_inc_ref(v_inst_1474_);
lean_inc(v_toPure_1481_);
v___f_1490_ = lean_alloc_closure((void*)(lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_chunk_go___redArg___lam__1___boxed), 6, 5);
lean_closure_set(v___f_1490_, 0, v_acc_1477_);
lean_closure_set(v___f_1490_, 1, v_toPure_1481_);
lean_closure_set(v___f_1490_, 2, v_inst_1474_);
lean_closure_set(v___f_1490_, 3, v_n_1475_);
lean_closure_set(v___f_1490_, 4, v_n_1489_);
v___f_1491_ = lean_alloc_closure((void*)(lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_chunk_go___redArg___lam__2), 5, 4);
lean_closure_set(v___f_1491_, 0, v_inst_1474_);
lean_closure_set(v___f_1491_, 1, v_M_1478_);
lean_closure_set(v___f_1491_, 2, v_toBind_1480_);
lean_closure_set(v___f_1491_, 3, v___f_1490_);
v___x_1492_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1492_, 0, v___f_1491_);
return v___x_1492_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_chunk_go___redArg___lam__0(lean_object* v_inst_1493_, lean_object* v_n_1494_, lean_object* v_M_1495_, lean_object* v_x_1496_){
_start:
{
lean_object* v___x_1497_; lean_object* v___x_1498_; 
v___x_1497_ = ((lean_object*)(lp_batteries_MLList_asArray___redArg___closed__0));
lean_inc(v_n_1494_);
v___x_1498_ = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_chunk_go___redArg(v_inst_1493_, v_n_1494_, v_n_1494_, v___x_1497_, v_M_1495_);
lean_dec(v_n_1494_);
return v___x_1498_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_chunk_go___redArg___boxed(lean_object* v_inst_1499_, lean_object* v_n_1500_, lean_object* v_r_1501_, lean_object* v_acc_1502_, lean_object* v_M_1503_){
_start:
{
lean_object* v_res_1504_; 
v_res_1504_ = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_chunk_go___redArg(v_inst_1499_, v_n_1500_, v_r_1501_, v_acc_1502_, v_M_1503_);
lean_dec(v_r_1501_);
return v_res_1504_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_chunk_go(lean_object* v_m_1505_, lean_object* v_00_u03b1_1506_, lean_object* v_inst_1507_, lean_object* v_n_1508_, lean_object* v_r_1509_, lean_object* v_acc_1510_, lean_object* v_M_1511_){
_start:
{
lean_object* v___x_1512_; 
v___x_1512_ = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_chunk_go___redArg(v_inst_1507_, v_n_1508_, v_r_1509_, v_acc_1510_, v_M_1511_);
return v___x_1512_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_chunk_go___boxed(lean_object* v_m_1513_, lean_object* v_00_u03b1_1514_, lean_object* v_inst_1515_, lean_object* v_n_1516_, lean_object* v_r_1517_, lean_object* v_acc_1518_, lean_object* v_M_1519_){
_start:
{
lean_object* v_res_1520_; 
v_res_1520_ = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_chunk_go(v_m_1513_, v_00_u03b1_1514_, v_inst_1515_, v_n_1516_, v_r_1517_, v_acc_1518_, v_M_1519_);
lean_dec(v_r_1517_);
return v_res_1520_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_chunk___redArg(lean_object* v_inst_1521_, lean_object* v_L_1522_, lean_object* v_n_1523_){
_start:
{
lean_object* v___x_1524_; lean_object* v___x_1525_; 
v___x_1524_ = ((lean_object*)(lp_batteries_MLList_asArray___redArg___closed__0));
lean_inc(v_n_1523_);
v___x_1525_ = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_chunk_go___redArg(v_inst_1521_, v_n_1523_, v_n_1523_, v___x_1524_, v_L_1522_);
lean_dec(v_n_1523_);
return v___x_1525_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_chunk(lean_object* v_m_1526_, lean_object* v_00_u03b1_1527_, lean_object* v_inst_1528_, lean_object* v_L_1529_, lean_object* v_n_1530_){
_start:
{
lean_object* v___x_1531_; 
v___x_1531_ = lp_batteries_MLList_chunk___redArg(v_inst_1528_, v_L_1529_, v_n_1530_);
return v___x_1531_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_concat___redArg___lam__0(lean_object* v_a_1532_, lean_object* v_x_1533_){
_start:
{
lean_object* v___x_1534_; lean_object* v___x_1535_; 
v___x_1534_ = lean_box(0);
v___x_1535_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1535_, 0, v_a_1532_);
lean_ctor_set(v___x_1535_, 1, v___x_1534_);
return v___x_1535_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_concat___redArg(lean_object* v_inst_1536_, lean_object* v_L_1537_, lean_object* v_a_1538_){
_start:
{
lean_object* v___f_1539_; lean_object* v___x_1540_; 
v___f_1539_ = lean_alloc_closure((void*)(lp_batteries_MLList_concat___redArg___lam__0), 2, 1);
lean_closure_set(v___f_1539_, 0, v_a_1538_);
v___x_1540_ = lp_batteries_MLList_append___redArg(v_inst_1536_, v_L_1537_, v___f_1539_);
return v___x_1540_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_concat(lean_object* v_m_1541_, lean_object* v_00_u03b1_1542_, lean_object* v_inst_1543_, lean_object* v_L_1544_, lean_object* v_a_1545_){
_start:
{
lean_object* v___x_1546_; 
v___x_1546_ = lp_batteries_MLList_concat___redArg(v_inst_1543_, v_L_1544_, v_a_1545_);
return v___x_1546_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_zip___redArg___lam__0(lean_object* v_x_1547_){
_start:
{
lean_object* v___x_1548_; 
v___x_1548_ = lean_box(0);
return v___x_1548_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_zip___redArg(lean_object* v_inst_1550_, lean_object* v_L_1551_, lean_object* v_M_1552_){
_start:
{
lean_object* v___f_1553_; lean_object* v___f_1554_; lean_object* v___x_1555_; 
v___f_1553_ = ((lean_object*)(lp_batteries_MLList_zip___redArg___closed__0));
lean_inc_ref(v_inst_1550_);
v___f_1554_ = lean_alloc_closure((void*)(lp_batteries_MLList_zip___redArg___lam__2), 5, 3);
lean_closure_set(v___f_1554_, 0, v_inst_1550_);
lean_closure_set(v___f_1554_, 1, v_M_1552_);
lean_closure_set(v___f_1554_, 2, v___f_1553_);
v___x_1555_ = lp_batteries_MLList_cases___redArg(v_inst_1550_, v_L_1551_, v___f_1553_, v___f_1554_);
return v___x_1555_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_zip___redArg___lam__1(lean_object* v_a_1556_, lean_object* v_inst_1557_, lean_object* v_L_1558_, lean_object* v_b_1559_, lean_object* v_M_1560_){
_start:
{
lean_object* v___x_1561_; lean_object* v___x_1562_; lean_object* v___x_1563_; 
v___x_1561_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1561_, 0, v_a_1556_);
lean_ctor_set(v___x_1561_, 1, v_b_1559_);
v___x_1562_ = lp_batteries_MLList_zip___redArg(v_inst_1557_, v_L_1558_, v_M_1560_);
v___x_1563_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1563_, 0, v___x_1561_);
lean_ctor_set(v___x_1563_, 1, v___x_1562_);
return v___x_1563_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_zip___redArg___lam__2(lean_object* v_inst_1564_, lean_object* v_M_1565_, lean_object* v___f_1566_, lean_object* v_a_1567_, lean_object* v_L_1568_){
_start:
{
lean_object* v___f_1569_; lean_object* v___x_1570_; 
lean_inc_ref(v_inst_1564_);
v___f_1569_ = lean_alloc_closure((void*)(lp_batteries_MLList_zip___redArg___lam__1), 5, 3);
lean_closure_set(v___f_1569_, 0, v_a_1567_);
lean_closure_set(v___f_1569_, 1, v_inst_1564_);
lean_closure_set(v___f_1569_, 2, v_L_1568_);
v___x_1570_ = lp_batteries_MLList_cases___redArg(v_inst_1564_, v_M_1565_, v___f_1566_, v___f_1569_);
return v___x_1570_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_zip(lean_object* v_m_1571_, lean_object* v_00_u03b1_1572_, lean_object* v_00_u03b2_1573_, lean_object* v_inst_1574_, lean_object* v_L_1575_, lean_object* v_M_1576_){
_start:
{
lean_object* v___x_1577_; 
v___x_1577_ = lp_batteries_MLList_zip___redArg(v_inst_1574_, v_L_1575_, v_M_1576_);
return v___x_1577_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_bind___redArg___lam__0(lean_object* v_inst_1578_, lean_object* v_f_1579_, lean_object* v_x_1580_, lean_object* v_xs_1581_){
_start:
{
lean_object* v___f_1582_; lean_object* v___x_1586_; 
lean_inc(v_f_1579_);
lean_inc(v_xs_1581_);
lean_inc_ref(v_inst_1578_);
v___f_1582_ = lean_alloc_closure((void*)(lp_batteries_MLList_bind___redArg___lam__1), 4, 3);
lean_closure_set(v___f_1582_, 0, v_inst_1578_);
lean_closure_set(v___f_1582_, 1, v_xs_1581_);
lean_closure_set(v___f_1582_, 2, v_f_1579_);
v___x_1586_ = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_uncons_x3fImpl___redArg(v_xs_1581_);
if (lean_obj_tag(v___x_1586_) == 1)
{
lean_object* v_val_1587_; 
v_val_1587_ = lean_ctor_get(v___x_1586_, 0);
lean_inc(v_val_1587_);
lean_dec_ref_known(v___x_1586_, 1);
if (lean_obj_tag(v_val_1587_) == 0)
{
lean_object* v___x_1588_; 
lean_dec_ref(v___f_1582_);
lean_dec_ref(v_inst_1578_);
v___x_1588_ = lean_apply_1(v_f_1579_, v_x_1580_);
return v___x_1588_;
}
else
{
lean_dec(v_val_1587_);
goto v___jp_1583_;
}
}
else
{
lean_dec(v___x_1586_);
goto v___jp_1583_;
}
v___jp_1583_:
{
lean_object* v___x_1584_; lean_object* v___x_1585_; 
v___x_1584_ = lean_apply_1(v_f_1579_, v_x_1580_);
v___x_1585_ = lp_batteries_MLList_append___redArg(v_inst_1578_, v___x_1584_, v___f_1582_);
return v___x_1585_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_bind___redArg(lean_object* v_inst_1589_, lean_object* v_xs_1590_, lean_object* v_f_1591_){
_start:
{
lean_object* v___f_1592_; lean_object* v___f_1593_; lean_object* v___x_1594_; 
v___f_1592_ = ((lean_object*)(lp_batteries_MLList_take___redArg___closed__0));
lean_inc_ref(v_inst_1589_);
v___f_1593_ = lean_alloc_closure((void*)(lp_batteries_MLList_bind___redArg___lam__0), 4, 2);
lean_closure_set(v___f_1593_, 0, v_inst_1589_);
lean_closure_set(v___f_1593_, 1, v_f_1591_);
v___x_1594_ = lp_batteries_MLList_cases___redArg(v_inst_1589_, v_xs_1590_, v___f_1592_, v___f_1593_);
return v___x_1594_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_bind___redArg___lam__1(lean_object* v_inst_1595_, lean_object* v_xs_1596_, lean_object* v_f_1597_, lean_object* v_x_1598_){
_start:
{
lean_object* v___x_1599_; 
v___x_1599_ = lp_batteries_MLList_bind___redArg(v_inst_1595_, v_xs_1596_, v_f_1597_);
return v___x_1599_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_bind(lean_object* v_m_1600_, lean_object* v_00_u03b1_1601_, lean_object* v_00_u03b2_1602_, lean_object* v_inst_1603_, lean_object* v_xs_1604_, lean_object* v_f_1605_){
_start:
{
lean_object* v___x_1606_; 
v___x_1606_ = lp_batteries_MLList_bind___redArg(v_inst_1603_, v_xs_1604_, v_f_1605_);
return v___x_1606_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_monadLift___redArg(lean_object* v_inst_1607_, lean_object* v_x_1608_){
_start:
{
lean_object* v_toApplicative_1609_; lean_object* v_toBind_1610_; lean_object* v_toPure_1611_; lean_object* v___f_1612_; lean_object* v___f_1613_; lean_object* v___x_1614_; 
v_toApplicative_1609_ = lean_ctor_get(v_inst_1607_, 0);
lean_inc_ref(v_toApplicative_1609_);
v_toBind_1610_ = lean_ctor_get(v_inst_1607_, 1);
lean_inc(v_toBind_1610_);
lean_dec_ref(v_inst_1607_);
v_toPure_1611_ = lean_ctor_get(v_toApplicative_1609_, 1);
lean_inc(v_toPure_1611_);
lean_dec_ref(v_toApplicative_1609_);
v___f_1612_ = lean_alloc_closure((void*)(lp_batteries_MLList_singletonM___redArg___lam__0), 2, 1);
lean_closure_set(v___f_1612_, 0, v_toPure_1611_);
v___f_1613_ = lean_alloc_closure((void*)(lp_batteries_MLList_singletonM___redArg___lam__1), 4, 3);
lean_closure_set(v___f_1613_, 0, v_toBind_1610_);
lean_closure_set(v___f_1613_, 1, v_x_1608_);
lean_closure_set(v___f_1613_, 2, v___f_1612_);
v___x_1614_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1614_, 0, v___f_1613_);
return v___x_1614_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_monadLift(lean_object* v_m_1615_, lean_object* v_00_u03b1_1616_, lean_object* v_inst_1617_, lean_object* v_x_1618_){
_start:
{
lean_object* v___x_1619_; 
v___x_1619_ = lp_batteries_MLList_monadLift___redArg(v_inst_1617_, v_x_1618_);
return v___x_1619_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_liftM___redArg___lam__1(lean_object* v_inst_1620_, lean_object* v_L_1621_, lean_object* v_inst_1622_, lean_object* v_toBind_1623_, lean_object* v___f_1624_, lean_object* v_x_1625_){
_start:
{
lean_object* v___x_1626_; lean_object* v___x_1627_; lean_object* v___x_1628_; 
v___x_1626_ = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_unconsImpl___redArg(v_inst_1620_, v_L_1621_);
v___x_1627_ = lean_apply_2(v_inst_1622_, lean_box(0), v___x_1626_);
v___x_1628_ = lean_apply_4(v_toBind_1623_, lean_box(0), lean_box(0), v___x_1627_, v___f_1624_);
return v___x_1628_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_liftM___redArg___lam__0(lean_object* v_toPure_1629_, lean_object* v_inst_1630_, lean_object* v_inst_1631_, lean_object* v_inst_1632_, lean_object* v_____do__lift_1633_){
_start:
{
if (lean_obj_tag(v_____do__lift_1633_) == 0)
{
lean_object* v___x_1634_; lean_object* v___x_1635_; 
lean_dec(v_inst_1632_);
lean_dec_ref(v_inst_1631_);
lean_dec_ref(v_inst_1630_);
v___x_1634_ = lean_box(0);
v___x_1635_ = lean_apply_2(v_toPure_1629_, lean_box(0), v___x_1634_);
return v___x_1635_;
}
else
{
lean_object* v_val_1636_; lean_object* v_fst_1637_; lean_object* v_snd_1638_; lean_object* v___x_1640_; uint8_t v_isShared_1641_; uint8_t v_isSharedCheck_1647_; 
v_val_1636_ = lean_ctor_get(v_____do__lift_1633_, 0);
lean_inc(v_val_1636_);
lean_dec_ref_known(v_____do__lift_1633_, 1);
v_fst_1637_ = lean_ctor_get(v_val_1636_, 0);
v_snd_1638_ = lean_ctor_get(v_val_1636_, 1);
v_isSharedCheck_1647_ = !lean_is_exclusive(v_val_1636_);
if (v_isSharedCheck_1647_ == 0)
{
v___x_1640_ = v_val_1636_;
v_isShared_1641_ = v_isSharedCheck_1647_;
goto v_resetjp_1639_;
}
else
{
lean_inc(v_snd_1638_);
lean_inc(v_fst_1637_);
lean_dec(v_val_1636_);
v___x_1640_ = lean_box(0);
v_isShared_1641_ = v_isSharedCheck_1647_;
goto v_resetjp_1639_;
}
v_resetjp_1639_:
{
lean_object* v___x_1642_; lean_object* v___x_1644_; 
v___x_1642_ = lp_batteries_MLList_liftM___redArg(v_inst_1630_, v_inst_1631_, v_inst_1632_, v_snd_1638_);
if (v_isShared_1641_ == 0)
{
lean_ctor_set_tag(v___x_1640_, 1);
lean_ctor_set(v___x_1640_, 1, v___x_1642_);
v___x_1644_ = v___x_1640_;
goto v_reusejp_1643_;
}
else
{
lean_object* v_reuseFailAlloc_1646_; 
v_reuseFailAlloc_1646_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1646_, 0, v_fst_1637_);
lean_ctor_set(v_reuseFailAlloc_1646_, 1, v___x_1642_);
v___x_1644_ = v_reuseFailAlloc_1646_;
goto v_reusejp_1643_;
}
v_reusejp_1643_:
{
lean_object* v___x_1645_; 
v___x_1645_ = lean_apply_2(v_toPure_1629_, lean_box(0), v___x_1644_);
return v___x_1645_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_liftM___redArg(lean_object* v_inst_1648_, lean_object* v_inst_1649_, lean_object* v_inst_1650_, lean_object* v_L_1651_){
_start:
{
lean_object* v_toApplicative_1652_; lean_object* v_toBind_1653_; lean_object* v_toPure_1654_; lean_object* v___f_1655_; lean_object* v___f_1656_; lean_object* v___x_1657_; 
v_toApplicative_1652_ = lean_ctor_get(v_inst_1649_, 0);
v_toBind_1653_ = lean_ctor_get(v_inst_1649_, 1);
lean_inc(v_toBind_1653_);
v_toPure_1654_ = lean_ctor_get(v_toApplicative_1652_, 1);
lean_inc(v_toPure_1654_);
lean_inc(v_inst_1650_);
lean_inc_ref(v_inst_1648_);
v___f_1655_ = lean_alloc_closure((void*)(lp_batteries_MLList_liftM___redArg___lam__0), 5, 4);
lean_closure_set(v___f_1655_, 0, v_toPure_1654_);
lean_closure_set(v___f_1655_, 1, v_inst_1648_);
lean_closure_set(v___f_1655_, 2, v_inst_1649_);
lean_closure_set(v___f_1655_, 3, v_inst_1650_);
v___f_1656_ = lean_alloc_closure((void*)(lp_batteries_MLList_liftM___redArg___lam__1), 6, 5);
lean_closure_set(v___f_1656_, 0, v_inst_1648_);
lean_closure_set(v___f_1656_, 1, v_L_1651_);
lean_closure_set(v___f_1656_, 2, v_inst_1650_);
lean_closure_set(v___f_1656_, 3, v_toBind_1653_);
lean_closure_set(v___f_1656_, 4, v___f_1655_);
v___x_1657_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1657_, 0, v___f_1656_);
return v___x_1657_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_liftM(lean_object* v_m_1658_, lean_object* v_n_1659_, lean_object* v_00_u03b1_1660_, lean_object* v_inst_1661_, lean_object* v_inst_1662_, lean_object* v_inst_1663_, lean_object* v_L_1664_){
_start:
{
lean_object* v___x_1665_; 
v___x_1665_ = lp_batteries_MLList_liftM___redArg(v_inst_1661_, v_inst_1662_, v_inst_1663_, v_L_1664_);
return v___x_1665_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_runState___redArg___lam__1(lean_object* v___x_1666_, lean_object* v_L_1667_, lean_object* v_s_1668_, lean_object* v_toBind_1669_, lean_object* v___f_1670_, lean_object* v_x_1671_){
_start:
{
lean_object* v___x_87__overap_1672_; lean_object* v___x_1673_; lean_object* v___x_1674_; 
v___x_87__overap_1672_ = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_unconsImpl___redArg(v___x_1666_, v_L_1667_);
v___x_1673_ = lean_apply_1(v___x_87__overap_1672_, v_s_1668_);
v___x_1674_ = lean_apply_4(v_toBind_1669_, lean_box(0), lean_box(0), v___x_1673_, v___f_1670_);
return v___x_1674_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_runState___redArg(lean_object* v_inst_1675_, lean_object* v_L_1676_, lean_object* v_s_1677_){
_start:
{
lean_object* v_toApplicative_1678_; lean_object* v_toBind_1679_; lean_object* v___f_1680_; lean_object* v___f_1681_; lean_object* v___f_1682_; lean_object* v___f_1683_; lean_object* v___x_1684_; lean_object* v___x_1685_; lean_object* v___x_1686_; lean_object* v___x_1687_; lean_object* v___x_1688_; lean_object* v___x_1689_; lean_object* v_toPure_1690_; lean_object* v___f_1691_; lean_object* v___f_1692_; lean_object* v___x_1693_; 
v_toApplicative_1678_ = lean_ctor_get(v_inst_1675_, 0);
v_toBind_1679_ = lean_ctor_get(v_inst_1675_, 1);
lean_inc(v_toBind_1679_);
lean_inc_ref_n(v_inst_1675_, 7);
v___f_1680_ = lean_alloc_closure((void*)(l_StateT_instMonad___redArg___lam__1), 6, 1);
lean_closure_set(v___f_1680_, 0, v_inst_1675_);
v___f_1681_ = lean_alloc_closure((void*)(l_StateT_instMonad___redArg___lam__4), 6, 1);
lean_closure_set(v___f_1681_, 0, v_inst_1675_);
v___f_1682_ = lean_alloc_closure((void*)(l_StateT_instMonad___redArg___lam__7), 6, 1);
lean_closure_set(v___f_1682_, 0, v_inst_1675_);
v___f_1683_ = lean_alloc_closure((void*)(l_StateT_instMonad___redArg___lam__9), 6, 1);
lean_closure_set(v___f_1683_, 0, v_inst_1675_);
v___x_1684_ = lean_alloc_closure((void*)(l_StateT_map), 8, 3);
lean_closure_set(v___x_1684_, 0, lean_box(0));
lean_closure_set(v___x_1684_, 1, lean_box(0));
lean_closure_set(v___x_1684_, 2, v_inst_1675_);
v___x_1685_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1685_, 0, v___x_1684_);
lean_ctor_set(v___x_1685_, 1, v___f_1680_);
v___x_1686_ = lean_alloc_closure((void*)(l_StateT_pure), 6, 3);
lean_closure_set(v___x_1686_, 0, lean_box(0));
lean_closure_set(v___x_1686_, 1, lean_box(0));
lean_closure_set(v___x_1686_, 2, v_inst_1675_);
v___x_1687_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_1687_, 0, v___x_1685_);
lean_ctor_set(v___x_1687_, 1, v___x_1686_);
lean_ctor_set(v___x_1687_, 2, v___f_1681_);
lean_ctor_set(v___x_1687_, 3, v___f_1682_);
lean_ctor_set(v___x_1687_, 4, v___f_1683_);
v___x_1688_ = lean_alloc_closure((void*)(l_StateT_bind), 8, 3);
lean_closure_set(v___x_1688_, 0, lean_box(0));
lean_closure_set(v___x_1688_, 1, lean_box(0));
lean_closure_set(v___x_1688_, 2, v_inst_1675_);
v___x_1689_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1689_, 0, v___x_1687_);
lean_ctor_set(v___x_1689_, 1, v___x_1688_);
v_toPure_1690_ = lean_ctor_get(v_toApplicative_1678_, 1);
lean_inc(v_toPure_1690_);
v___f_1691_ = lean_alloc_closure((void*)(lp_batteries_MLList_runState___redArg___lam__0), 3, 2);
lean_closure_set(v___f_1691_, 0, v_toPure_1690_);
lean_closure_set(v___f_1691_, 1, v_inst_1675_);
v___f_1692_ = lean_alloc_closure((void*)(lp_batteries_MLList_runState___redArg___lam__1), 6, 5);
lean_closure_set(v___f_1692_, 0, v___x_1689_);
lean_closure_set(v___f_1692_, 1, v_L_1676_);
lean_closure_set(v___f_1692_, 2, v_s_1677_);
lean_closure_set(v___f_1692_, 3, v_toBind_1679_);
lean_closure_set(v___f_1692_, 4, v___f_1691_);
v___x_1693_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1693_, 0, v___f_1692_);
return v___x_1693_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_runState___redArg___lam__0(lean_object* v_toPure_1694_, lean_object* v_inst_1695_, lean_object* v_____do__lift_1696_){
_start:
{
lean_object* v_fst_1697_; 
v_fst_1697_ = lean_ctor_get(v_____do__lift_1696_, 0);
if (lean_obj_tag(v_fst_1697_) == 0)
{
lean_object* v___x_1698_; lean_object* v___x_1699_; 
lean_dec_ref(v_____do__lift_1696_);
lean_dec_ref(v_inst_1695_);
v___x_1698_ = lean_box(0);
v___x_1699_ = lean_apply_2(v_toPure_1694_, lean_box(0), v___x_1698_);
return v___x_1699_;
}
else
{
lean_object* v_val_1700_; lean_object* v_snd_1701_; lean_object* v___x_1703_; uint8_t v_isShared_1704_; uint8_t v_isSharedCheck_1719_; 
v_val_1700_ = lean_ctor_get(v_fst_1697_, 0);
lean_inc(v_val_1700_);
v_snd_1701_ = lean_ctor_get(v_____do__lift_1696_, 1);
v_isSharedCheck_1719_ = !lean_is_exclusive(v_____do__lift_1696_);
if (v_isSharedCheck_1719_ == 0)
{
lean_object* v_unused_1720_; 
v_unused_1720_ = lean_ctor_get(v_____do__lift_1696_, 0);
lean_dec(v_unused_1720_);
v___x_1703_ = v_____do__lift_1696_;
v_isShared_1704_ = v_isSharedCheck_1719_;
goto v_resetjp_1702_;
}
else
{
lean_inc(v_snd_1701_);
lean_dec(v_____do__lift_1696_);
v___x_1703_ = lean_box(0);
v_isShared_1704_ = v_isSharedCheck_1719_;
goto v_resetjp_1702_;
}
v_resetjp_1702_:
{
lean_object* v_fst_1705_; lean_object* v_snd_1706_; lean_object* v___x_1708_; uint8_t v_isShared_1709_; uint8_t v_isSharedCheck_1718_; 
v_fst_1705_ = lean_ctor_get(v_val_1700_, 0);
v_snd_1706_ = lean_ctor_get(v_val_1700_, 1);
v_isSharedCheck_1718_ = !lean_is_exclusive(v_val_1700_);
if (v_isSharedCheck_1718_ == 0)
{
v___x_1708_ = v_val_1700_;
v_isShared_1709_ = v_isSharedCheck_1718_;
goto v_resetjp_1707_;
}
else
{
lean_inc(v_snd_1706_);
lean_inc(v_fst_1705_);
lean_dec(v_val_1700_);
v___x_1708_ = lean_box(0);
v_isShared_1709_ = v_isSharedCheck_1718_;
goto v_resetjp_1707_;
}
v_resetjp_1707_:
{
lean_object* v___x_1711_; 
lean_inc(v_snd_1701_);
if (v_isShared_1709_ == 0)
{
lean_ctor_set(v___x_1708_, 1, v_snd_1701_);
v___x_1711_ = v___x_1708_;
goto v_reusejp_1710_;
}
else
{
lean_object* v_reuseFailAlloc_1717_; 
v_reuseFailAlloc_1717_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1717_, 0, v_fst_1705_);
lean_ctor_set(v_reuseFailAlloc_1717_, 1, v_snd_1701_);
v___x_1711_ = v_reuseFailAlloc_1717_;
goto v_reusejp_1710_;
}
v_reusejp_1710_:
{
lean_object* v___x_1712_; lean_object* v___x_1714_; 
v___x_1712_ = lp_batteries_MLList_runState___redArg(v_inst_1695_, v_snd_1706_, v_snd_1701_);
if (v_isShared_1704_ == 0)
{
lean_ctor_set_tag(v___x_1703_, 1);
lean_ctor_set(v___x_1703_, 1, v___x_1712_);
lean_ctor_set(v___x_1703_, 0, v___x_1711_);
v___x_1714_ = v___x_1703_;
goto v_reusejp_1713_;
}
else
{
lean_object* v_reuseFailAlloc_1716_; 
v_reuseFailAlloc_1716_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1716_, 0, v___x_1711_);
lean_ctor_set(v_reuseFailAlloc_1716_, 1, v___x_1712_);
v___x_1714_ = v_reuseFailAlloc_1716_;
goto v_reusejp_1713_;
}
v_reusejp_1713_:
{
lean_object* v___x_1715_; 
v___x_1715_ = lean_apply_2(v_toPure_1694_, lean_box(0), v___x_1714_);
return v___x_1715_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_runState(lean_object* v_m_1721_, lean_object* v_00_u03c3_1722_, lean_object* v_00_u03b1_1723_, lean_object* v_inst_1724_, lean_object* v_L_1725_, lean_object* v_s_1726_){
_start:
{
lean_object* v___x_1727_; 
v___x_1727_ = lp_batteries_MLList_runState___redArg(v_inst_1724_, v_L_1725_, v_s_1726_);
return v___x_1727_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_runState_x27___redArg___lam__0(lean_object* v_x_1728_){
_start:
{
lean_object* v_fst_1729_; 
v_fst_1729_ = lean_ctor_get(v_x_1728_, 0);
lean_inc(v_fst_1729_);
return v_fst_1729_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_runState_x27___redArg___lam__0___boxed(lean_object* v_x_1730_){
_start:
{
lean_object* v_res_1731_; 
v_res_1731_ = lp_batteries_MLList_runState_x27___redArg___lam__0(v_x_1730_);
lean_dec_ref(v_x_1730_);
return v_res_1731_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_runState_x27___redArg(lean_object* v_inst_1733_, lean_object* v_L_1734_, lean_object* v_s_1735_){
_start:
{
lean_object* v___f_1736_; lean_object* v___x_1737_; lean_object* v___x_1738_; 
v___f_1736_ = ((lean_object*)(lp_batteries_MLList_runState_x27___redArg___closed__0));
lean_inc_ref(v_inst_1733_);
v___x_1737_ = lp_batteries_MLList_runState___redArg(v_inst_1733_, v_L_1734_, v_s_1735_);
v___x_1738_ = lp_batteries_MLList_map___redArg(v_inst_1733_, v___f_1736_, v___x_1737_);
return v___x_1738_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_runState_x27(lean_object* v_m_1739_, lean_object* v_00_u03c3_1740_, lean_object* v_00_u03b1_1741_, lean_object* v_inst_1742_, lean_object* v_L_1743_, lean_object* v_s_1744_){
_start:
{
lean_object* v___x_1745_; 
v___x_1745_ = lp_batteries_MLList_runState_x27___redArg(v_inst_1742_, v_L_1743_, v_s_1744_);
return v___x_1745_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_runReader___redArg___lam__1(lean_object* v___x_1746_, lean_object* v_L_1747_, lean_object* v_r_1748_, lean_object* v_toBind_1749_, lean_object* v___f_1750_, lean_object* v_x_1751_){
_start:
{
lean_object* v___x_69__overap_1752_; lean_object* v___x_1753_; lean_object* v___x_1754_; 
v___x_69__overap_1752_ = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_unconsImpl___redArg(v___x_1746_, v_L_1747_);
v___x_1753_ = lean_apply_1(v___x_69__overap_1752_, v_r_1748_);
v___x_1754_ = lean_apply_4(v_toBind_1749_, lean_box(0), lean_box(0), v___x_1753_, v___f_1750_);
return v___x_1754_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_runReader___redArg___lam__0(lean_object* v_toPure_1755_, lean_object* v_inst_1756_, lean_object* v_r_1757_, lean_object* v_____do__lift_1758_){
_start:
{
if (lean_obj_tag(v_____do__lift_1758_) == 0)
{
lean_object* v___x_1759_; lean_object* v___x_1760_; 
lean_dec(v_r_1757_);
lean_dec_ref(v_inst_1756_);
v___x_1759_ = lean_box(0);
v___x_1760_ = lean_apply_2(v_toPure_1755_, lean_box(0), v___x_1759_);
return v___x_1760_;
}
else
{
lean_object* v_val_1761_; lean_object* v_fst_1762_; lean_object* v_snd_1763_; lean_object* v___x_1765_; uint8_t v_isShared_1766_; uint8_t v_isSharedCheck_1772_; 
v_val_1761_ = lean_ctor_get(v_____do__lift_1758_, 0);
lean_inc(v_val_1761_);
lean_dec_ref_known(v_____do__lift_1758_, 1);
v_fst_1762_ = lean_ctor_get(v_val_1761_, 0);
v_snd_1763_ = lean_ctor_get(v_val_1761_, 1);
v_isSharedCheck_1772_ = !lean_is_exclusive(v_val_1761_);
if (v_isSharedCheck_1772_ == 0)
{
v___x_1765_ = v_val_1761_;
v_isShared_1766_ = v_isSharedCheck_1772_;
goto v_resetjp_1764_;
}
else
{
lean_inc(v_snd_1763_);
lean_inc(v_fst_1762_);
lean_dec(v_val_1761_);
v___x_1765_ = lean_box(0);
v_isShared_1766_ = v_isSharedCheck_1772_;
goto v_resetjp_1764_;
}
v_resetjp_1764_:
{
lean_object* v___x_1767_; lean_object* v___x_1769_; 
v___x_1767_ = lp_batteries_MLList_runReader___redArg(v_inst_1756_, v_snd_1763_, v_r_1757_);
if (v_isShared_1766_ == 0)
{
lean_ctor_set_tag(v___x_1765_, 1);
lean_ctor_set(v___x_1765_, 1, v___x_1767_);
v___x_1769_ = v___x_1765_;
goto v_reusejp_1768_;
}
else
{
lean_object* v_reuseFailAlloc_1771_; 
v_reuseFailAlloc_1771_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1771_, 0, v_fst_1762_);
lean_ctor_set(v_reuseFailAlloc_1771_, 1, v___x_1767_);
v___x_1769_ = v_reuseFailAlloc_1771_;
goto v_reusejp_1768_;
}
v_reusejp_1768_:
{
lean_object* v___x_1770_; 
v___x_1770_ = lean_apply_2(v_toPure_1755_, lean_box(0), v___x_1769_);
return v___x_1770_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_runReader___redArg(lean_object* v_inst_1773_, lean_object* v_L_1774_, lean_object* v_r_1775_){
_start:
{
lean_object* v_toApplicative_1776_; lean_object* v_toBind_1777_; lean_object* v___x_1778_; lean_object* v_toPure_1779_; lean_object* v___f_1780_; lean_object* v___f_1781_; lean_object* v___x_1782_; 
v_toApplicative_1776_ = lean_ctor_get(v_inst_1773_, 0);
v_toBind_1777_ = lean_ctor_get(v_inst_1773_, 1);
lean_inc(v_toBind_1777_);
lean_inc_ref(v_inst_1773_);
v___x_1778_ = l_ReaderT_instMonad___redArg(v_inst_1773_);
v_toPure_1779_ = lean_ctor_get(v_toApplicative_1776_, 1);
lean_inc(v_toPure_1779_);
lean_inc(v_r_1775_);
v___f_1780_ = lean_alloc_closure((void*)(lp_batteries_MLList_runReader___redArg___lam__0), 4, 3);
lean_closure_set(v___f_1780_, 0, v_toPure_1779_);
lean_closure_set(v___f_1780_, 1, v_inst_1773_);
lean_closure_set(v___f_1780_, 2, v_r_1775_);
v___f_1781_ = lean_alloc_closure((void*)(lp_batteries_MLList_runReader___redArg___lam__1), 6, 5);
lean_closure_set(v___f_1781_, 0, v___x_1778_);
lean_closure_set(v___f_1781_, 1, v_L_1774_);
lean_closure_set(v___f_1781_, 2, v_r_1775_);
lean_closure_set(v___f_1781_, 3, v_toBind_1777_);
lean_closure_set(v___f_1781_, 4, v___f_1780_);
v___x_1782_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1782_, 0, v___f_1781_);
return v___x_1782_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_runReader(lean_object* v_m_1783_, lean_object* v_00_u03c1_1784_, lean_object* v_00_u03b1_1785_, lean_object* v_inst_1786_, lean_object* v_L_1787_, lean_object* v_r_1788_){
_start:
{
lean_object* v___x_1789_; 
v___x_1789_ = lp_batteries_MLList_runReader___redArg(v_inst_1786_, v_L_1787_, v_r_1788_);
return v___x_1789_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_runStateRef___redArg___lam__0(lean_object* v_a_1790_, lean_object* v_toPure_1791_, lean_object* v_s_1792_){
_start:
{
lean_object* v___x_1793_; lean_object* v___x_1794_; 
v___x_1793_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1793_, 0, v_a_1790_);
lean_ctor_set(v___x_1793_, 1, v_s_1792_);
v___x_1794_ = lean_apply_2(v_toPure_1791_, lean_box(0), v___x_1793_);
return v___x_1794_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_runStateRef___redArg___lam__1(lean_object* v_toPure_1795_, lean_object* v_ref_1796_, lean_object* v_inst_1797_, lean_object* v_toBind_1798_, lean_object* v_a_1799_){
_start:
{
lean_object* v___f_1800_; lean_object* v___x_1801_; lean_object* v___x_1802_; lean_object* v___x_1803_; 
v___f_1800_ = lean_alloc_closure((void*)(lp_batteries_MLList_runStateRef___redArg___lam__0), 3, 2);
lean_closure_set(v___f_1800_, 0, v_a_1799_);
lean_closure_set(v___f_1800_, 1, v_toPure_1795_);
v___x_1801_ = lean_alloc_closure((void*)(l_ST_Prim_Ref_get___boxed), 4, 3);
lean_closure_set(v___x_1801_, 0, lean_box(0));
lean_closure_set(v___x_1801_, 1, lean_box(0));
lean_closure_set(v___x_1801_, 2, v_ref_1796_);
v___x_1802_ = lean_apply_2(v_inst_1797_, lean_box(0), v___x_1801_);
v___x_1803_ = lean_apply_4(v_toBind_1798_, lean_box(0), lean_box(0), v___x_1802_, v___f_1800_);
return v___x_1803_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_runStateRef___redArg___lam__2(lean_object* v_toPure_1804_, lean_object* v_inst_1805_, lean_object* v_toBind_1806_, lean_object* v___x_1807_, lean_object* v_L_1808_, lean_object* v_ref_1809_){
_start:
{
lean_object* v___f_1810_; lean_object* v___x_89__overap_1811_; lean_object* v___x_1812_; lean_object* v___x_1813_; 
lean_inc(v_toBind_1806_);
lean_inc(v_ref_1809_);
v___f_1810_ = lean_alloc_closure((void*)(lp_batteries_MLList_runStateRef___redArg___lam__1), 5, 4);
lean_closure_set(v___f_1810_, 0, v_toPure_1804_);
lean_closure_set(v___f_1810_, 1, v_ref_1809_);
lean_closure_set(v___f_1810_, 2, v_inst_1805_);
lean_closure_set(v___f_1810_, 3, v_toBind_1806_);
v___x_89__overap_1811_ = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_unconsImpl___redArg(v___x_1807_, v_L_1808_);
v___x_1812_ = lean_apply_1(v___x_89__overap_1811_, v_ref_1809_);
v___x_1813_ = lean_apply_4(v_toBind_1806_, lean_box(0), lean_box(0), v___x_1812_, v___f_1810_);
return v___x_1813_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_runStateRef___redArg___lam__4(lean_object* v_s_1814_, lean_object* v_inst_1815_, lean_object* v_toBind_1816_, lean_object* v___f_1817_, lean_object* v___f_1818_, lean_object* v_x_1819_){
_start:
{
lean_object* v___x_1820_; lean_object* v___x_1821_; lean_object* v___x_1822_; lean_object* v___x_1823_; 
v___x_1820_ = lean_alloc_closure((void*)(l_ST_Prim_mkRef___boxed), 4, 3);
lean_closure_set(v___x_1820_, 0, lean_box(0));
lean_closure_set(v___x_1820_, 1, lean_box(0));
lean_closure_set(v___x_1820_, 2, v_s_1814_);
v___x_1821_ = lean_apply_2(v_inst_1815_, lean_box(0), v___x_1820_);
lean_inc(v_toBind_1816_);
v___x_1822_ = lean_apply_4(v_toBind_1816_, lean_box(0), lean_box(0), v___x_1821_, v___f_1817_);
v___x_1823_ = lean_apply_4(v_toBind_1816_, lean_box(0), lean_box(0), v___x_1822_, v___f_1818_);
return v___x_1823_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_runStateRef___redArg___lam__3(lean_object* v_toPure_1824_, lean_object* v_inst_1825_, lean_object* v_inst_1826_, lean_object* v_____do__lift_1827_){
_start:
{
lean_object* v_fst_1828_; 
v_fst_1828_ = lean_ctor_get(v_____do__lift_1827_, 0);
if (lean_obj_tag(v_fst_1828_) == 0)
{
lean_object* v___x_1829_; lean_object* v___x_1830_; 
lean_dec_ref(v_____do__lift_1827_);
lean_dec(v_inst_1826_);
lean_dec_ref(v_inst_1825_);
v___x_1829_ = lean_box(0);
v___x_1830_ = lean_apply_2(v_toPure_1824_, lean_box(0), v___x_1829_);
return v___x_1830_;
}
else
{
lean_object* v_val_1831_; lean_object* v_snd_1832_; lean_object* v_fst_1833_; lean_object* v_snd_1834_; lean_object* v___x_1836_; uint8_t v_isShared_1837_; uint8_t v_isSharedCheck_1843_; 
v_val_1831_ = lean_ctor_get(v_fst_1828_, 0);
lean_inc(v_val_1831_);
v_snd_1832_ = lean_ctor_get(v_____do__lift_1827_, 1);
lean_inc(v_snd_1832_);
lean_dec_ref(v_____do__lift_1827_);
v_fst_1833_ = lean_ctor_get(v_val_1831_, 0);
v_snd_1834_ = lean_ctor_get(v_val_1831_, 1);
v_isSharedCheck_1843_ = !lean_is_exclusive(v_val_1831_);
if (v_isSharedCheck_1843_ == 0)
{
v___x_1836_ = v_val_1831_;
v_isShared_1837_ = v_isSharedCheck_1843_;
goto v_resetjp_1835_;
}
else
{
lean_inc(v_snd_1834_);
lean_inc(v_fst_1833_);
lean_dec(v_val_1831_);
v___x_1836_ = lean_box(0);
v_isShared_1837_ = v_isSharedCheck_1843_;
goto v_resetjp_1835_;
}
v_resetjp_1835_:
{
lean_object* v___x_1838_; lean_object* v___x_1840_; 
v___x_1838_ = lp_batteries_MLList_runStateRef___redArg(v_inst_1825_, v_inst_1826_, v_snd_1834_, v_snd_1832_);
if (v_isShared_1837_ == 0)
{
lean_ctor_set_tag(v___x_1836_, 1);
lean_ctor_set(v___x_1836_, 1, v___x_1838_);
v___x_1840_ = v___x_1836_;
goto v_reusejp_1839_;
}
else
{
lean_object* v_reuseFailAlloc_1842_; 
v_reuseFailAlloc_1842_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1842_, 0, v_fst_1833_);
lean_ctor_set(v_reuseFailAlloc_1842_, 1, v___x_1838_);
v___x_1840_ = v_reuseFailAlloc_1842_;
goto v_reusejp_1839_;
}
v_reusejp_1839_:
{
lean_object* v___x_1841_; 
v___x_1841_ = lean_apply_2(v_toPure_1824_, lean_box(0), v___x_1840_);
return v___x_1841_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_runStateRef___redArg(lean_object* v_inst_1844_, lean_object* v_inst_1845_, lean_object* v_L_1846_, lean_object* v_s_1847_){
_start:
{
lean_object* v_toApplicative_1848_; lean_object* v_toBind_1849_; lean_object* v___x_1850_; lean_object* v_toPure_1851_; lean_object* v___f_1852_; lean_object* v___f_1853_; lean_object* v___f_1854_; lean_object* v___x_1855_; 
v_toApplicative_1848_ = lean_ctor_get(v_inst_1844_, 0);
v_toBind_1849_ = lean_ctor_get(v_inst_1844_, 1);
lean_inc_n(v_toBind_1849_, 2);
lean_inc_ref(v_inst_1844_);
v___x_1850_ = l_StateRefT_x27_instMonad___redArg(v_inst_1844_);
v_toPure_1851_ = lean_ctor_get(v_toApplicative_1848_, 1);
lean_inc_n(v_toPure_1851_, 2);
lean_inc_n(v_inst_1845_, 2);
v___f_1852_ = lean_alloc_closure((void*)(lp_batteries_MLList_runStateRef___redArg___lam__2), 6, 5);
lean_closure_set(v___f_1852_, 0, v_toPure_1851_);
lean_closure_set(v___f_1852_, 1, v_inst_1845_);
lean_closure_set(v___f_1852_, 2, v_toBind_1849_);
lean_closure_set(v___f_1852_, 3, v___x_1850_);
lean_closure_set(v___f_1852_, 4, v_L_1846_);
v___f_1853_ = lean_alloc_closure((void*)(lp_batteries_MLList_runStateRef___redArg___lam__3), 4, 3);
lean_closure_set(v___f_1853_, 0, v_toPure_1851_);
lean_closure_set(v___f_1853_, 1, v_inst_1844_);
lean_closure_set(v___f_1853_, 2, v_inst_1845_);
v___f_1854_ = lean_alloc_closure((void*)(lp_batteries_MLList_runStateRef___redArg___lam__4), 6, 5);
lean_closure_set(v___f_1854_, 0, v_s_1847_);
lean_closure_set(v___f_1854_, 1, v_inst_1845_);
lean_closure_set(v___f_1854_, 2, v_toBind_1849_);
lean_closure_set(v___f_1854_, 3, v___f_1852_);
lean_closure_set(v___f_1854_, 4, v___f_1853_);
v___x_1855_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1855_, 0, v___f_1854_);
return v___x_1855_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_runStateRef(lean_object* v_m_1856_, lean_object* v_00_u03c9_1857_, lean_object* v_00_u03c3_1858_, lean_object* v_00_u03b1_1859_, lean_object* v_inst_1860_, lean_object* v_inst_1861_, lean_object* v_L_1862_, lean_object* v_s_1863_){
_start:
{
lean_object* v___x_1864_; 
v___x_1864_ = lp_batteries_MLList_runStateRef___redArg(v_inst_1860_, v_inst_1861_, v_L_1862_, v_s_1863_);
return v___x_1864_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_head_x3f___redArg___lam__0(lean_object* v_toPure_1865_, lean_object* v_____do__lift_1866_){
_start:
{
if (lean_obj_tag(v_____do__lift_1866_) == 0)
{
lean_object* v___x_1867_; lean_object* v___x_1868_; 
v___x_1867_ = lean_box(0);
v___x_1868_ = lean_apply_2(v_toPure_1865_, lean_box(0), v___x_1867_);
return v___x_1868_;
}
else
{
lean_object* v_val_1869_; lean_object* v___x_1871_; uint8_t v_isShared_1872_; uint8_t v_isSharedCheck_1878_; 
v_val_1869_ = lean_ctor_get(v_____do__lift_1866_, 0);
v_isSharedCheck_1878_ = !lean_is_exclusive(v_____do__lift_1866_);
if (v_isSharedCheck_1878_ == 0)
{
v___x_1871_ = v_____do__lift_1866_;
v_isShared_1872_ = v_isSharedCheck_1878_;
goto v_resetjp_1870_;
}
else
{
lean_inc(v_val_1869_);
lean_dec(v_____do__lift_1866_);
v___x_1871_ = lean_box(0);
v_isShared_1872_ = v_isSharedCheck_1878_;
goto v_resetjp_1870_;
}
v_resetjp_1870_:
{
lean_object* v_fst_1873_; lean_object* v___x_1875_; 
v_fst_1873_ = lean_ctor_get(v_val_1869_, 0);
lean_inc(v_fst_1873_);
lean_dec(v_val_1869_);
if (v_isShared_1872_ == 0)
{
lean_ctor_set(v___x_1871_, 0, v_fst_1873_);
v___x_1875_ = v___x_1871_;
goto v_reusejp_1874_;
}
else
{
lean_object* v_reuseFailAlloc_1877_; 
v_reuseFailAlloc_1877_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1877_, 0, v_fst_1873_);
v___x_1875_ = v_reuseFailAlloc_1877_;
goto v_reusejp_1874_;
}
v_reusejp_1874_:
{
lean_object* v___x_1876_; 
v___x_1876_ = lean_apply_2(v_toPure_1865_, lean_box(0), v___x_1875_);
return v___x_1876_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_head_x3f___redArg(lean_object* v_inst_1879_, lean_object* v_L_1880_){
_start:
{
lean_object* v_toApplicative_1881_; lean_object* v_toBind_1882_; lean_object* v_toPure_1883_; lean_object* v___x_1884_; lean_object* v___f_1885_; lean_object* v___x_1886_; 
v_toApplicative_1881_ = lean_ctor_get(v_inst_1879_, 0);
v_toBind_1882_ = lean_ctor_get(v_inst_1879_, 1);
lean_inc(v_toBind_1882_);
v_toPure_1883_ = lean_ctor_get(v_toApplicative_1881_, 1);
lean_inc(v_toPure_1883_);
v___x_1884_ = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_unconsImpl___redArg(v_inst_1879_, v_L_1880_);
v___f_1885_ = lean_alloc_closure((void*)(lp_batteries_MLList_head_x3f___redArg___lam__0), 2, 1);
lean_closure_set(v___f_1885_, 0, v_toPure_1883_);
v___x_1886_ = lean_apply_4(v_toBind_1882_, lean_box(0), lean_box(0), v___x_1884_, v___f_1885_);
return v___x_1886_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_head_x3f(lean_object* v_m_1887_, lean_object* v_00_u03b1_1888_, lean_object* v_inst_1889_, lean_object* v_L_1890_){
_start:
{
lean_object* v___x_1891_; 
v___x_1891_ = lp_batteries_MLList_head_x3f___redArg(v_inst_1889_, v_L_1890_);
return v___x_1891_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_takeUpToFirstM___redArg(lean_object* v_inst_1892_, lean_object* v_L_1893_, lean_object* v_f_1894_){
_start:
{
lean_object* v_toApplicative_1895_; lean_object* v_toBind_1896_; lean_object* v_toPure_1897_; lean_object* v___f_1898_; lean_object* v___f_1899_; lean_object* v___x_1900_; 
v_toApplicative_1895_ = lean_ctor_get(v_inst_1892_, 0);
v_toBind_1896_ = lean_ctor_get(v_inst_1892_, 1);
v_toPure_1897_ = lean_ctor_get(v_toApplicative_1895_, 1);
lean_inc(v_toBind_1896_);
lean_inc_ref(v_inst_1892_);
lean_inc_n(v_toPure_1897_, 2);
v___f_1898_ = lean_alloc_closure((void*)(lp_batteries_MLList_takeUpToFirstM___redArg___lam__1), 6, 4);
lean_closure_set(v___f_1898_, 0, v_toPure_1897_);
lean_closure_set(v___f_1898_, 1, v_inst_1892_);
lean_closure_set(v___f_1898_, 2, v_f_1894_);
lean_closure_set(v___f_1898_, 3, v_toBind_1896_);
v___f_1899_ = lean_alloc_closure((void*)(lp_batteries_MLList_filterM___redArg___lam__2), 2, 1);
lean_closure_set(v___f_1899_, 0, v_toPure_1897_);
v___x_1900_ = lp_batteries_MLList_casesM___redArg(v_inst_1892_, v_L_1893_, v___f_1899_, v___f_1898_);
return v___x_1900_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_takeUpToFirstM___redArg___lam__0(lean_object* v_x_1901_, lean_object* v_toPure_1902_, lean_object* v_inst_1903_, lean_object* v_xs_1904_, lean_object* v_f_1905_, uint8_t v_____do__lift_1906_){
_start:
{
lean_object* v___y_1908_; 
if (v_____do__lift_1906_ == 0)
{
lean_object* v___x_1911_; 
v___x_1911_ = lp_batteries_MLList_takeUpToFirstM___redArg(v_inst_1903_, v_xs_1904_, v_f_1905_);
v___y_1908_ = v___x_1911_;
goto v___jp_1907_;
}
else
{
lean_object* v___x_1912_; 
lean_dec(v_f_1905_);
lean_dec(v_xs_1904_);
lean_dec_ref(v_inst_1903_);
v___x_1912_ = lean_box(0);
v___y_1908_ = v___x_1912_;
goto v___jp_1907_;
}
v___jp_1907_:
{
lean_object* v___x_1909_; lean_object* v___x_1910_; 
v___x_1909_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1909_, 0, v_x_1901_);
lean_ctor_set(v___x_1909_, 1, v___y_1908_);
v___x_1910_ = lean_apply_2(v_toPure_1902_, lean_box(0), v___x_1909_);
return v___x_1910_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_takeUpToFirstM___redArg___lam__0___boxed(lean_object* v_x_1913_, lean_object* v_toPure_1914_, lean_object* v_inst_1915_, lean_object* v_xs_1916_, lean_object* v_f_1917_, lean_object* v_____do__lift_1918_){
_start:
{
uint8_t v_____do__lift_75__boxed_1919_; lean_object* v_res_1920_; 
v_____do__lift_75__boxed_1919_ = lean_unbox(v_____do__lift_1918_);
v_res_1920_ = lp_batteries_MLList_takeUpToFirstM___redArg___lam__0(v_x_1913_, v_toPure_1914_, v_inst_1915_, v_xs_1916_, v_f_1917_, v_____do__lift_75__boxed_1919_);
return v_res_1920_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_takeUpToFirstM___redArg___lam__1(lean_object* v_toPure_1921_, lean_object* v_inst_1922_, lean_object* v_f_1923_, lean_object* v_toBind_1924_, lean_object* v_x_1925_, lean_object* v_xs_1926_){
_start:
{
lean_object* v___f_1927_; lean_object* v___x_1928_; lean_object* v___x_1929_; 
lean_inc(v_f_1923_);
lean_inc(v_x_1925_);
v___f_1927_ = lean_alloc_closure((void*)(lp_batteries_MLList_takeUpToFirstM___redArg___lam__0___boxed), 6, 5);
lean_closure_set(v___f_1927_, 0, v_x_1925_);
lean_closure_set(v___f_1927_, 1, v_toPure_1921_);
lean_closure_set(v___f_1927_, 2, v_inst_1922_);
lean_closure_set(v___f_1927_, 3, v_xs_1926_);
lean_closure_set(v___f_1927_, 4, v_f_1923_);
v___x_1928_ = lean_apply_1(v_f_1923_, v_x_1925_);
v___x_1929_ = lean_apply_4(v_toBind_1924_, lean_box(0), lean_box(0), v___x_1928_, v___f_1927_);
return v___x_1929_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_takeUpToFirstM(lean_object* v_m_1930_, lean_object* v_00_u03b1_1931_, lean_object* v_inst_1932_, lean_object* v_L_1933_, lean_object* v_f_1934_){
_start:
{
lean_object* v___x_1935_; 
v___x_1935_ = lp_batteries_MLList_takeUpToFirstM___redArg(v_inst_1932_, v_L_1933_, v_f_1934_);
return v___x_1935_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_takeUpToFirst___redArg(lean_object* v_inst_1936_, lean_object* v_L_1937_, lean_object* v_f_1938_){
_start:
{
lean_object* v_toApplicative_1939_; lean_object* v_toPure_1940_; lean_object* v___f_1941_; lean_object* v___x_1942_; 
v_toApplicative_1939_ = lean_ctor_get(v_inst_1936_, 0);
v_toPure_1940_ = lean_ctor_get(v_toApplicative_1939_, 1);
lean_inc(v_toPure_1940_);
v___f_1941_ = lean_alloc_closure((void*)(lp_batteries_MLList_takeWhile___redArg___lam__0), 3, 2);
lean_closure_set(v___f_1941_, 0, v_f_1938_);
lean_closure_set(v___f_1941_, 1, v_toPure_1940_);
v___x_1942_ = lp_batteries_MLList_takeUpToFirstM___redArg(v_inst_1936_, v_L_1937_, v___f_1941_);
return v___x_1942_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_takeUpToFirst(lean_object* v_m_1943_, lean_object* v_00_u03b1_1944_, lean_object* v_inst_1945_, lean_object* v_L_1946_, lean_object* v_f_1947_){
_start:
{
lean_object* v___x_1948_; 
v___x_1948_ = lp_batteries_MLList_takeUpToFirst___redArg(v_inst_1945_, v_L_1946_, v_f_1947_);
return v___x_1948_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_getLast_x3f_aux___redArg___lam__0(lean_object* v_x_1949_, lean_object* v_toPure_1950_, lean_object* v_inst_1951_, lean_object* v_____do__lift_1952_){
_start:
{
if (lean_obj_tag(v_____do__lift_1952_) == 0)
{
lean_object* v___x_1953_; lean_object* v___x_1954_; 
lean_dec_ref(v_inst_1951_);
v___x_1953_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1953_, 0, v_x_1949_);
v___x_1954_ = lean_apply_2(v_toPure_1950_, lean_box(0), v___x_1953_);
return v___x_1954_;
}
else
{
lean_object* v_val_1955_; lean_object* v_fst_1956_; lean_object* v_snd_1957_; lean_object* v___x_1958_; 
lean_dec(v_toPure_1950_);
lean_dec(v_x_1949_);
v_val_1955_ = lean_ctor_get(v_____do__lift_1952_, 0);
lean_inc(v_val_1955_);
lean_dec_ref_known(v_____do__lift_1952_, 1);
v_fst_1956_ = lean_ctor_get(v_val_1955_, 0);
lean_inc(v_fst_1956_);
v_snd_1957_ = lean_ctor_get(v_val_1955_, 1);
lean_inc(v_snd_1957_);
lean_dec(v_val_1955_);
v___x_1958_ = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_getLast_x3f_aux___redArg(v_inst_1951_, v_fst_1956_, v_snd_1957_);
return v___x_1958_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_getLast_x3f_aux___redArg(lean_object* v_inst_1959_, lean_object* v_x_1960_, lean_object* v_L_1961_){
_start:
{
lean_object* v_toApplicative_1962_; lean_object* v_toBind_1963_; lean_object* v_toPure_1964_; lean_object* v___x_1965_; lean_object* v___f_1966_; lean_object* v___x_1967_; 
v_toApplicative_1962_ = lean_ctor_get(v_inst_1959_, 0);
v_toBind_1963_ = lean_ctor_get(v_inst_1959_, 1);
lean_inc(v_toBind_1963_);
v_toPure_1964_ = lean_ctor_get(v_toApplicative_1962_, 1);
lean_inc(v_toPure_1964_);
lean_inc_ref(v_inst_1959_);
v___x_1965_ = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_unconsImpl___redArg(v_inst_1959_, v_L_1961_);
v___f_1966_ = lean_alloc_closure((void*)(lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_getLast_x3f_aux___redArg___lam__0), 4, 3);
lean_closure_set(v___f_1966_, 0, v_x_1960_);
lean_closure_set(v___f_1966_, 1, v_toPure_1964_);
lean_closure_set(v___f_1966_, 2, v_inst_1959_);
v___x_1967_ = lean_apply_4(v_toBind_1963_, lean_box(0), lean_box(0), v___x_1965_, v___f_1966_);
return v___x_1967_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_getLast_x3f_aux(lean_object* v_m_1968_, lean_object* v_00_u03b1_1969_, lean_object* v_inst_1970_, lean_object* v_x_1971_, lean_object* v_L_1972_){
_start:
{
lean_object* v___x_1973_; 
v___x_1973_ = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_getLast_x3f_aux___redArg(v_inst_1970_, v_x_1971_, v_L_1972_);
return v___x_1973_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_getLast_x3f___redArg___lam__0(lean_object* v_toPure_1974_, lean_object* v_inst_1975_, lean_object* v_____do__lift_1976_){
_start:
{
if (lean_obj_tag(v_____do__lift_1976_) == 0)
{
lean_object* v___x_1977_; lean_object* v___x_1978_; 
lean_dec_ref(v_inst_1975_);
v___x_1977_ = lean_box(0);
v___x_1978_ = lean_apply_2(v_toPure_1974_, lean_box(0), v___x_1977_);
return v___x_1978_;
}
else
{
lean_object* v_val_1979_; lean_object* v_fst_1980_; lean_object* v_snd_1981_; lean_object* v___x_1982_; 
lean_dec(v_toPure_1974_);
v_val_1979_ = lean_ctor_get(v_____do__lift_1976_, 0);
lean_inc(v_val_1979_);
lean_dec_ref_known(v_____do__lift_1976_, 1);
v_fst_1980_ = lean_ctor_get(v_val_1979_, 0);
lean_inc(v_fst_1980_);
v_snd_1981_ = lean_ctor_get(v_val_1979_, 1);
lean_inc(v_snd_1981_);
lean_dec(v_val_1979_);
v___x_1982_ = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_getLast_x3f_aux___redArg(v_inst_1975_, v_fst_1980_, v_snd_1981_);
return v___x_1982_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_getLast_x3f___redArg(lean_object* v_inst_1983_, lean_object* v_L_1984_){
_start:
{
lean_object* v_toApplicative_1985_; lean_object* v_toBind_1986_; lean_object* v_toPure_1987_; lean_object* v___x_1988_; lean_object* v___f_1989_; lean_object* v___x_1990_; 
v_toApplicative_1985_ = lean_ctor_get(v_inst_1983_, 0);
v_toBind_1986_ = lean_ctor_get(v_inst_1983_, 1);
lean_inc(v_toBind_1986_);
v_toPure_1987_ = lean_ctor_get(v_toApplicative_1985_, 1);
lean_inc(v_toPure_1987_);
lean_inc_ref(v_inst_1983_);
v___x_1988_ = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_unconsImpl___redArg(v_inst_1983_, v_L_1984_);
v___f_1989_ = lean_alloc_closure((void*)(lp_batteries_MLList_getLast_x3f___redArg___lam__0), 3, 2);
lean_closure_set(v___f_1989_, 0, v_toPure_1987_);
lean_closure_set(v___f_1989_, 1, v_inst_1983_);
v___x_1990_ = lean_apply_4(v_toBind_1986_, lean_box(0), lean_box(0), v___x_1988_, v___f_1989_);
return v___x_1990_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_getLast_x3f(lean_object* v_m_1991_, lean_object* v_00_u03b1_1992_, lean_object* v_inst_1993_, lean_object* v_L_1994_){
_start:
{
lean_object* v___x_1995_; 
v___x_1995_ = lp_batteries_MLList_getLast_x3f___redArg(v_inst_1993_, v_L_1994_);
return v___x_1995_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_getLast_x21___redArg(lean_object* v_inst_1996_, lean_object* v_inst_1997_, lean_object* v_L_1998_){
_start:
{
lean_object* v_toApplicative_1999_; lean_object* v_toFunctor_2000_; lean_object* v_map_2001_; lean_object* v___x_2002_; lean_object* v___x_2003_; lean_object* v___x_2004_; 
v_toApplicative_1999_ = lean_ctor_get(v_inst_1996_, 0);
v_toFunctor_2000_ = lean_ctor_get(v_toApplicative_1999_, 0);
v_map_2001_ = lean_ctor_get(v_toFunctor_2000_, 0);
lean_inc(v_map_2001_);
v___x_2002_ = lean_alloc_closure((void*)(l_Option_get_x21___boxed), 3, 2);
lean_closure_set(v___x_2002_, 0, lean_box(0));
lean_closure_set(v___x_2002_, 1, v_inst_1997_);
v___x_2003_ = lp_batteries_MLList_getLast_x3f___redArg(v_inst_1996_, v_L_1998_);
v___x_2004_ = lean_apply_4(v_map_2001_, lean_box(0), lean_box(0), v___x_2002_, v___x_2003_);
return v___x_2004_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_getLast_x21(lean_object* v_m_2005_, lean_object* v_00_u03b1_2006_, lean_object* v_inst_2007_, lean_object* v_inst_2008_, lean_object* v_L_2009_){
_start:
{
lean_object* v___x_2010_; 
v___x_2010_ = lp_batteries_MLList_getLast_x21___redArg(v_inst_2007_, v_inst_2008_, v_L_2009_);
return v___x_2010_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_foldM___redArg___lam__0(lean_object* v_toPure_2011_, lean_object* v_init_2012_, lean_object* v_____do__lift_2013_){
_start:
{
if (lean_obj_tag(v_____do__lift_2013_) == 0)
{
lean_object* v___x_2014_; 
v___x_2014_ = lean_apply_2(v_toPure_2011_, lean_box(0), v_init_2012_);
return v___x_2014_;
}
else
{
lean_object* v_val_2015_; lean_object* v___x_2016_; 
lean_dec(v_init_2012_);
v_val_2015_ = lean_ctor_get(v_____do__lift_2013_, 0);
lean_inc(v_val_2015_);
lean_dec_ref_known(v_____do__lift_2013_, 1);
v___x_2016_ = lean_apply_2(v_toPure_2011_, lean_box(0), v_val_2015_);
return v___x_2016_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_foldM___redArg(lean_object* v_inst_2017_, lean_object* v_f_2018_, lean_object* v_init_2019_, lean_object* v_L_2020_){
_start:
{
lean_object* v_toApplicative_2021_; lean_object* v_toBind_2022_; lean_object* v_toPure_2023_; lean_object* v___x_2024_; lean_object* v___x_2025_; lean_object* v___f_2026_; lean_object* v___x_2027_; 
v_toApplicative_2021_ = lean_ctor_get(v_inst_2017_, 0);
v_toBind_2022_ = lean_ctor_get(v_inst_2017_, 1);
lean_inc(v_toBind_2022_);
v_toPure_2023_ = lean_ctor_get(v_toApplicative_2021_, 1);
lean_inc(v_toPure_2023_);
lean_inc(v_init_2019_);
lean_inc_ref(v_inst_2017_);
v___x_2024_ = lp_batteries_MLList_foldsM___redArg(v_inst_2017_, v_f_2018_, v_init_2019_, v_L_2020_);
v___x_2025_ = lp_batteries_MLList_getLast_x3f___redArg(v_inst_2017_, v___x_2024_);
v___f_2026_ = lean_alloc_closure((void*)(lp_batteries_MLList_foldM___redArg___lam__0), 3, 2);
lean_closure_set(v___f_2026_, 0, v_toPure_2023_);
lean_closure_set(v___f_2026_, 1, v_init_2019_);
v___x_2027_ = lean_apply_4(v_toBind_2022_, lean_box(0), lean_box(0), v___x_2025_, v___f_2026_);
return v___x_2027_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_foldM(lean_object* v_m_2028_, lean_object* v_00_u03b2_2029_, lean_object* v_00_u03b1_2030_, lean_object* v_inst_2031_, lean_object* v_f_2032_, lean_object* v_init_2033_, lean_object* v_L_2034_){
_start:
{
lean_object* v___x_2035_; 
v___x_2035_ = lp_batteries_MLList_foldM___redArg(v_inst_2031_, v_f_2032_, v_init_2033_, v_L_2034_);
return v___x_2035_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_fold___redArg(lean_object* v_inst_2036_, lean_object* v_f_2037_, lean_object* v_init_2038_, lean_object* v_L_2039_){
_start:
{
lean_object* v_toApplicative_2040_; lean_object* v_toPure_2041_; lean_object* v___f_2042_; lean_object* v___x_2043_; 
v_toApplicative_2040_ = lean_ctor_get(v_inst_2036_, 0);
v_toPure_2041_ = lean_ctor_get(v_toApplicative_2040_, 1);
lean_inc(v_toPure_2041_);
v___f_2042_ = lean_alloc_closure((void*)(lp_batteries_MLList_folds___redArg___lam__0), 4, 2);
lean_closure_set(v___f_2042_, 0, v_f_2037_);
lean_closure_set(v___f_2042_, 1, v_toPure_2041_);
v___x_2043_ = lp_batteries_MLList_foldM___redArg(v_inst_2036_, v___f_2042_, v_init_2038_, v_L_2039_);
return v___x_2043_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_fold(lean_object* v_m_2044_, lean_object* v_00_u03b2_2045_, lean_object* v_00_u03b1_2046_, lean_object* v_inst_2047_, lean_object* v_f_2048_, lean_object* v_init_2049_, lean_object* v_L_2050_){
_start:
{
lean_object* v___x_2051_; 
v___x_2051_ = lp_batteries_MLList_fold___redArg(v_inst_2047_, v_f_2048_, v_init_2049_, v_L_2050_);
return v___x_2051_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_head___redArg___lam__0(lean_object* v_toPure_2052_, lean_object* v_failure_2053_, lean_object* v_____x_2054_){
_start:
{
if (lean_obj_tag(v_____x_2054_) == 1)
{
lean_object* v_val_2055_; lean_object* v_fst_2056_; lean_object* v___x_2057_; 
lean_dec(v_failure_2053_);
v_val_2055_ = lean_ctor_get(v_____x_2054_, 0);
lean_inc(v_val_2055_);
lean_dec_ref_known(v_____x_2054_, 1);
v_fst_2056_ = lean_ctor_get(v_val_2055_, 0);
lean_inc(v_fst_2056_);
lean_dec(v_val_2055_);
v___x_2057_ = lean_apply_2(v_toPure_2052_, lean_box(0), v_fst_2056_);
return v___x_2057_;
}
else
{
lean_object* v___x_2058_; 
lean_dec(v_____x_2054_);
lean_dec(v_toPure_2052_);
v___x_2058_ = lean_apply_1(v_failure_2053_, lean_box(0));
return v___x_2058_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_head___redArg(lean_object* v_inst_2059_, lean_object* v_L_2060_){
_start:
{
lean_object* v___x_2061_; lean_object* v_toAlternative_2062_; lean_object* v_toApplicative_2063_; lean_object* v_toBind_2064_; lean_object* v_failure_2065_; lean_object* v_toPure_2066_; lean_object* v___x_2067_; lean_object* v___f_2068_; lean_object* v___x_2069_; 
lean_inc_ref(v_inst_2059_);
v___x_2061_ = lp_batteries_AlternativeMonad_toMonad___redArg(v_inst_2059_);
v_toAlternative_2062_ = lean_ctor_get(v_inst_2059_, 0);
lean_inc_ref(v_toAlternative_2062_);
lean_dec_ref(v_inst_2059_);
v_toApplicative_2063_ = lean_ctor_get(v_toAlternative_2062_, 0);
lean_inc_ref(v_toApplicative_2063_);
v_toBind_2064_ = lean_ctor_get(v___x_2061_, 1);
lean_inc(v_toBind_2064_);
v_failure_2065_ = lean_ctor_get(v_toAlternative_2062_, 1);
lean_inc(v_failure_2065_);
lean_dec_ref(v_toAlternative_2062_);
v_toPure_2066_ = lean_ctor_get(v_toApplicative_2063_, 1);
lean_inc(v_toPure_2066_);
lean_dec_ref(v_toApplicative_2063_);
v___x_2067_ = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_unconsImpl___redArg(v___x_2061_, v_L_2060_);
v___f_2068_ = lean_alloc_closure((void*)(lp_batteries_MLList_head___redArg___lam__0), 3, 2);
lean_closure_set(v___f_2068_, 0, v_toPure_2066_);
lean_closure_set(v___f_2068_, 1, v_failure_2065_);
v___x_2069_ = lean_apply_4(v_toBind_2064_, lean_box(0), lean_box(0), v___x_2067_, v___f_2068_);
return v___x_2069_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_head(lean_object* v_m_2070_, lean_object* v_00_u03b1_2071_, lean_object* v_inst_2072_, lean_object* v_L_2073_){
_start:
{
lean_object* v___x_2074_; 
v___x_2074_ = lp_batteries_MLList_head___redArg(v_inst_2072_, v_L_2073_);
return v___x_2074_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_firstM___redArg(lean_object* v_inst_2075_, lean_object* v_L_2076_, lean_object* v_f_2077_){
_start:
{
lean_object* v___x_2078_; lean_object* v___x_2079_; lean_object* v___x_2080_; 
lean_inc_ref(v_inst_2075_);
v___x_2078_ = lp_batteries_AlternativeMonad_toMonad___redArg(v_inst_2075_);
v___x_2079_ = lp_batteries_MLList_filterMapM___redArg(v___x_2078_, v_f_2077_, v_L_2076_);
v___x_2080_ = lp_batteries_MLList_head___redArg(v_inst_2075_, v___x_2079_);
return v___x_2080_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_firstM(lean_object* v_m_2081_, lean_object* v_00_u03b1_2082_, lean_object* v_00_u03b2_2083_, lean_object* v_inst_2084_, lean_object* v_L_2085_, lean_object* v_f_2086_){
_start:
{
lean_object* v___x_2087_; 
v___x_2087_ = lp_batteries_MLList_firstM___redArg(v_inst_2084_, v_L_2085_, v_f_2086_);
return v___x_2087_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_first___redArg(lean_object* v_inst_2088_, lean_object* v_L_2089_, lean_object* v_p_2090_){
_start:
{
lean_object* v___x_2091_; lean_object* v___x_2092_; lean_object* v___x_2093_; 
lean_inc_ref(v_inst_2088_);
v___x_2091_ = lp_batteries_AlternativeMonad_toMonad___redArg(v_inst_2088_);
v___x_2092_ = lp_batteries_MLList_filter___redArg(v___x_2091_, v_p_2090_, v_L_2089_);
v___x_2093_ = lp_batteries_MLList_head___redArg(v_inst_2088_, v___x_2092_);
return v___x_2093_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_first(lean_object* v_m_2094_, lean_object* v_00_u03b1_2095_, lean_object* v_inst_2096_, lean_object* v_L_2097_, lean_object* v_p_2098_){
_start:
{
lean_object* v___x_2099_; 
v___x_2099_ = lp_batteries_MLList_first___redArg(v_inst_2096_, v_L_2097_, v_p_2098_);
return v___x_2099_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_instAlternativeMonadOfMonad___redArg___lam__0(lean_object* v_inst_2100_, lean_object* v_00_u03b1_2101_, lean_object* v_00_u03b2_2102_, lean_object* v___y_2103_, lean_object* v___y_2104_){
_start:
{
lean_object* v___x_2105_; 
v___x_2105_ = lp_batteries_MLList_map___redArg(v_inst_2100_, v___y_2103_, v___y_2104_);
return v___x_2105_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_instAlternativeMonadOfMonad___redArg___lam__1(lean_object* v_inst_2106_, lean_object* v_00_u03b1_2107_, lean_object* v_00_u03b2_2108_, lean_object* v___y_2109_, lean_object* v___y_2110_){
_start:
{
lean_object* v___x_2111_; lean_object* v___x_2112_; 
v___x_2111_ = lean_alloc_closure((void*)(l_Function_const___boxed), 4, 3);
lean_closure_set(v___x_2111_, 0, lean_box(0));
lean_closure_set(v___x_2111_, 1, lean_box(0));
lean_closure_set(v___x_2111_, 2, v___y_2109_);
v___x_2112_ = lp_batteries_MLList_map___redArg(v_inst_2106_, v___x_2111_, v___y_2110_);
return v___x_2112_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_instAlternativeMonadOfMonad___redArg___lam__2(lean_object* v_00_u03b1_2113_, lean_object* v_a_2114_){
_start:
{
lean_object* v___x_2115_; lean_object* v___x_2116_; 
v___x_2115_ = lean_box(0);
v___x_2116_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_2116_, 0, v_a_2114_);
lean_ctor_set(v___x_2116_, 1, v___x_2115_);
return v___x_2116_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_instAlternativeMonadOfMonad___redArg___lam__3(lean_object* v_x_2117_, lean_object* v_inst_2118_, lean_object* v_y_2119_){
_start:
{
lean_object* v___x_2120_; lean_object* v___x_2121_; lean_object* v___x_2122_; 
v___x_2120_ = lean_box(0);
v___x_2121_ = lean_apply_1(v_x_2117_, v___x_2120_);
v___x_2122_ = lp_batteries_MLList_map___redArg(v_inst_2118_, v_y_2119_, v___x_2121_);
return v___x_2122_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_instAlternativeMonadOfMonad___redArg___lam__4(lean_object* v_inst_2123_, lean_object* v_00_u03b1_2124_, lean_object* v_00_u03b2_2125_, lean_object* v_f_2126_, lean_object* v_x_2127_){
_start:
{
lean_object* v___f_2128_; lean_object* v___x_2129_; 
lean_inc_ref(v_inst_2123_);
v___f_2128_ = lean_alloc_closure((void*)(lp_batteries_MLList_instAlternativeMonadOfMonad___redArg___lam__3), 3, 2);
lean_closure_set(v___f_2128_, 0, v_x_2127_);
lean_closure_set(v___f_2128_, 1, v_inst_2123_);
v___x_2129_ = lp_batteries_MLList_bind___redArg(v_inst_2123_, v_f_2126_, v___f_2128_);
return v___x_2129_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_instAlternativeMonadOfMonad___redArg___lam__5(lean_object* v___f_2130_, lean_object* v_a_2131_, lean_object* v_x_2132_){
_start:
{
lean_object* v___x_2133_; 
v___x_2133_ = lean_apply_2(v___f_2130_, lean_box(0), v_a_2131_);
return v___x_2133_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_instAlternativeMonadOfMonad___redArg___lam__5___boxed(lean_object* v___f_2134_, lean_object* v_a_2135_, lean_object* v_x_2136_){
_start:
{
lean_object* v_res_2137_; 
v_res_2137_ = lp_batteries_MLList_instAlternativeMonadOfMonad___redArg___lam__5(v___f_2134_, v_a_2135_, v_x_2136_);
lean_dec(v_x_2136_);
return v_res_2137_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_instAlternativeMonadOfMonad___redArg___lam__6(lean_object* v___f_2138_, lean_object* v_y_2139_, lean_object* v_inst_2140_, lean_object* v_a_2141_){
_start:
{
lean_object* v___f_2142_; lean_object* v___x_2143_; lean_object* v___x_2144_; lean_object* v___x_2145_; 
v___f_2142_ = lean_alloc_closure((void*)(lp_batteries_MLList_instAlternativeMonadOfMonad___redArg___lam__5___boxed), 3, 2);
lean_closure_set(v___f_2142_, 0, v___f_2138_);
lean_closure_set(v___f_2142_, 1, v_a_2141_);
v___x_2143_ = lean_box(0);
v___x_2144_ = lean_apply_1(v_y_2139_, v___x_2143_);
v___x_2145_ = lp_batteries_MLList_bind___redArg(v_inst_2140_, v___x_2144_, v___f_2142_);
return v___x_2145_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_instAlternativeMonadOfMonad___redArg___lam__7(lean_object* v___f_2146_, lean_object* v_inst_2147_, lean_object* v_00_u03b1_2148_, lean_object* v_00_u03b2_2149_, lean_object* v_x_2150_, lean_object* v_y_2151_){
_start:
{
lean_object* v___f_2152_; lean_object* v___x_2153_; 
lean_inc_ref(v_inst_2147_);
v___f_2152_ = lean_alloc_closure((void*)(lp_batteries_MLList_instAlternativeMonadOfMonad___redArg___lam__6), 4, 3);
lean_closure_set(v___f_2152_, 0, v___f_2146_);
lean_closure_set(v___f_2152_, 1, v_y_2151_);
lean_closure_set(v___f_2152_, 2, v_inst_2147_);
v___x_2153_ = lp_batteries_MLList_bind___redArg(v_inst_2147_, v_x_2150_, v___f_2152_);
return v___x_2153_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_instAlternativeMonadOfMonad___redArg___lam__8(lean_object* v_y_2154_, lean_object* v_x_2155_){
_start:
{
lean_object* v___x_2156_; lean_object* v___x_2157_; 
v___x_2156_ = lean_box(0);
v___x_2157_ = lean_apply_1(v_y_2154_, v___x_2156_);
return v___x_2157_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_instAlternativeMonadOfMonad___redArg___lam__8___boxed(lean_object* v_y_2158_, lean_object* v_x_2159_){
_start:
{
lean_object* v_res_2160_; 
v_res_2160_ = lp_batteries_MLList_instAlternativeMonadOfMonad___redArg___lam__8(v_y_2158_, v_x_2159_);
lean_dec(v_x_2159_);
return v_res_2160_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_instAlternativeMonadOfMonad___redArg___lam__9(lean_object* v_inst_2161_, lean_object* v_00_u03b1_2162_, lean_object* v_00_u03b2_2163_, lean_object* v_x_2164_, lean_object* v_y_2165_){
_start:
{
lean_object* v___f_2166_; lean_object* v___x_2167_; 
v___f_2166_ = lean_alloc_closure((void*)(lp_batteries_MLList_instAlternativeMonadOfMonad___redArg___lam__8___boxed), 2, 1);
lean_closure_set(v___f_2166_, 0, v_y_2165_);
v___x_2167_ = lp_batteries_MLList_bind___redArg(v_inst_2161_, v_x_2164_, v___f_2166_);
return v___x_2167_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_instAlternativeMonadOfMonad___redArg___lam__10(lean_object* v_inst_2168_, lean_object* v_00_u03b1_2169_, lean_object* v___y_2170_, lean_object* v___y_2171_){
_start:
{
lean_object* v___x_2172_; 
v___x_2172_ = lp_batteries_MLList_append___redArg(v_inst_2168_, v___y_2170_, v___y_2171_);
return v___x_2172_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_instAlternativeMonadOfMonad___redArg___lam__11(lean_object* v_inst_2173_, lean_object* v_00_u03b1_2174_, lean_object* v_00_u03b2_2175_, lean_object* v___y_2176_, lean_object* v___y_2177_){
_start:
{
lean_object* v___x_2178_; 
v___x_2178_ = lp_batteries_MLList_bind___redArg(v_inst_2173_, v___y_2176_, v___y_2177_);
return v___x_2178_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_instAlternativeMonadOfMonad___redArg(lean_object* v_inst_2181_){
_start:
{
lean_object* v___f_2182_; lean_object* v___f_2183_; lean_object* v___f_2184_; lean_object* v___f_2185_; lean_object* v___f_2186_; lean_object* v___f_2187_; lean_object* v___f_2188_; lean_object* v___f_2189_; lean_object* v___x_2190_; lean_object* v___x_2191_; lean_object* v___x_2192_; lean_object* v___x_2193_; lean_object* v___x_2194_; 
lean_inc_ref_n(v_inst_2181_, 6);
v___f_2182_ = lean_alloc_closure((void*)(lp_batteries_MLList_instAlternativeMonadOfMonad___redArg___lam__0), 5, 1);
lean_closure_set(v___f_2182_, 0, v_inst_2181_);
v___f_2183_ = lean_alloc_closure((void*)(lp_batteries_MLList_instAlternativeMonadOfMonad___redArg___lam__1), 5, 1);
lean_closure_set(v___f_2183_, 0, v_inst_2181_);
v___f_2184_ = ((lean_object*)(lp_batteries_MLList_instAlternativeMonadOfMonad___redArg___closed__0));
v___f_2185_ = lean_alloc_closure((void*)(lp_batteries_MLList_instAlternativeMonadOfMonad___redArg___lam__4), 5, 1);
lean_closure_set(v___f_2185_, 0, v_inst_2181_);
v___f_2186_ = lean_alloc_closure((void*)(lp_batteries_MLList_instAlternativeMonadOfMonad___redArg___lam__7), 6, 2);
lean_closure_set(v___f_2186_, 0, v___f_2184_);
lean_closure_set(v___f_2186_, 1, v_inst_2181_);
v___f_2187_ = lean_alloc_closure((void*)(lp_batteries_MLList_instAlternativeMonadOfMonad___redArg___lam__9), 5, 1);
lean_closure_set(v___f_2187_, 0, v_inst_2181_);
v___f_2188_ = lean_alloc_closure((void*)(lp_batteries_MLList_instAlternativeMonadOfMonad___redArg___lam__10), 4, 1);
lean_closure_set(v___f_2188_, 0, v_inst_2181_);
v___f_2189_ = lean_alloc_closure((void*)(lp_batteries_MLList_instAlternativeMonadOfMonad___redArg___lam__11), 5, 1);
lean_closure_set(v___f_2189_, 0, v_inst_2181_);
v___x_2190_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2190_, 0, v___f_2182_);
lean_ctor_set(v___x_2190_, 1, v___f_2183_);
v___x_2191_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_2191_, 0, v___x_2190_);
lean_ctor_set(v___x_2191_, 1, v___f_2184_);
lean_ctor_set(v___x_2191_, 2, v___f_2185_);
lean_ctor_set(v___x_2191_, 3, v___f_2186_);
lean_ctor_set(v___x_2191_, 4, v___f_2187_);
v___x_2192_ = ((lean_object*)(lp_batteries_MLList_instAlternativeMonadOfMonad___redArg___closed__1));
v___x_2193_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_2193_, 0, v___x_2191_);
lean_ctor_set(v___x_2193_, 1, v___x_2192_);
lean_ctor_set(v___x_2193_, 2, v___f_2188_);
v___x_2194_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2194_, 0, v___x_2193_);
lean_ctor_set(v___x_2194_, 1, v___f_2189_);
return v___x_2194_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_instAlternativeMonadOfMonad(lean_object* v_m_2195_, lean_object* v_inst_2196_){
_start:
{
lean_object* v___x_2197_; 
v___x_2197_ = lp_batteries_MLList_instAlternativeMonadOfMonad___redArg(v_inst_2196_);
return v___x_2197_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_instMonadLiftOfMonad___redArg___lam__0(lean_object* v_inst_2198_, lean_object* v_00_u03b1_2199_, lean_object* v___y_2200_){
_start:
{
lean_object* v___x_2201_; 
v___x_2201_ = lp_batteries_MLList_monadLift___redArg(v_inst_2198_, v___y_2200_);
return v___x_2201_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_instMonadLiftOfMonad___redArg(lean_object* v_inst_2202_){
_start:
{
lean_object* v___f_2203_; 
v___f_2203_ = lean_alloc_closure((void*)(lp_batteries_MLList_instMonadLiftOfMonad___redArg___lam__0), 3, 1);
lean_closure_set(v___f_2203_, 0, v_inst_2202_);
return v___f_2203_;
}
}
LEAN_EXPORT lean_object* lp_batteries_MLList_instMonadLiftOfMonad(lean_object* v_m_2204_, lean_object* v_inst_2205_){
_start:
{
lean_object* v___f_2206_; 
v___f_2206_ = lean_alloc_closure((void*)(lp_batteries_MLList_instMonadLiftOfMonad___redArg___lam__0), 3, 1);
lean_closure_set(v___f_2206_, 0, v_inst_2205_);
return v___f_2206_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Control_AlternativeMonad(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_batteries_Batteries_Data_MLList_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Control_AlternativeMonad(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_batteries_Batteries_Data_MLList_Basic(uint8_t builtin) {
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
lean_object* initialize_batteries_Batteries_Control_AlternativeMonad(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_batteries_Batteries_Data_MLList_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Control_AlternativeMonad(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Data_MLList_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_batteries_Batteries_Data_MLList_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_batteries_Batteries_Data_MLList_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
