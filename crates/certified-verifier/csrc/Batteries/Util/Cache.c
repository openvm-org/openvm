// Lean compiler output
// Module: Batteries.Util.Cache
// Imports: public import Init public meta import Init public import Lean.Meta.DiscrTree
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
lean_object* lean_task_get_own(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* lean_st_mk_ref(lean_object*);
lean_object* l_Lean_Core_wrapAsync___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_io_as_task(lean_object*, lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* l_Lean_Environment_constants(lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
size_t lean_usize_add(size_t, size_t);
lean_object* l_Lean_profileitIOUnsafe___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* l_Lean_Meta_DiscrTree_mapArrays___redArg(lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_DiscrTree_insertKeyValue___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__6(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__2___boxed(lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_Lean_Meta_DiscrTree_getMatch___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_reverse___redArg(lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Cache_mk___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Cache_mk___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Cache_mk(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Cache_mk___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_wrapAsync___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_wrapAsync___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_wrapAsync___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_wrapAsync___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_wrapAsync(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_wrapAsync___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Cache_get___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Cache_get___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Cache_get___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Cache_get___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Cache_get___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Cache_get___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Cache_get(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Cache_get___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_profileitM___at___00Batteries_Tactic_DeclCache_mk_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_profileitM___at___00Batteries_Tactic_DeclCache_mk_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_profileitM___at___00Batteries_Tactic_DeclCache_mk_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_profileitM___at___00Batteries_Tactic_DeclCache_mk_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_DeclCache_mk_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_DeclCache_mk_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_DeclCache_mk_spec__1___redArg(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_DeclCache_mk_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_DeclCache_mk___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_DeclCache_mk___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_DeclCache_mk___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_DeclCache_mk___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_DeclCache_mk___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_DeclCache_mk___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_DeclCache_mk(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_DeclCache_mk___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_DeclCache_mk_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_DeclCache_mk_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_DeclCache_mk_spec__1(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_DeclCache_mk_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_DeclCache_get___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_DeclCache_get___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Batteries_Tactic_DeclCache_get_spec__0_spec__0_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Batteries_Tactic_DeclCache_get_spec__0_spec__0_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Batteries_Tactic_DeclCache_get_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Batteries_Tactic_DeclCache_get_spec__0_spec__0_spec__1___redArg(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Batteries_Tactic_DeclCache_get_spec__0_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Batteries_Tactic_DeclCache_get_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_DeclCache_get___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_DeclCache_get___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_DeclCache_get(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_DeclCache_get___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlM___at___00Batteries_Tactic_DeclCache_get_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlM___at___00Batteries_Tactic_DeclCache_get_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlM___at___00Batteries_Tactic_DeclCache_get_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlM___at___00Batteries_Tactic_DeclCache_get_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Batteries_Tactic_DeclCache_get_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Batteries_Tactic_DeclCache_get_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Batteries_Tactic_DeclCache_get_spec__0_spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Batteries_Tactic_DeclCache_get_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Batteries_Tactic_DeclCache_get_spec__0_spec__0_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Batteries_Tactic_DeclCache_get_spec__0_spec__0_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__1(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__0, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__2___closed__0 = (const lean_object*)&lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__2___closed__0_value;
static const lean_closure_object lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__1___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__2___closed__1 = (const lean_object*)&lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__2___closed__1_value;
static const lean_closure_object lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__2___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__2___closed__2 = (const lean_object*)&lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__2___closed__2_value;
static const lean_closure_object lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__2___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__3, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__2___closed__3 = (const lean_object*)&lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__2___closed__3_value;
static const lean_closure_object lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__2___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__4___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__2___closed__4 = (const lean_object*)&lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__2___closed__4_value;
static const lean_closure_object lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__2___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__5___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__2___closed__5 = (const lean_object*)&lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__2___closed__5_value;
static const lean_closure_object lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__2___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__6, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__2___closed__6 = (const lean_object*)&lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__2___closed__6_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__2___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__2___closed__0_value),((lean_object*)&lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__2___closed__1_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__2___closed__7 = (const lean_object*)&lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__2___closed__7_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__2___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*5 + 0, .m_other = 5, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__2___closed__7_value),((lean_object*)&lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__2___closed__2_value),((lean_object*)&lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__2___closed__3_value),((lean_object*)&lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__2___closed__4_value),((lean_object*)&lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__2___closed__5_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__2___closed__8 = (const lean_object*)&lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__2___closed__8_value;
static const lean_ctor_object lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__2___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__2___closed__8_value),((lean_object*)&lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__2___closed__6_value)}};
static const lean_object* lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__2___closed__9 = (const lean_object*)&lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__2___closed__9_value;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__5___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__5___closed__0;
static lean_once_cell_t lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__5___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__5___closed__1;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___closed__0;
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_DiscrTreeCache_mk(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_DiscrTreeCache_getMatch___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_DiscrTreeCache_getMatch___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_DiscrTreeCache_getMatch(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_DiscrTreeCache_getMatch___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Cache_mk___redArg(lean_object* v_init_1_){
_start:
{
lean_object* v___x_3_; lean_object* v___x_4_; lean_object* v___x_5_; 
v___x_3_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_3_, 0, v_init_1_);
v___x_4_ = lean_st_mk_ref(v___x_3_);
v___x_5_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_5_, 0, v___x_4_);
return v___x_5_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Cache_mk___redArg___boxed(lean_object* v_init_6_, lean_object* v_a_7_){
_start:
{
lean_object* v_res_8_; 
v_res_8_ = lp_batteries_Batteries_Tactic_Cache_mk___redArg(v_init_6_);
return v_res_8_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Cache_mk(lean_object* v_00_u03b1_9_, lean_object* v_init_10_){
_start:
{
lean_object* v___x_12_; 
v___x_12_ = lp_batteries_Batteries_Tactic_Cache_mk___redArg(v_init_10_);
return v___x_12_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Cache_mk___boxed(lean_object* v_00_u03b1_13_, lean_object* v_init_14_, lean_object* v_a_15_){
_start:
{
lean_object* v_res_16_; 
v_res_16_ = lp_batteries_Batteries_Tactic_Cache_mk(v_00_u03b1_13_, v_init_14_);
return v_res_16_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_wrapAsync___redArg___lam__0(lean_object* v_val_17_, lean_object* v_act_18_, lean_object* v_a_19_, lean_object* v_a_20_, lean_object* v___y_21_, lean_object* v___y_22_){
_start:
{
lean_object* v___x_24_; lean_object* v___x_25_; 
v___x_24_ = lean_st_mk_ref(v_val_17_);
lean_inc(v___y_22_);
lean_inc_ref(v___y_21_);
lean_inc(v___x_24_);
lean_inc_ref(v_a_19_);
v___x_25_ = lean_apply_6(v_act_18_, v_a_20_, v_a_19_, v___x_24_, v___y_21_, v___y_22_, lean_box(0));
if (lean_obj_tag(v___x_25_) == 0)
{
lean_object* v_a_26_; lean_object* v___x_28_; uint8_t v_isShared_29_; uint8_t v_isSharedCheck_34_; 
v_a_26_ = lean_ctor_get(v___x_25_, 0);
v_isSharedCheck_34_ = !lean_is_exclusive(v___x_25_);
if (v_isSharedCheck_34_ == 0)
{
v___x_28_ = v___x_25_;
v_isShared_29_ = v_isSharedCheck_34_;
goto v_resetjp_27_;
}
else
{
lean_inc(v_a_26_);
lean_dec(v___x_25_);
v___x_28_ = lean_box(0);
v_isShared_29_ = v_isSharedCheck_34_;
goto v_resetjp_27_;
}
v_resetjp_27_:
{
lean_object* v___x_30_; lean_object* v___x_32_; 
v___x_30_ = lean_st_ref_get(v___x_24_);
lean_dec(v___x_24_);
lean_dec(v___x_30_);
if (v_isShared_29_ == 0)
{
v___x_32_ = v___x_28_;
goto v_reusejp_31_;
}
else
{
lean_object* v_reuseFailAlloc_33_; 
v_reuseFailAlloc_33_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_33_, 0, v_a_26_);
v___x_32_ = v_reuseFailAlloc_33_;
goto v_reusejp_31_;
}
v_reusejp_31_:
{
return v___x_32_;
}
}
}
else
{
lean_dec(v___x_24_);
return v___x_25_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_wrapAsync___redArg___lam__0___boxed(lean_object* v_val_35_, lean_object* v_act_36_, lean_object* v_a_37_, lean_object* v_a_38_, lean_object* v___y_39_, lean_object* v___y_40_, lean_object* v___y_41_){
_start:
{
lean_object* v_res_42_; 
v_res_42_ = lp_batteries_Lean_Meta_wrapAsync___redArg___lam__0(v_val_35_, v_act_36_, v_a_37_, v_a_38_, v___y_39_, v___y_40_);
lean_dec(v___y_40_);
lean_dec_ref(v___y_39_);
lean_dec_ref(v_a_37_);
return v_res_42_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_wrapAsync___redArg(lean_object* v_act_43_, lean_object* v_cancelTk_x3f_44_, lean_object* v_a_45_, lean_object* v_a_46_, lean_object* v_a_47_, lean_object* v_a_48_){
_start:
{
lean_object* v___x_50_; lean_object* v___f_51_; lean_object* v___x_52_; 
v___x_50_ = lean_st_ref_get(v_a_46_);
lean_inc_ref(v_a_45_);
v___f_51_ = lean_alloc_closure((void*)(lp_batteries_Lean_Meta_wrapAsync___redArg___lam__0___boxed), 7, 3);
lean_closure_set(v___f_51_, 0, v___x_50_);
lean_closure_set(v___f_51_, 1, v_act_43_);
lean_closure_set(v___f_51_, 2, v_a_45_);
v___x_52_ = l_Lean_Core_wrapAsync___redArg(v___f_51_, v_cancelTk_x3f_44_, v_a_47_, v_a_48_);
return v___x_52_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_wrapAsync___redArg___boxed(lean_object* v_act_53_, lean_object* v_cancelTk_x3f_54_, lean_object* v_a_55_, lean_object* v_a_56_, lean_object* v_a_57_, lean_object* v_a_58_, lean_object* v_a_59_){
_start:
{
lean_object* v_res_60_; 
v_res_60_ = lp_batteries_Lean_Meta_wrapAsync___redArg(v_act_53_, v_cancelTk_x3f_54_, v_a_55_, v_a_56_, v_a_57_, v_a_58_);
lean_dec(v_a_58_);
lean_dec_ref(v_a_57_);
lean_dec(v_a_56_);
lean_dec_ref(v_a_55_);
return v_res_60_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_wrapAsync(lean_object* v_00_u03b2_61_, lean_object* v_00_u03b1_62_, lean_object* v_act_63_, lean_object* v_cancelTk_x3f_64_, lean_object* v_a_65_, lean_object* v_a_66_, lean_object* v_a_67_, lean_object* v_a_68_){
_start:
{
lean_object* v___x_70_; 
v___x_70_ = lp_batteries_Lean_Meta_wrapAsync___redArg(v_act_63_, v_cancelTk_x3f_64_, v_a_65_, v_a_66_, v_a_67_, v_a_68_);
return v___x_70_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_wrapAsync___boxed(lean_object* v_00_u03b2_71_, lean_object* v_00_u03b1_72_, lean_object* v_act_73_, lean_object* v_cancelTk_x3f_74_, lean_object* v_a_75_, lean_object* v_a_76_, lean_object* v_a_77_, lean_object* v_a_78_, lean_object* v_a_79_){
_start:
{
lean_object* v_res_80_; 
v_res_80_ = lp_batteries_Lean_Meta_wrapAsync(v_00_u03b2_71_, v_00_u03b1_72_, v_act_73_, v_cancelTk_x3f_74_, v_a_75_, v_a_76_, v_a_77_, v_a_78_);
lean_dec(v_a_78_);
lean_dec_ref(v_a_77_);
lean_dec(v_a_76_);
lean_dec_ref(v_a_75_);
return v_res_80_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Cache_get___redArg___lam__0(lean_object* v_val_81_, lean_object* v_x_82_, lean_object* v___y_83_, lean_object* v___y_84_, lean_object* v___y_85_, lean_object* v___y_86_){
_start:
{
lean_object* v___x_88_; 
lean_inc(v___y_86_);
lean_inc_ref(v___y_85_);
lean_inc(v___y_84_);
lean_inc_ref(v___y_83_);
v___x_88_ = lean_apply_5(v_val_81_, v___y_83_, v___y_84_, v___y_85_, v___y_86_, lean_box(0));
return v___x_88_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Cache_get___redArg___lam__0___boxed(lean_object* v_val_89_, lean_object* v_x_90_, lean_object* v___y_91_, lean_object* v___y_92_, lean_object* v___y_93_, lean_object* v___y_94_, lean_object* v___y_95_){
_start:
{
lean_object* v_res_96_; 
v_res_96_ = lp_batteries_Batteries_Tactic_Cache_get___redArg___lam__0(v_val_89_, v_x_90_, v___y_91_, v___y_92_, v___y_93_, v___y_94_);
lean_dec(v___y_94_);
lean_dec_ref(v___y_93_);
lean_dec(v___y_92_);
lean_dec_ref(v___y_91_);
return v_res_96_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Cache_get___redArg___lam__1(lean_object* v_a_97_, lean_object* v___x_98_){
_start:
{
lean_object* v___x_100_; 
v___x_100_ = lean_apply_2(v_a_97_, v___x_98_, lean_box(0));
if (lean_obj_tag(v___x_100_) == 0)
{
lean_object* v_a_101_; lean_object* v___x_103_; uint8_t v_isShared_104_; uint8_t v_isSharedCheck_108_; 
v_a_101_ = lean_ctor_get(v___x_100_, 0);
v_isSharedCheck_108_ = !lean_is_exclusive(v___x_100_);
if (v_isSharedCheck_108_ == 0)
{
v___x_103_ = v___x_100_;
v_isShared_104_ = v_isSharedCheck_108_;
goto v_resetjp_102_;
}
else
{
lean_inc(v_a_101_);
lean_dec(v___x_100_);
v___x_103_ = lean_box(0);
v_isShared_104_ = v_isSharedCheck_108_;
goto v_resetjp_102_;
}
v_resetjp_102_:
{
lean_object* v___x_106_; 
if (v_isShared_104_ == 0)
{
lean_ctor_set_tag(v___x_103_, 1);
v___x_106_ = v___x_103_;
goto v_reusejp_105_;
}
else
{
lean_object* v_reuseFailAlloc_107_; 
v_reuseFailAlloc_107_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_107_, 0, v_a_101_);
v___x_106_ = v_reuseFailAlloc_107_;
goto v_reusejp_105_;
}
v_reusejp_105_:
{
return v___x_106_;
}
}
}
else
{
lean_object* v_a_109_; lean_object* v___x_111_; uint8_t v_isShared_112_; uint8_t v_isSharedCheck_116_; 
v_a_109_ = lean_ctor_get(v___x_100_, 0);
v_isSharedCheck_116_ = !lean_is_exclusive(v___x_100_);
if (v_isSharedCheck_116_ == 0)
{
v___x_111_ = v___x_100_;
v_isShared_112_ = v_isSharedCheck_116_;
goto v_resetjp_110_;
}
else
{
lean_inc(v_a_109_);
lean_dec(v___x_100_);
v___x_111_ = lean_box(0);
v_isShared_112_ = v_isSharedCheck_116_;
goto v_resetjp_110_;
}
v_resetjp_110_:
{
lean_object* v___x_114_; 
if (v_isShared_112_ == 0)
{
lean_ctor_set_tag(v___x_111_, 0);
v___x_114_ = v___x_111_;
goto v_reusejp_113_;
}
else
{
lean_object* v_reuseFailAlloc_115_; 
v_reuseFailAlloc_115_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_115_, 0, v_a_109_);
v___x_114_ = v_reuseFailAlloc_115_;
goto v_reusejp_113_;
}
v_reusejp_113_:
{
return v___x_114_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Cache_get___redArg___lam__1___boxed(lean_object* v_a_117_, lean_object* v___x_118_, lean_object* v___y_119_){
_start:
{
lean_object* v_res_120_; 
v_res_120_ = lp_batteries_Batteries_Tactic_Cache_get___redArg___lam__1(v_a_117_, v___x_118_);
return v_res_120_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Cache_get___redArg(lean_object* v_cache_121_, lean_object* v_a_122_, lean_object* v_a_123_, lean_object* v_a_124_, lean_object* v_a_125_){
_start:
{
lean_object* v_t_128_; lean_object* v___x_146_; 
v___x_146_ = lean_st_ref_get(v_cache_121_);
if (lean_obj_tag(v___x_146_) == 0)
{
lean_object* v_val_147_; lean_object* v___x_149_; uint8_t v_isShared_150_; uint8_t v_isSharedCheck_171_; 
v_val_147_ = lean_ctor_get(v___x_146_, 0);
v_isSharedCheck_171_ = !lean_is_exclusive(v___x_146_);
if (v_isSharedCheck_171_ == 0)
{
v___x_149_ = v___x_146_;
v_isShared_150_ = v_isSharedCheck_171_;
goto v_resetjp_148_;
}
else
{
lean_inc(v_val_147_);
lean_dec(v___x_146_);
v___x_149_ = lean_box(0);
v_isShared_150_ = v_isSharedCheck_171_;
goto v_resetjp_148_;
}
v_resetjp_148_:
{
lean_object* v___f_151_; lean_object* v___x_152_; lean_object* v___x_153_; 
v___f_151_ = lean_alloc_closure((void*)(lp_batteries_Batteries_Tactic_Cache_get___redArg___lam__0___boxed), 7, 1);
lean_closure_set(v___f_151_, 0, v_val_147_);
v___x_152_ = lean_box(0);
v___x_153_ = lp_batteries_Lean_Meta_wrapAsync___redArg(v___f_151_, v___x_152_, v_a_122_, v_a_123_, v_a_124_, v_a_125_);
if (lean_obj_tag(v___x_153_) == 0)
{
lean_object* v_a_154_; lean_object* v___x_155_; lean_object* v___f_156_; lean_object* v___x_157_; lean_object* v___x_158_; lean_object* v___x_160_; 
v_a_154_ = lean_ctor_get(v___x_153_, 0);
lean_inc(v_a_154_);
lean_dec_ref_known(v___x_153_, 1);
v___x_155_ = lean_box(0);
v___f_156_ = lean_alloc_closure((void*)(lp_batteries_Batteries_Tactic_Cache_get___redArg___lam__1___boxed), 3, 2);
lean_closure_set(v___f_156_, 0, v_a_154_);
lean_closure_set(v___f_156_, 1, v___x_155_);
v___x_157_ = lean_unsigned_to_nat(0u);
v___x_158_ = lean_io_as_task(v___f_156_, v___x_157_);
lean_inc_ref(v___x_158_);
if (v_isShared_150_ == 0)
{
lean_ctor_set_tag(v___x_149_, 1);
lean_ctor_set(v___x_149_, 0, v___x_158_);
v___x_160_ = v___x_149_;
goto v_reusejp_159_;
}
else
{
lean_object* v_reuseFailAlloc_162_; 
v_reuseFailAlloc_162_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_162_, 0, v___x_158_);
v___x_160_ = v_reuseFailAlloc_162_;
goto v_reusejp_159_;
}
v_reusejp_159_:
{
lean_object* v___x_161_; 
v___x_161_ = lean_st_ref_set(v_cache_121_, v___x_160_);
v_t_128_ = v___x_158_;
goto v___jp_127_;
}
}
else
{
lean_object* v_a_163_; lean_object* v___x_165_; uint8_t v_isShared_166_; uint8_t v_isSharedCheck_170_; 
lean_del_object(v___x_149_);
v_a_163_ = lean_ctor_get(v___x_153_, 0);
v_isSharedCheck_170_ = !lean_is_exclusive(v___x_153_);
if (v_isSharedCheck_170_ == 0)
{
v___x_165_ = v___x_153_;
v_isShared_166_ = v_isSharedCheck_170_;
goto v_resetjp_164_;
}
else
{
lean_inc(v_a_163_);
lean_dec(v___x_153_);
v___x_165_ = lean_box(0);
v_isShared_166_ = v_isSharedCheck_170_;
goto v_resetjp_164_;
}
v_resetjp_164_:
{
lean_object* v___x_168_; 
if (v_isShared_166_ == 0)
{
v___x_168_ = v___x_165_;
goto v_reusejp_167_;
}
else
{
lean_object* v_reuseFailAlloc_169_; 
v_reuseFailAlloc_169_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_169_, 0, v_a_163_);
v___x_168_ = v_reuseFailAlloc_169_;
goto v_reusejp_167_;
}
v_reusejp_167_:
{
return v___x_168_;
}
}
}
}
}
else
{
lean_object* v_val_172_; 
v_val_172_ = lean_ctor_get(v___x_146_, 0);
lean_inc(v_val_172_);
lean_dec_ref_known(v___x_146_, 1);
v_t_128_ = v_val_172_;
goto v___jp_127_;
}
v___jp_127_:
{
lean_object* v___x_129_; 
v___x_129_ = lean_task_get_own(v_t_128_);
if (lean_obj_tag(v___x_129_) == 0)
{
lean_object* v_a_130_; lean_object* v___x_132_; uint8_t v_isShared_133_; uint8_t v_isSharedCheck_137_; 
v_a_130_ = lean_ctor_get(v___x_129_, 0);
v_isSharedCheck_137_ = !lean_is_exclusive(v___x_129_);
if (v_isSharedCheck_137_ == 0)
{
v___x_132_ = v___x_129_;
v_isShared_133_ = v_isSharedCheck_137_;
goto v_resetjp_131_;
}
else
{
lean_inc(v_a_130_);
lean_dec(v___x_129_);
v___x_132_ = lean_box(0);
v_isShared_133_ = v_isSharedCheck_137_;
goto v_resetjp_131_;
}
v_resetjp_131_:
{
lean_object* v___x_135_; 
if (v_isShared_133_ == 0)
{
lean_ctor_set_tag(v___x_132_, 1);
v___x_135_ = v___x_132_;
goto v_reusejp_134_;
}
else
{
lean_object* v_reuseFailAlloc_136_; 
v_reuseFailAlloc_136_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_136_, 0, v_a_130_);
v___x_135_ = v_reuseFailAlloc_136_;
goto v_reusejp_134_;
}
v_reusejp_134_:
{
return v___x_135_;
}
}
}
else
{
lean_object* v_a_138_; lean_object* v___x_140_; uint8_t v_isShared_141_; uint8_t v_isSharedCheck_145_; 
v_a_138_ = lean_ctor_get(v___x_129_, 0);
v_isSharedCheck_145_ = !lean_is_exclusive(v___x_129_);
if (v_isSharedCheck_145_ == 0)
{
v___x_140_ = v___x_129_;
v_isShared_141_ = v_isSharedCheck_145_;
goto v_resetjp_139_;
}
else
{
lean_inc(v_a_138_);
lean_dec(v___x_129_);
v___x_140_ = lean_box(0);
v_isShared_141_ = v_isSharedCheck_145_;
goto v_resetjp_139_;
}
v_resetjp_139_:
{
lean_object* v___x_143_; 
if (v_isShared_141_ == 0)
{
lean_ctor_set_tag(v___x_140_, 0);
v___x_143_ = v___x_140_;
goto v_reusejp_142_;
}
else
{
lean_object* v_reuseFailAlloc_144_; 
v_reuseFailAlloc_144_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_144_, 0, v_a_138_);
v___x_143_ = v_reuseFailAlloc_144_;
goto v_reusejp_142_;
}
v_reusejp_142_:
{
return v___x_143_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Cache_get___redArg___boxed(lean_object* v_cache_173_, lean_object* v_a_174_, lean_object* v_a_175_, lean_object* v_a_176_, lean_object* v_a_177_, lean_object* v_a_178_){
_start:
{
lean_object* v_res_179_; 
v_res_179_ = lp_batteries_Batteries_Tactic_Cache_get___redArg(v_cache_173_, v_a_174_, v_a_175_, v_a_176_, v_a_177_);
lean_dec(v_a_177_);
lean_dec_ref(v_a_176_);
lean_dec(v_a_175_);
lean_dec_ref(v_a_174_);
lean_dec(v_cache_173_);
return v_res_179_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Cache_get(lean_object* v_00_u03b1_180_, lean_object* v_cache_181_, lean_object* v_a_182_, lean_object* v_a_183_, lean_object* v_a_184_, lean_object* v_a_185_){
_start:
{
lean_object* v___x_187_; 
v___x_187_ = lp_batteries_Batteries_Tactic_Cache_get___redArg(v_cache_181_, v_a_182_, v_a_183_, v_a_184_, v_a_185_);
return v___x_187_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_Cache_get___boxed(lean_object* v_00_u03b1_188_, lean_object* v_cache_189_, lean_object* v_a_190_, lean_object* v_a_191_, lean_object* v_a_192_, lean_object* v_a_193_, lean_object* v_a_194_){
_start:
{
lean_object* v_res_195_; 
v_res_195_ = lp_batteries_Batteries_Tactic_Cache_get(v_00_u03b1_188_, v_cache_189_, v_a_190_, v_a_191_, v_a_192_, v_a_193_);
lean_dec(v_a_193_);
lean_dec_ref(v_a_192_);
lean_dec(v_a_191_);
lean_dec_ref(v_a_190_);
lean_dec(v_cache_189_);
return v_res_195_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_profileitM___at___00Batteries_Tactic_DeclCache_mk_spec__2___redArg(lean_object* v_category_196_, lean_object* v_opts_197_, lean_object* v_act_198_, lean_object* v_decl_199_, lean_object* v___y_200_, lean_object* v___y_201_, lean_object* v___y_202_, lean_object* v___y_203_){
_start:
{
lean_object* v___x_205_; lean_object* v___x_206_; 
lean_inc(v___y_203_);
lean_inc_ref(v___y_202_);
lean_inc(v___y_201_);
lean_inc_ref(v___y_200_);
v___x_205_ = lean_apply_4(v_act_198_, v___y_200_, v___y_201_, v___y_202_, v___y_203_);
v___x_206_ = l_Lean_profileitIOUnsafe___redArg(v_category_196_, v_opts_197_, v___x_205_, v_decl_199_);
return v___x_206_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_profileitM___at___00Batteries_Tactic_DeclCache_mk_spec__2___redArg___boxed(lean_object* v_category_207_, lean_object* v_opts_208_, lean_object* v_act_209_, lean_object* v_decl_210_, lean_object* v___y_211_, lean_object* v___y_212_, lean_object* v___y_213_, lean_object* v___y_214_, lean_object* v___y_215_){
_start:
{
lean_object* v_res_216_; 
v_res_216_ = lp_batteries_Lean_profileitM___at___00Batteries_Tactic_DeclCache_mk_spec__2___redArg(v_category_207_, v_opts_208_, v_act_209_, v_decl_210_, v___y_211_, v___y_212_, v___y_213_, v___y_214_);
lean_dec(v___y_214_);
lean_dec_ref(v___y_213_);
lean_dec(v___y_212_);
lean_dec_ref(v___y_211_);
lean_dec_ref(v_opts_208_);
lean_dec_ref(v_category_207_);
return v_res_216_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_profileitM___at___00Batteries_Tactic_DeclCache_mk_spec__2(lean_object* v_00_u03b1_217_, lean_object* v_category_218_, lean_object* v_opts_219_, lean_object* v_act_220_, lean_object* v_decl_221_, lean_object* v___y_222_, lean_object* v___y_223_, lean_object* v___y_224_, lean_object* v___y_225_){
_start:
{
lean_object* v___x_227_; 
v___x_227_ = lp_batteries_Lean_profileitM___at___00Batteries_Tactic_DeclCache_mk_spec__2___redArg(v_category_218_, v_opts_219_, v_act_220_, v_decl_221_, v___y_222_, v___y_223_, v___y_224_, v___y_225_);
return v___x_227_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_profileitM___at___00Batteries_Tactic_DeclCache_mk_spec__2___boxed(lean_object* v_00_u03b1_228_, lean_object* v_category_229_, lean_object* v_opts_230_, lean_object* v_act_231_, lean_object* v_decl_232_, lean_object* v___y_233_, lean_object* v___y_234_, lean_object* v___y_235_, lean_object* v___y_236_, lean_object* v___y_237_){
_start:
{
lean_object* v_res_238_; 
v_res_238_ = lp_batteries_Lean_profileitM___at___00Batteries_Tactic_DeclCache_mk_spec__2(v_00_u03b1_228_, v_category_229_, v_opts_230_, v_act_231_, v_decl_232_, v___y_233_, v___y_234_, v___y_235_, v___y_236_);
lean_dec(v___y_236_);
lean_dec_ref(v___y_235_);
lean_dec(v___y_234_);
lean_dec_ref(v___y_233_);
lean_dec_ref(v_opts_230_);
lean_dec_ref(v_category_229_);
return v_res_238_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_DeclCache_mk_spec__0___redArg(lean_object* v_addLibraryDecl_239_, lean_object* v_x_240_, lean_object* v_x_241_, lean_object* v___y_242_, lean_object* v___y_243_, lean_object* v___y_244_, lean_object* v___y_245_){
_start:
{
if (lean_obj_tag(v_x_241_) == 0)
{
lean_object* v___x_247_; 
lean_dec_ref(v_addLibraryDecl_239_);
v___x_247_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_247_, 0, v_x_240_);
return v___x_247_;
}
else
{
lean_object* v_key_248_; lean_object* v_value_249_; lean_object* v_tail_250_; lean_object* v___x_251_; 
v_key_248_ = lean_ctor_get(v_x_241_, 0);
lean_inc(v_key_248_);
v_value_249_ = lean_ctor_get(v_x_241_, 1);
lean_inc(v_value_249_);
v_tail_250_ = lean_ctor_get(v_x_241_, 2);
lean_inc(v_tail_250_);
lean_dec_ref_known(v_x_241_, 3);
lean_inc_ref(v_addLibraryDecl_239_);
lean_inc(v___y_245_);
lean_inc_ref(v___y_244_);
lean_inc(v___y_243_);
lean_inc_ref(v___y_242_);
v___x_251_ = lean_apply_8(v_addLibraryDecl_239_, v_key_248_, v_value_249_, v_x_240_, v___y_242_, v___y_243_, v___y_244_, v___y_245_, lean_box(0));
if (lean_obj_tag(v___x_251_) == 0)
{
lean_object* v_a_252_; 
v_a_252_ = lean_ctor_get(v___x_251_, 0);
lean_inc(v_a_252_);
lean_dec_ref_known(v___x_251_, 1);
v_x_240_ = v_a_252_;
v_x_241_ = v_tail_250_;
goto _start;
}
else
{
lean_dec(v_tail_250_);
lean_dec_ref(v_addLibraryDecl_239_);
return v___x_251_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_DeclCache_mk_spec__0___redArg___boxed(lean_object* v_addLibraryDecl_254_, lean_object* v_x_255_, lean_object* v_x_256_, lean_object* v___y_257_, lean_object* v___y_258_, lean_object* v___y_259_, lean_object* v___y_260_, lean_object* v___y_261_){
_start:
{
lean_object* v_res_262_; 
v_res_262_ = lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_DeclCache_mk_spec__0___redArg(v_addLibraryDecl_254_, v_x_255_, v_x_256_, v___y_257_, v___y_258_, v___y_259_, v___y_260_);
lean_dec(v___y_260_);
lean_dec_ref(v___y_259_);
lean_dec(v___y_258_);
lean_dec_ref(v___y_257_);
return v_res_262_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_DeclCache_mk_spec__1___redArg(lean_object* v_addLibraryDecl_263_, lean_object* v_as_264_, size_t v_i_265_, size_t v_stop_266_, lean_object* v_b_267_, lean_object* v___y_268_, lean_object* v___y_269_, lean_object* v___y_270_, lean_object* v___y_271_){
_start:
{
uint8_t v___x_273_; 
v___x_273_ = lean_usize_dec_eq(v_i_265_, v_stop_266_);
if (v___x_273_ == 0)
{
lean_object* v___x_274_; lean_object* v___x_275_; 
v___x_274_ = lean_array_uget_borrowed(v_as_264_, v_i_265_);
lean_inc(v___x_274_);
lean_inc_ref(v_addLibraryDecl_263_);
v___x_275_ = lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_DeclCache_mk_spec__0___redArg(v_addLibraryDecl_263_, v_b_267_, v___x_274_, v___y_268_, v___y_269_, v___y_270_, v___y_271_);
if (lean_obj_tag(v___x_275_) == 0)
{
lean_object* v_a_276_; size_t v___x_277_; size_t v___x_278_; 
v_a_276_ = lean_ctor_get(v___x_275_, 0);
lean_inc(v_a_276_);
lean_dec_ref_known(v___x_275_, 1);
v___x_277_ = ((size_t)1ULL);
v___x_278_ = lean_usize_add(v_i_265_, v___x_277_);
v_i_265_ = v___x_278_;
v_b_267_ = v_a_276_;
goto _start;
}
else
{
lean_dec_ref(v_addLibraryDecl_263_);
return v___x_275_;
}
}
else
{
lean_object* v___x_280_; 
lean_dec_ref(v_addLibraryDecl_263_);
v___x_280_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_280_, 0, v_b_267_);
return v___x_280_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_DeclCache_mk_spec__1___redArg___boxed(lean_object* v_addLibraryDecl_281_, lean_object* v_as_282_, lean_object* v_i_283_, lean_object* v_stop_284_, lean_object* v_b_285_, lean_object* v___y_286_, lean_object* v___y_287_, lean_object* v___y_288_, lean_object* v___y_289_, lean_object* v___y_290_){
_start:
{
size_t v_i_boxed_291_; size_t v_stop_boxed_292_; lean_object* v_res_293_; 
v_i_boxed_291_ = lean_unbox_usize(v_i_283_);
lean_dec(v_i_283_);
v_stop_boxed_292_ = lean_unbox_usize(v_stop_284_);
lean_dec(v_stop_284_);
v_res_293_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_DeclCache_mk_spec__1___redArg(v_addLibraryDecl_281_, v_as_282_, v_i_boxed_291_, v_stop_boxed_292_, v_b_285_, v___y_286_, v___y_287_, v___y_288_, v___y_289_);
lean_dec(v___y_289_);
lean_dec_ref(v___y_288_);
lean_dec(v___y_287_);
lean_dec_ref(v___y_286_);
lean_dec_ref(v_as_282_);
return v_res_293_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_DeclCache_mk___redArg___lam__0(lean_object* v_post_294_, lean_object* v_empty_295_, lean_object* v_addLibraryDecl_296_, lean_object* v___y_297_, lean_object* v___y_298_, lean_object* v___y_299_, lean_object* v___y_300_){
_start:
{
lean_object* v___y_303_; lean_object* v___x_306_; lean_object* v_env_307_; lean_object* v___x_308_; lean_object* v_map_u2081_309_; lean_object* v_buckets_310_; lean_object* v___x_311_; lean_object* v___x_312_; uint8_t v___x_313_; 
v___x_306_ = lean_st_ref_get(v___y_300_);
v_env_307_ = lean_ctor_get(v___x_306_, 0);
lean_inc_ref(v_env_307_);
lean_dec(v___x_306_);
v___x_308_ = l_Lean_Environment_constants(v_env_307_);
v_map_u2081_309_ = lean_ctor_get(v___x_308_, 0);
lean_inc_ref(v_map_u2081_309_);
lean_dec_ref(v___x_308_);
v_buckets_310_ = lean_ctor_get(v_map_u2081_309_, 1);
lean_inc_ref(v_buckets_310_);
lean_dec_ref(v_map_u2081_309_);
v___x_311_ = lean_unsigned_to_nat(0u);
v___x_312_ = lean_array_get_size(v_buckets_310_);
v___x_313_ = lean_nat_dec_lt(v___x_311_, v___x_312_);
if (v___x_313_ == 0)
{
lean_object* v___x_314_; 
lean_dec_ref(v_buckets_310_);
lean_dec_ref(v_addLibraryDecl_296_);
v___x_314_ = lean_apply_6(v_post_294_, v_empty_295_, v___y_297_, v___y_298_, v___y_299_, v___y_300_, lean_box(0));
return v___x_314_;
}
else
{
uint8_t v___x_315_; 
v___x_315_ = lean_nat_dec_le(v___x_312_, v___x_312_);
if (v___x_315_ == 0)
{
if (v___x_313_ == 0)
{
lean_object* v___x_316_; 
lean_dec_ref(v_buckets_310_);
lean_dec_ref(v_addLibraryDecl_296_);
v___x_316_ = lean_apply_6(v_post_294_, v_empty_295_, v___y_297_, v___y_298_, v___y_299_, v___y_300_, lean_box(0));
return v___x_316_;
}
else
{
size_t v___x_317_; size_t v___x_318_; lean_object* v___x_319_; 
v___x_317_ = ((size_t)0ULL);
v___x_318_ = lean_usize_of_nat(v___x_312_);
v___x_319_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_DeclCache_mk_spec__1___redArg(v_addLibraryDecl_296_, v_buckets_310_, v___x_317_, v___x_318_, v_empty_295_, v___y_297_, v___y_298_, v___y_299_, v___y_300_);
lean_dec_ref(v_buckets_310_);
v___y_303_ = v___x_319_;
goto v___jp_302_;
}
}
else
{
size_t v___x_320_; size_t v___x_321_; lean_object* v___x_322_; 
v___x_320_ = ((size_t)0ULL);
v___x_321_ = lean_usize_of_nat(v___x_312_);
v___x_322_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_DeclCache_mk_spec__1___redArg(v_addLibraryDecl_296_, v_buckets_310_, v___x_320_, v___x_321_, v_empty_295_, v___y_297_, v___y_298_, v___y_299_, v___y_300_);
lean_dec_ref(v_buckets_310_);
v___y_303_ = v___x_322_;
goto v___jp_302_;
}
}
v___jp_302_:
{
if (lean_obj_tag(v___y_303_) == 0)
{
lean_object* v_a_304_; lean_object* v___x_305_; 
v_a_304_ = lean_ctor_get(v___y_303_, 0);
lean_inc(v_a_304_);
lean_dec_ref_known(v___y_303_, 1);
v___x_305_ = lean_apply_6(v_post_294_, v_a_304_, v___y_297_, v___y_298_, v___y_299_, v___y_300_, lean_box(0));
return v___x_305_;
}
else
{
lean_dec(v___y_300_);
lean_dec_ref(v___y_299_);
lean_dec(v___y_298_);
lean_dec_ref(v___y_297_);
lean_dec_ref(v_post_294_);
return v___y_303_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_DeclCache_mk___redArg___lam__0___boxed(lean_object* v_post_323_, lean_object* v_empty_324_, lean_object* v_addLibraryDecl_325_, lean_object* v___y_326_, lean_object* v___y_327_, lean_object* v___y_328_, lean_object* v___y_329_, lean_object* v___y_330_){
_start:
{
lean_object* v_res_331_; 
v_res_331_ = lp_batteries_Batteries_Tactic_DeclCache_mk___redArg___lam__0(v_post_323_, v_empty_324_, v_addLibraryDecl_325_, v___y_326_, v___y_327_, v___y_328_, v___y_329_);
return v_res_331_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_DeclCache_mk___redArg___lam__1(lean_object* v_pre_332_, lean_object* v_profilingName_333_, lean_object* v___f_334_, lean_object* v___y_335_, lean_object* v___y_336_, lean_object* v___y_337_, lean_object* v___y_338_){
_start:
{
lean_object* v___x_340_; 
lean_inc(v___y_338_);
lean_inc_ref(v___y_337_);
lean_inc(v___y_336_);
lean_inc_ref(v___y_335_);
v___x_340_ = lean_apply_5(v_pre_332_, v___y_335_, v___y_336_, v___y_337_, v___y_338_, lean_box(0));
if (lean_obj_tag(v___x_340_) == 0)
{
lean_dec_ref(v___y_335_);
lean_dec_ref(v___f_334_);
return v___x_340_;
}
else
{
lean_object* v_a_341_; uint8_t v___y_343_; uint8_t v___x_347_; 
v_a_341_ = lean_ctor_get(v___x_340_, 0);
lean_inc(v_a_341_);
v___x_347_ = l_Lean_Exception_isInterrupt(v_a_341_);
if (v___x_347_ == 0)
{
uint8_t v___x_348_; 
v___x_348_ = l_Lean_Exception_isRuntime(v_a_341_);
v___y_343_ = v___x_348_;
goto v___jp_342_;
}
else
{
lean_dec(v_a_341_);
v___y_343_ = v___x_347_;
goto v___jp_342_;
}
v___jp_342_:
{
if (v___y_343_ == 0)
{
lean_object* v_options_344_; lean_object* v___x_345_; lean_object* v___x_346_; 
lean_dec_ref_known(v___x_340_, 1);
v_options_344_ = lean_ctor_get(v___y_337_, 2);
v___x_345_ = lean_box(0);
v___x_346_ = lp_batteries_Lean_profileitM___at___00Batteries_Tactic_DeclCache_mk_spec__2___redArg(v_profilingName_333_, v_options_344_, v___f_334_, v___x_345_, v___y_335_, v___y_336_, v___y_337_, v___y_338_);
lean_dec_ref(v___y_335_);
return v___x_346_;
}
else
{
lean_dec_ref(v___y_335_);
lean_dec_ref(v___f_334_);
return v___x_340_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_DeclCache_mk___redArg___lam__1___boxed(lean_object* v_pre_349_, lean_object* v_profilingName_350_, lean_object* v___f_351_, lean_object* v___y_352_, lean_object* v___y_353_, lean_object* v___y_354_, lean_object* v___y_355_, lean_object* v___y_356_){
_start:
{
lean_object* v_res_357_; 
v_res_357_ = lp_batteries_Batteries_Tactic_DeclCache_mk___redArg___lam__1(v_pre_349_, v_profilingName_350_, v___f_351_, v___y_352_, v___y_353_, v___y_354_, v___y_355_);
lean_dec(v___y_355_);
lean_dec_ref(v___y_354_);
lean_dec(v___y_353_);
lean_dec_ref(v_profilingName_350_);
return v_res_357_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_DeclCache_mk___redArg(lean_object* v_profilingName_358_, lean_object* v_pre_359_, lean_object* v_empty_360_, lean_object* v_addDecl_361_, lean_object* v_addLibraryDecl_362_, lean_object* v_post_363_){
_start:
{
lean_object* v___f_365_; lean_object* v___f_366_; lean_object* v___x_367_; lean_object* v_a_368_; lean_object* v___x_370_; uint8_t v_isShared_371_; uint8_t v_isSharedCheck_376_; 
v___f_365_ = lean_alloc_closure((void*)(lp_batteries_Batteries_Tactic_DeclCache_mk___redArg___lam__0___boxed), 8, 3);
lean_closure_set(v___f_365_, 0, v_post_363_);
lean_closure_set(v___f_365_, 1, v_empty_360_);
lean_closure_set(v___f_365_, 2, v_addLibraryDecl_362_);
v___f_366_ = lean_alloc_closure((void*)(lp_batteries_Batteries_Tactic_DeclCache_mk___redArg___lam__1___boxed), 8, 3);
lean_closure_set(v___f_366_, 0, v_pre_359_);
lean_closure_set(v___f_366_, 1, v_profilingName_358_);
lean_closure_set(v___f_366_, 2, v___f_365_);
v___x_367_ = lp_batteries_Batteries_Tactic_Cache_mk___redArg(v___f_366_);
v_a_368_ = lean_ctor_get(v___x_367_, 0);
v_isSharedCheck_376_ = !lean_is_exclusive(v___x_367_);
if (v_isSharedCheck_376_ == 0)
{
v___x_370_ = v___x_367_;
v_isShared_371_ = v_isSharedCheck_376_;
goto v_resetjp_369_;
}
else
{
lean_inc(v_a_368_);
lean_dec(v___x_367_);
v___x_370_ = lean_box(0);
v_isShared_371_ = v_isSharedCheck_376_;
goto v_resetjp_369_;
}
v_resetjp_369_:
{
lean_object* v___x_372_; lean_object* v___x_374_; 
lean_inc_ref(v_addDecl_361_);
v___x_372_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_372_, 0, v_a_368_);
lean_ctor_set(v___x_372_, 1, v_addDecl_361_);
lean_ctor_set(v___x_372_, 2, v_addDecl_361_);
if (v_isShared_371_ == 0)
{
lean_ctor_set(v___x_370_, 0, v___x_372_);
v___x_374_ = v___x_370_;
goto v_reusejp_373_;
}
else
{
lean_object* v_reuseFailAlloc_375_; 
v_reuseFailAlloc_375_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_375_, 0, v___x_372_);
v___x_374_ = v_reuseFailAlloc_375_;
goto v_reusejp_373_;
}
v_reusejp_373_:
{
return v___x_374_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_DeclCache_mk___redArg___boxed(lean_object* v_profilingName_377_, lean_object* v_pre_378_, lean_object* v_empty_379_, lean_object* v_addDecl_380_, lean_object* v_addLibraryDecl_381_, lean_object* v_post_382_, lean_object* v_a_383_){
_start:
{
lean_object* v_res_384_; 
v_res_384_ = lp_batteries_Batteries_Tactic_DeclCache_mk___redArg(v_profilingName_377_, v_pre_378_, v_empty_379_, v_addDecl_380_, v_addLibraryDecl_381_, v_post_382_);
return v_res_384_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_DeclCache_mk(lean_object* v_00_u03b1_385_, lean_object* v_profilingName_386_, lean_object* v_pre_387_, lean_object* v_empty_388_, lean_object* v_addDecl_389_, lean_object* v_addLibraryDecl_390_, lean_object* v_post_391_){
_start:
{
lean_object* v___x_393_; 
v___x_393_ = lp_batteries_Batteries_Tactic_DeclCache_mk___redArg(v_profilingName_386_, v_pre_387_, v_empty_388_, v_addDecl_389_, v_addLibraryDecl_390_, v_post_391_);
return v___x_393_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_DeclCache_mk___boxed(lean_object* v_00_u03b1_394_, lean_object* v_profilingName_395_, lean_object* v_pre_396_, lean_object* v_empty_397_, lean_object* v_addDecl_398_, lean_object* v_addLibraryDecl_399_, lean_object* v_post_400_, lean_object* v_a_401_){
_start:
{
lean_object* v_res_402_; 
v_res_402_ = lp_batteries_Batteries_Tactic_DeclCache_mk(v_00_u03b1_394_, v_profilingName_395_, v_pre_396_, v_empty_397_, v_addDecl_398_, v_addLibraryDecl_399_, v_post_400_);
return v_res_402_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_DeclCache_mk_spec__0(lean_object* v_00_u03b1_403_, lean_object* v_addLibraryDecl_404_, lean_object* v_x_405_, lean_object* v_x_406_, lean_object* v___y_407_, lean_object* v___y_408_, lean_object* v___y_409_, lean_object* v___y_410_){
_start:
{
lean_object* v___x_412_; 
v___x_412_ = lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_DeclCache_mk_spec__0___redArg(v_addLibraryDecl_404_, v_x_405_, v_x_406_, v___y_407_, v___y_408_, v___y_409_, v___y_410_);
return v___x_412_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_DeclCache_mk_spec__0___boxed(lean_object* v_00_u03b1_413_, lean_object* v_addLibraryDecl_414_, lean_object* v_x_415_, lean_object* v_x_416_, lean_object* v___y_417_, lean_object* v___y_418_, lean_object* v___y_419_, lean_object* v___y_420_, lean_object* v___y_421_){
_start:
{
lean_object* v_res_422_; 
v_res_422_ = lp_batteries_Std_DHashMap_Internal_AssocList_foldlM___at___00Batteries_Tactic_DeclCache_mk_spec__0(v_00_u03b1_413_, v_addLibraryDecl_414_, v_x_415_, v_x_416_, v___y_417_, v___y_418_, v___y_419_, v___y_420_);
lean_dec(v___y_420_);
lean_dec_ref(v___y_419_);
lean_dec(v___y_418_);
lean_dec_ref(v___y_417_);
return v_res_422_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_DeclCache_mk_spec__1(lean_object* v_00_u03b1_423_, lean_object* v_addLibraryDecl_424_, lean_object* v_as_425_, size_t v_i_426_, size_t v_stop_427_, lean_object* v_b_428_, lean_object* v___y_429_, lean_object* v___y_430_, lean_object* v___y_431_, lean_object* v___y_432_){
_start:
{
lean_object* v___x_434_; 
v___x_434_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_DeclCache_mk_spec__1___redArg(v_addLibraryDecl_424_, v_as_425_, v_i_426_, v_stop_427_, v_b_428_, v___y_429_, v___y_430_, v___y_431_, v___y_432_);
return v___x_434_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_DeclCache_mk_spec__1___boxed(lean_object* v_00_u03b1_435_, lean_object* v_addLibraryDecl_436_, lean_object* v_as_437_, lean_object* v_i_438_, lean_object* v_stop_439_, lean_object* v_b_440_, lean_object* v___y_441_, lean_object* v___y_442_, lean_object* v___y_443_, lean_object* v___y_444_, lean_object* v___y_445_){
_start:
{
size_t v_i_boxed_446_; size_t v_stop_boxed_447_; lean_object* v_res_448_; 
v_i_boxed_446_ = lean_unbox_usize(v_i_438_);
lean_dec(v_i_438_);
v_stop_boxed_447_ = lean_unbox_usize(v_stop_439_);
lean_dec(v_stop_439_);
v_res_448_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Batteries_Tactic_DeclCache_mk_spec__1(v_00_u03b1_435_, v_addLibraryDecl_436_, v_as_437_, v_i_boxed_446_, v_stop_boxed_447_, v_b_440_, v___y_441_, v___y_442_, v___y_443_, v___y_444_);
lean_dec(v___y_444_);
lean_dec_ref(v___y_443_);
lean_dec(v___y_442_);
lean_dec_ref(v___y_441_);
lean_dec_ref(v_as_437_);
return v_res_448_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_DeclCache_get___redArg___lam__0(lean_object* v_addDecl_449_, lean_object* v_a_450_, lean_object* v_n_451_, lean_object* v_c_452_, lean_object* v___y_453_, lean_object* v___y_454_, lean_object* v___y_455_, lean_object* v___y_456_){
_start:
{
lean_object* v___x_458_; 
lean_inc(v___y_456_);
lean_inc_ref(v___y_455_);
lean_inc(v___y_454_);
lean_inc_ref(v___y_453_);
v___x_458_ = lean_apply_8(v_addDecl_449_, v_n_451_, v_c_452_, v_a_450_, v___y_453_, v___y_454_, v___y_455_, v___y_456_, lean_box(0));
return v___x_458_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_DeclCache_get___redArg___lam__0___boxed(lean_object* v_addDecl_459_, lean_object* v_a_460_, lean_object* v_n_461_, lean_object* v_c_462_, lean_object* v___y_463_, lean_object* v___y_464_, lean_object* v___y_465_, lean_object* v___y_466_, lean_object* v___y_467_){
_start:
{
lean_object* v_res_468_; 
v_res_468_ = lp_batteries_Batteries_Tactic_DeclCache_get___redArg___lam__0(v_addDecl_459_, v_a_460_, v_n_461_, v_c_462_, v___y_463_, v___y_464_, v___y_465_, v___y_466_);
lean_dec(v___y_466_);
lean_dec_ref(v___y_465_);
lean_dec(v___y_464_);
lean_dec_ref(v___y_463_);
return v_res_468_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Batteries_Tactic_DeclCache_get_spec__0_spec__0_spec__2___redArg(lean_object* v_f_469_, lean_object* v_keys_470_, lean_object* v_vals_471_, lean_object* v_i_472_, lean_object* v_acc_473_, lean_object* v___y_474_, lean_object* v___y_475_, lean_object* v___y_476_, lean_object* v___y_477_){
_start:
{
lean_object* v___x_479_; uint8_t v___x_480_; 
v___x_479_ = lean_array_get_size(v_keys_470_);
v___x_480_ = lean_nat_dec_lt(v_i_472_, v___x_479_);
if (v___x_480_ == 0)
{
lean_object* v___x_481_; 
lean_dec(v_i_472_);
lean_dec_ref(v_f_469_);
v___x_481_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_481_, 0, v_acc_473_);
return v___x_481_;
}
else
{
lean_object* v_k_482_; lean_object* v_v_483_; lean_object* v___x_484_; 
v_k_482_ = lean_array_fget_borrowed(v_keys_470_, v_i_472_);
v_v_483_ = lean_array_fget_borrowed(v_vals_471_, v_i_472_);
lean_inc_ref(v_f_469_);
lean_inc(v___y_477_);
lean_inc_ref(v___y_476_);
lean_inc(v___y_475_);
lean_inc_ref(v___y_474_);
lean_inc(v_v_483_);
lean_inc(v_k_482_);
v___x_484_ = lean_apply_8(v_f_469_, v_acc_473_, v_k_482_, v_v_483_, v___y_474_, v___y_475_, v___y_476_, v___y_477_, lean_box(0));
if (lean_obj_tag(v___x_484_) == 0)
{
lean_object* v_a_485_; lean_object* v___x_486_; lean_object* v___x_487_; 
v_a_485_ = lean_ctor_get(v___x_484_, 0);
lean_inc(v_a_485_);
lean_dec_ref_known(v___x_484_, 1);
v___x_486_ = lean_unsigned_to_nat(1u);
v___x_487_ = lean_nat_add(v_i_472_, v___x_486_);
lean_dec(v_i_472_);
v_i_472_ = v___x_487_;
v_acc_473_ = v_a_485_;
goto _start;
}
else
{
lean_dec(v_i_472_);
lean_dec_ref(v_f_469_);
return v___x_484_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Batteries_Tactic_DeclCache_get_spec__0_spec__0_spec__2___redArg___boxed(lean_object* v_f_489_, lean_object* v_keys_490_, lean_object* v_vals_491_, lean_object* v_i_492_, lean_object* v_acc_493_, lean_object* v___y_494_, lean_object* v___y_495_, lean_object* v___y_496_, lean_object* v___y_497_, lean_object* v___y_498_){
_start:
{
lean_object* v_res_499_; 
v_res_499_ = lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Batteries_Tactic_DeclCache_get_spec__0_spec__0_spec__2___redArg(v_f_489_, v_keys_490_, v_vals_491_, v_i_492_, v_acc_493_, v___y_494_, v___y_495_, v___y_496_, v___y_497_);
lean_dec(v___y_497_);
lean_dec_ref(v___y_496_);
lean_dec(v___y_495_);
lean_dec_ref(v___y_494_);
lean_dec_ref(v_vals_491_);
lean_dec_ref(v_keys_490_);
return v_res_499_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Batteries_Tactic_DeclCache_get_spec__0_spec__0___redArg(lean_object* v_f_500_, lean_object* v_x_501_, lean_object* v_x_502_, lean_object* v___y_503_, lean_object* v___y_504_, lean_object* v___y_505_, lean_object* v___y_506_){
_start:
{
if (lean_obj_tag(v_x_501_) == 0)
{
lean_object* v_es_508_; lean_object* v___x_510_; uint8_t v_isShared_511_; uint8_t v_isSharedCheck_528_; 
v_es_508_ = lean_ctor_get(v_x_501_, 0);
v_isSharedCheck_528_ = !lean_is_exclusive(v_x_501_);
if (v_isSharedCheck_528_ == 0)
{
v___x_510_ = v_x_501_;
v_isShared_511_ = v_isSharedCheck_528_;
goto v_resetjp_509_;
}
else
{
lean_inc(v_es_508_);
lean_dec(v_x_501_);
v___x_510_ = lean_box(0);
v_isShared_511_ = v_isSharedCheck_528_;
goto v_resetjp_509_;
}
v_resetjp_509_:
{
lean_object* v___x_512_; lean_object* v___x_513_; uint8_t v___x_514_; 
v___x_512_ = lean_unsigned_to_nat(0u);
v___x_513_ = lean_array_get_size(v_es_508_);
v___x_514_ = lean_nat_dec_lt(v___x_512_, v___x_513_);
if (v___x_514_ == 0)
{
lean_object* v___x_516_; 
lean_dec_ref(v_es_508_);
lean_dec_ref(v_f_500_);
if (v_isShared_511_ == 0)
{
lean_ctor_set(v___x_510_, 0, v_x_502_);
v___x_516_ = v___x_510_;
goto v_reusejp_515_;
}
else
{
lean_object* v_reuseFailAlloc_517_; 
v_reuseFailAlloc_517_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_517_, 0, v_x_502_);
v___x_516_ = v_reuseFailAlloc_517_;
goto v_reusejp_515_;
}
v_reusejp_515_:
{
return v___x_516_;
}
}
else
{
uint8_t v___x_518_; 
v___x_518_ = lean_nat_dec_le(v___x_513_, v___x_513_);
if (v___x_518_ == 0)
{
if (v___x_514_ == 0)
{
lean_object* v___x_520_; 
lean_dec_ref(v_es_508_);
lean_dec_ref(v_f_500_);
if (v_isShared_511_ == 0)
{
lean_ctor_set(v___x_510_, 0, v_x_502_);
v___x_520_ = v___x_510_;
goto v_reusejp_519_;
}
else
{
lean_object* v_reuseFailAlloc_521_; 
v_reuseFailAlloc_521_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_521_, 0, v_x_502_);
v___x_520_ = v_reuseFailAlloc_521_;
goto v_reusejp_519_;
}
v_reusejp_519_:
{
return v___x_520_;
}
}
else
{
size_t v___x_522_; size_t v___x_523_; lean_object* v___x_524_; 
lean_del_object(v___x_510_);
v___x_522_ = ((size_t)0ULL);
v___x_523_ = lean_usize_of_nat(v___x_513_);
v___x_524_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Batteries_Tactic_DeclCache_get_spec__0_spec__0_spec__1___redArg(v_f_500_, v_es_508_, v___x_522_, v___x_523_, v_x_502_, v___y_503_, v___y_504_, v___y_505_, v___y_506_);
lean_dec_ref(v_es_508_);
return v___x_524_;
}
}
else
{
size_t v___x_525_; size_t v___x_526_; lean_object* v___x_527_; 
lean_del_object(v___x_510_);
v___x_525_ = ((size_t)0ULL);
v___x_526_ = lean_usize_of_nat(v___x_513_);
v___x_527_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Batteries_Tactic_DeclCache_get_spec__0_spec__0_spec__1___redArg(v_f_500_, v_es_508_, v___x_525_, v___x_526_, v_x_502_, v___y_503_, v___y_504_, v___y_505_, v___y_506_);
lean_dec_ref(v_es_508_);
return v___x_527_;
}
}
}
}
else
{
lean_object* v_ks_529_; lean_object* v_vs_530_; lean_object* v___x_531_; lean_object* v___x_532_; 
v_ks_529_ = lean_ctor_get(v_x_501_, 0);
lean_inc_ref(v_ks_529_);
v_vs_530_ = lean_ctor_get(v_x_501_, 1);
lean_inc_ref(v_vs_530_);
lean_dec_ref_known(v_x_501_, 2);
v___x_531_ = lean_unsigned_to_nat(0u);
v___x_532_ = lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Batteries_Tactic_DeclCache_get_spec__0_spec__0_spec__2___redArg(v_f_500_, v_ks_529_, v_vs_530_, v___x_531_, v_x_502_, v___y_503_, v___y_504_, v___y_505_, v___y_506_);
lean_dec_ref(v_vs_530_);
lean_dec_ref(v_ks_529_);
return v___x_532_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Batteries_Tactic_DeclCache_get_spec__0_spec__0_spec__1___redArg(lean_object* v_f_533_, lean_object* v_as_534_, size_t v_i_535_, size_t v_stop_536_, lean_object* v_b_537_, lean_object* v___y_538_, lean_object* v___y_539_, lean_object* v___y_540_, lean_object* v___y_541_){
_start:
{
lean_object* v_a_544_; lean_object* v___y_549_; uint8_t v___x_551_; 
v___x_551_ = lean_usize_dec_eq(v_i_535_, v_stop_536_);
if (v___x_551_ == 0)
{
lean_object* v___x_552_; 
v___x_552_ = lean_array_uget_borrowed(v_as_534_, v_i_535_);
switch(lean_obj_tag(v___x_552_))
{
case 0:
{
lean_object* v_key_553_; lean_object* v_val_554_; lean_object* v___x_555_; 
v_key_553_ = lean_ctor_get(v___x_552_, 0);
v_val_554_ = lean_ctor_get(v___x_552_, 1);
lean_inc_ref(v_f_533_);
lean_inc(v___y_541_);
lean_inc_ref(v___y_540_);
lean_inc(v___y_539_);
lean_inc_ref(v___y_538_);
lean_inc(v_val_554_);
lean_inc(v_key_553_);
v___x_555_ = lean_apply_8(v_f_533_, v_b_537_, v_key_553_, v_val_554_, v___y_538_, v___y_539_, v___y_540_, v___y_541_, lean_box(0));
v___y_549_ = v___x_555_;
goto v___jp_548_;
}
case 1:
{
lean_object* v_node_556_; lean_object* v___x_557_; 
v_node_556_ = lean_ctor_get(v___x_552_, 0);
lean_inc(v_node_556_);
lean_inc_ref(v_f_533_);
v___x_557_ = lp_batteries_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Batteries_Tactic_DeclCache_get_spec__0_spec__0___redArg(v_f_533_, v_node_556_, v_b_537_, v___y_538_, v___y_539_, v___y_540_, v___y_541_);
v___y_549_ = v___x_557_;
goto v___jp_548_;
}
default: 
{
v_a_544_ = v_b_537_;
goto v___jp_543_;
}
}
}
else
{
lean_object* v___x_558_; 
lean_dec_ref(v_f_533_);
v___x_558_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_558_, 0, v_b_537_);
return v___x_558_;
}
v___jp_543_:
{
size_t v___x_545_; size_t v___x_546_; 
v___x_545_ = ((size_t)1ULL);
v___x_546_ = lean_usize_add(v_i_535_, v___x_545_);
v_i_535_ = v___x_546_;
v_b_537_ = v_a_544_;
goto _start;
}
v___jp_548_:
{
if (lean_obj_tag(v___y_549_) == 0)
{
lean_object* v_a_550_; 
v_a_550_ = lean_ctor_get(v___y_549_, 0);
lean_inc(v_a_550_);
lean_dec_ref_known(v___y_549_, 1);
v_a_544_ = v_a_550_;
goto v___jp_543_;
}
else
{
lean_dec_ref(v_f_533_);
return v___y_549_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Batteries_Tactic_DeclCache_get_spec__0_spec__0_spec__1___redArg___boxed(lean_object* v_f_559_, lean_object* v_as_560_, lean_object* v_i_561_, lean_object* v_stop_562_, lean_object* v_b_563_, lean_object* v___y_564_, lean_object* v___y_565_, lean_object* v___y_566_, lean_object* v___y_567_, lean_object* v___y_568_){
_start:
{
size_t v_i_boxed_569_; size_t v_stop_boxed_570_; lean_object* v_res_571_; 
v_i_boxed_569_ = lean_unbox_usize(v_i_561_);
lean_dec(v_i_561_);
v_stop_boxed_570_ = lean_unbox_usize(v_stop_562_);
lean_dec(v_stop_562_);
v_res_571_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Batteries_Tactic_DeclCache_get_spec__0_spec__0_spec__1___redArg(v_f_559_, v_as_560_, v_i_boxed_569_, v_stop_boxed_570_, v_b_563_, v___y_564_, v___y_565_, v___y_566_, v___y_567_);
lean_dec(v___y_567_);
lean_dec_ref(v___y_566_);
lean_dec(v___y_565_);
lean_dec_ref(v___y_564_);
lean_dec_ref(v_as_560_);
return v_res_571_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Batteries_Tactic_DeclCache_get_spec__0_spec__0___redArg___boxed(lean_object* v_f_572_, lean_object* v_x_573_, lean_object* v_x_574_, lean_object* v___y_575_, lean_object* v___y_576_, lean_object* v___y_577_, lean_object* v___y_578_, lean_object* v___y_579_){
_start:
{
lean_object* v_res_580_; 
v_res_580_ = lp_batteries_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Batteries_Tactic_DeclCache_get_spec__0_spec__0___redArg(v_f_572_, v_x_573_, v_x_574_, v___y_575_, v___y_576_, v___y_577_, v___y_578_);
lean_dec(v___y_578_);
lean_dec_ref(v___y_577_);
lean_dec(v___y_576_);
lean_dec_ref(v___y_575_);
return v_res_580_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_DeclCache_get___redArg(lean_object* v_cache_581_, lean_object* v_a_582_, lean_object* v_a_583_, lean_object* v_a_584_, lean_object* v_a_585_){
_start:
{
lean_object* v___x_587_; lean_object* v_cache_588_; lean_object* v_addDecl_589_; lean_object* v___x_590_; 
v___x_587_ = lean_st_ref_get(v_a_585_);
v_cache_588_ = lean_ctor_get(v_cache_581_, 0);
lean_inc(v_cache_588_);
v_addDecl_589_ = lean_ctor_get(v_cache_581_, 1);
lean_inc_ref(v_addDecl_589_);
lean_dec_ref(v_cache_581_);
v___x_590_ = lp_batteries_Batteries_Tactic_Cache_get___redArg(v_cache_588_, v_a_582_, v_a_583_, v_a_584_, v_a_585_);
lean_dec(v_cache_588_);
if (lean_obj_tag(v___x_590_) == 0)
{
lean_object* v_a_591_; lean_object* v_env_592_; lean_object* v___x_593_; lean_object* v_map_u2082_594_; lean_object* v___f_595_; lean_object* v___x_596_; 
v_a_591_ = lean_ctor_get(v___x_590_, 0);
lean_inc(v_a_591_);
lean_dec_ref_known(v___x_590_, 1);
v_env_592_ = lean_ctor_get(v___x_587_, 0);
lean_inc_ref(v_env_592_);
lean_dec(v___x_587_);
v___x_593_ = l_Lean_Environment_constants(v_env_592_);
v_map_u2082_594_ = lean_ctor_get(v___x_593_, 1);
lean_inc_ref(v_map_u2082_594_);
lean_dec_ref(v___x_593_);
v___f_595_ = lean_alloc_closure((void*)(lp_batteries_Batteries_Tactic_DeclCache_get___redArg___lam__0___boxed), 9, 1);
lean_closure_set(v___f_595_, 0, v_addDecl_589_);
v___x_596_ = lp_batteries_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Batteries_Tactic_DeclCache_get_spec__0_spec__0___redArg(v___f_595_, v_map_u2082_594_, v_a_591_, v_a_582_, v_a_583_, v_a_584_, v_a_585_);
return v___x_596_;
}
else
{
lean_dec_ref(v_addDecl_589_);
lean_dec(v___x_587_);
return v___x_590_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_DeclCache_get___redArg___boxed(lean_object* v_cache_597_, lean_object* v_a_598_, lean_object* v_a_599_, lean_object* v_a_600_, lean_object* v_a_601_, lean_object* v_a_602_){
_start:
{
lean_object* v_res_603_; 
v_res_603_ = lp_batteries_Batteries_Tactic_DeclCache_get___redArg(v_cache_597_, v_a_598_, v_a_599_, v_a_600_, v_a_601_);
lean_dec(v_a_601_);
lean_dec_ref(v_a_600_);
lean_dec(v_a_599_);
lean_dec_ref(v_a_598_);
return v_res_603_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_DeclCache_get(lean_object* v_00_u03b1_604_, lean_object* v_cache_605_, lean_object* v_a_606_, lean_object* v_a_607_, lean_object* v_a_608_, lean_object* v_a_609_){
_start:
{
lean_object* v___x_611_; 
v___x_611_ = lp_batteries_Batteries_Tactic_DeclCache_get___redArg(v_cache_605_, v_a_606_, v_a_607_, v_a_608_, v_a_609_);
return v___x_611_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_DeclCache_get___boxed(lean_object* v_00_u03b1_612_, lean_object* v_cache_613_, lean_object* v_a_614_, lean_object* v_a_615_, lean_object* v_a_616_, lean_object* v_a_617_, lean_object* v_a_618_){
_start:
{
lean_object* v_res_619_; 
v_res_619_ = lp_batteries_Batteries_Tactic_DeclCache_get(v_00_u03b1_612_, v_cache_613_, v_a_614_, v_a_615_, v_a_616_, v_a_617_);
lean_dec(v_a_617_);
lean_dec_ref(v_a_616_);
lean_dec(v_a_615_);
lean_dec_ref(v_a_614_);
return v_res_619_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlM___at___00Batteries_Tactic_DeclCache_get_spec__0___redArg(lean_object* v_map_620_, lean_object* v_f_621_, lean_object* v_init_622_, lean_object* v___y_623_, lean_object* v___y_624_, lean_object* v___y_625_, lean_object* v___y_626_){
_start:
{
lean_object* v___x_628_; 
v___x_628_ = lp_batteries_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Batteries_Tactic_DeclCache_get_spec__0_spec__0___redArg(v_f_621_, v_map_620_, v_init_622_, v___y_623_, v___y_624_, v___y_625_, v___y_626_);
return v___x_628_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlM___at___00Batteries_Tactic_DeclCache_get_spec__0___redArg___boxed(lean_object* v_map_629_, lean_object* v_f_630_, lean_object* v_init_631_, lean_object* v___y_632_, lean_object* v___y_633_, lean_object* v___y_634_, lean_object* v___y_635_, lean_object* v___y_636_){
_start:
{
lean_object* v_res_637_; 
v_res_637_ = lp_batteries_Lean_PersistentHashMap_foldlM___at___00Batteries_Tactic_DeclCache_get_spec__0___redArg(v_map_629_, v_f_630_, v_init_631_, v___y_632_, v___y_633_, v___y_634_, v___y_635_);
lean_dec(v___y_635_);
lean_dec_ref(v___y_634_);
lean_dec(v___y_633_);
lean_dec_ref(v___y_632_);
return v_res_637_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlM___at___00Batteries_Tactic_DeclCache_get_spec__0(lean_object* v_00_u03c3_638_, lean_object* v_00_u03b2_639_, lean_object* v_map_640_, lean_object* v_f_641_, lean_object* v_init_642_, lean_object* v___y_643_, lean_object* v___y_644_, lean_object* v___y_645_, lean_object* v___y_646_){
_start:
{
lean_object* v___x_648_; 
v___x_648_ = lp_batteries_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Batteries_Tactic_DeclCache_get_spec__0_spec__0___redArg(v_f_641_, v_map_640_, v_init_642_, v___y_643_, v___y_644_, v___y_645_, v___y_646_);
return v___x_648_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlM___at___00Batteries_Tactic_DeclCache_get_spec__0___boxed(lean_object* v_00_u03c3_649_, lean_object* v_00_u03b2_650_, lean_object* v_map_651_, lean_object* v_f_652_, lean_object* v_init_653_, lean_object* v___y_654_, lean_object* v___y_655_, lean_object* v___y_656_, lean_object* v___y_657_, lean_object* v___y_658_){
_start:
{
lean_object* v_res_659_; 
v_res_659_ = lp_batteries_Lean_PersistentHashMap_foldlM___at___00Batteries_Tactic_DeclCache_get_spec__0(v_00_u03c3_649_, v_00_u03b2_650_, v_map_651_, v_f_652_, v_init_653_, v___y_654_, v___y_655_, v___y_656_, v___y_657_);
lean_dec(v___y_657_);
lean_dec_ref(v___y_656_);
lean_dec(v___y_655_);
lean_dec_ref(v___y_654_);
return v_res_659_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Batteries_Tactic_DeclCache_get_spec__0_spec__0(lean_object* v_00_u03c3_660_, lean_object* v_00_u03b1_661_, lean_object* v_00_u03b2_662_, lean_object* v_f_663_, lean_object* v_x_664_, lean_object* v_x_665_, lean_object* v___y_666_, lean_object* v___y_667_, lean_object* v___y_668_, lean_object* v___y_669_){
_start:
{
lean_object* v___x_671_; 
v___x_671_ = lp_batteries_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Batteries_Tactic_DeclCache_get_spec__0_spec__0___redArg(v_f_663_, v_x_664_, v_x_665_, v___y_666_, v___y_667_, v___y_668_, v___y_669_);
return v___x_671_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Batteries_Tactic_DeclCache_get_spec__0_spec__0___boxed(lean_object* v_00_u03c3_672_, lean_object* v_00_u03b1_673_, lean_object* v_00_u03b2_674_, lean_object* v_f_675_, lean_object* v_x_676_, lean_object* v_x_677_, lean_object* v___y_678_, lean_object* v___y_679_, lean_object* v___y_680_, lean_object* v___y_681_, lean_object* v___y_682_){
_start:
{
lean_object* v_res_683_; 
v_res_683_ = lp_batteries_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Batteries_Tactic_DeclCache_get_spec__0_spec__0(v_00_u03c3_672_, v_00_u03b1_673_, v_00_u03b2_674_, v_f_675_, v_x_676_, v_x_677_, v___y_678_, v___y_679_, v___y_680_, v___y_681_);
lean_dec(v___y_681_);
lean_dec_ref(v___y_680_);
lean_dec(v___y_679_);
lean_dec_ref(v___y_678_);
return v_res_683_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Batteries_Tactic_DeclCache_get_spec__0_spec__0_spec__1(lean_object* v_00_u03b1_684_, lean_object* v_00_u03b2_685_, lean_object* v_00_u03c3_686_, lean_object* v_f_687_, lean_object* v_as_688_, size_t v_i_689_, size_t v_stop_690_, lean_object* v_b_691_, lean_object* v___y_692_, lean_object* v___y_693_, lean_object* v___y_694_, lean_object* v___y_695_){
_start:
{
lean_object* v___x_697_; 
v___x_697_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Batteries_Tactic_DeclCache_get_spec__0_spec__0_spec__1___redArg(v_f_687_, v_as_688_, v_i_689_, v_stop_690_, v_b_691_, v___y_692_, v___y_693_, v___y_694_, v___y_695_);
return v___x_697_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Batteries_Tactic_DeclCache_get_spec__0_spec__0_spec__1___boxed(lean_object* v_00_u03b1_698_, lean_object* v_00_u03b2_699_, lean_object* v_00_u03c3_700_, lean_object* v_f_701_, lean_object* v_as_702_, lean_object* v_i_703_, lean_object* v_stop_704_, lean_object* v_b_705_, lean_object* v___y_706_, lean_object* v___y_707_, lean_object* v___y_708_, lean_object* v___y_709_, lean_object* v___y_710_){
_start:
{
size_t v_i_boxed_711_; size_t v_stop_boxed_712_; lean_object* v_res_713_; 
v_i_boxed_711_ = lean_unbox_usize(v_i_703_);
lean_dec(v_i_703_);
v_stop_boxed_712_ = lean_unbox_usize(v_stop_704_);
lean_dec(v_stop_704_);
v_res_713_ = lp_batteries___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Batteries_Tactic_DeclCache_get_spec__0_spec__0_spec__1(v_00_u03b1_698_, v_00_u03b2_699_, v_00_u03c3_700_, v_f_701_, v_as_702_, v_i_boxed_711_, v_stop_boxed_712_, v_b_705_, v___y_706_, v___y_707_, v___y_708_, v___y_709_);
lean_dec(v___y_709_);
lean_dec_ref(v___y_708_);
lean_dec(v___y_707_);
lean_dec_ref(v___y_706_);
lean_dec_ref(v_as_702_);
return v_res_713_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Batteries_Tactic_DeclCache_get_spec__0_spec__0_spec__2(lean_object* v_00_u03c3_714_, lean_object* v_00_u03b1_715_, lean_object* v_00_u03b2_716_, lean_object* v_f_717_, lean_object* v_keys_718_, lean_object* v_vals_719_, lean_object* v_heq_720_, lean_object* v_i_721_, lean_object* v_acc_722_, lean_object* v___y_723_, lean_object* v___y_724_, lean_object* v___y_725_, lean_object* v___y_726_){
_start:
{
lean_object* v___x_728_; 
v___x_728_ = lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Batteries_Tactic_DeclCache_get_spec__0_spec__0_spec__2___redArg(v_f_717_, v_keys_718_, v_vals_719_, v_i_721_, v_acc_722_, v___y_723_, v___y_724_, v___y_725_, v___y_726_);
return v___x_728_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Batteries_Tactic_DeclCache_get_spec__0_spec__0_spec__2___boxed(lean_object* v_00_u03c3_729_, lean_object* v_00_u03b1_730_, lean_object* v_00_u03b2_731_, lean_object* v_f_732_, lean_object* v_keys_733_, lean_object* v_vals_734_, lean_object* v_heq_735_, lean_object* v_i_736_, lean_object* v_acc_737_, lean_object* v___y_738_, lean_object* v___y_739_, lean_object* v___y_740_, lean_object* v___y_741_, lean_object* v___y_742_){
_start:
{
lean_object* v_res_743_; 
v_res_743_ = lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Batteries_Tactic_DeclCache_get_spec__0_spec__0_spec__2(v_00_u03c3_729_, v_00_u03b1_730_, v_00_u03b2_731_, v_f_732_, v_keys_733_, v_vals_734_, v_heq_735_, v_i_736_, v_acc_737_, v___y_738_, v___y_739_, v___y_740_, v___y_741_);
lean_dec(v___y_741_);
lean_dec_ref(v___y_740_);
lean_dec(v___y_739_);
lean_dec_ref(v___y_738_);
lean_dec_ref(v_vals_734_);
lean_dec_ref(v_keys_733_);
return v_res_743_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__0(lean_object* v_post_x3f_744_, lean_object* v___y_745_, lean_object* v___y_746_, lean_object* v___y_747_, lean_object* v___y_748_, lean_object* v___y_749_){
_start:
{
if (lean_obj_tag(v_post_x3f_744_) == 0)
{
lean_object* v___x_751_; 
v___x_751_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_751_, 0, v___y_745_);
return v___x_751_;
}
else
{
lean_object* v_val_752_; lean_object* v___x_754_; uint8_t v_isShared_755_; uint8_t v_isSharedCheck_769_; 
v_val_752_ = lean_ctor_get(v_post_x3f_744_, 0);
v_isSharedCheck_769_ = !lean_is_exclusive(v_post_x3f_744_);
if (v_isSharedCheck_769_ == 0)
{
v___x_754_ = v_post_x3f_744_;
v_isShared_755_ = v_isSharedCheck_769_;
goto v_resetjp_753_;
}
else
{
lean_inc(v_val_752_);
lean_dec(v_post_x3f_744_);
v___x_754_ = lean_box(0);
v_isShared_755_ = v_isSharedCheck_769_;
goto v_resetjp_753_;
}
v_resetjp_753_:
{
lean_object* v_fst_756_; lean_object* v_snd_757_; lean_object* v___x_759_; uint8_t v_isShared_760_; uint8_t v_isSharedCheck_768_; 
v_fst_756_ = lean_ctor_get(v___y_745_, 0);
v_snd_757_ = lean_ctor_get(v___y_745_, 1);
v_isSharedCheck_768_ = !lean_is_exclusive(v___y_745_);
if (v_isSharedCheck_768_ == 0)
{
v___x_759_ = v___y_745_;
v_isShared_760_ = v_isSharedCheck_768_;
goto v_resetjp_758_;
}
else
{
lean_inc(v_snd_757_);
lean_inc(v_fst_756_);
lean_dec(v___y_745_);
v___x_759_ = lean_box(0);
v_isShared_760_ = v_isSharedCheck_768_;
goto v_resetjp_758_;
}
v_resetjp_758_:
{
lean_object* v___x_761_; lean_object* v___x_763_; 
v___x_761_ = l_Lean_Meta_DiscrTree_mapArrays___redArg(v_snd_757_, v_val_752_);
if (v_isShared_760_ == 0)
{
lean_ctor_set(v___x_759_, 1, v___x_761_);
v___x_763_ = v___x_759_;
goto v_reusejp_762_;
}
else
{
lean_object* v_reuseFailAlloc_767_; 
v_reuseFailAlloc_767_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_767_, 0, v_fst_756_);
lean_ctor_set(v_reuseFailAlloc_767_, 1, v___x_761_);
v___x_763_ = v_reuseFailAlloc_767_;
goto v_reusejp_762_;
}
v_reusejp_762_:
{
lean_object* v___x_765_; 
if (v_isShared_755_ == 0)
{
lean_ctor_set_tag(v___x_754_, 0);
lean_ctor_set(v___x_754_, 0, v___x_763_);
v___x_765_ = v___x_754_;
goto v_reusejp_764_;
}
else
{
lean_object* v_reuseFailAlloc_766_; 
v_reuseFailAlloc_766_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_766_, 0, v___x_763_);
v___x_765_ = v_reuseFailAlloc_766_;
goto v_reusejp_764_;
}
v_reusejp_764_:
{
return v___x_765_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__0___boxed(lean_object* v_post_x3f_770_, lean_object* v___y_771_, lean_object* v___y_772_, lean_object* v___y_773_, lean_object* v___y_774_, lean_object* v___y_775_, lean_object* v___y_776_){
_start:
{
lean_object* v_res_777_; 
v_res_777_ = lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__0(v_post_x3f_770_, v___y_771_, v___y_772_, v___y_773_, v___y_774_, v___y_775_);
lean_dec(v___y_775_);
lean_dec_ref(v___y_774_);
lean_dec(v___y_773_);
lean_dec_ref(v___y_772_);
return v_res_777_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__1(lean_object* v_inst_778_, lean_object* v_x1_779_, lean_object* v_x2_780_){
_start:
{
lean_object* v_fst_781_; lean_object* v_snd_782_; lean_object* v___x_783_; 
v_fst_781_ = lean_ctor_get(v_x2_780_, 0);
lean_inc(v_fst_781_);
v_snd_782_ = lean_ctor_get(v_x2_780_, 1);
lean_inc(v_snd_782_);
lean_dec_ref(v_x2_780_);
v___x_783_ = l_Lean_Meta_DiscrTree_insertKeyValue___redArg(v_inst_778_, v_x1_779_, v_fst_781_, v_snd_782_);
return v___x_783_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__2(lean_object* v_processDecl_803_, lean_object* v___f_804_, lean_object* v_name_805_, lean_object* v_constInfo_806_, lean_object* v_tree_807_, lean_object* v___y_808_, lean_object* v___y_809_, lean_object* v___y_810_, lean_object* v___y_811_){
_start:
{
lean_object* v___x_813_; 
lean_inc(v___y_811_);
lean_inc_ref(v___y_810_);
lean_inc(v___y_809_);
lean_inc_ref(v___y_808_);
v___x_813_ = lean_apply_7(v_processDecl_803_, v_name_805_, v_constInfo_806_, v___y_808_, v___y_809_, v___y_810_, v___y_811_, lean_box(0));
if (lean_obj_tag(v___x_813_) == 0)
{
lean_object* v_a_814_; lean_object* v___x_816_; uint8_t v_isShared_817_; uint8_t v_isSharedCheck_841_; 
v_a_814_ = lean_ctor_get(v___x_813_, 0);
v_isSharedCheck_841_ = !lean_is_exclusive(v___x_813_);
if (v_isSharedCheck_841_ == 0)
{
v___x_816_ = v___x_813_;
v_isShared_817_ = v_isSharedCheck_841_;
goto v_resetjp_815_;
}
else
{
lean_inc(v_a_814_);
lean_dec(v___x_813_);
v___x_816_ = lean_box(0);
v_isShared_817_ = v_isSharedCheck_841_;
goto v_resetjp_815_;
}
v_resetjp_815_:
{
lean_object* v___x_818_; lean_object* v___x_819_; lean_object* v___x_820_; uint8_t v___x_821_; 
v___x_818_ = lean_unsigned_to_nat(0u);
v___x_819_ = lean_array_get_size(v_a_814_);
v___x_820_ = ((lean_object*)(lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__2___closed__9));
v___x_821_ = lean_nat_dec_lt(v___x_818_, v___x_819_);
if (v___x_821_ == 0)
{
lean_object* v___x_823_; 
lean_dec(v_a_814_);
lean_dec_ref(v___f_804_);
if (v_isShared_817_ == 0)
{
lean_ctor_set(v___x_816_, 0, v_tree_807_);
v___x_823_ = v___x_816_;
goto v_reusejp_822_;
}
else
{
lean_object* v_reuseFailAlloc_824_; 
v_reuseFailAlloc_824_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_824_, 0, v_tree_807_);
v___x_823_ = v_reuseFailAlloc_824_;
goto v_reusejp_822_;
}
v_reusejp_822_:
{
return v___x_823_;
}
}
else
{
uint8_t v___x_825_; 
v___x_825_ = lean_nat_dec_le(v___x_819_, v___x_819_);
if (v___x_825_ == 0)
{
if (v___x_821_ == 0)
{
lean_object* v___x_827_; 
lean_dec(v_a_814_);
lean_dec_ref(v___f_804_);
if (v_isShared_817_ == 0)
{
lean_ctor_set(v___x_816_, 0, v_tree_807_);
v___x_827_ = v___x_816_;
goto v_reusejp_826_;
}
else
{
lean_object* v_reuseFailAlloc_828_; 
v_reuseFailAlloc_828_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_828_, 0, v_tree_807_);
v___x_827_ = v_reuseFailAlloc_828_;
goto v_reusejp_826_;
}
v_reusejp_826_:
{
return v___x_827_;
}
}
else
{
size_t v___x_829_; size_t v___x_830_; lean_object* v___x_831_; lean_object* v___x_833_; 
v___x_829_ = ((size_t)0ULL);
v___x_830_ = lean_usize_of_nat(v___x_819_);
v___x_831_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_820_, v___f_804_, v_a_814_, v___x_829_, v___x_830_, v_tree_807_);
if (v_isShared_817_ == 0)
{
lean_ctor_set(v___x_816_, 0, v___x_831_);
v___x_833_ = v___x_816_;
goto v_reusejp_832_;
}
else
{
lean_object* v_reuseFailAlloc_834_; 
v_reuseFailAlloc_834_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_834_, 0, v___x_831_);
v___x_833_ = v_reuseFailAlloc_834_;
goto v_reusejp_832_;
}
v_reusejp_832_:
{
return v___x_833_;
}
}
}
else
{
size_t v___x_835_; size_t v___x_836_; lean_object* v___x_837_; lean_object* v___x_839_; 
v___x_835_ = ((size_t)0ULL);
v___x_836_ = lean_usize_of_nat(v___x_819_);
v___x_837_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_820_, v___f_804_, v_a_814_, v___x_835_, v___x_836_, v_tree_807_);
if (v_isShared_817_ == 0)
{
lean_ctor_set(v___x_816_, 0, v___x_837_);
v___x_839_ = v___x_816_;
goto v_reusejp_838_;
}
else
{
lean_object* v_reuseFailAlloc_840_; 
v_reuseFailAlloc_840_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_840_, 0, v___x_837_);
v___x_839_ = v_reuseFailAlloc_840_;
goto v_reusejp_838_;
}
v_reusejp_838_:
{
return v___x_839_;
}
}
}
}
}
else
{
lean_object* v_a_842_; lean_object* v___x_844_; uint8_t v_isShared_845_; uint8_t v_isSharedCheck_849_; 
lean_dec_ref(v_tree_807_);
lean_dec_ref(v___f_804_);
v_a_842_ = lean_ctor_get(v___x_813_, 0);
v_isSharedCheck_849_ = !lean_is_exclusive(v___x_813_);
if (v_isSharedCheck_849_ == 0)
{
v___x_844_ = v___x_813_;
v_isShared_845_ = v_isSharedCheck_849_;
goto v_resetjp_843_;
}
else
{
lean_inc(v_a_842_);
lean_dec(v___x_813_);
v___x_844_ = lean_box(0);
v_isShared_845_ = v_isSharedCheck_849_;
goto v_resetjp_843_;
}
v_resetjp_843_:
{
lean_object* v___x_847_; 
if (v_isShared_845_ == 0)
{
v___x_847_ = v___x_844_;
goto v_reusejp_846_;
}
else
{
lean_object* v_reuseFailAlloc_848_; 
v_reuseFailAlloc_848_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_848_, 0, v_a_842_);
v___x_847_ = v_reuseFailAlloc_848_;
goto v_reusejp_846_;
}
v_reusejp_846_:
{
return v___x_847_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__2___boxed(lean_object* v_processDecl_850_, lean_object* v___f_851_, lean_object* v_name_852_, lean_object* v_constInfo_853_, lean_object* v_tree_854_, lean_object* v___y_855_, lean_object* v___y_856_, lean_object* v___y_857_, lean_object* v___y_858_, lean_object* v___y_859_){
_start:
{
lean_object* v_res_860_; 
v_res_860_ = lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__2(v_processDecl_850_, v___f_851_, v_name_852_, v_constInfo_853_, v_tree_854_, v___y_855_, v___y_856_, v___y_857_, v___y_858_);
lean_dec(v___y_858_);
lean_dec_ref(v___y_857_);
lean_dec(v___y_856_);
lean_dec_ref(v___y_855_);
return v_res_860_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__3(lean_object* v_updateTree_861_, lean_object* v_name_862_, lean_object* v_constInfo_863_, lean_object* v_x_864_, lean_object* v___y_865_, lean_object* v___y_866_, lean_object* v___y_867_, lean_object* v___y_868_){
_start:
{
lean_object* v_fst_870_; lean_object* v_snd_871_; lean_object* v___x_873_; uint8_t v_isShared_874_; uint8_t v_isSharedCheck_895_; 
v_fst_870_ = lean_ctor_get(v_x_864_, 0);
v_snd_871_ = lean_ctor_get(v_x_864_, 1);
v_isSharedCheck_895_ = !lean_is_exclusive(v_x_864_);
if (v_isSharedCheck_895_ == 0)
{
v___x_873_ = v_x_864_;
v_isShared_874_ = v_isSharedCheck_895_;
goto v_resetjp_872_;
}
else
{
lean_inc(v_snd_871_);
lean_inc(v_fst_870_);
lean_dec(v_x_864_);
v___x_873_ = lean_box(0);
v_isShared_874_ = v_isSharedCheck_895_;
goto v_resetjp_872_;
}
v_resetjp_872_:
{
lean_object* v___x_875_; 
lean_inc(v___y_868_);
lean_inc_ref(v___y_867_);
lean_inc(v___y_866_);
lean_inc_ref(v___y_865_);
v___x_875_ = lean_apply_8(v_updateTree_861_, v_name_862_, v_constInfo_863_, v_snd_871_, v___y_865_, v___y_866_, v___y_867_, v___y_868_, lean_box(0));
if (lean_obj_tag(v___x_875_) == 0)
{
lean_object* v_a_876_; lean_object* v___x_878_; uint8_t v_isShared_879_; uint8_t v_isSharedCheck_886_; 
v_a_876_ = lean_ctor_get(v___x_875_, 0);
v_isSharedCheck_886_ = !lean_is_exclusive(v___x_875_);
if (v_isSharedCheck_886_ == 0)
{
v___x_878_ = v___x_875_;
v_isShared_879_ = v_isSharedCheck_886_;
goto v_resetjp_877_;
}
else
{
lean_inc(v_a_876_);
lean_dec(v___x_875_);
v___x_878_ = lean_box(0);
v_isShared_879_ = v_isSharedCheck_886_;
goto v_resetjp_877_;
}
v_resetjp_877_:
{
lean_object* v___x_881_; 
if (v_isShared_874_ == 0)
{
lean_ctor_set(v___x_873_, 1, v_a_876_);
v___x_881_ = v___x_873_;
goto v_reusejp_880_;
}
else
{
lean_object* v_reuseFailAlloc_885_; 
v_reuseFailAlloc_885_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_885_, 0, v_fst_870_);
lean_ctor_set(v_reuseFailAlloc_885_, 1, v_a_876_);
v___x_881_ = v_reuseFailAlloc_885_;
goto v_reusejp_880_;
}
v_reusejp_880_:
{
lean_object* v___x_883_; 
if (v_isShared_879_ == 0)
{
lean_ctor_set(v___x_878_, 0, v___x_881_);
v___x_883_ = v___x_878_;
goto v_reusejp_882_;
}
else
{
lean_object* v_reuseFailAlloc_884_; 
v_reuseFailAlloc_884_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_884_, 0, v___x_881_);
v___x_883_ = v_reuseFailAlloc_884_;
goto v_reusejp_882_;
}
v_reusejp_882_:
{
return v___x_883_;
}
}
}
}
else
{
lean_object* v_a_887_; lean_object* v___x_889_; uint8_t v_isShared_890_; uint8_t v_isSharedCheck_894_; 
lean_del_object(v___x_873_);
lean_dec(v_fst_870_);
v_a_887_ = lean_ctor_get(v___x_875_, 0);
v_isSharedCheck_894_ = !lean_is_exclusive(v___x_875_);
if (v_isSharedCheck_894_ == 0)
{
v___x_889_ = v___x_875_;
v_isShared_890_ = v_isSharedCheck_894_;
goto v_resetjp_888_;
}
else
{
lean_inc(v_a_887_);
lean_dec(v___x_875_);
v___x_889_ = lean_box(0);
v_isShared_890_ = v_isSharedCheck_894_;
goto v_resetjp_888_;
}
v_resetjp_888_:
{
lean_object* v___x_892_; 
if (v_isShared_890_ == 0)
{
v___x_892_ = v___x_889_;
goto v_reusejp_891_;
}
else
{
lean_object* v_reuseFailAlloc_893_; 
v_reuseFailAlloc_893_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_893_, 0, v_a_887_);
v___x_892_ = v_reuseFailAlloc_893_;
goto v_reusejp_891_;
}
v_reusejp_891_:
{
return v___x_892_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__3___boxed(lean_object* v_updateTree_896_, lean_object* v_name_897_, lean_object* v_constInfo_898_, lean_object* v_x_899_, lean_object* v___y_900_, lean_object* v___y_901_, lean_object* v___y_902_, lean_object* v___y_903_, lean_object* v___y_904_){
_start:
{
lean_object* v_res_905_; 
v_res_905_ = lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__3(v_updateTree_896_, v_name_897_, v_constInfo_898_, v_x_899_, v___y_900_, v___y_901_, v___y_902_, v___y_903_);
lean_dec(v___y_903_);
lean_dec_ref(v___y_902_);
lean_dec(v___y_901_);
lean_dec_ref(v___y_900_);
return v_res_905_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__4(lean_object* v_updateTree_906_, lean_object* v_name_907_, lean_object* v_constInfo_908_, lean_object* v_x_909_, lean_object* v___y_910_, lean_object* v___y_911_, lean_object* v___y_912_, lean_object* v___y_913_){
_start:
{
lean_object* v_fst_915_; lean_object* v_snd_916_; lean_object* v___x_918_; uint8_t v_isShared_919_; uint8_t v_isSharedCheck_940_; 
v_fst_915_ = lean_ctor_get(v_x_909_, 0);
v_snd_916_ = lean_ctor_get(v_x_909_, 1);
v_isSharedCheck_940_ = !lean_is_exclusive(v_x_909_);
if (v_isSharedCheck_940_ == 0)
{
v___x_918_ = v_x_909_;
v_isShared_919_ = v_isSharedCheck_940_;
goto v_resetjp_917_;
}
else
{
lean_inc(v_snd_916_);
lean_inc(v_fst_915_);
lean_dec(v_x_909_);
v___x_918_ = lean_box(0);
v_isShared_919_ = v_isSharedCheck_940_;
goto v_resetjp_917_;
}
v_resetjp_917_:
{
lean_object* v___x_920_; 
lean_inc(v___y_913_);
lean_inc_ref(v___y_912_);
lean_inc(v___y_911_);
lean_inc_ref(v___y_910_);
v___x_920_ = lean_apply_8(v_updateTree_906_, v_name_907_, v_constInfo_908_, v_fst_915_, v___y_910_, v___y_911_, v___y_912_, v___y_913_, lean_box(0));
if (lean_obj_tag(v___x_920_) == 0)
{
lean_object* v_a_921_; lean_object* v___x_923_; uint8_t v_isShared_924_; uint8_t v_isSharedCheck_931_; 
v_a_921_ = lean_ctor_get(v___x_920_, 0);
v_isSharedCheck_931_ = !lean_is_exclusive(v___x_920_);
if (v_isSharedCheck_931_ == 0)
{
v___x_923_ = v___x_920_;
v_isShared_924_ = v_isSharedCheck_931_;
goto v_resetjp_922_;
}
else
{
lean_inc(v_a_921_);
lean_dec(v___x_920_);
v___x_923_ = lean_box(0);
v_isShared_924_ = v_isSharedCheck_931_;
goto v_resetjp_922_;
}
v_resetjp_922_:
{
lean_object* v___x_926_; 
if (v_isShared_919_ == 0)
{
lean_ctor_set(v___x_918_, 0, v_a_921_);
v___x_926_ = v___x_918_;
goto v_reusejp_925_;
}
else
{
lean_object* v_reuseFailAlloc_930_; 
v_reuseFailAlloc_930_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_930_, 0, v_a_921_);
lean_ctor_set(v_reuseFailAlloc_930_, 1, v_snd_916_);
v___x_926_ = v_reuseFailAlloc_930_;
goto v_reusejp_925_;
}
v_reusejp_925_:
{
lean_object* v___x_928_; 
if (v_isShared_924_ == 0)
{
lean_ctor_set(v___x_923_, 0, v___x_926_);
v___x_928_ = v___x_923_;
goto v_reusejp_927_;
}
else
{
lean_object* v_reuseFailAlloc_929_; 
v_reuseFailAlloc_929_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_929_, 0, v___x_926_);
v___x_928_ = v_reuseFailAlloc_929_;
goto v_reusejp_927_;
}
v_reusejp_927_:
{
return v___x_928_;
}
}
}
}
else
{
lean_object* v_a_932_; lean_object* v___x_934_; uint8_t v_isShared_935_; uint8_t v_isSharedCheck_939_; 
lean_del_object(v___x_918_);
lean_dec(v_snd_916_);
v_a_932_ = lean_ctor_get(v___x_920_, 0);
v_isSharedCheck_939_ = !lean_is_exclusive(v___x_920_);
if (v_isSharedCheck_939_ == 0)
{
v___x_934_ = v___x_920_;
v_isShared_935_ = v_isSharedCheck_939_;
goto v_resetjp_933_;
}
else
{
lean_inc(v_a_932_);
lean_dec(v___x_920_);
v___x_934_ = lean_box(0);
v_isShared_935_ = v_isSharedCheck_939_;
goto v_resetjp_933_;
}
v_resetjp_933_:
{
lean_object* v___x_937_; 
if (v_isShared_935_ == 0)
{
v___x_937_ = v___x_934_;
goto v_reusejp_936_;
}
else
{
lean_object* v_reuseFailAlloc_938_; 
v_reuseFailAlloc_938_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_938_, 0, v_a_932_);
v___x_937_ = v_reuseFailAlloc_938_;
goto v_reusejp_936_;
}
v_reusejp_936_:
{
return v___x_937_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__4___boxed(lean_object* v_updateTree_941_, lean_object* v_name_942_, lean_object* v_constInfo_943_, lean_object* v_x_944_, lean_object* v___y_945_, lean_object* v___y_946_, lean_object* v___y_947_, lean_object* v___y_948_, lean_object* v___y_949_){
_start:
{
lean_object* v_res_950_; 
v_res_950_ = lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__4(v_updateTree_941_, v_name_942_, v_constInfo_943_, v_x_944_, v___y_945_, v___y_946_, v___y_947_, v___y_948_);
lean_dec(v___y_948_);
lean_dec_ref(v___y_947_);
lean_dec(v___y_946_);
lean_dec_ref(v___y_945_);
return v_res_950_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__5___closed__0(void){
_start:
{
lean_object* v___x_951_; 
v___x_951_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_951_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__5___closed__1(void){
_start:
{
lean_object* v___x_952_; lean_object* v___x_953_; 
v___x_952_ = lean_obj_once(&lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__5___closed__0, &lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__5___closed__0_once, _init_lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__5___closed__0);
v___x_953_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_953_, 0, v___x_952_);
return v___x_953_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__5(lean_object* v_init_954_, lean_object* v___y_955_, lean_object* v___y_956_, lean_object* v___y_957_, lean_object* v___y_958_){
_start:
{
lean_object* v___x_960_; 
v___x_960_ = lean_apply_5(v_init_954_, v___y_955_, v___y_956_, v___y_957_, v___y_958_, lean_box(0));
if (lean_obj_tag(v___x_960_) == 0)
{
lean_object* v_a_961_; lean_object* v___x_963_; uint8_t v_isShared_964_; uint8_t v_isSharedCheck_970_; 
v_a_961_ = lean_ctor_get(v___x_960_, 0);
v_isSharedCheck_970_ = !lean_is_exclusive(v___x_960_);
if (v_isSharedCheck_970_ == 0)
{
v___x_963_ = v___x_960_;
v_isShared_964_ = v_isSharedCheck_970_;
goto v_resetjp_962_;
}
else
{
lean_inc(v_a_961_);
lean_dec(v___x_960_);
v___x_963_ = lean_box(0);
v_isShared_964_ = v_isSharedCheck_970_;
goto v_resetjp_962_;
}
v_resetjp_962_:
{
lean_object* v___x_965_; lean_object* v___x_966_; lean_object* v___x_968_; 
v___x_965_ = lean_obj_once(&lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__5___closed__1, &lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__5___closed__1_once, _init_lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__5___closed__1);
v___x_966_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_966_, 0, v___x_965_);
lean_ctor_set(v___x_966_, 1, v_a_961_);
if (v_isShared_964_ == 0)
{
lean_ctor_set(v___x_963_, 0, v___x_966_);
v___x_968_ = v___x_963_;
goto v_reusejp_967_;
}
else
{
lean_object* v_reuseFailAlloc_969_; 
v_reuseFailAlloc_969_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_969_, 0, v___x_966_);
v___x_968_ = v_reuseFailAlloc_969_;
goto v_reusejp_967_;
}
v_reusejp_967_:
{
return v___x_968_;
}
}
}
else
{
lean_object* v_a_971_; lean_object* v___x_973_; uint8_t v_isShared_974_; uint8_t v_isSharedCheck_978_; 
v_a_971_ = lean_ctor_get(v___x_960_, 0);
v_isSharedCheck_978_ = !lean_is_exclusive(v___x_960_);
if (v_isSharedCheck_978_ == 0)
{
v___x_973_ = v___x_960_;
v_isShared_974_ = v_isSharedCheck_978_;
goto v_resetjp_972_;
}
else
{
lean_inc(v_a_971_);
lean_dec(v___x_960_);
v___x_973_ = lean_box(0);
v_isShared_974_ = v_isSharedCheck_978_;
goto v_resetjp_972_;
}
v_resetjp_972_:
{
lean_object* v___x_976_; 
if (v_isShared_974_ == 0)
{
v___x_976_ = v___x_973_;
goto v_reusejp_975_;
}
else
{
lean_object* v_reuseFailAlloc_977_; 
v_reuseFailAlloc_977_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_977_, 0, v_a_971_);
v___x_976_ = v_reuseFailAlloc_977_;
goto v_reusejp_975_;
}
v_reusejp_975_:
{
return v___x_976_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__5___boxed(lean_object* v_init_979_, lean_object* v___y_980_, lean_object* v___y_981_, lean_object* v___y_982_, lean_object* v___y_983_, lean_object* v___y_984_){
_start:
{
lean_object* v_res_985_; 
v_res_985_ = lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__5(v_init_979_, v___y_980_, v___y_981_, v___y_982_, v___y_983_);
return v_res_985_;
}
}
static lean_object* _init_lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___closed__0(void){
_start:
{
lean_object* v___x_986_; lean_object* v___x_987_; 
v___x_986_ = lean_obj_once(&lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__5___closed__1, &lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__5___closed__1_once, _init_lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__5___closed__1);
v___x_987_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_987_, 0, v___x_986_);
lean_ctor_set(v___x_987_, 1, v___x_986_);
return v___x_987_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg(lean_object* v_inst_988_, lean_object* v_profilingName_989_, lean_object* v_processDecl_990_, lean_object* v_post_x3f_991_, lean_object* v_init_992_){
_start:
{
lean_object* v_post_994_; lean_object* v___f_995_; lean_object* v_updateTree_996_; lean_object* v_addLibraryDecl_997_; lean_object* v_addDecl_998_; lean_object* v_init_x27_999_; lean_object* v___x_1000_; lean_object* v___x_1001_; 
v_post_994_ = lean_alloc_closure((void*)(lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__0___boxed), 7, 1);
lean_closure_set(v_post_994_, 0, v_post_x3f_991_);
v___f_995_ = lean_alloc_closure((void*)(lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__1), 3, 1);
lean_closure_set(v___f_995_, 0, v_inst_988_);
v_updateTree_996_ = lean_alloc_closure((void*)(lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__2___boxed), 10, 2);
lean_closure_set(v_updateTree_996_, 0, v_processDecl_990_);
lean_closure_set(v_updateTree_996_, 1, v___f_995_);
lean_inc_ref(v_updateTree_996_);
v_addLibraryDecl_997_ = lean_alloc_closure((void*)(lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__3___boxed), 9, 1);
lean_closure_set(v_addLibraryDecl_997_, 0, v_updateTree_996_);
v_addDecl_998_ = lean_alloc_closure((void*)(lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__4___boxed), 9, 1);
lean_closure_set(v_addDecl_998_, 0, v_updateTree_996_);
v_init_x27_999_ = lean_alloc_closure((void*)(lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___lam__5___boxed), 6, 1);
lean_closure_set(v_init_x27_999_, 0, v_init_992_);
v___x_1000_ = lean_obj_once(&lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___closed__0, &lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___closed__0_once, _init_lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___closed__0);
v___x_1001_ = lp_batteries_Batteries_Tactic_DeclCache_mk___redArg(v_profilingName_989_, v_init_x27_999_, v___x_1000_, v_addDecl_998_, v_addLibraryDecl_997_, v_post_994_);
return v___x_1001_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg___boxed(lean_object* v_inst_1002_, lean_object* v_profilingName_1003_, lean_object* v_processDecl_1004_, lean_object* v_post_x3f_1005_, lean_object* v_init_1006_, lean_object* v_a_1007_){
_start:
{
lean_object* v_res_1008_; 
v_res_1008_ = lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg(v_inst_1002_, v_profilingName_1003_, v_processDecl_1004_, v_post_x3f_1005_, v_init_1006_);
return v_res_1008_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_DiscrTreeCache_mk(lean_object* v_00_u03b1_1009_, lean_object* v_inst_1010_, lean_object* v_profilingName_1011_, lean_object* v_processDecl_1012_, lean_object* v_post_x3f_1013_, lean_object* v_init_1014_){
_start:
{
lean_object* v___x_1016_; 
v___x_1016_ = lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___redArg(v_inst_1010_, v_profilingName_1011_, v_processDecl_1012_, v_post_x3f_1013_, v_init_1014_);
return v___x_1016_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_DiscrTreeCache_mk___boxed(lean_object* v_00_u03b1_1017_, lean_object* v_inst_1018_, lean_object* v_profilingName_1019_, lean_object* v_processDecl_1020_, lean_object* v_post_x3f_1021_, lean_object* v_init_1022_, lean_object* v_a_1023_){
_start:
{
lean_object* v_res_1024_; 
v_res_1024_ = lp_batteries_Batteries_Tactic_DiscrTreeCache_mk(v_00_u03b1_1017_, v_inst_1018_, v_profilingName_1019_, v_processDecl_1020_, v_post_x3f_1021_, v_init_1022_);
return v_res_1024_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_DiscrTreeCache_getMatch___redArg(lean_object* v_c_1025_, lean_object* v_e_1026_, lean_object* v_a_1027_, lean_object* v_a_1028_, lean_object* v_a_1029_, lean_object* v_a_1030_){
_start:
{
lean_object* v___x_1032_; 
v___x_1032_ = lp_batteries_Batteries_Tactic_DeclCache_get___redArg(v_c_1025_, v_a_1027_, v_a_1028_, v_a_1029_, v_a_1030_);
if (lean_obj_tag(v___x_1032_) == 0)
{
lean_object* v_a_1033_; lean_object* v_fst_1034_; lean_object* v_snd_1035_; lean_object* v___x_1036_; 
v_a_1033_ = lean_ctor_get(v___x_1032_, 0);
lean_inc(v_a_1033_);
lean_dec_ref_known(v___x_1032_, 1);
v_fst_1034_ = lean_ctor_get(v_a_1033_, 0);
lean_inc(v_fst_1034_);
v_snd_1035_ = lean_ctor_get(v_a_1033_, 1);
lean_inc(v_snd_1035_);
lean_dec(v_a_1033_);
lean_inc_ref(v_e_1026_);
v___x_1036_ = l_Lean_Meta_DiscrTree_getMatch___redArg(v_fst_1034_, v_e_1026_, v_a_1027_, v_a_1028_, v_a_1029_, v_a_1030_);
lean_dec(v_fst_1034_);
if (lean_obj_tag(v___x_1036_) == 0)
{
lean_object* v_a_1037_; lean_object* v___x_1038_; 
v_a_1037_ = lean_ctor_get(v___x_1036_, 0);
lean_inc(v_a_1037_);
lean_dec_ref_known(v___x_1036_, 1);
v___x_1038_ = l_Lean_Meta_DiscrTree_getMatch___redArg(v_snd_1035_, v_e_1026_, v_a_1027_, v_a_1028_, v_a_1029_, v_a_1030_);
lean_dec(v_snd_1035_);
if (lean_obj_tag(v___x_1038_) == 0)
{
lean_object* v_a_1039_; lean_object* v___x_1041_; uint8_t v_isShared_1042_; uint8_t v_isSharedCheck_1049_; 
v_a_1039_ = lean_ctor_get(v___x_1038_, 0);
v_isSharedCheck_1049_ = !lean_is_exclusive(v___x_1038_);
if (v_isSharedCheck_1049_ == 0)
{
v___x_1041_ = v___x_1038_;
v_isShared_1042_ = v_isSharedCheck_1049_;
goto v_resetjp_1040_;
}
else
{
lean_inc(v_a_1039_);
lean_dec(v___x_1038_);
v___x_1041_ = lean_box(0);
v_isShared_1042_ = v_isSharedCheck_1049_;
goto v_resetjp_1040_;
}
v_resetjp_1040_:
{
lean_object* v___x_1043_; lean_object* v___x_1044_; lean_object* v___x_1045_; lean_object* v___x_1047_; 
v___x_1043_ = l_Array_reverse___redArg(v_a_1037_);
v___x_1044_ = l_Array_reverse___redArg(v_a_1039_);
v___x_1045_ = l_Array_append___redArg(v___x_1043_, v___x_1044_);
lean_dec_ref(v___x_1044_);
if (v_isShared_1042_ == 0)
{
lean_ctor_set(v___x_1041_, 0, v___x_1045_);
v___x_1047_ = v___x_1041_;
goto v_reusejp_1046_;
}
else
{
lean_object* v_reuseFailAlloc_1048_; 
v_reuseFailAlloc_1048_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1048_, 0, v___x_1045_);
v___x_1047_ = v_reuseFailAlloc_1048_;
goto v_reusejp_1046_;
}
v_reusejp_1046_:
{
return v___x_1047_;
}
}
}
else
{
lean_dec(v_a_1037_);
return v___x_1038_;
}
}
else
{
lean_dec(v_snd_1035_);
lean_dec_ref(v_e_1026_);
return v___x_1036_;
}
}
else
{
lean_object* v_a_1050_; lean_object* v___x_1052_; uint8_t v_isShared_1053_; uint8_t v_isSharedCheck_1057_; 
lean_dec_ref(v_e_1026_);
v_a_1050_ = lean_ctor_get(v___x_1032_, 0);
v_isSharedCheck_1057_ = !lean_is_exclusive(v___x_1032_);
if (v_isSharedCheck_1057_ == 0)
{
v___x_1052_ = v___x_1032_;
v_isShared_1053_ = v_isSharedCheck_1057_;
goto v_resetjp_1051_;
}
else
{
lean_inc(v_a_1050_);
lean_dec(v___x_1032_);
v___x_1052_ = lean_box(0);
v_isShared_1053_ = v_isSharedCheck_1057_;
goto v_resetjp_1051_;
}
v_resetjp_1051_:
{
lean_object* v___x_1055_; 
if (v_isShared_1053_ == 0)
{
v___x_1055_ = v___x_1052_;
goto v_reusejp_1054_;
}
else
{
lean_object* v_reuseFailAlloc_1056_; 
v_reuseFailAlloc_1056_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1056_, 0, v_a_1050_);
v___x_1055_ = v_reuseFailAlloc_1056_;
goto v_reusejp_1054_;
}
v_reusejp_1054_:
{
return v___x_1055_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_DiscrTreeCache_getMatch___redArg___boxed(lean_object* v_c_1058_, lean_object* v_e_1059_, lean_object* v_a_1060_, lean_object* v_a_1061_, lean_object* v_a_1062_, lean_object* v_a_1063_, lean_object* v_a_1064_){
_start:
{
lean_object* v_res_1065_; 
v_res_1065_ = lp_batteries_Batteries_Tactic_DiscrTreeCache_getMatch___redArg(v_c_1058_, v_e_1059_, v_a_1060_, v_a_1061_, v_a_1062_, v_a_1063_);
lean_dec(v_a_1063_);
lean_dec_ref(v_a_1062_);
lean_dec(v_a_1061_);
lean_dec_ref(v_a_1060_);
return v_res_1065_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_DiscrTreeCache_getMatch(lean_object* v_00_u03b1_1066_, lean_object* v_c_1067_, lean_object* v_e_1068_, lean_object* v_a_1069_, lean_object* v_a_1070_, lean_object* v_a_1071_, lean_object* v_a_1072_){
_start:
{
lean_object* v___x_1074_; 
v___x_1074_ = lp_batteries_Batteries_Tactic_DiscrTreeCache_getMatch___redArg(v_c_1067_, v_e_1068_, v_a_1069_, v_a_1070_, v_a_1071_, v_a_1072_);
return v___x_1074_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Batteries_Tactic_DiscrTreeCache_getMatch___boxed(lean_object* v_00_u03b1_1075_, lean_object* v_c_1076_, lean_object* v_e_1077_, lean_object* v_a_1078_, lean_object* v_a_1079_, lean_object* v_a_1080_, lean_object* v_a_1081_, lean_object* v_a_1082_){
_start:
{
lean_object* v_res_1083_; 
v_res_1083_ = lp_batteries_Batteries_Tactic_DiscrTreeCache_getMatch(v_00_u03b1_1075_, v_c_1076_, v_e_1077_, v_a_1078_, v_a_1079_, v_a_1080_, v_a_1081_);
lean_dec(v_a_1081_);
lean_dec_ref(v_a_1080_);
lean_dec(v_a_1079_);
lean_dec_ref(v_a_1078_);
return v_res_1083_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_DiscrTree(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_batteries_Batteries_Util_Cache(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_DiscrTree(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_batteries_Batteries_Util_Cache(uint8_t builtin) {
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
lean_object* initialize_Lean_Meta_DiscrTree(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_batteries_Batteries_Util_Cache(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_DiscrTree(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Util_Cache(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_batteries_Batteries_Util_Cache(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_batteries_Batteries_Util_Cache(builtin);
}
#ifdef __cplusplus
}
#endif
