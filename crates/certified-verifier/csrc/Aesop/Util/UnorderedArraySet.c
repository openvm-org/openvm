// Lean compiler output
// Module: Aesop.Util.UnorderedArraySet
// Imports: public import Init public meta import Init public import Batteries.Data.Array.Merge public import Lean.Message
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
lean_object* l_Id_instMonad___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__6(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__2___boxed(lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Std_DHashMap_Internal_AssocList_foldlM___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_List_mapTR_loop___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofList(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_List_toString___redArg(lean_object*, lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
lean_object* lp_batteries_Array_sortDedup___redArg(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
uint8_t l_Array_contains___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Array_erase___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_batteries_Array_mergeUnsortedDedup___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_List_format___redArg(lean_object*, lean_object*);
lean_object* lp_batteries_Array_dedupSorted___redArg(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_foldlMAux___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_aesop_Aesop_instInhabitedUnorderedArraySet_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_instInhabitedUnorderedArraySet_default___closed__0 = (const lean_object*)&lp_aesop_Aesop_instInhabitedUnorderedArraySet_default___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedUnorderedArraySet_default(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedUnorderedArraySet_default___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedUnorderedArraySet___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedUnorderedArraySet___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedUnorderedArraySet(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedUnorderedArraySet___boxed(lean_object*, lean_object*);
static const lean_array_object lp_aesop_Aesop_UnorderedArraySet_empty___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_UnorderedArraySet_empty___closed__0 = (const lean_object*)&lp_aesop_Aesop_UnorderedArraySet_empty___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_empty(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_empty___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_instEmptyCollection___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_instEmptyCollection___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_instEmptyCollection(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_instEmptyCollection___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_singleton___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_singleton(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_singleton___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_insert___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_insert(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_ofDeduplicatedArray___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_ofDeduplicatedArray___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_ofDeduplicatedArray(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_ofDeduplicatedArray___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_ofSortedArray___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_ofSortedArray___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_ofSortedArray(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_ofSortedArray___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_ofArray___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_ofArray(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_ofArray___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_ofArraySlow___redArg___lam__0(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_UnorderedArraySet_ofArraySlow___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__0, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_UnorderedArraySet_ofArraySlow___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_UnorderedArraySet_ofArraySlow___redArg___closed__0_value;
static const lean_closure_object lp_aesop_Aesop_UnorderedArraySet_ofArraySlow___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__1___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_UnorderedArraySet_ofArraySlow___redArg___closed__1 = (const lean_object*)&lp_aesop_Aesop_UnorderedArraySet_ofArraySlow___redArg___closed__1_value;
static const lean_closure_object lp_aesop_Aesop_UnorderedArraySet_ofArraySlow___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__2___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_UnorderedArraySet_ofArraySlow___redArg___closed__2 = (const lean_object*)&lp_aesop_Aesop_UnorderedArraySet_ofArraySlow___redArg___closed__2_value;
static const lean_closure_object lp_aesop_Aesop_UnorderedArraySet_ofArraySlow___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__3, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_UnorderedArraySet_ofArraySlow___redArg___closed__3 = (const lean_object*)&lp_aesop_Aesop_UnorderedArraySet_ofArraySlow___redArg___closed__3_value;
static const lean_closure_object lp_aesop_Aesop_UnorderedArraySet_ofArraySlow___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__4___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_UnorderedArraySet_ofArraySlow___redArg___closed__4 = (const lean_object*)&lp_aesop_Aesop_UnorderedArraySet_ofArraySlow___redArg___closed__4_value;
static const lean_closure_object lp_aesop_Aesop_UnorderedArraySet_ofArraySlow___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__5___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_UnorderedArraySet_ofArraySlow___redArg___closed__5 = (const lean_object*)&lp_aesop_Aesop_UnorderedArraySet_ofArraySlow___redArg___closed__5_value;
static const lean_closure_object lp_aesop_Aesop_UnorderedArraySet_ofArraySlow___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__6, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_UnorderedArraySet_ofArraySlow___redArg___closed__6 = (const lean_object*)&lp_aesop_Aesop_UnorderedArraySet_ofArraySlow___redArg___closed__6_value;
static const lean_ctor_object lp_aesop_Aesop_UnorderedArraySet_ofArraySlow___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop_Aesop_UnorderedArraySet_ofArraySlow___redArg___closed__0_value),((lean_object*)&lp_aesop_Aesop_UnorderedArraySet_ofArraySlow___redArg___closed__1_value)}};
static const lean_object* lp_aesop_Aesop_UnorderedArraySet_ofArraySlow___redArg___closed__7 = (const lean_object*)&lp_aesop_Aesop_UnorderedArraySet_ofArraySlow___redArg___closed__7_value;
static const lean_ctor_object lp_aesop_Aesop_UnorderedArraySet_ofArraySlow___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*5 + 0, .m_other = 5, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop_Aesop_UnorderedArraySet_ofArraySlow___redArg___closed__7_value),((lean_object*)&lp_aesop_Aesop_UnorderedArraySet_ofArraySlow___redArg___closed__2_value),((lean_object*)&lp_aesop_Aesop_UnorderedArraySet_ofArraySlow___redArg___closed__3_value),((lean_object*)&lp_aesop_Aesop_UnorderedArraySet_ofArraySlow___redArg___closed__4_value),((lean_object*)&lp_aesop_Aesop_UnorderedArraySet_ofArraySlow___redArg___closed__5_value)}};
static const lean_object* lp_aesop_Aesop_UnorderedArraySet_ofArraySlow___redArg___closed__8 = (const lean_object*)&lp_aesop_Aesop_UnorderedArraySet_ofArraySlow___redArg___closed__8_value;
static const lean_ctor_object lp_aesop_Aesop_UnorderedArraySet_ofArraySlow___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop_Aesop_UnorderedArraySet_ofArraySlow___redArg___closed__8_value),((lean_object*)&lp_aesop_Aesop_UnorderedArraySet_ofArraySlow___redArg___closed__6_value)}};
static const lean_object* lp_aesop_Aesop_UnorderedArraySet_ofArraySlow___redArg___closed__9 = (const lean_object*)&lp_aesop_Aesop_UnorderedArraySet_ofArraySlow___redArg___closed__9_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_ofArraySlow___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_ofArraySlow(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_ofHashSet___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_ofHashSet___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_UnorderedArraySet_ofHashSet___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_UnorderedArraySet_ofHashSet___redArg___lam__0, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_UnorderedArraySet_ofHashSet___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_UnorderedArraySet_ofHashSet___redArg___closed__0_value;
static const lean_closure_object lp_aesop_Aesop_UnorderedArraySet_ofHashSet___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_UnorderedArraySet_ofHashSet___redArg___lam__1, .m_arity = 4, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_aesop_Aesop_UnorderedArraySet_ofArraySlow___redArg___closed__9_value),((lean_object*)&lp_aesop_Aesop_UnorderedArraySet_ofHashSet___redArg___closed__0_value)} };
static const lean_object* lp_aesop_Aesop_UnorderedArraySet_ofHashSet___redArg___closed__1 = (const lean_object*)&lp_aesop_Aesop_UnorderedArraySet_ofHashSet___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_ofHashSet___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_ofHashSet(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_ofHashSet___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_ofPersistentHashSet___redArg___lam__0(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_UnorderedArraySet_ofPersistentHashSet___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_UnorderedArraySet_ofPersistentHashSet___redArg___lam__0, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_UnorderedArraySet_ofPersistentHashSet___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_UnorderedArraySet_ofPersistentHashSet___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_ofPersistentHashSet___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_ofPersistentHashSet(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_ofPersistentHashSet___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_toArray___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_toArray___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_toArray(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_toArray___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_erase___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_erase(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_filterM___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_filterM___redArg___lam__1(lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_filterM___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_filterM___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_filterM___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_filterM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_filterM___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_filter___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_filter___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_filter(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_filter___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_merge___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_merge(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_instAppend___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_instAppend(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_UnorderedArraySet_contains___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_contains___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_UnorderedArraySet_contains(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_contains___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_foldM___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_foldM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_foldM___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_instForInOfMonad___private__1___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_instForInOfMonad___private__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_instForInOfMonad___private__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_instForInOfMonad___private__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_instForInOfMonad___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_instForInOfMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_instForInOfMonad___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_instForInOfMonad(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_instForInOfMonad___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_fold___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_fold___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_fold(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_fold___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_partition___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_aesop_Aesop_UnorderedArraySet_partition___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop_Aesop_UnorderedArraySet_empty___closed__0_value),((lean_object*)&lp_aesop_Aesop_UnorderedArraySet_empty___closed__0_value)}};
static const lean_object* lp_aesop_Aesop_UnorderedArraySet_partition___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_UnorderedArraySet_partition___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_partition___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_partition(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_partition___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_size___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_size___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_size(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_size___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_UnorderedArraySet_isEmpty___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_isEmpty___redArg___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_UnorderedArraySet_isEmpty(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_isEmpty___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_anyM___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_anyM___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_anyM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_anyM___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_UnorderedArraySet_any___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_any___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_UnorderedArraySet_any___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_any___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_UnorderedArraySet_any(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_any___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_allM___redArg___lam__0(lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_allM___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_allM___redArg___lam__1(lean_object*, uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_allM___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_allM___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_allM___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_allM___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_allM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_allM___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_UnorderedArraySet_all___redArg___lam__0(lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_all___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_UnorderedArraySet_all___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_all___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_UnorderedArraySet_all(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_all___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_UnorderedArraySet_instToString___private__1___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "#"};
static const lean_object* lp_aesop_Aesop_UnorderedArraySet_instToString___private__1___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_UnorderedArraySet_instToString___private__1___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_instToString___private__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_instToString___private__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_instToString___private__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_instToString___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_instToString___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_instToString(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_instToString___boxed(lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_aesop_Aesop_UnorderedArraySet_instToFormat___private__1___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_UnorderedArraySet_instToString___private__1___redArg___closed__0_value)}};
static const lean_object* lp_aesop_Aesop_UnorderedArraySet_instToFormat___private__1___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_UnorderedArraySet_instToFormat___private__1___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_instToFormat___private__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_instToFormat___private__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_instToFormat___private__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_instToFormat___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_instToFormat___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_instToFormat(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_instToFormat___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_instToMessageData___private__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_instToMessageData___private__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_instToMessageData___private__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_instToMessageData___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_instToMessageData___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_instToMessageData(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_instToMessageData___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedUnorderedArraySet_default(lean_object* v_00_u03b1_3_, lean_object* v_inst_4_){
_start:
{
lean_object* v___x_5_; 
v___x_5_ = ((lean_object*)(lp_aesop_Aesop_instInhabitedUnorderedArraySet_default___closed__0));
return v___x_5_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedUnorderedArraySet_default___boxed(lean_object* v_00_u03b1_6_, lean_object* v_inst_7_){
_start:
{
lean_object* v_res_8_; 
v_res_8_ = lp_aesop_Aesop_instInhabitedUnorderedArraySet_default(v_00_u03b1_6_, v_inst_7_);
lean_dec_ref(v_inst_7_);
return v_res_8_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedUnorderedArraySet___redArg(lean_object* v_a_9_){
_start:
{
lean_object* v___x_10_; 
v___x_10_ = lp_aesop_Aesop_instInhabitedUnorderedArraySet_default(lean_box(0), v_a_9_);
return v___x_10_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedUnorderedArraySet___redArg___boxed(lean_object* v_a_11_){
_start:
{
lean_object* v_res_12_; 
v_res_12_ = lp_aesop_Aesop_instInhabitedUnorderedArraySet___redArg(v_a_11_);
lean_dec_ref(v_a_11_);
return v_res_12_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedUnorderedArraySet(lean_object* v_a_13_, lean_object* v_a_14_){
_start:
{
lean_object* v___x_15_; 
v___x_15_ = lp_aesop_Aesop_instInhabitedUnorderedArraySet_default(lean_box(0), v_a_14_);
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedUnorderedArraySet___boxed(lean_object* v_a_16_, lean_object* v_a_17_){
_start:
{
lean_object* v_res_18_; 
v_res_18_ = lp_aesop_Aesop_instInhabitedUnorderedArraySet(v_a_16_, v_a_17_);
lean_dec_ref(v_a_17_);
return v_res_18_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_empty(lean_object* v_00_u03b1_21_, lean_object* v_inst_22_){
_start:
{
lean_object* v___x_23_; 
v___x_23_ = ((lean_object*)(lp_aesop_Aesop_UnorderedArraySet_empty___closed__0));
return v___x_23_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_empty___boxed(lean_object* v_00_u03b1_24_, lean_object* v_inst_25_){
_start:
{
lean_object* v_res_26_; 
v_res_26_ = lp_aesop_Aesop_UnorderedArraySet_empty(v_00_u03b1_24_, v_inst_25_);
lean_dec_ref(v_inst_25_);
return v_res_26_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_instEmptyCollection___redArg(lean_object* v_inst_27_){
_start:
{
lean_object* v___x_28_; 
v___x_28_ = lp_aesop_Aesop_UnorderedArraySet_empty(lean_box(0), v_inst_27_);
return v___x_28_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_instEmptyCollection___redArg___boxed(lean_object* v_inst_29_){
_start:
{
lean_object* v_res_30_; 
v_res_30_ = lp_aesop_Aesop_UnorderedArraySet_instEmptyCollection___redArg(v_inst_29_);
lean_dec_ref(v_inst_29_);
return v_res_30_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_instEmptyCollection(lean_object* v_00_u03b1_31_, lean_object* v_inst_32_){
_start:
{
lean_object* v___x_33_; 
v___x_33_ = lp_aesop_Aesop_UnorderedArraySet_empty(lean_box(0), v_inst_32_);
return v___x_33_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_instEmptyCollection___boxed(lean_object* v_00_u03b1_34_, lean_object* v_inst_35_){
_start:
{
lean_object* v_res_36_; 
v_res_36_ = lp_aesop_Aesop_UnorderedArraySet_instEmptyCollection(v_00_u03b1_34_, v_inst_35_);
lean_dec_ref(v_inst_35_);
return v_res_36_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_singleton___redArg(lean_object* v_a_37_){
_start:
{
lean_object* v___x_38_; lean_object* v___x_39_; lean_object* v___x_40_; 
v___x_38_ = lean_unsigned_to_nat(1u);
v___x_39_ = lean_mk_empty_array_with_capacity(v___x_38_);
v___x_40_ = lean_array_push(v___x_39_, v_a_37_);
return v___x_40_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_singleton(lean_object* v_00_u03b1_41_, lean_object* v_inst_42_, lean_object* v_a_43_){
_start:
{
lean_object* v___x_44_; 
v___x_44_ = lp_aesop_Aesop_UnorderedArraySet_singleton___redArg(v_a_43_);
return v___x_44_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_singleton___boxed(lean_object* v_00_u03b1_45_, lean_object* v_inst_46_, lean_object* v_a_47_){
_start:
{
lean_object* v_res_48_; 
v_res_48_ = lp_aesop_Aesop_UnorderedArraySet_singleton(v_00_u03b1_45_, v_inst_46_, v_a_47_);
lean_dec_ref(v_inst_46_);
return v_res_48_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_insert___redArg(lean_object* v_inst_49_, lean_object* v_x_50_, lean_object* v_x_51_){
_start:
{
uint8_t v___x_52_; 
lean_inc(v_x_50_);
lean_inc_ref(v_x_51_);
v___x_52_ = l_Array_contains___redArg(v_inst_49_, v_x_51_, v_x_50_);
if (v___x_52_ == 0)
{
lean_object* v___x_53_; 
v___x_53_ = lean_array_push(v_x_51_, v_x_50_);
return v___x_53_;
}
else
{
lean_dec(v_x_50_);
return v_x_51_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_insert(lean_object* v_00_u03b1_54_, lean_object* v_inst_55_, lean_object* v_x_56_, lean_object* v_x_57_){
_start:
{
lean_object* v___x_58_; 
v___x_58_ = lp_aesop_Aesop_UnorderedArraySet_insert___redArg(v_inst_55_, v_x_56_, v_x_57_);
return v___x_58_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_ofDeduplicatedArray___redArg(lean_object* v_xs_59_){
_start:
{
lean_inc_ref(v_xs_59_);
return v_xs_59_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_ofDeduplicatedArray___redArg___boxed(lean_object* v_xs_60_){
_start:
{
lean_object* v_res_61_; 
v_res_61_ = lp_aesop_Aesop_UnorderedArraySet_ofDeduplicatedArray___redArg(v_xs_60_);
lean_dec_ref(v_xs_60_);
return v_res_61_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_ofDeduplicatedArray(lean_object* v_00_u03b1_62_, lean_object* v_inst_63_, lean_object* v_xs_64_){
_start:
{
lean_inc_ref(v_xs_64_);
return v_xs_64_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_ofDeduplicatedArray___boxed(lean_object* v_00_u03b1_65_, lean_object* v_inst_66_, lean_object* v_xs_67_){
_start:
{
lean_object* v_res_68_; 
v_res_68_ = lp_aesop_Aesop_UnorderedArraySet_ofDeduplicatedArray(v_00_u03b1_65_, v_inst_66_, v_xs_67_);
lean_dec_ref(v_xs_67_);
lean_dec_ref(v_inst_66_);
return v_res_68_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_ofSortedArray___redArg(lean_object* v_inst_69_, lean_object* v_xs_70_){
_start:
{
lean_object* v___x_71_; 
v___x_71_ = lp_batteries_Array_dedupSorted___redArg(v_inst_69_, v_xs_70_);
return v___x_71_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_ofSortedArray___redArg___boxed(lean_object* v_inst_72_, lean_object* v_xs_73_){
_start:
{
lean_object* v_res_74_; 
v_res_74_ = lp_aesop_Aesop_UnorderedArraySet_ofSortedArray___redArg(v_inst_72_, v_xs_73_);
lean_dec_ref(v_xs_73_);
return v_res_74_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_ofSortedArray(lean_object* v_00_u03b1_75_, lean_object* v_inst_76_, lean_object* v_xs_77_){
_start:
{
lean_object* v___x_78_; 
v___x_78_ = lp_batteries_Array_dedupSorted___redArg(v_inst_76_, v_xs_77_);
return v___x_78_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_ofSortedArray___boxed(lean_object* v_00_u03b1_79_, lean_object* v_inst_80_, lean_object* v_xs_81_){
_start:
{
lean_object* v_res_82_; 
v_res_82_ = lp_aesop_Aesop_UnorderedArraySet_ofSortedArray(v_00_u03b1_79_, v_inst_80_, v_xs_81_);
lean_dec_ref(v_xs_81_);
return v_res_82_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_ofArray___redArg(lean_object* v_ord_83_, lean_object* v_xs_84_){
_start:
{
lean_object* v___x_85_; 
v___x_85_ = lp_batteries_Array_sortDedup___redArg(v_ord_83_, v_xs_84_);
return v___x_85_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_ofArray(lean_object* v_00_u03b1_86_, lean_object* v_inst_87_, lean_object* v_ord_88_, lean_object* v_xs_89_){
_start:
{
lean_object* v___x_90_; 
v___x_90_ = lp_batteries_Array_sortDedup___redArg(v_ord_88_, v_xs_89_);
return v___x_90_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_ofArray___boxed(lean_object* v_00_u03b1_91_, lean_object* v_inst_92_, lean_object* v_ord_93_, lean_object* v_xs_94_){
_start:
{
lean_object* v_res_95_; 
v_res_95_ = lp_aesop_Aesop_UnorderedArraySet_ofArray(v_00_u03b1_91_, v_inst_92_, v_ord_93_, v_xs_94_);
lean_dec_ref(v_inst_92_);
return v_res_95_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_ofArraySlow___redArg___lam__0(lean_object* v_inst_96_, lean_object* v_x1_97_, lean_object* v_x2_98_){
_start:
{
lean_object* v___x_99_; 
v___x_99_ = lp_aesop_Aesop_UnorderedArraySet_insert___redArg(v_inst_96_, v_x2_98_, v_x1_97_);
return v___x_99_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_ofArraySlow___redArg(lean_object* v_inst_119_, lean_object* v_xs_120_){
_start:
{
lean_object* v___x_121_; lean_object* v___x_122_; lean_object* v___x_123_; lean_object* v___x_124_; uint8_t v___x_125_; 
v___x_121_ = lp_aesop_Aesop_UnorderedArraySet_empty(lean_box(0), v_inst_119_);
v___x_122_ = lean_unsigned_to_nat(0u);
v___x_123_ = lean_array_get_size(v_xs_120_);
v___x_124_ = ((lean_object*)(lp_aesop_Aesop_UnorderedArraySet_ofArraySlow___redArg___closed__9));
v___x_125_ = lean_nat_dec_lt(v___x_122_, v___x_123_);
if (v___x_125_ == 0)
{
lean_dec_ref(v_xs_120_);
lean_dec_ref(v_inst_119_);
return v___x_121_;
}
else
{
lean_object* v___f_126_; uint8_t v___x_127_; 
v___f_126_ = lean_alloc_closure((void*)(lp_aesop_Aesop_UnorderedArraySet_ofArraySlow___redArg___lam__0), 3, 1);
lean_closure_set(v___f_126_, 0, v_inst_119_);
v___x_127_ = lean_nat_dec_le(v___x_123_, v___x_123_);
if (v___x_127_ == 0)
{
if (v___x_125_ == 0)
{
lean_dec_ref(v___f_126_);
lean_dec_ref(v_xs_120_);
return v___x_121_;
}
else
{
size_t v___x_128_; size_t v___x_129_; lean_object* v___x_130_; 
v___x_128_ = ((size_t)0ULL);
v___x_129_ = lean_usize_of_nat(v___x_123_);
v___x_130_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_124_, v___f_126_, v_xs_120_, v___x_128_, v___x_129_, v___x_121_);
return v___x_130_;
}
}
else
{
size_t v___x_131_; size_t v___x_132_; lean_object* v___x_133_; 
v___x_131_ = ((size_t)0ULL);
v___x_132_ = lean_usize_of_nat(v___x_123_);
v___x_133_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_124_, v___f_126_, v_xs_120_, v___x_131_, v___x_132_, v___x_121_);
return v___x_133_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_ofArraySlow(lean_object* v_00_u03b1_134_, lean_object* v_inst_135_, lean_object* v_xs_136_){
_start:
{
lean_object* v___x_137_; 
v___x_137_ = lp_aesop_Aesop_UnorderedArraySet_ofArraySlow___redArg(v_inst_135_, v_xs_136_);
return v___x_137_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_ofHashSet___redArg___lam__0(lean_object* v_x1_138_, lean_object* v_x2_139_, lean_object* v_x3_140_){
_start:
{
lean_object* v___x_141_; 
v___x_141_ = lean_array_push(v_x1_138_, v_x2_139_);
return v___x_141_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_ofHashSet___redArg___lam__1(lean_object* v___x_142_, lean_object* v___f_143_, lean_object* v_acc_144_, lean_object* v_l_145_){
_start:
{
lean_object* v___x_146_; 
v___x_146_ = l_Std_DHashMap_Internal_AssocList_foldlM___redArg(v___x_142_, v___f_143_, v_acc_144_, v_l_145_);
return v___x_146_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_ofHashSet___redArg(lean_object* v_xs_151_){
_start:
{
lean_object* v_size_152_; lean_object* v_buckets_153_; lean_object* v___x_154_; lean_object* v___x_155_; lean_object* v___x_156_; lean_object* v___x_157_; uint8_t v___x_158_; 
v_size_152_ = lean_ctor_get(v_xs_151_, 0);
lean_inc(v_size_152_);
v_buckets_153_ = lean_ctor_get(v_xs_151_, 1);
lean_inc_ref(v_buckets_153_);
lean_dec_ref(v_xs_151_);
v___x_154_ = lean_mk_empty_array_with_capacity(v_size_152_);
lean_dec(v_size_152_);
v___x_155_ = ((lean_object*)(lp_aesop_Aesop_UnorderedArraySet_ofArraySlow___redArg___closed__9));
v___x_156_ = lean_unsigned_to_nat(0u);
v___x_157_ = lean_array_get_size(v_buckets_153_);
v___x_158_ = lean_nat_dec_lt(v___x_156_, v___x_157_);
if (v___x_158_ == 0)
{
lean_dec_ref(v_buckets_153_);
return v___x_154_;
}
else
{
lean_object* v___f_159_; uint8_t v___x_160_; 
v___f_159_ = ((lean_object*)(lp_aesop_Aesop_UnorderedArraySet_ofHashSet___redArg___closed__1));
v___x_160_ = lean_nat_dec_le(v___x_157_, v___x_157_);
if (v___x_160_ == 0)
{
if (v___x_158_ == 0)
{
lean_dec_ref(v_buckets_153_);
return v___x_154_;
}
else
{
size_t v___x_161_; size_t v___x_162_; lean_object* v___x_163_; 
v___x_161_ = ((size_t)0ULL);
v___x_162_ = lean_usize_of_nat(v___x_157_);
v___x_163_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_155_, v___f_159_, v_buckets_153_, v___x_161_, v___x_162_, v___x_154_);
return v___x_163_;
}
}
else
{
size_t v___x_164_; size_t v___x_165_; lean_object* v___x_166_; 
v___x_164_ = ((size_t)0ULL);
v___x_165_ = lean_usize_of_nat(v___x_157_);
v___x_166_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_155_, v___f_159_, v_buckets_153_, v___x_164_, v___x_165_, v___x_154_);
return v___x_166_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_ofHashSet(lean_object* v_00_u03b1_167_, lean_object* v_inst_168_, lean_object* v_inst_169_, lean_object* v_xs_170_){
_start:
{
lean_object* v___x_171_; 
v___x_171_ = lp_aesop_Aesop_UnorderedArraySet_ofHashSet___redArg(v_xs_170_);
return v___x_171_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_ofHashSet___boxed(lean_object* v_00_u03b1_172_, lean_object* v_inst_173_, lean_object* v_inst_174_, lean_object* v_xs_175_){
_start:
{
lean_object* v_res_176_; 
v_res_176_ = lp_aesop_Aesop_UnorderedArraySet_ofHashSet(v_00_u03b1_172_, v_inst_173_, v_inst_174_, v_xs_175_);
lean_dec_ref(v_inst_174_);
lean_dec_ref(v_inst_173_);
return v_res_176_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_ofPersistentHashSet___redArg___lam__0(lean_object* v_d_177_, lean_object* v_a_178_, lean_object* v_x_179_){
_start:
{
lean_object* v___x_180_; 
v___x_180_ = lean_array_push(v_d_177_, v_a_178_);
return v___x_180_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_ofPersistentHashSet___redArg(lean_object* v_xs_182_){
_start:
{
lean_object* v___f_183_; lean_object* v___x_184_; lean_object* v___x_185_; lean_object* v___x_186_; 
v___f_183_ = ((lean_object*)(lp_aesop_Aesop_UnorderedArraySet_ofPersistentHashSet___redArg___closed__0));
v___x_184_ = ((lean_object*)(lp_aesop_Aesop_UnorderedArraySet_empty___closed__0));
v___x_185_ = ((lean_object*)(lp_aesop_Aesop_UnorderedArraySet_ofArraySlow___redArg___closed__9));
v___x_186_ = l_Lean_PersistentHashMap_foldlMAux___redArg(v___x_185_, v___f_183_, v_xs_182_, v___x_184_);
return v___x_186_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_ofPersistentHashSet(lean_object* v_00_u03b1_187_, lean_object* v_inst_188_, lean_object* v_inst_189_, lean_object* v_xs_190_){
_start:
{
lean_object* v___x_191_; 
v___x_191_ = lp_aesop_Aesop_UnorderedArraySet_ofPersistentHashSet___redArg(v_xs_190_);
return v___x_191_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_ofPersistentHashSet___boxed(lean_object* v_00_u03b1_192_, lean_object* v_inst_193_, lean_object* v_inst_194_, lean_object* v_xs_195_){
_start:
{
lean_object* v_res_196_; 
v_res_196_ = lp_aesop_Aesop_UnorderedArraySet_ofPersistentHashSet(v_00_u03b1_192_, v_inst_193_, v_inst_194_, v_xs_195_);
lean_dec_ref(v_inst_194_);
lean_dec_ref(v_inst_193_);
return v_res_196_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_toArray___redArg(lean_object* v_s_197_){
_start:
{
lean_inc_ref(v_s_197_);
return v_s_197_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_toArray___redArg___boxed(lean_object* v_s_198_){
_start:
{
lean_object* v_res_199_; 
v_res_199_ = lp_aesop_Aesop_UnorderedArraySet_toArray___redArg(v_s_198_);
lean_dec_ref(v_s_198_);
return v_res_199_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_toArray(lean_object* v_00_u03b1_200_, lean_object* v_inst_201_, lean_object* v_s_202_){
_start:
{
lean_inc_ref(v_s_202_);
return v_s_202_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_toArray___boxed(lean_object* v_00_u03b1_203_, lean_object* v_inst_204_, lean_object* v_s_205_){
_start:
{
lean_object* v_res_206_; 
v_res_206_ = lp_aesop_Aesop_UnorderedArraySet_toArray(v_00_u03b1_203_, v_inst_204_, v_s_205_);
lean_dec_ref(v_s_205_);
lean_dec_ref(v_inst_204_);
return v_res_206_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_erase___redArg(lean_object* v_inst_207_, lean_object* v_x_208_, lean_object* v_s_209_){
_start:
{
lean_object* v___x_210_; 
v___x_210_ = l_Array_erase___redArg(v_inst_207_, v_s_209_, v_x_208_);
return v___x_210_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_erase(lean_object* v_00_u03b1_211_, lean_object* v_inst_212_, lean_object* v_x_213_, lean_object* v_s_214_){
_start:
{
lean_object* v___x_215_; 
v___x_215_ = l_Array_erase___redArg(v_inst_212_, v_s_214_, v_x_213_);
return v___x_215_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_filterM___redArg___lam__0(lean_object* v_toPure_216_, lean_object* v_____do__lift_217_){
_start:
{
lean_object* v___x_218_; 
v___x_218_ = lean_apply_2(v_toPure_216_, lean_box(0), v_____do__lift_217_);
return v___x_218_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_filterM___redArg___lam__1(lean_object* v_toApplicative_219_, lean_object* v_acc_220_, lean_object* v_a_221_, uint8_t v_____do__lift_222_){
_start:
{
if (v_____do__lift_222_ == 0)
{
lean_object* v_toPure_223_; lean_object* v___x_224_; 
lean_dec(v_a_221_);
v_toPure_223_ = lean_ctor_get(v_toApplicative_219_, 1);
lean_inc(v_toPure_223_);
lean_dec_ref(v_toApplicative_219_);
v___x_224_ = lean_apply_2(v_toPure_223_, lean_box(0), v_acc_220_);
return v___x_224_;
}
else
{
lean_object* v_toPure_225_; lean_object* v___x_226_; lean_object* v___x_227_; 
v_toPure_225_ = lean_ctor_get(v_toApplicative_219_, 1);
lean_inc(v_toPure_225_);
lean_dec_ref(v_toApplicative_219_);
v___x_226_ = lean_array_push(v_acc_220_, v_a_221_);
v___x_227_ = lean_apply_2(v_toPure_225_, lean_box(0), v___x_226_);
return v___x_227_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_filterM___redArg___lam__1___boxed(lean_object* v_toApplicative_228_, lean_object* v_acc_229_, lean_object* v_a_230_, lean_object* v_____do__lift_231_){
_start:
{
uint8_t v_____do__lift_81__boxed_232_; lean_object* v_res_233_; 
v_____do__lift_81__boxed_232_ = lean_unbox(v_____do__lift_231_);
v_res_233_ = lp_aesop_Aesop_UnorderedArraySet_filterM___redArg___lam__1(v_toApplicative_228_, v_acc_229_, v_a_230_, v_____do__lift_81__boxed_232_);
return v_res_233_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_filterM___redArg___lam__2(lean_object* v_toApplicative_234_, lean_object* v_p_235_, lean_object* v_toBind_236_, lean_object* v_acc_237_, lean_object* v_a_238_){
_start:
{
lean_object* v___f_239_; lean_object* v___x_240_; lean_object* v___x_241_; 
lean_inc(v_a_238_);
v___f_239_ = lean_alloc_closure((void*)(lp_aesop_Aesop_UnorderedArraySet_filterM___redArg___lam__1___boxed), 4, 3);
lean_closure_set(v___f_239_, 0, v_toApplicative_234_);
lean_closure_set(v___f_239_, 1, v_acc_237_);
lean_closure_set(v___f_239_, 2, v_a_238_);
v___x_240_ = lean_apply_1(v_p_235_, v_a_238_);
v___x_241_ = lean_apply_4(v_toBind_236_, lean_box(0), lean_box(0), v___x_240_, v___f_239_);
return v___x_241_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_filterM___redArg(lean_object* v_inst_242_, lean_object* v_p_243_, lean_object* v_s_244_){
_start:
{
lean_object* v_toApplicative_245_; lean_object* v_toBind_246_; lean_object* v___y_248_; lean_object* v___x_252_; lean_object* v___x_253_; lean_object* v___x_254_; uint8_t v___x_255_; 
v_toApplicative_245_ = lean_ctor_get(v_inst_242_, 0);
lean_inc_ref(v_toApplicative_245_);
v_toBind_246_ = lean_ctor_get(v_inst_242_, 1);
lean_inc(v_toBind_246_);
v___x_252_ = lean_unsigned_to_nat(0u);
v___x_253_ = lean_array_get_size(v_s_244_);
v___x_254_ = ((lean_object*)(lp_aesop_Aesop_UnorderedArraySet_empty___closed__0));
v___x_255_ = lean_nat_dec_lt(v___x_252_, v___x_253_);
if (v___x_255_ == 0)
{
lean_object* v_toPure_256_; lean_object* v___x_257_; 
lean_dec_ref(v_s_244_);
lean_dec(v_p_243_);
lean_dec_ref(v_inst_242_);
v_toPure_256_ = lean_ctor_get(v_toApplicative_245_, 1);
lean_inc(v_toPure_256_);
v___x_257_ = lean_apply_2(v_toPure_256_, lean_box(0), v___x_254_);
v___y_248_ = v___x_257_;
goto v___jp_247_;
}
else
{
lean_object* v___f_258_; uint8_t v___x_259_; 
lean_inc(v_toBind_246_);
lean_inc_ref(v_toApplicative_245_);
v___f_258_ = lean_alloc_closure((void*)(lp_aesop_Aesop_UnorderedArraySet_filterM___redArg___lam__2), 5, 3);
lean_closure_set(v___f_258_, 0, v_toApplicative_245_);
lean_closure_set(v___f_258_, 1, v_p_243_);
lean_closure_set(v___f_258_, 2, v_toBind_246_);
v___x_259_ = lean_nat_dec_le(v___x_253_, v___x_253_);
if (v___x_259_ == 0)
{
if (v___x_255_ == 0)
{
lean_object* v_toPure_260_; lean_object* v___x_261_; 
lean_dec_ref(v___f_258_);
lean_dec_ref(v_s_244_);
lean_dec_ref(v_inst_242_);
v_toPure_260_ = lean_ctor_get(v_toApplicative_245_, 1);
lean_inc(v_toPure_260_);
v___x_261_ = lean_apply_2(v_toPure_260_, lean_box(0), v___x_254_);
v___y_248_ = v___x_261_;
goto v___jp_247_;
}
else
{
size_t v___x_262_; size_t v___x_263_; lean_object* v___x_264_; 
v___x_262_ = ((size_t)0ULL);
v___x_263_ = lean_usize_of_nat(v___x_253_);
v___x_264_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v_inst_242_, v___f_258_, v_s_244_, v___x_262_, v___x_263_, v___x_254_);
v___y_248_ = v___x_264_;
goto v___jp_247_;
}
}
else
{
size_t v___x_265_; size_t v___x_266_; lean_object* v___x_267_; 
v___x_265_ = ((size_t)0ULL);
v___x_266_ = lean_usize_of_nat(v___x_253_);
v___x_267_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v_inst_242_, v___f_258_, v_s_244_, v___x_265_, v___x_266_, v___x_254_);
v___y_248_ = v___x_267_;
goto v___jp_247_;
}
}
v___jp_247_:
{
lean_object* v_toPure_249_; lean_object* v___f_250_; lean_object* v___x_251_; 
v_toPure_249_ = lean_ctor_get(v_toApplicative_245_, 1);
lean_inc(v_toPure_249_);
lean_dec_ref(v_toApplicative_245_);
v___f_250_ = lean_alloc_closure((void*)(lp_aesop_Aesop_UnorderedArraySet_filterM___redArg___lam__0), 2, 1);
lean_closure_set(v___f_250_, 0, v_toPure_249_);
v___x_251_ = lean_apply_4(v_toBind_246_, lean_box(0), lean_box(0), v___y_248_, v___f_250_);
return v___x_251_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_filterM(lean_object* v_00_u03b1_268_, lean_object* v_inst_269_, lean_object* v_m_270_, lean_object* v_inst_271_, lean_object* v_p_272_, lean_object* v_s_273_){
_start:
{
lean_object* v___x_274_; 
v___x_274_ = lp_aesop_Aesop_UnorderedArraySet_filterM___redArg(v_inst_271_, v_p_272_, v_s_273_);
return v___x_274_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_filterM___boxed(lean_object* v_00_u03b1_275_, lean_object* v_inst_276_, lean_object* v_m_277_, lean_object* v_inst_278_, lean_object* v_p_279_, lean_object* v_s_280_){
_start:
{
lean_object* v_res_281_; 
v_res_281_ = lp_aesop_Aesop_UnorderedArraySet_filterM(v_00_u03b1_275_, v_inst_276_, v_m_277_, v_inst_278_, v_p_279_, v_s_280_);
lean_dec_ref(v_inst_276_);
return v_res_281_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_filter___redArg___lam__0(lean_object* v_p_282_, lean_object* v_x1_283_, lean_object* v_x2_284_){
_start:
{
lean_object* v___x_285_; uint8_t v___x_286_; 
lean_inc(v_x2_284_);
v___x_285_ = lean_apply_1(v_p_282_, v_x2_284_);
v___x_286_ = lean_unbox(v___x_285_);
if (v___x_286_ == 0)
{
lean_dec(v_x2_284_);
return v_x1_283_;
}
else
{
lean_object* v___x_287_; 
v___x_287_ = lean_array_push(v_x1_283_, v_x2_284_);
return v___x_287_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_filter___redArg(lean_object* v_p_288_, lean_object* v_s_289_){
_start:
{
lean_object* v___x_290_; lean_object* v___x_291_; lean_object* v___x_292_; lean_object* v___x_293_; uint8_t v___x_294_; 
v___x_290_ = lean_unsigned_to_nat(0u);
v___x_291_ = lean_array_get_size(v_s_289_);
v___x_292_ = ((lean_object*)(lp_aesop_Aesop_UnorderedArraySet_empty___closed__0));
v___x_293_ = ((lean_object*)(lp_aesop_Aesop_UnorderedArraySet_ofArraySlow___redArg___closed__9));
v___x_294_ = lean_nat_dec_lt(v___x_290_, v___x_291_);
if (v___x_294_ == 0)
{
lean_dec_ref(v_s_289_);
lean_dec_ref(v_p_288_);
return v___x_292_;
}
else
{
lean_object* v___f_295_; uint8_t v___x_296_; 
v___f_295_ = lean_alloc_closure((void*)(lp_aesop_Aesop_UnorderedArraySet_filter___redArg___lam__0), 3, 1);
lean_closure_set(v___f_295_, 0, v_p_288_);
v___x_296_ = lean_nat_dec_le(v___x_291_, v___x_291_);
if (v___x_296_ == 0)
{
if (v___x_294_ == 0)
{
lean_dec_ref(v___f_295_);
lean_dec_ref(v_s_289_);
return v___x_292_;
}
else
{
size_t v___x_297_; size_t v___x_298_; lean_object* v___x_299_; 
v___x_297_ = ((size_t)0ULL);
v___x_298_ = lean_usize_of_nat(v___x_291_);
v___x_299_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_293_, v___f_295_, v_s_289_, v___x_297_, v___x_298_, v___x_292_);
return v___x_299_;
}
}
else
{
size_t v___x_300_; size_t v___x_301_; lean_object* v___x_302_; 
v___x_300_ = ((size_t)0ULL);
v___x_301_ = lean_usize_of_nat(v___x_291_);
v___x_302_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_293_, v___f_295_, v_s_289_, v___x_300_, v___x_301_, v___x_292_);
return v___x_302_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_filter(lean_object* v_00_u03b1_303_, lean_object* v_inst_304_, lean_object* v_p_305_, lean_object* v_s_306_){
_start:
{
lean_object* v___x_307_; 
v___x_307_ = lp_aesop_Aesop_UnorderedArraySet_filter___redArg(v_p_305_, v_s_306_);
return v___x_307_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_filter___boxed(lean_object* v_00_u03b1_308_, lean_object* v_inst_309_, lean_object* v_p_310_, lean_object* v_s_311_){
_start:
{
lean_object* v_res_312_; 
v_res_312_ = lp_aesop_Aesop_UnorderedArraySet_filter(v_00_u03b1_308_, v_inst_309_, v_p_310_, v_s_311_);
lean_dec_ref(v_inst_309_);
return v_res_312_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_merge___redArg(lean_object* v_inst_313_, lean_object* v_s_314_, lean_object* v_t_315_){
_start:
{
lean_object* v___x_316_; 
v___x_316_ = lp_batteries_Array_mergeUnsortedDedup___redArg(v_inst_313_, v_s_314_, v_t_315_);
return v___x_316_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_merge(lean_object* v_00_u03b1_317_, lean_object* v_inst_318_, lean_object* v_s_319_, lean_object* v_t_320_){
_start:
{
lean_object* v___x_321_; 
v___x_321_ = lp_batteries_Array_mergeUnsortedDedup___redArg(v_inst_318_, v_s_319_, v_t_320_);
return v___x_321_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_instAppend___redArg(lean_object* v_inst_322_){
_start:
{
lean_object* v___x_323_; 
v___x_323_ = lean_alloc_closure((void*)(lp_aesop_Aesop_UnorderedArraySet_merge), 4, 2);
lean_closure_set(v___x_323_, 0, lean_box(0));
lean_closure_set(v___x_323_, 1, v_inst_322_);
return v___x_323_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_instAppend(lean_object* v_00_u03b1_324_, lean_object* v_inst_325_){
_start:
{
lean_object* v___x_326_; 
v___x_326_ = lean_alloc_closure((void*)(lp_aesop_Aesop_UnorderedArraySet_merge), 4, 2);
lean_closure_set(v___x_326_, 0, lean_box(0));
lean_closure_set(v___x_326_, 1, v_inst_325_);
return v___x_326_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_UnorderedArraySet_contains___redArg(lean_object* v_inst_327_, lean_object* v_x_328_, lean_object* v_s_329_){
_start:
{
uint8_t v___x_330_; 
v___x_330_ = l_Array_contains___redArg(v_inst_327_, v_s_329_, v_x_328_);
return v___x_330_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_contains___redArg___boxed(lean_object* v_inst_331_, lean_object* v_x_332_, lean_object* v_s_333_){
_start:
{
uint8_t v_res_334_; lean_object* v_r_335_; 
v_res_334_ = lp_aesop_Aesop_UnorderedArraySet_contains___redArg(v_inst_331_, v_x_332_, v_s_333_);
v_r_335_ = lean_box(v_res_334_);
return v_r_335_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_UnorderedArraySet_contains(lean_object* v_00_u03b1_336_, lean_object* v_inst_337_, lean_object* v_x_338_, lean_object* v_s_339_){
_start:
{
uint8_t v___x_340_; 
v___x_340_ = l_Array_contains___redArg(v_inst_337_, v_s_339_, v_x_338_);
return v___x_340_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_contains___boxed(lean_object* v_00_u03b1_341_, lean_object* v_inst_342_, lean_object* v_x_343_, lean_object* v_s_344_){
_start:
{
uint8_t v_res_345_; lean_object* v_r_346_; 
v_res_345_ = lp_aesop_Aesop_UnorderedArraySet_contains(v_00_u03b1_341_, v_inst_342_, v_x_343_, v_s_344_);
v_r_346_ = lean_box(v_res_345_);
return v_r_346_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_foldM___redArg(lean_object* v_inst_347_, lean_object* v_f_348_, lean_object* v_init_349_, lean_object* v_s_350_){
_start:
{
lean_object* v___x_351_; lean_object* v___x_352_; uint8_t v___x_353_; 
v___x_351_ = lean_unsigned_to_nat(0u);
v___x_352_ = lean_array_get_size(v_s_350_);
v___x_353_ = lean_nat_dec_lt(v___x_351_, v___x_352_);
if (v___x_353_ == 0)
{
lean_object* v_toApplicative_354_; lean_object* v_toPure_355_; lean_object* v___x_356_; 
lean_dec_ref(v_s_350_);
lean_dec(v_f_348_);
v_toApplicative_354_ = lean_ctor_get(v_inst_347_, 0);
lean_inc_ref(v_toApplicative_354_);
lean_dec_ref(v_inst_347_);
v_toPure_355_ = lean_ctor_get(v_toApplicative_354_, 1);
lean_inc(v_toPure_355_);
lean_dec_ref(v_toApplicative_354_);
v___x_356_ = lean_apply_2(v_toPure_355_, lean_box(0), v_init_349_);
return v___x_356_;
}
else
{
uint8_t v___x_357_; 
v___x_357_ = lean_nat_dec_le(v___x_352_, v___x_352_);
if (v___x_357_ == 0)
{
if (v___x_353_ == 0)
{
lean_object* v_toApplicative_358_; lean_object* v_toPure_359_; lean_object* v___x_360_; 
lean_dec_ref(v_s_350_);
lean_dec(v_f_348_);
v_toApplicative_358_ = lean_ctor_get(v_inst_347_, 0);
lean_inc_ref(v_toApplicative_358_);
lean_dec_ref(v_inst_347_);
v_toPure_359_ = lean_ctor_get(v_toApplicative_358_, 1);
lean_inc(v_toPure_359_);
lean_dec_ref(v_toApplicative_358_);
v___x_360_ = lean_apply_2(v_toPure_359_, lean_box(0), v_init_349_);
return v___x_360_;
}
else
{
size_t v___x_361_; size_t v___x_362_; lean_object* v___x_363_; 
v___x_361_ = ((size_t)0ULL);
v___x_362_ = lean_usize_of_nat(v___x_352_);
v___x_363_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v_inst_347_, v_f_348_, v_s_350_, v___x_361_, v___x_362_, v_init_349_);
return v___x_363_;
}
}
else
{
size_t v___x_364_; size_t v___x_365_; lean_object* v___x_366_; 
v___x_364_ = ((size_t)0ULL);
v___x_365_ = lean_usize_of_nat(v___x_352_);
v___x_366_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v_inst_347_, v_f_348_, v_s_350_, v___x_364_, v___x_365_, v_init_349_);
return v___x_366_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_foldM(lean_object* v_00_u03b1_367_, lean_object* v_inst_368_, lean_object* v_m_369_, lean_object* v_00_u03c3_370_, lean_object* v_inst_371_, lean_object* v_f_372_, lean_object* v_init_373_, lean_object* v_s_374_){
_start:
{
lean_object* v___x_375_; 
v___x_375_ = lp_aesop_Aesop_UnorderedArraySet_foldM___redArg(v_inst_371_, v_f_372_, v_init_373_, v_s_374_);
return v___x_375_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_foldM___boxed(lean_object* v_00_u03b1_376_, lean_object* v_inst_377_, lean_object* v_m_378_, lean_object* v_00_u03c3_379_, lean_object* v_inst_380_, lean_object* v_f_381_, lean_object* v_init_382_, lean_object* v_s_383_){
_start:
{
lean_object* v_res_384_; 
v_res_384_ = lp_aesop_Aesop_UnorderedArraySet_foldM(v_00_u03b1_376_, v_inst_377_, v_m_378_, v_00_u03c3_379_, v_inst_380_, v_f_381_, v_init_382_, v_s_383_);
lean_dec_ref(v_inst_377_);
return v_res_384_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_instForInOfMonad___private__1___redArg___lam__0(lean_object* v_f_385_, lean_object* v_a_386_, lean_object* v_x_387_, lean_object* v___y_388_){
_start:
{
lean_object* v___x_389_; 
v___x_389_ = lean_apply_2(v_f_385_, v_a_386_, v___y_388_);
return v___x_389_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_instForInOfMonad___private__1___redArg(lean_object* v_inst_390_, lean_object* v_s_391_, lean_object* v_b_392_, lean_object* v_f_393_){
_start:
{
lean_object* v___f_394_; size_t v_sz_395_; size_t v___x_396_; lean_object* v___x_397_; 
v___f_394_ = lean_alloc_closure((void*)(lp_aesop_Aesop_UnorderedArraySet_instForInOfMonad___private__1___redArg___lam__0), 4, 1);
lean_closure_set(v___f_394_, 0, v_f_393_);
v_sz_395_ = lean_array_size(v_s_391_);
v___x_396_ = ((size_t)0ULL);
v___x_397_ = l___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop(lean_box(0), lean_box(0), lean_box(0), v_inst_390_, v_s_391_, v___f_394_, v_sz_395_, v___x_396_, v_b_392_);
return v___x_397_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_instForInOfMonad___private__1(lean_object* v_00_u03b1_398_, lean_object* v_inst_399_, lean_object* v_m_400_, lean_object* v_inst_401_, lean_object* v_00_u03b2_402_, lean_object* v_s_403_, lean_object* v_b_404_, lean_object* v_f_405_){
_start:
{
lean_object* v___f_406_; size_t v_sz_407_; size_t v___x_408_; lean_object* v___x_409_; 
v___f_406_ = lean_alloc_closure((void*)(lp_aesop_Aesop_UnorderedArraySet_instForInOfMonad___private__1___redArg___lam__0), 4, 1);
lean_closure_set(v___f_406_, 0, v_f_405_);
v_sz_407_ = lean_array_size(v_s_403_);
v___x_408_ = ((size_t)0ULL);
v___x_409_ = l___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop(lean_box(0), lean_box(0), lean_box(0), v_inst_401_, v_s_403_, v___f_406_, v_sz_407_, v___x_408_, v_b_404_);
return v___x_409_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_instForInOfMonad___private__1___boxed(lean_object* v_00_u03b1_410_, lean_object* v_inst_411_, lean_object* v_m_412_, lean_object* v_inst_413_, lean_object* v_00_u03b2_414_, lean_object* v_s_415_, lean_object* v_b_416_, lean_object* v_f_417_){
_start:
{
lean_object* v_res_418_; 
v_res_418_ = lp_aesop_Aesop_UnorderedArraySet_instForInOfMonad___private__1(v_00_u03b1_410_, v_inst_411_, v_m_412_, v_inst_413_, v_00_u03b2_414_, v_s_415_, v_b_416_, v_f_417_);
lean_dec_ref(v_inst_411_);
return v_res_418_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_instForInOfMonad___redArg___lam__0(lean_object* v___y_419_, lean_object* v_a_420_, lean_object* v_x_421_, lean_object* v___y_422_){
_start:
{
lean_object* v___x_423_; 
v___x_423_ = lean_apply_2(v___y_419_, v_a_420_, v___y_422_);
return v___x_423_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_instForInOfMonad___redArg___lam__1(lean_object* v_inst_424_, lean_object* v_00_u03b2_425_, lean_object* v_s_426_, lean_object* v___y_427_, lean_object* v___y_428_){
_start:
{
lean_object* v___f_429_; size_t v_sz_430_; size_t v___x_431_; lean_object* v___x_432_; 
v___f_429_ = lean_alloc_closure((void*)(lp_aesop_Aesop_UnorderedArraySet_instForInOfMonad___redArg___lam__0), 4, 1);
lean_closure_set(v___f_429_, 0, v___y_428_);
v_sz_430_ = lean_array_size(v_s_426_);
v___x_431_ = ((size_t)0ULL);
v___x_432_ = l___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop(lean_box(0), lean_box(0), lean_box(0), v_inst_424_, v_s_426_, v___f_429_, v_sz_430_, v___x_431_, v___y_427_);
return v___x_432_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_instForInOfMonad___redArg(lean_object* v_inst_433_){
_start:
{
lean_object* v___f_434_; 
v___f_434_ = lean_alloc_closure((void*)(lp_aesop_Aesop_UnorderedArraySet_instForInOfMonad___redArg___lam__1), 5, 1);
lean_closure_set(v___f_434_, 0, v_inst_433_);
return v___f_434_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_instForInOfMonad(lean_object* v_00_u03b1_435_, lean_object* v_inst_436_, lean_object* v_m_437_, lean_object* v_inst_438_){
_start:
{
lean_object* v___f_439_; 
v___f_439_ = lean_alloc_closure((void*)(lp_aesop_Aesop_UnorderedArraySet_instForInOfMonad___redArg___lam__1), 5, 1);
lean_closure_set(v___f_439_, 0, v_inst_438_);
return v___f_439_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_instForInOfMonad___boxed(lean_object* v_00_u03b1_440_, lean_object* v_inst_441_, lean_object* v_m_442_, lean_object* v_inst_443_){
_start:
{
lean_object* v_res_444_; 
v_res_444_ = lp_aesop_Aesop_UnorderedArraySet_instForInOfMonad(v_00_u03b1_440_, v_inst_441_, v_m_442_, v_inst_443_);
lean_dec_ref(v_inst_441_);
return v_res_444_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_fold___redArg___lam__0(lean_object* v_f_445_, lean_object* v_x1_446_, lean_object* v_x2_447_){
_start:
{
lean_object* v___x_448_; 
v___x_448_ = lean_apply_2(v_f_445_, v_x1_446_, v_x2_447_);
return v___x_448_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_fold___redArg(lean_object* v_f_449_, lean_object* v_init_450_, lean_object* v_s_451_){
_start:
{
lean_object* v___x_452_; lean_object* v___x_453_; lean_object* v___x_454_; uint8_t v___x_455_; 
v___x_452_ = lean_unsigned_to_nat(0u);
v___x_453_ = lean_array_get_size(v_s_451_);
v___x_454_ = ((lean_object*)(lp_aesop_Aesop_UnorderedArraySet_ofArraySlow___redArg___closed__9));
v___x_455_ = lean_nat_dec_lt(v___x_452_, v___x_453_);
if (v___x_455_ == 0)
{
lean_dec_ref(v_s_451_);
lean_dec(v_f_449_);
return v_init_450_;
}
else
{
lean_object* v___f_456_; uint8_t v___x_457_; 
v___f_456_ = lean_alloc_closure((void*)(lp_aesop_Aesop_UnorderedArraySet_fold___redArg___lam__0), 3, 1);
lean_closure_set(v___f_456_, 0, v_f_449_);
v___x_457_ = lean_nat_dec_le(v___x_453_, v___x_453_);
if (v___x_457_ == 0)
{
if (v___x_455_ == 0)
{
lean_dec_ref(v___f_456_);
lean_dec_ref(v_s_451_);
return v_init_450_;
}
else
{
size_t v___x_458_; size_t v___x_459_; lean_object* v___x_460_; 
v___x_458_ = ((size_t)0ULL);
v___x_459_ = lean_usize_of_nat(v___x_453_);
v___x_460_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_454_, v___f_456_, v_s_451_, v___x_458_, v___x_459_, v_init_450_);
return v___x_460_;
}
}
else
{
size_t v___x_461_; size_t v___x_462_; lean_object* v___x_463_; 
v___x_461_ = ((size_t)0ULL);
v___x_462_ = lean_usize_of_nat(v___x_453_);
v___x_463_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_454_, v___f_456_, v_s_451_, v___x_461_, v___x_462_, v_init_450_);
return v___x_463_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_fold(lean_object* v_00_u03b1_464_, lean_object* v_inst_465_, lean_object* v_00_u03c3_466_, lean_object* v_f_467_, lean_object* v_init_468_, lean_object* v_s_469_){
_start:
{
lean_object* v___x_470_; 
v___x_470_ = lp_aesop_Aesop_UnorderedArraySet_fold___redArg(v_f_467_, v_init_468_, v_s_469_);
return v___x_470_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_fold___boxed(lean_object* v_00_u03b1_471_, lean_object* v_inst_472_, lean_object* v_00_u03c3_473_, lean_object* v_f_474_, lean_object* v_init_475_, lean_object* v_s_476_){
_start:
{
lean_object* v_res_477_; 
v_res_477_ = lp_aesop_Aesop_UnorderedArraySet_fold(v_00_u03b1_471_, v_inst_472_, v_00_u03c3_473_, v_f_474_, v_init_475_, v_s_476_);
lean_dec_ref(v_inst_472_);
return v_res_477_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_partition___redArg___lam__0(lean_object* v_f_478_, lean_object* v_a_479_, lean_object* v_x_480_, lean_object* v___y_481_){
_start:
{
lean_object* v_fst_482_; lean_object* v_snd_483_; lean_object* v___x_485_; uint8_t v_isShared_486_; uint8_t v_isSharedCheck_499_; 
v_fst_482_ = lean_ctor_get(v___y_481_, 0);
v_snd_483_ = lean_ctor_get(v___y_481_, 1);
v_isSharedCheck_499_ = !lean_is_exclusive(v___y_481_);
if (v_isSharedCheck_499_ == 0)
{
v___x_485_ = v___y_481_;
v_isShared_486_ = v_isSharedCheck_499_;
goto v_resetjp_484_;
}
else
{
lean_inc(v_snd_483_);
lean_inc(v_fst_482_);
lean_dec(v___y_481_);
v___x_485_ = lean_box(0);
v_isShared_486_ = v_isSharedCheck_499_;
goto v_resetjp_484_;
}
v_resetjp_484_:
{
lean_object* v___x_487_; uint8_t v___x_488_; 
lean_inc(v_a_479_);
v___x_487_ = lean_apply_1(v_f_478_, v_a_479_);
v___x_488_ = lean_unbox(v___x_487_);
if (v___x_488_ == 0)
{
lean_object* v___x_489_; lean_object* v___x_491_; 
v___x_489_ = lean_array_push(v_snd_483_, v_a_479_);
if (v_isShared_486_ == 0)
{
lean_ctor_set(v___x_485_, 1, v___x_489_);
v___x_491_ = v___x_485_;
goto v_reusejp_490_;
}
else
{
lean_object* v_reuseFailAlloc_493_; 
v_reuseFailAlloc_493_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_493_, 0, v_fst_482_);
lean_ctor_set(v_reuseFailAlloc_493_, 1, v___x_489_);
v___x_491_ = v_reuseFailAlloc_493_;
goto v_reusejp_490_;
}
v_reusejp_490_:
{
lean_object* v___x_492_; 
v___x_492_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_492_, 0, v___x_491_);
return v___x_492_;
}
}
else
{
lean_object* v___x_494_; lean_object* v___x_496_; 
v___x_494_ = lean_array_push(v_fst_482_, v_a_479_);
if (v_isShared_486_ == 0)
{
lean_ctor_set(v___x_485_, 0, v___x_494_);
v___x_496_ = v___x_485_;
goto v_reusejp_495_;
}
else
{
lean_object* v_reuseFailAlloc_498_; 
v_reuseFailAlloc_498_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_498_, 0, v___x_494_);
lean_ctor_set(v_reuseFailAlloc_498_, 1, v_snd_483_);
v___x_496_ = v_reuseFailAlloc_498_;
goto v_reusejp_495_;
}
v_reusejp_495_:
{
lean_object* v___x_497_; 
v___x_497_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_497_, 0, v___x_496_);
return v___x_497_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_partition___redArg(lean_object* v_f_502_, lean_object* v_s_503_){
_start:
{
lean_object* v___f_504_; lean_object* v___x_505_; lean_object* v___x_506_; size_t v_sz_507_; size_t v___x_508_; lean_object* v___x_509_; lean_object* v_fst_510_; lean_object* v_snd_511_; lean_object* v___x_513_; uint8_t v_isShared_514_; uint8_t v_isSharedCheck_518_; 
v___f_504_ = lean_alloc_closure((void*)(lp_aesop_Aesop_UnorderedArraySet_partition___redArg___lam__0), 4, 1);
lean_closure_set(v___f_504_, 0, v_f_502_);
v___x_505_ = ((lean_object*)(lp_aesop_Aesop_UnorderedArraySet_ofArraySlow___redArg___closed__9));
v___x_506_ = ((lean_object*)(lp_aesop_Aesop_UnorderedArraySet_partition___redArg___closed__0));
v_sz_507_ = lean_array_size(v_s_503_);
v___x_508_ = ((size_t)0ULL);
v___x_509_ = l___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop(lean_box(0), lean_box(0), lean_box(0), v___x_505_, v_s_503_, v___f_504_, v_sz_507_, v___x_508_, v___x_506_);
v_fst_510_ = lean_ctor_get(v___x_509_, 0);
v_snd_511_ = lean_ctor_get(v___x_509_, 1);
v_isSharedCheck_518_ = !lean_is_exclusive(v___x_509_);
if (v_isSharedCheck_518_ == 0)
{
v___x_513_ = v___x_509_;
v_isShared_514_ = v_isSharedCheck_518_;
goto v_resetjp_512_;
}
else
{
lean_inc(v_snd_511_);
lean_inc(v_fst_510_);
lean_dec(v___x_509_);
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
v_reuseFailAlloc_517_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_517_, 0, v_fst_510_);
lean_ctor_set(v_reuseFailAlloc_517_, 1, v_snd_511_);
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
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_partition(lean_object* v_00_u03b1_519_, lean_object* v_inst_520_, lean_object* v_f_521_, lean_object* v_s_522_){
_start:
{
lean_object* v___x_523_; 
v___x_523_ = lp_aesop_Aesop_UnorderedArraySet_partition___redArg(v_f_521_, v_s_522_);
return v___x_523_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_partition___boxed(lean_object* v_00_u03b1_524_, lean_object* v_inst_525_, lean_object* v_f_526_, lean_object* v_s_527_){
_start:
{
lean_object* v_res_528_; 
v_res_528_ = lp_aesop_Aesop_UnorderedArraySet_partition(v_00_u03b1_524_, v_inst_525_, v_f_526_, v_s_527_);
lean_dec_ref(v_inst_525_);
return v_res_528_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_size___redArg(lean_object* v_s_529_){
_start:
{
lean_object* v___x_530_; 
v___x_530_ = lean_array_get_size(v_s_529_);
return v___x_530_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_size___redArg___boxed(lean_object* v_s_531_){
_start:
{
lean_object* v_res_532_; 
v_res_532_ = lp_aesop_Aesop_UnorderedArraySet_size___redArg(v_s_531_);
lean_dec_ref(v_s_531_);
return v_res_532_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_size(lean_object* v_00_u03b1_533_, lean_object* v_inst_534_, lean_object* v_s_535_){
_start:
{
lean_object* v___x_536_; 
v___x_536_ = lean_array_get_size(v_s_535_);
return v___x_536_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_size___boxed(lean_object* v_00_u03b1_537_, lean_object* v_inst_538_, lean_object* v_s_539_){
_start:
{
lean_object* v_res_540_; 
v_res_540_ = lp_aesop_Aesop_UnorderedArraySet_size(v_00_u03b1_537_, v_inst_538_, v_s_539_);
lean_dec_ref(v_s_539_);
lean_dec_ref(v_inst_538_);
return v_res_540_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_UnorderedArraySet_isEmpty___redArg(lean_object* v_s_541_){
_start:
{
lean_object* v___x_542_; lean_object* v___x_543_; uint8_t v___x_544_; 
v___x_542_ = lean_array_get_size(v_s_541_);
v___x_543_ = lean_unsigned_to_nat(0u);
v___x_544_ = lean_nat_dec_eq(v___x_542_, v___x_543_);
return v___x_544_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_isEmpty___redArg___boxed(lean_object* v_s_545_){
_start:
{
uint8_t v_res_546_; lean_object* v_r_547_; 
v_res_546_ = lp_aesop_Aesop_UnorderedArraySet_isEmpty___redArg(v_s_545_);
lean_dec_ref(v_s_545_);
v_r_547_ = lean_box(v_res_546_);
return v_r_547_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_UnorderedArraySet_isEmpty(lean_object* v_00_u03b1_548_, lean_object* v_inst_549_, lean_object* v_s_550_){
_start:
{
uint8_t v___x_551_; 
v___x_551_ = lp_aesop_Aesop_UnorderedArraySet_isEmpty___redArg(v_s_550_);
return v___x_551_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_isEmpty___boxed(lean_object* v_00_u03b1_552_, lean_object* v_inst_553_, lean_object* v_s_554_){
_start:
{
uint8_t v_res_555_; lean_object* v_r_556_; 
v_res_555_ = lp_aesop_Aesop_UnorderedArraySet_isEmpty(v_00_u03b1_552_, v_inst_553_, v_s_554_);
lean_dec_ref(v_s_554_);
lean_dec_ref(v_inst_553_);
v_r_556_ = lean_box(v_res_555_);
return v_r_556_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_anyM___redArg(lean_object* v_inst_557_, lean_object* v_p_558_, lean_object* v_s_559_, lean_object* v_start_560_, lean_object* v_stop_561_){
_start:
{
lean_object* v___y_563_; uint8_t v___x_572_; 
v___x_572_ = lean_nat_dec_lt(v_start_560_, v_stop_561_);
if (v___x_572_ == 0)
{
lean_object* v_toApplicative_573_; lean_object* v_toPure_574_; lean_object* v___x_575_; lean_object* v___x_576_; 
lean_dec(v_stop_561_);
lean_dec_ref(v_s_559_);
lean_dec(v_p_558_);
v_toApplicative_573_ = lean_ctor_get(v_inst_557_, 0);
lean_inc_ref(v_toApplicative_573_);
lean_dec_ref(v_inst_557_);
v_toPure_574_ = lean_ctor_get(v_toApplicative_573_, 1);
lean_inc(v_toPure_574_);
lean_dec_ref(v_toApplicative_573_);
v___x_575_ = lean_box(v___x_572_);
v___x_576_ = lean_apply_2(v_toPure_574_, lean_box(0), v___x_575_);
return v___x_576_;
}
else
{
lean_object* v___x_577_; uint8_t v___x_578_; 
v___x_577_ = lean_array_get_size(v_s_559_);
v___x_578_ = lean_nat_dec_le(v_stop_561_, v___x_577_);
if (v___x_578_ == 0)
{
lean_dec(v_stop_561_);
v___y_563_ = v___x_577_;
goto v___jp_562_;
}
else
{
v___y_563_ = v_stop_561_;
goto v___jp_562_;
}
}
v___jp_562_:
{
uint8_t v___x_564_; 
v___x_564_ = lean_nat_dec_lt(v_start_560_, v___y_563_);
if (v___x_564_ == 0)
{
lean_object* v_toApplicative_565_; lean_object* v_toPure_566_; lean_object* v___x_567_; lean_object* v___x_568_; 
lean_dec(v___y_563_);
lean_dec_ref(v_s_559_);
lean_dec(v_p_558_);
v_toApplicative_565_ = lean_ctor_get(v_inst_557_, 0);
lean_inc_ref(v_toApplicative_565_);
lean_dec_ref(v_inst_557_);
v_toPure_566_ = lean_ctor_get(v_toApplicative_565_, 1);
lean_inc(v_toPure_566_);
lean_dec_ref(v_toApplicative_565_);
v___x_567_ = lean_box(v___x_564_);
v___x_568_ = lean_apply_2(v_toPure_566_, lean_box(0), v___x_567_);
return v___x_568_;
}
else
{
size_t v___x_569_; size_t v___x_570_; lean_object* v___x_571_; 
v___x_569_ = lean_usize_of_nat(v_start_560_);
v___x_570_ = lean_usize_of_nat(v___y_563_);
lean_dec(v___y_563_);
v___x_571_ = l___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any(lean_box(0), lean_box(0), v_inst_557_, v_p_558_, v_s_559_, v___x_569_, v___x_570_);
return v___x_571_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_anyM___redArg___boxed(lean_object* v_inst_579_, lean_object* v_p_580_, lean_object* v_s_581_, lean_object* v_start_582_, lean_object* v_stop_583_){
_start:
{
lean_object* v_res_584_; 
v_res_584_ = lp_aesop_Aesop_UnorderedArraySet_anyM___redArg(v_inst_579_, v_p_580_, v_s_581_, v_start_582_, v_stop_583_);
lean_dec(v_start_582_);
return v_res_584_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_anyM(lean_object* v_00_u03b1_585_, lean_object* v_inst_586_, lean_object* v_m_587_, lean_object* v_inst_588_, lean_object* v_p_589_, lean_object* v_s_590_, lean_object* v_start_591_, lean_object* v_stop_592_){
_start:
{
lean_object* v___x_593_; 
v___x_593_ = lp_aesop_Aesop_UnorderedArraySet_anyM___redArg(v_inst_588_, v_p_589_, v_s_590_, v_start_591_, v_stop_592_);
return v___x_593_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_anyM___boxed(lean_object* v_00_u03b1_594_, lean_object* v_inst_595_, lean_object* v_m_596_, lean_object* v_inst_597_, lean_object* v_p_598_, lean_object* v_s_599_, lean_object* v_start_600_, lean_object* v_stop_601_){
_start:
{
lean_object* v_res_602_; 
v_res_602_ = lp_aesop_Aesop_UnorderedArraySet_anyM(v_00_u03b1_594_, v_inst_595_, v_m_596_, v_inst_597_, v_p_598_, v_s_599_, v_start_600_, v_stop_601_);
lean_dec(v_start_600_);
lean_dec_ref(v_inst_595_);
return v_res_602_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_UnorderedArraySet_any___redArg___lam__0(lean_object* v_p_603_, lean_object* v_x_604_){
_start:
{
lean_object* v___x_605_; uint8_t v___x_606_; 
v___x_605_ = lean_apply_1(v_p_603_, v_x_604_);
v___x_606_ = lean_unbox(v___x_605_);
return v___x_606_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_any___redArg___lam__0___boxed(lean_object* v_p_607_, lean_object* v_x_608_){
_start:
{
uint8_t v_res_609_; lean_object* v_r_610_; 
v_res_609_ = lp_aesop_Aesop_UnorderedArraySet_any___redArg___lam__0(v_p_607_, v_x_608_);
v_r_610_ = lean_box(v_res_609_);
return v_r_610_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_UnorderedArraySet_any___redArg(lean_object* v_p_611_, lean_object* v_s_612_, lean_object* v_start_613_, lean_object* v_stop_614_){
_start:
{
lean_object* v___x_615_; uint8_t v___x_616_; 
v___x_615_ = ((lean_object*)(lp_aesop_Aesop_UnorderedArraySet_ofArraySlow___redArg___closed__9));
v___x_616_ = lean_nat_dec_lt(v_start_613_, v_stop_614_);
if (v___x_616_ == 0)
{
lean_dec(v_stop_614_);
lean_dec_ref(v_s_612_);
lean_dec_ref(v_p_611_);
return v___x_616_;
}
else
{
lean_object* v___f_617_; lean_object* v___y_619_; lean_object* v___x_625_; uint8_t v___x_626_; 
v___f_617_ = lean_alloc_closure((void*)(lp_aesop_Aesop_UnorderedArraySet_any___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_617_, 0, v_p_611_);
v___x_625_ = lean_array_get_size(v_s_612_);
v___x_626_ = lean_nat_dec_le(v_stop_614_, v___x_625_);
if (v___x_626_ == 0)
{
lean_dec(v_stop_614_);
v___y_619_ = v___x_625_;
goto v___jp_618_;
}
else
{
v___y_619_ = v_stop_614_;
goto v___jp_618_;
}
v___jp_618_:
{
uint8_t v___x_620_; 
v___x_620_ = lean_nat_dec_lt(v_start_613_, v___y_619_);
if (v___x_620_ == 0)
{
lean_dec(v___y_619_);
lean_dec_ref(v___f_617_);
lean_dec_ref(v_s_612_);
return v___x_620_;
}
else
{
size_t v___x_621_; size_t v___x_622_; lean_object* v___x_623_; uint8_t v___x_624_; 
v___x_621_ = lean_usize_of_nat(v_start_613_);
v___x_622_ = lean_usize_of_nat(v___y_619_);
lean_dec(v___y_619_);
v___x_623_ = l___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any(lean_box(0), lean_box(0), v___x_615_, v___f_617_, v_s_612_, v___x_621_, v___x_622_);
v___x_624_ = lean_unbox(v___x_623_);
lean_dec(v___x_623_);
return v___x_624_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_any___redArg___boxed(lean_object* v_p_627_, lean_object* v_s_628_, lean_object* v_start_629_, lean_object* v_stop_630_){
_start:
{
uint8_t v_res_631_; lean_object* v_r_632_; 
v_res_631_ = lp_aesop_Aesop_UnorderedArraySet_any___redArg(v_p_627_, v_s_628_, v_start_629_, v_stop_630_);
lean_dec(v_start_629_);
v_r_632_ = lean_box(v_res_631_);
return v_r_632_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_UnorderedArraySet_any(lean_object* v_00_u03b1_633_, lean_object* v_inst_634_, lean_object* v_p_635_, lean_object* v_s_636_, lean_object* v_start_637_, lean_object* v_stop_638_){
_start:
{
uint8_t v___x_639_; 
v___x_639_ = lp_aesop_Aesop_UnorderedArraySet_any___redArg(v_p_635_, v_s_636_, v_start_637_, v_stop_638_);
return v___x_639_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_any___boxed(lean_object* v_00_u03b1_640_, lean_object* v_inst_641_, lean_object* v_p_642_, lean_object* v_s_643_, lean_object* v_start_644_, lean_object* v_stop_645_){
_start:
{
uint8_t v_res_646_; lean_object* v_r_647_; 
v_res_646_ = lp_aesop_Aesop_UnorderedArraySet_any(v_00_u03b1_640_, v_inst_641_, v_p_642_, v_s_643_, v_start_644_, v_stop_645_);
lean_dec(v_start_644_);
lean_dec_ref(v_inst_641_);
v_r_647_ = lean_box(v_res_646_);
return v_r_647_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_allM___redArg___lam__0(lean_object* v_toPure_648_, uint8_t v_____do__lift_649_){
_start:
{
if (v_____do__lift_649_ == 0)
{
uint8_t v___x_650_; lean_object* v___x_651_; lean_object* v___x_652_; 
v___x_650_ = 1;
v___x_651_ = lean_box(v___x_650_);
v___x_652_ = lean_apply_2(v_toPure_648_, lean_box(0), v___x_651_);
return v___x_652_;
}
else
{
uint8_t v___x_653_; lean_object* v___x_654_; lean_object* v___x_655_; 
v___x_653_ = 0;
v___x_654_ = lean_box(v___x_653_);
v___x_655_ = lean_apply_2(v_toPure_648_, lean_box(0), v___x_654_);
return v___x_655_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_allM___redArg___lam__0___boxed(lean_object* v_toPure_656_, lean_object* v_____do__lift_657_){
_start:
{
uint8_t v_____do__lift_67__boxed_658_; lean_object* v_res_659_; 
v_____do__lift_67__boxed_658_ = lean_unbox(v_____do__lift_657_);
v_res_659_ = lp_aesop_Aesop_UnorderedArraySet_allM___redArg___lam__0(v_toPure_656_, v_____do__lift_67__boxed_658_);
return v_res_659_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_allM___redArg___lam__1(lean_object* v_toPure_660_, uint8_t v___x_661_, uint8_t v_____do__lift_662_){
_start:
{
if (v_____do__lift_662_ == 0)
{
lean_object* v___x_663_; lean_object* v___x_664_; 
v___x_663_ = lean_box(v___x_661_);
v___x_664_ = lean_apply_2(v_toPure_660_, lean_box(0), v___x_663_);
return v___x_664_;
}
else
{
uint8_t v___x_665_; lean_object* v___x_666_; lean_object* v___x_667_; 
v___x_665_ = 0;
v___x_666_ = lean_box(v___x_665_);
v___x_667_ = lean_apply_2(v_toPure_660_, lean_box(0), v___x_666_);
return v___x_667_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_allM___redArg___lam__1___boxed(lean_object* v_toPure_668_, lean_object* v___x_669_, lean_object* v_____do__lift_670_){
_start:
{
uint8_t v___x_82__boxed_671_; uint8_t v_____do__lift_83__boxed_672_; lean_object* v_res_673_; 
v___x_82__boxed_671_ = lean_unbox(v___x_669_);
v_____do__lift_83__boxed_672_ = lean_unbox(v_____do__lift_670_);
v_res_673_ = lp_aesop_Aesop_UnorderedArraySet_allM___redArg___lam__1(v_toPure_668_, v___x_82__boxed_671_, v_____do__lift_83__boxed_672_);
return v_res_673_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_allM___redArg___lam__2(lean_object* v_p_674_, lean_object* v_toBind_675_, lean_object* v___f_676_, lean_object* v_v_677_){
_start:
{
lean_object* v___x_678_; lean_object* v___x_679_; 
v___x_678_ = lean_apply_1(v_p_674_, v_v_677_);
v___x_679_ = lean_apply_4(v_toBind_675_, lean_box(0), lean_box(0), v___x_678_, v___f_676_);
return v___x_679_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_allM___redArg(lean_object* v_inst_680_, lean_object* v_p_681_, lean_object* v_s_682_, lean_object* v_start_683_, lean_object* v_stop_684_){
_start:
{
lean_object* v_toApplicative_685_; lean_object* v_toBind_686_; lean_object* v_toPure_687_; lean_object* v___f_688_; uint8_t v___x_689_; 
v_toApplicative_685_ = lean_ctor_get(v_inst_680_, 0);
v_toBind_686_ = lean_ctor_get(v_inst_680_, 1);
lean_inc(v_toBind_686_);
v_toPure_687_ = lean_ctor_get(v_toApplicative_685_, 1);
lean_inc(v_toPure_687_);
v___f_688_ = lean_alloc_closure((void*)(lp_aesop_Aesop_UnorderedArraySet_allM___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_688_, 0, v_toPure_687_);
v___x_689_ = lean_nat_dec_lt(v_start_683_, v_stop_684_);
if (v___x_689_ == 0)
{
lean_object* v___x_690_; lean_object* v___x_691_; lean_object* v___x_692_; 
lean_inc(v_toPure_687_);
lean_dec(v_stop_684_);
lean_dec_ref(v_s_682_);
lean_dec(v_p_681_);
lean_dec_ref(v_inst_680_);
v___x_690_ = lean_box(v___x_689_);
v___x_691_ = lean_apply_2(v_toPure_687_, lean_box(0), v___x_690_);
v___x_692_ = lean_apply_4(v_toBind_686_, lean_box(0), lean_box(0), v___x_691_, v___f_688_);
return v___x_692_;
}
else
{
lean_object* v___x_693_; lean_object* v___f_694_; lean_object* v___f_695_; lean_object* v___y_697_; lean_object* v___x_706_; uint8_t v___x_707_; 
v___x_693_ = lean_box(v___x_689_);
lean_inc(v_toPure_687_);
v___f_694_ = lean_alloc_closure((void*)(lp_aesop_Aesop_UnorderedArraySet_allM___redArg___lam__1___boxed), 3, 2);
lean_closure_set(v___f_694_, 0, v_toPure_687_);
lean_closure_set(v___f_694_, 1, v___x_693_);
lean_inc(v_toBind_686_);
v___f_695_ = lean_alloc_closure((void*)(lp_aesop_Aesop_UnorderedArraySet_allM___redArg___lam__2), 4, 3);
lean_closure_set(v___f_695_, 0, v_p_681_);
lean_closure_set(v___f_695_, 1, v_toBind_686_);
lean_closure_set(v___f_695_, 2, v___f_694_);
v___x_706_ = lean_array_get_size(v_s_682_);
v___x_707_ = lean_nat_dec_le(v_stop_684_, v___x_706_);
if (v___x_707_ == 0)
{
lean_dec(v_stop_684_);
v___y_697_ = v___x_706_;
goto v___jp_696_;
}
else
{
v___y_697_ = v_stop_684_;
goto v___jp_696_;
}
v___jp_696_:
{
uint8_t v___x_698_; 
v___x_698_ = lean_nat_dec_lt(v_start_683_, v___y_697_);
if (v___x_698_ == 0)
{
lean_object* v___x_699_; lean_object* v___x_700_; lean_object* v___x_701_; 
lean_inc(v_toPure_687_);
lean_dec(v___y_697_);
lean_dec_ref(v___f_695_);
lean_dec_ref(v_s_682_);
lean_dec_ref(v_inst_680_);
v___x_699_ = lean_box(v___x_698_);
v___x_700_ = lean_apply_2(v_toPure_687_, lean_box(0), v___x_699_);
v___x_701_ = lean_apply_4(v_toBind_686_, lean_box(0), lean_box(0), v___x_700_, v___f_688_);
return v___x_701_;
}
else
{
size_t v___x_702_; size_t v___x_703_; lean_object* v___x_704_; lean_object* v___x_705_; 
v___x_702_ = lean_usize_of_nat(v_start_683_);
v___x_703_ = lean_usize_of_nat(v___y_697_);
lean_dec(v___y_697_);
v___x_704_ = l___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any(lean_box(0), lean_box(0), v_inst_680_, v___f_695_, v_s_682_, v___x_702_, v___x_703_);
v___x_705_ = lean_apply_4(v_toBind_686_, lean_box(0), lean_box(0), v___x_704_, v___f_688_);
return v___x_705_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_allM___redArg___boxed(lean_object* v_inst_708_, lean_object* v_p_709_, lean_object* v_s_710_, lean_object* v_start_711_, lean_object* v_stop_712_){
_start:
{
lean_object* v_res_713_; 
v_res_713_ = lp_aesop_Aesop_UnorderedArraySet_allM___redArg(v_inst_708_, v_p_709_, v_s_710_, v_start_711_, v_stop_712_);
lean_dec(v_start_711_);
return v_res_713_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_allM(lean_object* v_00_u03b1_714_, lean_object* v_inst_715_, lean_object* v_m_716_, lean_object* v_inst_717_, lean_object* v_p_718_, lean_object* v_s_719_, lean_object* v_start_720_, lean_object* v_stop_721_){
_start:
{
lean_object* v___x_722_; 
v___x_722_ = lp_aesop_Aesop_UnorderedArraySet_allM___redArg(v_inst_717_, v_p_718_, v_s_719_, v_start_720_, v_stop_721_);
return v___x_722_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_allM___boxed(lean_object* v_00_u03b1_723_, lean_object* v_inst_724_, lean_object* v_m_725_, lean_object* v_inst_726_, lean_object* v_p_727_, lean_object* v_s_728_, lean_object* v_start_729_, lean_object* v_stop_730_){
_start:
{
lean_object* v_res_731_; 
v_res_731_ = lp_aesop_Aesop_UnorderedArraySet_allM(v_00_u03b1_723_, v_inst_724_, v_m_725_, v_inst_726_, v_p_727_, v_s_728_, v_start_729_, v_stop_730_);
lean_dec(v_start_729_);
lean_dec_ref(v_inst_724_);
return v_res_731_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_UnorderedArraySet_all___redArg___lam__0(lean_object* v_p_732_, uint8_t v___x_733_, lean_object* v_v_734_){
_start:
{
lean_object* v___x_735_; uint8_t v___x_736_; 
v___x_735_ = lean_apply_1(v_p_732_, v_v_734_);
v___x_736_ = lean_unbox(v___x_735_);
if (v___x_736_ == 0)
{
return v___x_733_;
}
else
{
uint8_t v___x_737_; 
v___x_737_ = 0;
return v___x_737_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_all___redArg___lam__0___boxed(lean_object* v_p_738_, lean_object* v___x_739_, lean_object* v_v_740_){
_start:
{
uint8_t v___x_44__boxed_741_; uint8_t v_res_742_; lean_object* v_r_743_; 
v___x_44__boxed_741_ = lean_unbox(v___x_739_);
v_res_742_ = lp_aesop_Aesop_UnorderedArraySet_all___redArg___lam__0(v_p_738_, v___x_44__boxed_741_, v_v_740_);
v_r_743_ = lean_box(v_res_742_);
return v_r_743_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_UnorderedArraySet_all___redArg(lean_object* v_p_744_, lean_object* v_s_745_, lean_object* v_start_746_, lean_object* v_stop_747_){
_start:
{
lean_object* v___x_748_; uint8_t v___x_749_; 
v___x_748_ = ((lean_object*)(lp_aesop_Aesop_UnorderedArraySet_ofArraySlow___redArg___closed__9));
v___x_749_ = lean_nat_dec_lt(v_start_746_, v_stop_747_);
if (v___x_749_ == 0)
{
uint8_t v___x_750_; 
lean_dec(v_stop_747_);
lean_dec_ref(v_s_745_);
lean_dec_ref(v_p_744_);
v___x_750_ = 1;
return v___x_750_;
}
else
{
lean_object* v___x_751_; lean_object* v___f_752_; lean_object* v___y_754_; lean_object* v___x_761_; uint8_t v___x_762_; 
v___x_751_ = lean_box(v___x_749_);
v___f_752_ = lean_alloc_closure((void*)(lp_aesop_Aesop_UnorderedArraySet_all___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_752_, 0, v_p_744_);
lean_closure_set(v___f_752_, 1, v___x_751_);
v___x_761_ = lean_array_get_size(v_s_745_);
v___x_762_ = lean_nat_dec_le(v_stop_747_, v___x_761_);
if (v___x_762_ == 0)
{
lean_dec(v_stop_747_);
v___y_754_ = v___x_761_;
goto v___jp_753_;
}
else
{
v___y_754_ = v_stop_747_;
goto v___jp_753_;
}
v___jp_753_:
{
uint8_t v___x_755_; 
v___x_755_ = lean_nat_dec_lt(v_start_746_, v___y_754_);
if (v___x_755_ == 0)
{
lean_dec(v___y_754_);
lean_dec_ref(v___f_752_);
lean_dec_ref(v_s_745_);
return v___x_749_;
}
else
{
size_t v___x_756_; size_t v___x_757_; lean_object* v___x_758_; uint8_t v___x_759_; 
v___x_756_ = lean_usize_of_nat(v_start_746_);
v___x_757_ = lean_usize_of_nat(v___y_754_);
lean_dec(v___y_754_);
v___x_758_ = l___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any(lean_box(0), lean_box(0), v___x_748_, v___f_752_, v_s_745_, v___x_756_, v___x_757_);
v___x_759_ = lean_unbox(v___x_758_);
lean_dec(v___x_758_);
if (v___x_759_ == 0)
{
return v___x_755_;
}
else
{
uint8_t v___x_760_; 
v___x_760_ = 0;
return v___x_760_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_all___redArg___boxed(lean_object* v_p_763_, lean_object* v_s_764_, lean_object* v_start_765_, lean_object* v_stop_766_){
_start:
{
uint8_t v_res_767_; lean_object* v_r_768_; 
v_res_767_ = lp_aesop_Aesop_UnorderedArraySet_all___redArg(v_p_763_, v_s_764_, v_start_765_, v_stop_766_);
lean_dec(v_start_765_);
v_r_768_ = lean_box(v_res_767_);
return v_r_768_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_UnorderedArraySet_all(lean_object* v_00_u03b1_769_, lean_object* v_inst_770_, lean_object* v_p_771_, lean_object* v_s_772_, lean_object* v_start_773_, lean_object* v_stop_774_){
_start:
{
uint8_t v___x_775_; 
v___x_775_ = lp_aesop_Aesop_UnorderedArraySet_all___redArg(v_p_771_, v_s_772_, v_start_773_, v_stop_774_);
return v___x_775_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_all___boxed(lean_object* v_00_u03b1_776_, lean_object* v_inst_777_, lean_object* v_p_778_, lean_object* v_s_779_, lean_object* v_start_780_, lean_object* v_stop_781_){
_start:
{
uint8_t v_res_782_; lean_object* v_r_783_; 
v_res_782_ = lp_aesop_Aesop_UnorderedArraySet_all(v_00_u03b1_776_, v_inst_777_, v_p_778_, v_s_779_, v_start_780_, v_stop_781_);
lean_dec(v_start_780_);
lean_dec_ref(v_inst_777_);
v_r_783_ = lean_box(v_res_782_);
return v_r_783_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_instToString___private__1___redArg(lean_object* v_inst_785_, lean_object* v_s_786_){
_start:
{
lean_object* v___x_787_; lean_object* v___x_788_; lean_object* v___x_789_; lean_object* v___x_790_; 
v___x_787_ = ((lean_object*)(lp_aesop_Aesop_UnorderedArraySet_instToString___private__1___redArg___closed__0));
v___x_788_ = lean_array_to_list(v_s_786_);
v___x_789_ = l_List_toString___redArg(v_inst_785_, v___x_788_);
v___x_790_ = lean_string_append(v___x_787_, v___x_789_);
lean_dec_ref(v___x_789_);
return v___x_790_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_instToString___private__1(lean_object* v_00_u03b1_791_, lean_object* v_inst_792_, lean_object* v_inst_793_, lean_object* v_s_794_){
_start:
{
lean_object* v___x_795_; lean_object* v___x_796_; lean_object* v___x_797_; lean_object* v___x_798_; 
v___x_795_ = ((lean_object*)(lp_aesop_Aesop_UnorderedArraySet_instToString___private__1___redArg___closed__0));
v___x_796_ = lean_array_to_list(v_s_794_);
v___x_797_ = l_List_toString___redArg(v_inst_793_, v___x_796_);
v___x_798_ = lean_string_append(v___x_795_, v___x_797_);
lean_dec_ref(v___x_797_);
return v___x_798_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_instToString___private__1___boxed(lean_object* v_00_u03b1_799_, lean_object* v_inst_800_, lean_object* v_inst_801_, lean_object* v_s_802_){
_start:
{
lean_object* v_res_803_; 
v_res_803_ = lp_aesop_Aesop_UnorderedArraySet_instToString___private__1(v_00_u03b1_799_, v_inst_800_, v_inst_801_, v_s_802_);
lean_dec_ref(v_inst_800_);
return v_res_803_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_instToString___redArg___lam__0(lean_object* v_inst_804_, lean_object* v_s_805_){
_start:
{
lean_object* v___x_806_; lean_object* v___x_807_; lean_object* v___x_808_; lean_object* v___x_809_; 
v___x_806_ = ((lean_object*)(lp_aesop_Aesop_UnorderedArraySet_instToString___private__1___redArg___closed__0));
v___x_807_ = lean_array_to_list(v_s_805_);
v___x_808_ = l_List_toString___redArg(v_inst_804_, v___x_807_);
v___x_809_ = lean_string_append(v___x_806_, v___x_808_);
lean_dec_ref(v___x_808_);
return v___x_809_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_instToString___redArg(lean_object* v_inst_810_){
_start:
{
lean_object* v___f_811_; 
v___f_811_ = lean_alloc_closure((void*)(lp_aesop_Aesop_UnorderedArraySet_instToString___redArg___lam__0), 2, 1);
lean_closure_set(v___f_811_, 0, v_inst_810_);
return v___f_811_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_instToString(lean_object* v_00_u03b1_812_, lean_object* v_inst_813_, lean_object* v_inst_814_){
_start:
{
lean_object* v___f_815_; 
v___f_815_ = lean_alloc_closure((void*)(lp_aesop_Aesop_UnorderedArraySet_instToString___redArg___lam__0), 2, 1);
lean_closure_set(v___f_815_, 0, v_inst_814_);
return v___f_815_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_instToString___boxed(lean_object* v_00_u03b1_816_, lean_object* v_inst_817_, lean_object* v_inst_818_){
_start:
{
lean_object* v_res_819_; 
v_res_819_ = lp_aesop_Aesop_UnorderedArraySet_instToString(v_00_u03b1_816_, v_inst_817_, v_inst_818_);
lean_dec_ref(v_inst_817_);
return v_res_819_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_instToFormat___private__1___redArg(lean_object* v_inst_822_, lean_object* v_s_823_){
_start:
{
lean_object* v___x_824_; lean_object* v___x_825_; lean_object* v___x_826_; lean_object* v___x_827_; 
v___x_824_ = ((lean_object*)(lp_aesop_Aesop_UnorderedArraySet_instToFormat___private__1___redArg___closed__0));
v___x_825_ = lean_array_to_list(v_s_823_);
v___x_826_ = l_List_format___redArg(v_inst_822_, v___x_825_);
v___x_827_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_827_, 0, v___x_824_);
lean_ctor_set(v___x_827_, 1, v___x_826_);
return v___x_827_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_instToFormat___private__1(lean_object* v_00_u03b1_828_, lean_object* v_inst_829_, lean_object* v_inst_830_, lean_object* v_s_831_){
_start:
{
lean_object* v___x_832_; lean_object* v___x_833_; lean_object* v___x_834_; lean_object* v___x_835_; 
v___x_832_ = ((lean_object*)(lp_aesop_Aesop_UnorderedArraySet_instToFormat___private__1___redArg___closed__0));
v___x_833_ = lean_array_to_list(v_s_831_);
v___x_834_ = l_List_format___redArg(v_inst_830_, v___x_833_);
v___x_835_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_835_, 0, v___x_832_);
lean_ctor_set(v___x_835_, 1, v___x_834_);
return v___x_835_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_instToFormat___private__1___boxed(lean_object* v_00_u03b1_836_, lean_object* v_inst_837_, lean_object* v_inst_838_, lean_object* v_s_839_){
_start:
{
lean_object* v_res_840_; 
v_res_840_ = lp_aesop_Aesop_UnorderedArraySet_instToFormat___private__1(v_00_u03b1_836_, v_inst_837_, v_inst_838_, v_s_839_);
lean_dec_ref(v_inst_837_);
return v_res_840_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_instToFormat___redArg___lam__0(lean_object* v_inst_841_, lean_object* v_s_842_){
_start:
{
lean_object* v___x_843_; lean_object* v___x_844_; lean_object* v___x_845_; lean_object* v___x_846_; 
v___x_843_ = ((lean_object*)(lp_aesop_Aesop_UnorderedArraySet_instToFormat___private__1___redArg___closed__0));
v___x_844_ = lean_array_to_list(v_s_842_);
v___x_845_ = l_List_format___redArg(v_inst_841_, v___x_844_);
v___x_846_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_846_, 0, v___x_843_);
lean_ctor_set(v___x_846_, 1, v___x_845_);
return v___x_846_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_instToFormat___redArg(lean_object* v_inst_847_){
_start:
{
lean_object* v___f_848_; 
v___f_848_ = lean_alloc_closure((void*)(lp_aesop_Aesop_UnorderedArraySet_instToFormat___redArg___lam__0), 2, 1);
lean_closure_set(v___f_848_, 0, v_inst_847_);
return v___f_848_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_instToFormat(lean_object* v_00_u03b1_849_, lean_object* v_inst_850_, lean_object* v_inst_851_){
_start:
{
lean_object* v___f_852_; 
v___f_852_ = lean_alloc_closure((void*)(lp_aesop_Aesop_UnorderedArraySet_instToFormat___redArg___lam__0), 2, 1);
lean_closure_set(v___f_852_, 0, v_inst_851_);
return v___f_852_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_instToFormat___boxed(lean_object* v_00_u03b1_853_, lean_object* v_inst_854_, lean_object* v_inst_855_){
_start:
{
lean_object* v_res_856_; 
v_res_856_ = lp_aesop_Aesop_UnorderedArraySet_instToFormat(v_00_u03b1_853_, v_inst_854_, v_inst_855_);
lean_dec_ref(v_inst_854_);
return v_res_856_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_instToMessageData___private__1___redArg(lean_object* v_inst_857_, lean_object* v_s_858_){
_start:
{
lean_object* v___x_859_; lean_object* v___x_860_; lean_object* v___x_861_; lean_object* v___x_862_; 
v___x_859_ = lean_array_to_list(v_s_858_);
v___x_860_ = lean_box(0);
v___x_861_ = l_List_mapTR_loop___redArg(v_inst_857_, v___x_859_, v___x_860_);
v___x_862_ = l_Lean_MessageData_ofList(v___x_861_);
return v___x_862_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_instToMessageData___private__1(lean_object* v_00_u03b1_863_, lean_object* v_inst_864_, lean_object* v_inst_865_, lean_object* v_s_866_){
_start:
{
lean_object* v___x_867_; lean_object* v___x_868_; lean_object* v___x_869_; lean_object* v___x_870_; 
v___x_867_ = lean_array_to_list(v_s_866_);
v___x_868_ = lean_box(0);
v___x_869_ = l_List_mapTR_loop___redArg(v_inst_865_, v___x_867_, v___x_868_);
v___x_870_ = l_Lean_MessageData_ofList(v___x_869_);
return v___x_870_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_instToMessageData___private__1___boxed(lean_object* v_00_u03b1_871_, lean_object* v_inst_872_, lean_object* v_inst_873_, lean_object* v_s_874_){
_start:
{
lean_object* v_res_875_; 
v_res_875_ = lp_aesop_Aesop_UnorderedArraySet_instToMessageData___private__1(v_00_u03b1_871_, v_inst_872_, v_inst_873_, v_s_874_);
lean_dec_ref(v_inst_872_);
return v_res_875_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_instToMessageData___redArg___lam__0(lean_object* v_inst_876_, lean_object* v_s_877_){
_start:
{
lean_object* v___x_878_; lean_object* v___x_879_; lean_object* v___x_880_; lean_object* v___x_881_; 
v___x_878_ = lean_array_to_list(v_s_877_);
v___x_879_ = lean_box(0);
v___x_880_ = l_List_mapTR_loop___redArg(v_inst_876_, v___x_878_, v___x_879_);
v___x_881_ = l_Lean_MessageData_ofList(v___x_880_);
return v___x_881_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_instToMessageData___redArg(lean_object* v_inst_882_){
_start:
{
lean_object* v___f_883_; 
v___f_883_ = lean_alloc_closure((void*)(lp_aesop_Aesop_UnorderedArraySet_instToMessageData___redArg___lam__0), 2, 1);
lean_closure_set(v___f_883_, 0, v_inst_882_);
return v___f_883_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_instToMessageData(lean_object* v_00_u03b1_884_, lean_object* v_inst_885_, lean_object* v_inst_886_){
_start:
{
lean_object* v___f_887_; 
v___f_887_ = lean_alloc_closure((void*)(lp_aesop_Aesop_UnorderedArraySet_instToMessageData___redArg___lam__0), 2, 1);
lean_closure_set(v___f_887_, 0, v_inst_886_);
return v___f_887_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnorderedArraySet_instToMessageData___boxed(lean_object* v_00_u03b1_888_, lean_object* v_inst_889_, lean_object* v_inst_890_){
_start:
{
lean_object* v_res_891_; 
v_res_891_ = lp_aesop_Aesop_UnorderedArraySet_instToMessageData(v_00_u03b1_888_, v_inst_889_, v_inst_890_);
lean_dec_ref(v_inst_889_);
return v_res_891_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Data_Array_Merge(uint8_t builtin);
lean_object* runtime_initialize_Lean_Message(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Util_UnorderedArraySet(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Data_Array_Merge(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Message(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Util_UnorderedArraySet(uint8_t builtin) {
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
lean_object* initialize_batteries_Batteries_Data_Array_Merge(uint8_t builtin);
lean_object* initialize_Lean_Message(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Util_UnorderedArraySet(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Data_Array_Merge(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Message(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Util_UnorderedArraySet(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Util_UnorderedArraySet(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Util_UnorderedArraySet(builtin);
}
#ifdef __cplusplus
}
#endif
