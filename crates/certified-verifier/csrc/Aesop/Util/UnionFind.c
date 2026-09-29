// Lean compiler output
// Module: Aesop.Util.UnionFind
// Imports: public import Init public meta import Init public import Std.Data.HashMap.Basic
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
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* l_instDecidableEqUSize___boxed(lean_object*, lean_object*);
lean_object* l_instBEqOfDecidableEq___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
lean_object* l_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Std_DHashMap_Internal_Raw_u2080_insert___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Id_instMonad___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__6(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__2___boxed(lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Std_DHashMap_Internal_AssocList_foldlM___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_USize_toUInt64___boxed(lean_object*);
uint8_t l_Std_DHashMap_Internal_Raw_u2080_contains___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
size_t lean_usize_add(size_t, size_t);
size_t lean_array_size(lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
static const lean_array_object lp_aesop_Aesop_instInhabitedUnionFind_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_instInhabitedUnionFind_default___closed__0 = (const lean_object*)&lp_aesop_Aesop_instInhabitedUnionFind_default___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_instInhabitedUnionFind_default___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instInhabitedUnionFind_default___closed__1;
static lean_once_cell_t lp_aesop_Aesop_instInhabitedUnionFind_default___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instInhabitedUnionFind_default___closed__2;
static lean_once_cell_t lp_aesop_Aesop_instInhabitedUnionFind_default___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instInhabitedUnionFind_default___closed__3;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedUnionFind_default(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedUnionFind_default___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedUnionFind___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedUnionFind___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedUnionFind(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedUnionFind___boxed(lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_aesop_Aesop_UnionFind_instEmptyCollection___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_UnionFind_instEmptyCollection___closed__0 = (const lean_object*)&lp_aesop_Aesop_UnionFind_instEmptyCollection___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_UnionFind_instEmptyCollection___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_UnionFind_instEmptyCollection___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnionFind_instEmptyCollection(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnionFind_instEmptyCollection___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnionFind_size___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnionFind_size___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnionFind_size(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnionFind_size___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_aesop_Aesop_UnionFind_add___redArg___boxed__const__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*0 + sizeof(size_t)*1, .m_other = 0, .m_tag = 0}, .m_objs = {(lean_object*)(size_t)(1ULL)}};
LEAN_EXPORT const lean_object* lp_aesop_Aesop_UnionFind_add___redArg___boxed__const__1 = (const lean_object*)&lp_aesop_Aesop_UnionFind_add___redArg___boxed__const__1_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnionFind_add___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnionFind_add(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnionFind_addArray___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_UnionFind_addArray___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__0, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_UnionFind_addArray___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_UnionFind_addArray___redArg___closed__0_value;
static const lean_closure_object lp_aesop_Aesop_UnionFind_addArray___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__1___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_UnionFind_addArray___redArg___closed__1 = (const lean_object*)&lp_aesop_Aesop_UnionFind_addArray___redArg___closed__1_value;
static const lean_closure_object lp_aesop_Aesop_UnionFind_addArray___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__2___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_UnionFind_addArray___redArg___closed__2 = (const lean_object*)&lp_aesop_Aesop_UnionFind_addArray___redArg___closed__2_value;
static const lean_closure_object lp_aesop_Aesop_UnionFind_addArray___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__3, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_UnionFind_addArray___redArg___closed__3 = (const lean_object*)&lp_aesop_Aesop_UnionFind_addArray___redArg___closed__3_value;
static const lean_closure_object lp_aesop_Aesop_UnionFind_addArray___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__4___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_UnionFind_addArray___redArg___closed__4 = (const lean_object*)&lp_aesop_Aesop_UnionFind_addArray___redArg___closed__4_value;
static const lean_closure_object lp_aesop_Aesop_UnionFind_addArray___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__5___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_UnionFind_addArray___redArg___closed__5 = (const lean_object*)&lp_aesop_Aesop_UnionFind_addArray___redArg___closed__5_value;
static const lean_closure_object lp_aesop_Aesop_UnionFind_addArray___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__6, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_UnionFind_addArray___redArg___closed__6 = (const lean_object*)&lp_aesop_Aesop_UnionFind_addArray___redArg___closed__6_value;
static const lean_ctor_object lp_aesop_Aesop_UnionFind_addArray___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop_Aesop_UnionFind_addArray___redArg___closed__0_value),((lean_object*)&lp_aesop_Aesop_UnionFind_addArray___redArg___closed__1_value)}};
static const lean_object* lp_aesop_Aesop_UnionFind_addArray___redArg___closed__7 = (const lean_object*)&lp_aesop_Aesop_UnionFind_addArray___redArg___closed__7_value;
static const lean_ctor_object lp_aesop_Aesop_UnionFind_addArray___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*5 + 0, .m_other = 5, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop_Aesop_UnionFind_addArray___redArg___closed__7_value),((lean_object*)&lp_aesop_Aesop_UnionFind_addArray___redArg___closed__2_value),((lean_object*)&lp_aesop_Aesop_UnionFind_addArray___redArg___closed__3_value),((lean_object*)&lp_aesop_Aesop_UnionFind_addArray___redArg___closed__4_value),((lean_object*)&lp_aesop_Aesop_UnionFind_addArray___redArg___closed__5_value)}};
static const lean_object* lp_aesop_Aesop_UnionFind_addArray___redArg___closed__8 = (const lean_object*)&lp_aesop_Aesop_UnionFind_addArray___redArg___closed__8_value;
static const lean_ctor_object lp_aesop_Aesop_UnionFind_addArray___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop_Aesop_UnionFind_addArray___redArg___closed__8_value),((lean_object*)&lp_aesop_Aesop_UnionFind_addArray___redArg___closed__6_value)}};
static const lean_object* lp_aesop_Aesop_UnionFind_addArray___redArg___closed__9 = (const lean_object*)&lp_aesop_Aesop_UnionFind_addArray___redArg___closed__9_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnionFind_addArray___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnionFind_addArray(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnionFind_ofArray___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnionFind_ofArray(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_UnionFind_0__Aesop_UnionFind_findRepUnsafe___redArg(size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_UnionFind_0__Aesop_UnionFind_findRepUnsafe___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_UnionFind_0__Aesop_UnionFind_findRepUnsafe(lean_object*, lean_object*, lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_UnionFind_0__Aesop_UnionFind_findRepUnsafe___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnionFind_find_x3f___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnionFind_find_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_UnionFind_0__Aesop_UnionFind_mergeUnsafe___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_UnionFind_0__Aesop_UnionFind_mergeUnsafe(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnionFind_sets___redArg___lam__0(lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnionFind_sets___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnionFind_sets___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Aesop_UnionFind_sets___redArg___lam__2___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_UnionFind_sets___redArg___lam__2___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnionFind_sets___redArg___lam__2(lean_object*, lean_object*, lean_object*, size_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnionFind_sets___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnionFind_sets___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_UnionFind_sets___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_UnionFind_sets___redArg___lam__0___boxed, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_UnionFind_sets___redArg___closed__0 = (const lean_object*)&lp_aesop_Aesop_UnionFind_sets___redArg___closed__0_value;
static const lean_closure_object lp_aesop_Aesop_UnionFind_sets___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_UnionFind_sets___redArg___lam__1, .m_arity = 4, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_aesop_Aesop_UnionFind_addArray___redArg___closed__9_value),((lean_object*)&lp_aesop_Aesop_UnionFind_sets___redArg___closed__0_value)} };
static const lean_object* lp_aesop_Aesop_UnionFind_sets___redArg___closed__1 = (const lean_object*)&lp_aesop_Aesop_UnionFind_sets___redArg___closed__1_value;
static lean_once_cell_t lp_aesop_Aesop_UnionFind_sets___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_UnionFind_sets___redArg___closed__2;
static lean_once_cell_t lp_aesop_Aesop_UnionFind_sets___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_UnionFind_sets___redArg___closed__3;
static const lean_closure_object lp_aesop_Aesop_UnionFind_sets___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_USize_toUInt64___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_UnionFind_sets___redArg___closed__4 = (const lean_object*)&lp_aesop_Aesop_UnionFind_sets___redArg___closed__4_value;
static const lean_closure_object lp_aesop_Aesop_UnionFind_sets___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_UnionFind_sets___redArg___lam__2___boxed, .m_arity = 4, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_aesop_Aesop_UnionFind_sets___redArg___closed__4_value)} };
static const lean_object* lp_aesop_Aesop_UnionFind_sets___redArg___closed__5 = (const lean_object*)&lp_aesop_Aesop_UnionFind_sets___redArg___closed__5_value;
static const lean_closure_object lp_aesop_Aesop_UnionFind_sets___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_UnionFind_sets___redArg___lam__3, .m_arity = 4, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_aesop_Aesop_UnionFind_addArray___redArg___closed__9_value),((lean_object*)&lp_aesop_Aesop_UnionFind_sets___redArg___closed__5_value)} };
static const lean_object* lp_aesop_Aesop_UnionFind_sets___redArg___closed__6 = (const lean_object*)&lp_aesop_Aesop_UnionFind_sets___redArg___closed__6_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnionFind_sets___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnionFind_sets(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnionFind_sets___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_cluster___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_cluster___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_cluster___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_cluster___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_cluster(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_aesop_Aesop_instInhabitedUnionFind_default___closed__1(void){
_start:
{
lean_object* v___x_3_; lean_object* v___x_4_; lean_object* v___x_5_; 
v___x_3_ = lean_box(0);
v___x_4_ = lean_unsigned_to_nat(16u);
v___x_5_ = lean_mk_array(v___x_4_, v___x_3_);
return v___x_5_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedUnionFind_default___closed__2(void){
_start:
{
lean_object* v___x_6_; lean_object* v___x_7_; lean_object* v___x_8_; 
v___x_6_ = lean_obj_once(&lp_aesop_Aesop_instInhabitedUnionFind_default___closed__1, &lp_aesop_Aesop_instInhabitedUnionFind_default___closed__1_once, _init_lp_aesop_Aesop_instInhabitedUnionFind_default___closed__1);
v___x_7_ = lean_unsigned_to_nat(0u);
v___x_8_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_8_, 0, v___x_7_);
lean_ctor_set(v___x_8_, 1, v___x_6_);
return v___x_8_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedUnionFind_default___closed__3(void){
_start:
{
lean_object* v___x_9_; lean_object* v___x_10_; lean_object* v___x_11_; 
v___x_9_ = lean_obj_once(&lp_aesop_Aesop_instInhabitedUnionFind_default___closed__2, &lp_aesop_Aesop_instInhabitedUnionFind_default___closed__2_once, _init_lp_aesop_Aesop_instInhabitedUnionFind_default___closed__2);
v___x_10_ = ((lean_object*)(lp_aesop_Aesop_instInhabitedUnionFind_default___closed__0));
v___x_11_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_11_, 0, v___x_10_);
lean_ctor_set(v___x_11_, 1, v___x_10_);
lean_ctor_set(v___x_11_, 2, v___x_9_);
return v___x_11_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedUnionFind_default(lean_object* v_00_u03b1_12_, lean_object* v_inst_13_, lean_object* v_inst_14_){
_start:
{
lean_object* v___x_15_; 
v___x_15_ = lean_obj_once(&lp_aesop_Aesop_instInhabitedUnionFind_default___closed__3, &lp_aesop_Aesop_instInhabitedUnionFind_default___closed__3_once, _init_lp_aesop_Aesop_instInhabitedUnionFind_default___closed__3);
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedUnionFind_default___boxed(lean_object* v_00_u03b1_16_, lean_object* v_inst_17_, lean_object* v_inst_18_){
_start:
{
lean_object* v_res_19_; 
v_res_19_ = lp_aesop_Aesop_instInhabitedUnionFind_default(v_00_u03b1_16_, v_inst_17_, v_inst_18_);
lean_dec_ref(v_inst_18_);
lean_dec_ref(v_inst_17_);
return v_res_19_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedUnionFind___redArg(lean_object* v_a_20_, lean_object* v_a_21_){
_start:
{
lean_object* v___x_22_; 
v___x_22_ = lp_aesop_Aesop_instInhabitedUnionFind_default(lean_box(0), v_a_20_, v_a_21_);
return v___x_22_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedUnionFind___redArg___boxed(lean_object* v_a_23_, lean_object* v_a_24_){
_start:
{
lean_object* v_res_25_; 
v_res_25_ = lp_aesop_Aesop_instInhabitedUnionFind___redArg(v_a_23_, v_a_24_);
lean_dec_ref(v_a_24_);
lean_dec_ref(v_a_23_);
return v_res_25_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedUnionFind(lean_object* v_a_26_, lean_object* v_a_27_, lean_object* v_a_28_){
_start:
{
lean_object* v___x_29_; 
v___x_29_ = lp_aesop_Aesop_instInhabitedUnionFind_default(lean_box(0), v_a_27_, v_a_28_);
return v___x_29_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedUnionFind___boxed(lean_object* v_a_30_, lean_object* v_a_31_, lean_object* v_a_32_){
_start:
{
lean_object* v_res_33_; 
v_res_33_ = lp_aesop_Aesop_instInhabitedUnionFind(v_a_30_, v_a_31_, v_a_32_);
lean_dec_ref(v_a_32_);
lean_dec_ref(v_a_31_);
return v_res_33_;
}
}
static lean_object* _init_lp_aesop_Aesop_UnionFind_instEmptyCollection___closed__1(void){
_start:
{
lean_object* v___x_36_; lean_object* v___x_37_; lean_object* v___x_38_; 
v___x_36_ = lean_obj_once(&lp_aesop_Aesop_instInhabitedUnionFind_default___closed__2, &lp_aesop_Aesop_instInhabitedUnionFind_default___closed__2_once, _init_lp_aesop_Aesop_instInhabitedUnionFind_default___closed__2);
v___x_37_ = ((lean_object*)(lp_aesop_Aesop_UnionFind_instEmptyCollection___closed__0));
v___x_38_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_38_, 0, v___x_37_);
lean_ctor_set(v___x_38_, 1, v___x_37_);
lean_ctor_set(v___x_38_, 2, v___x_36_);
return v___x_38_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnionFind_instEmptyCollection(lean_object* v_00_u03b1_39_, lean_object* v_inst_40_, lean_object* v_inst_41_){
_start:
{
lean_object* v___x_42_; 
v___x_42_ = lean_obj_once(&lp_aesop_Aesop_UnionFind_instEmptyCollection___closed__1, &lp_aesop_Aesop_UnionFind_instEmptyCollection___closed__1_once, _init_lp_aesop_Aesop_UnionFind_instEmptyCollection___closed__1);
return v___x_42_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnionFind_instEmptyCollection___boxed(lean_object* v_00_u03b1_43_, lean_object* v_inst_44_, lean_object* v_inst_45_){
_start:
{
lean_object* v_res_46_; 
v_res_46_ = lp_aesop_Aesop_UnionFind_instEmptyCollection(v_00_u03b1_43_, v_inst_44_, v_inst_45_);
lean_dec_ref(v_inst_45_);
lean_dec_ref(v_inst_44_);
return v_res_46_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnionFind_size___redArg(lean_object* v_u_47_){
_start:
{
lean_object* v_parents_48_; lean_object* v___x_49_; 
v_parents_48_ = lean_ctor_get(v_u_47_, 0);
v___x_49_ = lean_array_get_size(v_parents_48_);
return v___x_49_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnionFind_size___redArg___boxed(lean_object* v_u_50_){
_start:
{
lean_object* v_res_51_; 
v_res_51_ = lp_aesop_Aesop_UnionFind_size___redArg(v_u_50_);
lean_dec_ref(v_u_50_);
return v_res_51_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnionFind_size(lean_object* v_00_u03b1_52_, lean_object* v_inst_53_, lean_object* v_inst_54_, lean_object* v_u_55_){
_start:
{
lean_object* v___x_56_; 
v___x_56_ = lp_aesop_Aesop_UnionFind_size___redArg(v_u_55_);
return v___x_56_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnionFind_size___boxed(lean_object* v_00_u03b1_57_, lean_object* v_inst_58_, lean_object* v_inst_59_, lean_object* v_u_60_){
_start:
{
lean_object* v_res_61_; 
v_res_61_ = lp_aesop_Aesop_UnionFind_size(v_00_u03b1_57_, v_inst_58_, v_inst_59_, v_u_60_);
lean_dec_ref(v_u_60_);
lean_dec_ref(v_inst_59_);
lean_dec_ref(v_inst_58_);
return v_res_61_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnionFind_add___redArg(lean_object* v_inst_64_, lean_object* v_inst_65_, lean_object* v_x_66_, lean_object* v_u_67_){
_start:
{
lean_object* v_parents_68_; lean_object* v_toRep_69_; uint8_t v___x_70_; 
v_parents_68_ = lean_ctor_get(v_u_67_, 0);
v_toRep_69_ = lean_ctor_get(v_u_67_, 2);
lean_inc(v_x_66_);
lean_inc_ref(v_inst_65_);
lean_inc_ref(v_inst_64_);
v___x_70_ = l_Std_DHashMap_Internal_Raw_u2080_contains___redArg(v_inst_64_, v_inst_65_, v_toRep_69_, v_x_66_);
if (v___x_70_ == 0)
{
lean_object* v___x_72_; uint8_t v_isShared_73_; uint8_t v_isSharedCheck_85_; 
lean_inc_ref(v_toRep_69_);
lean_inc_ref(v_parents_68_);
v_isSharedCheck_85_ = !lean_is_exclusive(v_u_67_);
if (v_isSharedCheck_85_ == 0)
{
lean_object* v_unused_86_; lean_object* v_unused_87_; lean_object* v_unused_88_; 
v_unused_86_ = lean_ctor_get(v_u_67_, 2);
lean_dec(v_unused_86_);
v_unused_87_ = lean_ctor_get(v_u_67_, 1);
lean_dec(v_unused_87_);
v_unused_88_ = lean_ctor_get(v_u_67_, 0);
lean_dec(v_unused_88_);
v___x_72_ = v_u_67_;
v_isShared_73_ = v_isSharedCheck_85_;
goto v_resetjp_71_;
}
else
{
lean_dec(v_u_67_);
v___x_72_ = lean_box(0);
v_isShared_73_ = v_isSharedCheck_85_;
goto v_resetjp_71_;
}
v_resetjp_71_:
{
lean_object* v___x_74_; size_t v_rep_75_; lean_object* v___x_76_; lean_object* v___x_77_; lean_object* v___x_78_; lean_object* v___x_79_; lean_object* v___x_80_; lean_object* v___x_81_; lean_object* v___x_83_; 
v___x_74_ = lean_array_get_size(v_parents_68_);
v_rep_75_ = lean_usize_of_nat(v___x_74_);
v___x_76_ = lean_box_usize(v_rep_75_);
lean_inc_ref(v_parents_68_);
v___x_77_ = lean_array_push(v_parents_68_, v___x_76_);
v___x_78_ = ((lean_object*)(lp_aesop_Aesop_UnionFind_add___redArg___boxed__const__1));
v___x_79_ = lean_array_push(v_parents_68_, v___x_78_);
v___x_80_ = lean_box_usize(v_rep_75_);
v___x_81_ = l_Std_DHashMap_Internal_Raw_u2080_insert___redArg(v_inst_64_, v_inst_65_, v_toRep_69_, v_x_66_, v___x_80_);
if (v_isShared_73_ == 0)
{
lean_ctor_set(v___x_72_, 2, v___x_81_);
lean_ctor_set(v___x_72_, 1, v___x_79_);
lean_ctor_set(v___x_72_, 0, v___x_77_);
v___x_83_ = v___x_72_;
goto v_reusejp_82_;
}
else
{
lean_object* v_reuseFailAlloc_84_; 
v_reuseFailAlloc_84_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_84_, 0, v___x_77_);
lean_ctor_set(v_reuseFailAlloc_84_, 1, v___x_79_);
lean_ctor_set(v_reuseFailAlloc_84_, 2, v___x_81_);
v___x_83_ = v_reuseFailAlloc_84_;
goto v_reusejp_82_;
}
v_reusejp_82_:
{
return v___x_83_;
}
}
}
else
{
lean_dec(v_x_66_);
lean_dec_ref(v_inst_65_);
lean_dec_ref(v_inst_64_);
return v_u_67_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnionFind_add(lean_object* v_00_u03b1_89_, lean_object* v_inst_90_, lean_object* v_inst_91_, lean_object* v_x_92_, lean_object* v_u_93_){
_start:
{
lean_object* v___x_94_; 
v___x_94_ = lp_aesop_Aesop_UnionFind_add___redArg(v_inst_90_, v_inst_91_, v_x_92_, v_u_93_);
return v___x_94_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnionFind_addArray___redArg___lam__0(lean_object* v_inst_95_, lean_object* v_inst_96_, lean_object* v_x1_97_, lean_object* v_x2_98_){
_start:
{
lean_object* v___x_99_; 
v___x_99_ = lp_aesop_Aesop_UnionFind_add___redArg(v_inst_95_, v_inst_96_, v_x2_98_, v_x1_97_);
return v___x_99_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnionFind_addArray___redArg(lean_object* v_inst_119_, lean_object* v_inst_120_, lean_object* v_xs_121_, lean_object* v_u_122_){
_start:
{
lean_object* v___x_123_; lean_object* v___x_124_; lean_object* v___x_125_; uint8_t v___x_126_; 
v___x_123_ = lean_unsigned_to_nat(0u);
v___x_124_ = lean_array_get_size(v_xs_121_);
v___x_125_ = ((lean_object*)(lp_aesop_Aesop_UnionFind_addArray___redArg___closed__9));
v___x_126_ = lean_nat_dec_lt(v___x_123_, v___x_124_);
if (v___x_126_ == 0)
{
lean_dec_ref(v_xs_121_);
lean_dec_ref(v_inst_120_);
lean_dec_ref(v_inst_119_);
return v_u_122_;
}
else
{
lean_object* v___f_127_; uint8_t v___x_128_; 
v___f_127_ = lean_alloc_closure((void*)(lp_aesop_Aesop_UnionFind_addArray___redArg___lam__0), 4, 2);
lean_closure_set(v___f_127_, 0, v_inst_119_);
lean_closure_set(v___f_127_, 1, v_inst_120_);
v___x_128_ = lean_nat_dec_le(v___x_124_, v___x_124_);
if (v___x_128_ == 0)
{
if (v___x_126_ == 0)
{
lean_dec_ref(v___f_127_);
lean_dec_ref(v_xs_121_);
return v_u_122_;
}
else
{
size_t v___x_129_; size_t v___x_130_; lean_object* v___x_131_; 
v___x_129_ = ((size_t)0ULL);
v___x_130_ = lean_usize_of_nat(v___x_124_);
v___x_131_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_125_, v___f_127_, v_xs_121_, v___x_129_, v___x_130_, v_u_122_);
return v___x_131_;
}
}
else
{
size_t v___x_132_; size_t v___x_133_; lean_object* v___x_134_; 
v___x_132_ = ((size_t)0ULL);
v___x_133_ = lean_usize_of_nat(v___x_124_);
v___x_134_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_125_, v___f_127_, v_xs_121_, v___x_132_, v___x_133_, v_u_122_);
return v___x_134_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnionFind_addArray(lean_object* v_00_u03b1_135_, lean_object* v_inst_136_, lean_object* v_inst_137_, lean_object* v_xs_138_, lean_object* v_u_139_){
_start:
{
lean_object* v___x_140_; 
v___x_140_ = lp_aesop_Aesop_UnionFind_addArray___redArg(v_inst_136_, v_inst_137_, v_xs_138_, v_u_139_);
return v___x_140_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnionFind_ofArray___redArg(lean_object* v_inst_141_, lean_object* v_inst_142_, lean_object* v_xs_143_){
_start:
{
lean_object* v___x_144_; lean_object* v___x_145_; 
v___x_144_ = lean_obj_once(&lp_aesop_Aesop_UnionFind_instEmptyCollection___closed__1, &lp_aesop_Aesop_UnionFind_instEmptyCollection___closed__1_once, _init_lp_aesop_Aesop_UnionFind_instEmptyCollection___closed__1);
v___x_145_ = lp_aesop_Aesop_UnionFind_addArray___redArg(v_inst_141_, v_inst_142_, v_xs_143_, v___x_144_);
return v___x_145_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnionFind_ofArray(lean_object* v_00_u03b1_146_, lean_object* v_inst_147_, lean_object* v_inst_148_, lean_object* v_xs_149_){
_start:
{
lean_object* v___x_150_; 
v___x_150_ = lp_aesop_Aesop_UnionFind_ofArray___redArg(v_inst_147_, v_inst_148_, v_xs_149_);
return v___x_150_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_UnionFind_0__Aesop_UnionFind_findRepUnsafe___redArg(size_t v_i_151_, lean_object* v_u_152_){
_start:
{
lean_object* v_parents_153_; lean_object* v_parent_154_; size_t v___x_155_; uint8_t v___x_156_; 
v_parents_153_ = lean_ctor_get(v_u_152_, 0);
v_parent_154_ = lean_array_uget_borrowed(v_parents_153_, v_i_151_);
v___x_155_ = lean_unbox_usize(v_parent_154_);
v___x_156_ = lean_usize_dec_eq(v___x_155_, v_i_151_);
if (v___x_156_ == 0)
{
size_t v___x_157_; lean_object* v___x_158_; lean_object* v_snd_159_; lean_object* v_fst_160_; lean_object* v___x_162_; uint8_t v_isShared_163_; uint8_t v_isSharedCheck_178_; 
v___x_157_ = lean_unbox_usize(v_parent_154_);
v___x_158_ = lp_aesop___private_Aesop_Util_UnionFind_0__Aesop_UnionFind_findRepUnsafe___redArg(v___x_157_, v_u_152_);
v_snd_159_ = lean_ctor_get(v___x_158_, 1);
v_fst_160_ = lean_ctor_get(v___x_158_, 0);
v_isSharedCheck_178_ = !lean_is_exclusive(v___x_158_);
if (v_isSharedCheck_178_ == 0)
{
v___x_162_ = v___x_158_;
v_isShared_163_ = v_isSharedCheck_178_;
goto v_resetjp_161_;
}
else
{
lean_inc(v_snd_159_);
lean_inc(v_fst_160_);
lean_dec(v___x_158_);
v___x_162_ = lean_box(0);
v_isShared_163_ = v_isSharedCheck_178_;
goto v_resetjp_161_;
}
v_resetjp_161_:
{
lean_object* v_parents_164_; lean_object* v_sizes_165_; lean_object* v_toRep_166_; lean_object* v___x_168_; uint8_t v_isShared_169_; uint8_t v_isSharedCheck_177_; 
v_parents_164_ = lean_ctor_get(v_snd_159_, 0);
v_sizes_165_ = lean_ctor_get(v_snd_159_, 1);
v_toRep_166_ = lean_ctor_get(v_snd_159_, 2);
v_isSharedCheck_177_ = !lean_is_exclusive(v_snd_159_);
if (v_isSharedCheck_177_ == 0)
{
v___x_168_ = v_snd_159_;
v_isShared_169_ = v_isSharedCheck_177_;
goto v_resetjp_167_;
}
else
{
lean_inc(v_toRep_166_);
lean_inc(v_sizes_165_);
lean_inc(v_parents_164_);
lean_dec(v_snd_159_);
v___x_168_ = lean_box(0);
v_isShared_169_ = v_isSharedCheck_177_;
goto v_resetjp_167_;
}
v_resetjp_167_:
{
lean_object* v___x_170_; lean_object* v___x_172_; 
lean_inc(v_fst_160_);
v___x_170_ = lean_array_uset(v_parents_164_, v_i_151_, v_fst_160_);
if (v_isShared_169_ == 0)
{
lean_ctor_set(v___x_168_, 0, v___x_170_);
v___x_172_ = v___x_168_;
goto v_reusejp_171_;
}
else
{
lean_object* v_reuseFailAlloc_176_; 
v_reuseFailAlloc_176_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_176_, 0, v___x_170_);
lean_ctor_set(v_reuseFailAlloc_176_, 1, v_sizes_165_);
lean_ctor_set(v_reuseFailAlloc_176_, 2, v_toRep_166_);
v___x_172_ = v_reuseFailAlloc_176_;
goto v_reusejp_171_;
}
v_reusejp_171_:
{
lean_object* v___x_174_; 
if (v_isShared_163_ == 0)
{
lean_ctor_set(v___x_162_, 1, v___x_172_);
v___x_174_ = v___x_162_;
goto v_reusejp_173_;
}
else
{
lean_object* v_reuseFailAlloc_175_; 
v_reuseFailAlloc_175_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_175_, 0, v_fst_160_);
lean_ctor_set(v_reuseFailAlloc_175_, 1, v___x_172_);
v___x_174_ = v_reuseFailAlloc_175_;
goto v_reusejp_173_;
}
v_reusejp_173_:
{
return v___x_174_;
}
}
}
}
}
else
{
lean_object* v___x_179_; 
lean_inc(v_parent_154_);
v___x_179_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_179_, 0, v_parent_154_);
lean_ctor_set(v___x_179_, 1, v_u_152_);
return v___x_179_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_UnionFind_0__Aesop_UnionFind_findRepUnsafe___redArg___boxed(lean_object* v_i_180_, lean_object* v_u_181_){
_start:
{
size_t v_i_boxed_182_; lean_object* v_res_183_; 
v_i_boxed_182_ = lean_unbox_usize(v_i_180_);
lean_dec(v_i_180_);
v_res_183_ = lp_aesop___private_Aesop_Util_UnionFind_0__Aesop_UnionFind_findRepUnsafe___redArg(v_i_boxed_182_, v_u_181_);
return v_res_183_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_UnionFind_0__Aesop_UnionFind_findRepUnsafe(lean_object* v_00_u03b1_184_, lean_object* v_inst_185_, lean_object* v_inst_186_, size_t v_i_187_, lean_object* v_u_188_){
_start:
{
lean_object* v___x_189_; 
v___x_189_ = lp_aesop___private_Aesop_Util_UnionFind_0__Aesop_UnionFind_findRepUnsafe___redArg(v_i_187_, v_u_188_);
return v___x_189_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_UnionFind_0__Aesop_UnionFind_findRepUnsafe___boxed(lean_object* v_00_u03b1_190_, lean_object* v_inst_191_, lean_object* v_inst_192_, lean_object* v_i_193_, lean_object* v_u_194_){
_start:
{
size_t v_i_boxed_195_; lean_object* v_res_196_; 
v_i_boxed_195_ = lean_unbox_usize(v_i_193_);
lean_dec(v_i_193_);
v_res_196_ = lp_aesop___private_Aesop_Util_UnionFind_0__Aesop_UnionFind_findRepUnsafe(v_00_u03b1_190_, v_inst_191_, v_inst_192_, v_i_boxed_195_, v_u_194_);
lean_dec_ref(v_inst_192_);
lean_dec_ref(v_inst_191_);
return v_res_196_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnionFind_find_x3f___redArg(lean_object* v_inst_197_, lean_object* v_inst_198_, lean_object* v_x_199_, lean_object* v_u_200_){
_start:
{
lean_object* v_toRep_201_; lean_object* v___x_202_; 
v_toRep_201_ = lean_ctor_get(v_u_200_, 2);
v___x_202_ = l_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___redArg(v_inst_197_, v_inst_198_, v_toRep_201_, v_x_199_);
if (lean_obj_tag(v___x_202_) == 0)
{
lean_object* v___x_203_; 
v___x_203_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_203_, 0, v___x_202_);
lean_ctor_set(v___x_203_, 1, v_u_200_);
return v___x_203_;
}
else
{
lean_object* v_val_204_; lean_object* v___x_206_; uint8_t v_isShared_207_; uint8_t v_isSharedCheck_222_; 
v_val_204_ = lean_ctor_get(v___x_202_, 0);
v_isSharedCheck_222_ = !lean_is_exclusive(v___x_202_);
if (v_isSharedCheck_222_ == 0)
{
v___x_206_ = v___x_202_;
v_isShared_207_ = v_isSharedCheck_222_;
goto v_resetjp_205_;
}
else
{
lean_inc(v_val_204_);
lean_dec(v___x_202_);
v___x_206_ = lean_box(0);
v_isShared_207_ = v_isSharedCheck_222_;
goto v_resetjp_205_;
}
v_resetjp_205_:
{
size_t v___x_208_; lean_object* v___x_209_; lean_object* v_fst_210_; lean_object* v_snd_211_; lean_object* v___x_213_; uint8_t v_isShared_214_; uint8_t v_isSharedCheck_221_; 
v___x_208_ = lean_unbox_usize(v_val_204_);
lean_dec(v_val_204_);
v___x_209_ = lp_aesop___private_Aesop_Util_UnionFind_0__Aesop_UnionFind_findRepUnsafe___redArg(v___x_208_, v_u_200_);
v_fst_210_ = lean_ctor_get(v___x_209_, 0);
v_snd_211_ = lean_ctor_get(v___x_209_, 1);
v_isSharedCheck_221_ = !lean_is_exclusive(v___x_209_);
if (v_isSharedCheck_221_ == 0)
{
v___x_213_ = v___x_209_;
v_isShared_214_ = v_isSharedCheck_221_;
goto v_resetjp_212_;
}
else
{
lean_inc(v_snd_211_);
lean_inc(v_fst_210_);
lean_dec(v___x_209_);
v___x_213_ = lean_box(0);
v_isShared_214_ = v_isSharedCheck_221_;
goto v_resetjp_212_;
}
v_resetjp_212_:
{
lean_object* v___x_216_; 
if (v_isShared_207_ == 0)
{
lean_ctor_set(v___x_206_, 0, v_fst_210_);
v___x_216_ = v___x_206_;
goto v_reusejp_215_;
}
else
{
lean_object* v_reuseFailAlloc_220_; 
v_reuseFailAlloc_220_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_220_, 0, v_fst_210_);
v___x_216_ = v_reuseFailAlloc_220_;
goto v_reusejp_215_;
}
v_reusejp_215_:
{
lean_object* v___x_218_; 
if (v_isShared_214_ == 0)
{
lean_ctor_set(v___x_213_, 0, v___x_216_);
v___x_218_ = v___x_213_;
goto v_reusejp_217_;
}
else
{
lean_object* v_reuseFailAlloc_219_; 
v_reuseFailAlloc_219_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_219_, 0, v___x_216_);
lean_ctor_set(v_reuseFailAlloc_219_, 1, v_snd_211_);
v___x_218_ = v_reuseFailAlloc_219_;
goto v_reusejp_217_;
}
v_reusejp_217_:
{
return v___x_218_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnionFind_find_x3f(lean_object* v_00_u03b1_223_, lean_object* v_inst_224_, lean_object* v_inst_225_, lean_object* v_x_226_, lean_object* v_u_227_){
_start:
{
lean_object* v___x_228_; 
v___x_228_ = lp_aesop_Aesop_UnionFind_find_x3f___redArg(v_inst_224_, v_inst_225_, v_x_226_, v_u_227_);
return v___x_228_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_UnionFind_0__Aesop_UnionFind_mergeUnsafe___redArg(lean_object* v_inst_229_, lean_object* v_inst_230_, lean_object* v_x_231_, lean_object* v_y_232_, lean_object* v_u_233_){
_start:
{
lean_object* v___x_234_; lean_object* v_fst_235_; 
lean_inc_ref(v_u_233_);
lean_inc_ref(v_inst_230_);
lean_inc_ref(v_inst_229_);
v___x_234_ = lp_aesop_Aesop_UnionFind_find_x3f___redArg(v_inst_229_, v_inst_230_, v_x_231_, v_u_233_);
v_fst_235_ = lean_ctor_get(v___x_234_, 0);
lean_inc(v_fst_235_);
if (lean_obj_tag(v_fst_235_) == 1)
{
lean_object* v_snd_236_; lean_object* v_val_237_; lean_object* v___x_238_; lean_object* v_fst_239_; 
lean_dec_ref(v_u_233_);
v_snd_236_ = lean_ctor_get(v___x_234_, 1);
lean_inc_n(v_snd_236_, 2);
lean_dec_ref(v___x_234_);
v_val_237_ = lean_ctor_get(v_fst_235_, 0);
lean_inc(v_val_237_);
lean_dec_ref_known(v_fst_235_, 1);
v___x_238_ = lp_aesop_Aesop_UnionFind_find_x3f___redArg(v_inst_229_, v_inst_230_, v_y_232_, v_snd_236_);
v_fst_239_ = lean_ctor_get(v___x_238_, 0);
lean_inc(v_fst_239_);
if (lean_obj_tag(v_fst_239_) == 1)
{
lean_object* v_snd_240_; lean_object* v_val_241_; size_t v___x_242_; size_t v___x_243_; uint8_t v___x_244_; 
lean_dec(v_snd_236_);
v_snd_240_ = lean_ctor_get(v___x_238_, 1);
lean_inc(v_snd_240_);
lean_dec_ref(v___x_238_);
v_val_241_ = lean_ctor_get(v_fst_239_, 0);
lean_inc(v_val_241_);
lean_dec_ref_known(v_fst_239_, 1);
v___x_242_ = lean_unbox_usize(v_val_237_);
v___x_243_ = lean_unbox_usize(v_val_241_);
v___x_244_ = lean_usize_dec_eq(v___x_242_, v___x_243_);
if (v___x_244_ == 0)
{
lean_object* v_parents_245_; lean_object* v_sizes_246_; lean_object* v_toRep_247_; lean_object* v___x_249_; uint8_t v_isShared_250_; uint8_t v_isSharedCheck_280_; 
v_parents_245_ = lean_ctor_get(v_snd_240_, 0);
v_sizes_246_ = lean_ctor_get(v_snd_240_, 1);
v_toRep_247_ = lean_ctor_get(v_snd_240_, 2);
v_isSharedCheck_280_ = !lean_is_exclusive(v_snd_240_);
if (v_isSharedCheck_280_ == 0)
{
v___x_249_ = v_snd_240_;
v_isShared_250_ = v_isSharedCheck_280_;
goto v_resetjp_248_;
}
else
{
lean_inc(v_toRep_247_);
lean_inc(v_sizes_246_);
lean_inc(v_parents_245_);
lean_dec(v_snd_240_);
v___x_249_ = lean_box(0);
v_isShared_250_ = v_isSharedCheck_280_;
goto v_resetjp_248_;
}
v_resetjp_248_:
{
size_t v___x_251_; lean_object* v_xSize_252_; size_t v___x_253_; lean_object* v_ySize_254_; size_t v___x_255_; size_t v___x_256_; uint8_t v___x_257_; 
v___x_251_ = lean_unbox_usize(v_val_237_);
v_xSize_252_ = lean_array_uget_borrowed(v_sizes_246_, v___x_251_);
v___x_253_ = lean_unbox_usize(v_val_241_);
v_ySize_254_ = lean_array_uget_borrowed(v_sizes_246_, v___x_253_);
v___x_255_ = lean_unbox_usize(v_xSize_252_);
v___x_256_ = lean_unbox_usize(v_ySize_254_);
v___x_257_ = lean_usize_dec_lt(v___x_255_, v___x_256_);
if (v___x_257_ == 0)
{
size_t v___x_258_; lean_object* v___x_259_; size_t v___x_260_; size_t v___x_261_; size_t v___x_262_; size_t v___x_263_; lean_object* v___x_264_; lean_object* v___x_265_; lean_object* v___x_267_; 
v___x_258_ = lean_unbox_usize(v_val_241_);
lean_dec(v_val_241_);
lean_inc(v_val_237_);
v___x_259_ = lean_array_uset(v_parents_245_, v___x_258_, v_val_237_);
v___x_260_ = lean_unbox_usize(v_xSize_252_);
v___x_261_ = lean_unbox_usize(v_ySize_254_);
v___x_262_ = lean_usize_add(v___x_260_, v___x_261_);
v___x_263_ = lean_unbox_usize(v_val_237_);
lean_dec(v_val_237_);
v___x_264_ = lean_box_usize(v___x_262_);
v___x_265_ = lean_array_uset(v_sizes_246_, v___x_263_, v___x_264_);
if (v_isShared_250_ == 0)
{
lean_ctor_set(v___x_249_, 1, v___x_265_);
lean_ctor_set(v___x_249_, 0, v___x_259_);
v___x_267_ = v___x_249_;
goto v_reusejp_266_;
}
else
{
lean_object* v_reuseFailAlloc_268_; 
v_reuseFailAlloc_268_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_268_, 0, v___x_259_);
lean_ctor_set(v_reuseFailAlloc_268_, 1, v___x_265_);
lean_ctor_set(v_reuseFailAlloc_268_, 2, v_toRep_247_);
v___x_267_ = v_reuseFailAlloc_268_;
goto v_reusejp_266_;
}
v_reusejp_266_:
{
return v___x_267_;
}
}
else
{
size_t v___x_269_; lean_object* v___x_270_; size_t v___x_271_; size_t v___x_272_; size_t v___x_273_; size_t v___x_274_; lean_object* v___x_275_; lean_object* v___x_276_; lean_object* v___x_278_; 
v___x_269_ = lean_unbox_usize(v_val_237_);
lean_dec(v_val_237_);
lean_inc(v_val_241_);
v___x_270_ = lean_array_uset(v_parents_245_, v___x_269_, v_val_241_);
v___x_271_ = lean_unbox_usize(v_xSize_252_);
v___x_272_ = lean_unbox_usize(v_ySize_254_);
v___x_273_ = lean_usize_add(v___x_271_, v___x_272_);
v___x_274_ = lean_unbox_usize(v_val_241_);
lean_dec(v_val_241_);
v___x_275_ = lean_box_usize(v___x_273_);
v___x_276_ = lean_array_uset(v_sizes_246_, v___x_274_, v___x_275_);
if (v_isShared_250_ == 0)
{
lean_ctor_set(v___x_249_, 1, v___x_276_);
lean_ctor_set(v___x_249_, 0, v___x_270_);
v___x_278_ = v___x_249_;
goto v_reusejp_277_;
}
else
{
lean_object* v_reuseFailAlloc_279_; 
v_reuseFailAlloc_279_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_279_, 0, v___x_270_);
lean_ctor_set(v_reuseFailAlloc_279_, 1, v___x_276_);
lean_ctor_set(v_reuseFailAlloc_279_, 2, v_toRep_247_);
v___x_278_ = v_reuseFailAlloc_279_;
goto v_reusejp_277_;
}
v_reusejp_277_:
{
return v___x_278_;
}
}
}
}
else
{
lean_dec(v_val_241_);
lean_dec(v_val_237_);
return v_snd_240_;
}
}
else
{
lean_dec(v_fst_239_);
lean_dec_ref(v___x_238_);
lean_dec(v_val_237_);
return v_snd_236_;
}
}
else
{
lean_dec(v_fst_235_);
lean_dec_ref(v___x_234_);
lean_dec(v_y_232_);
lean_dec_ref(v_inst_230_);
lean_dec_ref(v_inst_229_);
return v_u_233_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Util_UnionFind_0__Aesop_UnionFind_mergeUnsafe(lean_object* v_00_u03b1_281_, lean_object* v_inst_282_, lean_object* v_inst_283_, lean_object* v_x_284_, lean_object* v_y_285_, lean_object* v_u_286_){
_start:
{
lean_object* v___x_287_; 
v___x_287_ = lp_aesop___private_Aesop_Util_UnionFind_0__Aesop_UnionFind_mergeUnsafe___redArg(v_inst_282_, v_inst_283_, v_x_284_, v_y_285_, v_u_286_);
return v___x_287_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnionFind_sets___redArg___lam__0(lean_object* v_x1_288_, size_t v_x2_289_, lean_object* v_x3_290_){
_start:
{
lean_object* v___x_291_; 
v___x_291_ = lean_array_push(v_x1_288_, v_x3_290_);
return v___x_291_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnionFind_sets___redArg___lam__0___boxed(lean_object* v_x1_292_, lean_object* v_x2_293_, lean_object* v_x3_294_){
_start:
{
size_t v_x2_469__boxed_295_; lean_object* v_res_296_; 
v_x2_469__boxed_295_ = lean_unbox_usize(v_x2_293_);
lean_dec(v_x2_293_);
v_res_296_ = lp_aesop_Aesop_UnionFind_sets___redArg___lam__0(v_x1_292_, v_x2_469__boxed_295_, v_x3_294_);
return v_res_296_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnionFind_sets___redArg___lam__1(lean_object* v___x_297_, lean_object* v___f_298_, lean_object* v_acc_299_, lean_object* v_l_300_){
_start:
{
lean_object* v___x_301_; 
v___x_301_ = l_Std_DHashMap_Internal_AssocList_foldlM___redArg(v___x_297_, v___f_298_, v_acc_299_, v_l_300_);
return v___x_301_;
}
}
static lean_object* _init_lp_aesop_Aesop_UnionFind_sets___redArg___lam__2___closed__0(void){
_start:
{
lean_object* v___x_302_; lean_object* v___f_303_; 
v___x_302_ = lean_alloc_closure((void*)(l_instDecidableEqUSize___boxed), 2, 0);
v___f_303_ = lean_alloc_closure((void*)(l_instBEqOfDecidableEq___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_303_, 0, v___x_302_);
return v___f_303_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnionFind_sets___redArg___lam__2(lean_object* v___f_304_, lean_object* v_x1_305_, lean_object* v_x2_306_, size_t v_x3_307_){
_start:
{
lean_object* v_fst_308_; lean_object* v_snd_309_; lean_object* v___x_310_; lean_object* v_fst_311_; lean_object* v_snd_312_; lean_object* v___x_314_; uint8_t v_isShared_315_; uint8_t v_isSharedCheck_331_; 
v_fst_308_ = lean_ctor_get(v_x1_305_, 0);
lean_inc(v_fst_308_);
v_snd_309_ = lean_ctor_get(v_x1_305_, 1);
lean_inc(v_snd_309_);
lean_dec_ref(v_x1_305_);
v___x_310_ = lp_aesop___private_Aesop_Util_UnionFind_0__Aesop_UnionFind_findRepUnsafe___redArg(v_x3_307_, v_snd_309_);
v_fst_311_ = lean_ctor_get(v___x_310_, 0);
v_snd_312_ = lean_ctor_get(v___x_310_, 1);
v_isSharedCheck_331_ = !lean_is_exclusive(v___x_310_);
if (v_isSharedCheck_331_ == 0)
{
v___x_314_ = v___x_310_;
v_isShared_315_ = v_isSharedCheck_331_;
goto v_resetjp_313_;
}
else
{
lean_inc(v_snd_312_);
lean_inc(v_fst_311_);
lean_dec(v___x_310_);
v___x_314_ = lean_box(0);
v_isShared_315_ = v_isSharedCheck_331_;
goto v_resetjp_313_;
}
v_resetjp_313_:
{
lean_object* v___f_316_; lean_object* v___x_317_; 
v___f_316_ = lean_obj_once(&lp_aesop_Aesop_UnionFind_sets___redArg___lam__2___closed__0, &lp_aesop_Aesop_UnionFind_sets___redArg___lam__2___closed__0_once, _init_lp_aesop_Aesop_UnionFind_sets___redArg___lam__2___closed__0);
lean_inc(v_fst_311_);
lean_inc_ref(v___f_304_);
v___x_317_ = l_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___redArg(v___f_316_, v___f_304_, v_fst_308_, v_fst_311_);
if (lean_obj_tag(v___x_317_) == 0)
{
lean_object* v___x_318_; lean_object* v___x_319_; lean_object* v___x_320_; lean_object* v___x_321_; lean_object* v___x_323_; 
v___x_318_ = lean_unsigned_to_nat(1u);
v___x_319_ = lean_mk_empty_array_with_capacity(v___x_318_);
v___x_320_ = lean_array_push(v___x_319_, v_x2_306_);
v___x_321_ = l_Std_DHashMap_Internal_Raw_u2080_insert___redArg(v___f_316_, v___f_304_, v_fst_308_, v_fst_311_, v___x_320_);
if (v_isShared_315_ == 0)
{
lean_ctor_set(v___x_314_, 0, v___x_321_);
v___x_323_ = v___x_314_;
goto v_reusejp_322_;
}
else
{
lean_object* v_reuseFailAlloc_324_; 
v_reuseFailAlloc_324_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_324_, 0, v___x_321_);
lean_ctor_set(v_reuseFailAlloc_324_, 1, v_snd_312_);
v___x_323_ = v_reuseFailAlloc_324_;
goto v_reusejp_322_;
}
v_reusejp_322_:
{
return v___x_323_;
}
}
else
{
lean_object* v_val_325_; lean_object* v___x_326_; lean_object* v___x_327_; lean_object* v___x_329_; 
v_val_325_ = lean_ctor_get(v___x_317_, 0);
lean_inc(v_val_325_);
lean_dec_ref_known(v___x_317_, 1);
v___x_326_ = lean_array_push(v_val_325_, v_x2_306_);
v___x_327_ = l_Std_DHashMap_Internal_Raw_u2080_insert___redArg(v___f_316_, v___f_304_, v_fst_308_, v_fst_311_, v___x_326_);
if (v_isShared_315_ == 0)
{
lean_ctor_set(v___x_314_, 0, v___x_327_);
v___x_329_ = v___x_314_;
goto v_reusejp_328_;
}
else
{
lean_object* v_reuseFailAlloc_330_; 
v_reuseFailAlloc_330_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_330_, 0, v___x_327_);
lean_ctor_set(v_reuseFailAlloc_330_, 1, v_snd_312_);
v___x_329_ = v_reuseFailAlloc_330_;
goto v_reusejp_328_;
}
v_reusejp_328_:
{
return v___x_329_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnionFind_sets___redArg___lam__2___boxed(lean_object* v___f_332_, lean_object* v_x1_333_, lean_object* v_x2_334_, lean_object* v_x3_335_){
_start:
{
size_t v_x3_492__boxed_336_; lean_object* v_res_337_; 
v_x3_492__boxed_336_ = lean_unbox_usize(v_x3_335_);
lean_dec(v_x3_335_);
v_res_337_ = lp_aesop_Aesop_UnionFind_sets___redArg___lam__2(v___f_332_, v_x1_333_, v_x2_334_, v_x3_492__boxed_336_);
return v_res_337_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnionFind_sets___redArg___lam__3(lean_object* v___x_338_, lean_object* v___f_339_, lean_object* v_acc_340_, lean_object* v_l_341_){
_start:
{
lean_object* v___x_342_; 
v___x_342_ = l_Std_DHashMap_Internal_AssocList_foldlM___redArg(v___x_338_, v___f_339_, v_acc_340_, v_l_341_);
return v___x_342_;
}
}
static lean_object* _init_lp_aesop_Aesop_UnionFind_sets___redArg___closed__2(void){
_start:
{
lean_object* v___x_347_; lean_object* v___x_348_; lean_object* v___x_349_; 
v___x_347_ = lean_box(0);
v___x_348_ = lean_unsigned_to_nat(16u);
v___x_349_ = lean_mk_array(v___x_348_, v___x_347_);
return v___x_349_;
}
}
static lean_object* _init_lp_aesop_Aesop_UnionFind_sets___redArg___closed__3(void){
_start:
{
lean_object* v___x_350_; lean_object* v___x_351_; lean_object* v___x_352_; 
v___x_350_ = lean_obj_once(&lp_aesop_Aesop_UnionFind_sets___redArg___closed__2, &lp_aesop_Aesop_UnionFind_sets___redArg___closed__2_once, _init_lp_aesop_Aesop_UnionFind_sets___redArg___closed__2);
v___x_351_ = lean_unsigned_to_nat(0u);
v___x_352_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_352_, 0, v___x_351_);
lean_ctor_set(v___x_352_, 1, v___x_350_);
return v___x_352_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnionFind_sets___redArg(lean_object* v_u_359_){
_start:
{
lean_object* v_toRep_360_; lean_object* v___x_361_; lean_object* v_buckets_362_; lean_object* v___x_364_; uint8_t v_isShared_365_; uint8_t v_isSharedCheck_414_; 
v_toRep_360_ = lean_ctor_get(v_u_359_, 2);
lean_inc_ref(v_toRep_360_);
v___x_361_ = ((lean_object*)(lp_aesop_Aesop_UnionFind_addArray___redArg___closed__9));
v_buckets_362_ = lean_ctor_get(v_toRep_360_, 1);
v_isSharedCheck_414_ = !lean_is_exclusive(v_toRep_360_);
if (v_isSharedCheck_414_ == 0)
{
lean_object* v_unused_415_; 
v_unused_415_ = lean_ctor_get(v_toRep_360_, 0);
lean_dec(v_unused_415_);
v___x_364_ = v_toRep_360_;
v_isShared_365_ = v_isSharedCheck_414_;
goto v_resetjp_363_;
}
else
{
lean_inc(v_buckets_362_);
lean_dec(v_toRep_360_);
v___x_364_ = lean_box(0);
v_isShared_365_ = v_isSharedCheck_414_;
goto v_resetjp_363_;
}
v_resetjp_363_:
{
lean_object* v_size_367_; lean_object* v_buckets_368_; lean_object* v_snd_369_; lean_object* v___y_395_; lean_object* v___x_400_; lean_object* v___x_401_; lean_object* v___x_402_; uint8_t v___x_403_; 
v___x_400_ = lean_unsigned_to_nat(0u);
v___x_401_ = lean_obj_once(&lp_aesop_Aesop_UnionFind_sets___redArg___closed__2, &lp_aesop_Aesop_UnionFind_sets___redArg___closed__2_once, _init_lp_aesop_Aesop_UnionFind_sets___redArg___closed__2);
v___x_402_ = lean_array_get_size(v_buckets_362_);
v___x_403_ = lean_nat_dec_lt(v___x_400_, v___x_402_);
if (v___x_403_ == 0)
{
lean_dec_ref(v_buckets_362_);
v_size_367_ = v___x_400_;
v_buckets_368_ = v___x_401_;
v_snd_369_ = v_u_359_;
goto v___jp_366_;
}
else
{
lean_object* v___x_404_; lean_object* v___f_405_; lean_object* v___x_406_; uint8_t v___x_407_; 
v___x_404_ = lean_obj_once(&lp_aesop_Aesop_UnionFind_sets___redArg___closed__3, &lp_aesop_Aesop_UnionFind_sets___redArg___closed__3_once, _init_lp_aesop_Aesop_UnionFind_sets___redArg___closed__3);
v___f_405_ = ((lean_object*)(lp_aesop_Aesop_UnionFind_sets___redArg___closed__6));
lean_inc_ref(v_u_359_);
v___x_406_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_406_, 0, v___x_404_);
lean_ctor_set(v___x_406_, 1, v_u_359_);
v___x_407_ = lean_nat_dec_le(v___x_402_, v___x_402_);
if (v___x_407_ == 0)
{
if (v___x_403_ == 0)
{
lean_dec_ref_known(v___x_406_, 2);
lean_dec_ref(v_buckets_362_);
v_size_367_ = v___x_400_;
v_buckets_368_ = v___x_401_;
v_snd_369_ = v_u_359_;
goto v___jp_366_;
}
else
{
size_t v___x_408_; size_t v___x_409_; lean_object* v___x_410_; 
lean_dec_ref(v_u_359_);
v___x_408_ = ((size_t)0ULL);
v___x_409_ = lean_usize_of_nat(v___x_402_);
v___x_410_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_361_, v___f_405_, v_buckets_362_, v___x_408_, v___x_409_, v___x_406_);
v___y_395_ = v___x_410_;
goto v___jp_394_;
}
}
else
{
size_t v___x_411_; size_t v___x_412_; lean_object* v___x_413_; 
lean_dec_ref(v_u_359_);
v___x_411_ = ((size_t)0ULL);
v___x_412_ = lean_usize_of_nat(v___x_402_);
v___x_413_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_361_, v___f_405_, v_buckets_362_, v___x_411_, v___x_412_, v___x_406_);
v___y_395_ = v___x_413_;
goto v___jp_394_;
}
}
v___jp_366_:
{
lean_object* v___x_370_; lean_object* v___x_371_; lean_object* v___x_372_; uint8_t v___x_373_; 
v___x_370_ = lean_mk_empty_array_with_capacity(v_size_367_);
lean_dec(v_size_367_);
v___x_371_ = lean_unsigned_to_nat(0u);
v___x_372_ = lean_array_get_size(v_buckets_368_);
v___x_373_ = lean_nat_dec_lt(v___x_371_, v___x_372_);
if (v___x_373_ == 0)
{
lean_object* v___x_375_; 
lean_dec_ref(v_buckets_368_);
if (v_isShared_365_ == 0)
{
lean_ctor_set(v___x_364_, 1, v_snd_369_);
lean_ctor_set(v___x_364_, 0, v___x_370_);
v___x_375_ = v___x_364_;
goto v_reusejp_374_;
}
else
{
lean_object* v_reuseFailAlloc_376_; 
v_reuseFailAlloc_376_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_376_, 0, v___x_370_);
lean_ctor_set(v_reuseFailAlloc_376_, 1, v_snd_369_);
v___x_375_ = v_reuseFailAlloc_376_;
goto v_reusejp_374_;
}
v_reusejp_374_:
{
return v___x_375_;
}
}
else
{
lean_object* v___f_377_; uint8_t v___x_378_; 
v___f_377_ = ((lean_object*)(lp_aesop_Aesop_UnionFind_sets___redArg___closed__1));
v___x_378_ = lean_nat_dec_le(v___x_372_, v___x_372_);
if (v___x_378_ == 0)
{
if (v___x_373_ == 0)
{
lean_object* v___x_380_; 
lean_dec_ref(v_buckets_368_);
if (v_isShared_365_ == 0)
{
lean_ctor_set(v___x_364_, 1, v_snd_369_);
lean_ctor_set(v___x_364_, 0, v___x_370_);
v___x_380_ = v___x_364_;
goto v_reusejp_379_;
}
else
{
lean_object* v_reuseFailAlloc_381_; 
v_reuseFailAlloc_381_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_381_, 0, v___x_370_);
lean_ctor_set(v_reuseFailAlloc_381_, 1, v_snd_369_);
v___x_380_ = v_reuseFailAlloc_381_;
goto v_reusejp_379_;
}
v_reusejp_379_:
{
return v___x_380_;
}
}
else
{
size_t v___x_382_; size_t v___x_383_; lean_object* v___x_384_; lean_object* v___x_386_; 
v___x_382_ = ((size_t)0ULL);
v___x_383_ = lean_usize_of_nat(v___x_372_);
v___x_384_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_361_, v___f_377_, v_buckets_368_, v___x_382_, v___x_383_, v___x_370_);
if (v_isShared_365_ == 0)
{
lean_ctor_set(v___x_364_, 1, v_snd_369_);
lean_ctor_set(v___x_364_, 0, v___x_384_);
v___x_386_ = v___x_364_;
goto v_reusejp_385_;
}
else
{
lean_object* v_reuseFailAlloc_387_; 
v_reuseFailAlloc_387_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_387_, 0, v___x_384_);
lean_ctor_set(v_reuseFailAlloc_387_, 1, v_snd_369_);
v___x_386_ = v_reuseFailAlloc_387_;
goto v_reusejp_385_;
}
v_reusejp_385_:
{
return v___x_386_;
}
}
}
else
{
size_t v___x_388_; size_t v___x_389_; lean_object* v___x_390_; lean_object* v___x_392_; 
v___x_388_ = ((size_t)0ULL);
v___x_389_ = lean_usize_of_nat(v___x_372_);
v___x_390_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_361_, v___f_377_, v_buckets_368_, v___x_388_, v___x_389_, v___x_370_);
if (v_isShared_365_ == 0)
{
lean_ctor_set(v___x_364_, 1, v_snd_369_);
lean_ctor_set(v___x_364_, 0, v___x_390_);
v___x_392_ = v___x_364_;
goto v_reusejp_391_;
}
else
{
lean_object* v_reuseFailAlloc_393_; 
v_reuseFailAlloc_393_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_393_, 0, v___x_390_);
lean_ctor_set(v_reuseFailAlloc_393_, 1, v_snd_369_);
v___x_392_ = v_reuseFailAlloc_393_;
goto v_reusejp_391_;
}
v_reusejp_391_:
{
return v___x_392_;
}
}
}
}
v___jp_394_:
{
lean_object* v_fst_396_; lean_object* v_snd_397_; lean_object* v_size_398_; lean_object* v_buckets_399_; 
v_fst_396_ = lean_ctor_get(v___y_395_, 0);
lean_inc(v_fst_396_);
v_snd_397_ = lean_ctor_get(v___y_395_, 1);
lean_inc(v_snd_397_);
lean_dec_ref(v___y_395_);
v_size_398_ = lean_ctor_get(v_fst_396_, 0);
lean_inc(v_size_398_);
v_buckets_399_ = lean_ctor_get(v_fst_396_, 1);
lean_inc_ref(v_buckets_399_);
lean_dec(v_fst_396_);
v_size_367_ = v_size_398_;
v_buckets_368_ = v_buckets_399_;
v_snd_369_ = v_snd_397_;
goto v___jp_366_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnionFind_sets(lean_object* v_00_u03b1_416_, lean_object* v_inst_417_, lean_object* v_inst_418_, lean_object* v_u_419_){
_start:
{
lean_object* v___x_420_; 
v___x_420_ = lp_aesop_Aesop_UnionFind_sets___redArg(v_u_419_);
return v___x_420_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_UnionFind_sets___boxed(lean_object* v_00_u03b1_421_, lean_object* v_inst_422_, lean_object* v_inst_423_, lean_object* v_u_424_){
_start:
{
lean_object* v_res_425_; 
v_res_425_ = lp_aesop_Aesop_UnionFind_sets(v_00_u03b1_421_, v_inst_422_, v_inst_423_, v_u_424_);
lean_dec_ref(v_inst_423_);
lean_dec_ref(v_inst_422_);
return v_res_425_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_cluster___redArg___lam__0(lean_object* v_inst_426_, lean_object* v_inst_427_, lean_object* v_a_428_, lean_object* v_a_429_, lean_object* v_x_430_, lean_object* v___y_431_){
_start:
{
lean_object* v___x_432_; lean_object* v___x_433_; 
v___x_432_ = lp_aesop___private_Aesop_Util_UnionFind_0__Aesop_UnionFind_mergeUnsafe___redArg(v_inst_426_, v_inst_427_, v_a_428_, v_a_429_, v___y_431_);
v___x_433_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_433_, 0, v___x_432_);
return v___x_433_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_cluster___redArg___lam__1(lean_object* v_inst_434_, lean_object* v_inst_435_, lean_object* v_a_436_, lean_object* v___x_437_, lean_object* v___f_438_, lean_object* v_a_439_, lean_object* v_x_440_, lean_object* v___y_441_){
_start:
{
lean_object* v_fst_442_; lean_object* v_snd_443_; lean_object* v___x_445_; uint8_t v_isShared_446_; uint8_t v_isSharedCheck_472_; 
v_fst_442_ = lean_ctor_get(v___y_441_, 0);
v_snd_443_ = lean_ctor_get(v___y_441_, 1);
v_isSharedCheck_472_ = !lean_is_exclusive(v___y_441_);
if (v_isSharedCheck_472_ == 0)
{
v___x_445_ = v___y_441_;
v_isShared_446_ = v_isSharedCheck_472_;
goto v_resetjp_444_;
}
else
{
lean_inc(v_snd_443_);
lean_inc(v_fst_442_);
lean_dec(v___y_441_);
v___x_445_ = lean_box(0);
v_isShared_446_ = v_isSharedCheck_472_;
goto v_resetjp_444_;
}
v_resetjp_444_:
{
lean_object* v___x_447_; 
lean_inc(v_a_439_);
lean_inc_ref(v_inst_435_);
lean_inc_ref(v_inst_434_);
v___x_447_ = l_Std_DHashMap_Internal_Raw_u2080_Const_get_x3f___redArg(v_inst_434_, v_inst_435_, v_snd_443_, v_a_439_);
if (lean_obj_tag(v___x_447_) == 0)
{
lean_object* v___x_448_; lean_object* v___x_449_; lean_object* v___x_450_; lean_object* v___x_451_; lean_object* v___x_453_; 
lean_dec_ref(v___f_438_);
lean_dec_ref(v___x_437_);
v___x_448_ = lean_unsigned_to_nat(1u);
v___x_449_ = lean_mk_empty_array_with_capacity(v___x_448_);
v___x_450_ = lean_array_push(v___x_449_, v_a_436_);
v___x_451_ = l_Std_DHashMap_Internal_Raw_u2080_insert___redArg(v_inst_434_, v_inst_435_, v_snd_443_, v_a_439_, v___x_450_);
if (v_isShared_446_ == 0)
{
lean_ctor_set(v___x_445_, 1, v___x_451_);
v___x_453_ = v___x_445_;
goto v_reusejp_452_;
}
else
{
lean_object* v_reuseFailAlloc_455_; 
v_reuseFailAlloc_455_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_455_, 0, v_fst_442_);
lean_ctor_set(v_reuseFailAlloc_455_, 1, v___x_451_);
v___x_453_ = v_reuseFailAlloc_455_;
goto v_reusejp_452_;
}
v_reusejp_452_:
{
lean_object* v___x_454_; 
v___x_454_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_454_, 0, v___x_453_);
return v___x_454_;
}
}
else
{
lean_object* v_val_456_; lean_object* v___x_458_; uint8_t v_isShared_459_; uint8_t v_isSharedCheck_471_; 
v_val_456_ = lean_ctor_get(v___x_447_, 0);
v_isSharedCheck_471_ = !lean_is_exclusive(v___x_447_);
if (v_isSharedCheck_471_ == 0)
{
v___x_458_ = v___x_447_;
v_isShared_459_ = v_isSharedCheck_471_;
goto v_resetjp_457_;
}
else
{
lean_inc(v_val_456_);
lean_dec(v___x_447_);
v___x_458_ = lean_box(0);
v_isShared_459_ = v_isSharedCheck_471_;
goto v_resetjp_457_;
}
v_resetjp_457_:
{
size_t v_sz_460_; size_t v___x_461_; lean_object* v___x_462_; lean_object* v___x_463_; lean_object* v___x_464_; lean_object* v___x_466_; 
v_sz_460_ = lean_array_size(v_val_456_);
v___x_461_ = ((size_t)0ULL);
lean_inc(v_val_456_);
v___x_462_ = l___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop(lean_box(0), lean_box(0), lean_box(0), v___x_437_, v_val_456_, v___f_438_, v_sz_460_, v___x_461_, v_fst_442_);
v___x_463_ = lean_array_push(v_val_456_, v_a_436_);
v___x_464_ = l_Std_DHashMap_Internal_Raw_u2080_insert___redArg(v_inst_434_, v_inst_435_, v_snd_443_, v_a_439_, v___x_463_);
if (v_isShared_446_ == 0)
{
lean_ctor_set(v___x_445_, 1, v___x_464_);
lean_ctor_set(v___x_445_, 0, v___x_462_);
v___x_466_ = v___x_445_;
goto v_reusejp_465_;
}
else
{
lean_object* v_reuseFailAlloc_470_; 
v_reuseFailAlloc_470_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_470_, 0, v___x_462_);
lean_ctor_set(v_reuseFailAlloc_470_, 1, v___x_464_);
v___x_466_ = v_reuseFailAlloc_470_;
goto v_reusejp_465_;
}
v_reusejp_465_:
{
lean_object* v___x_468_; 
if (v_isShared_459_ == 0)
{
lean_ctor_set(v___x_458_, 0, v___x_466_);
v___x_468_ = v___x_458_;
goto v_reusejp_467_;
}
else
{
lean_object* v_reuseFailAlloc_469_; 
v_reuseFailAlloc_469_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_469_, 0, v___x_466_);
v___x_468_ = v_reuseFailAlloc_469_;
goto v_reusejp_467_;
}
v_reusejp_467_:
{
return v___x_468_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_cluster___redArg___lam__2(lean_object* v_inst_473_, lean_object* v_inst_474_, lean_object* v_inst_475_, lean_object* v_inst_476_, lean_object* v___x_477_, lean_object* v_f_478_, lean_object* v_a_479_, lean_object* v_x_480_, lean_object* v___y_481_){
_start:
{
lean_object* v_fst_482_; lean_object* v_snd_483_; lean_object* v___x_485_; uint8_t v_isShared_486_; uint8_t v_isSharedCheck_506_; 
v_fst_482_ = lean_ctor_get(v___y_481_, 0);
v_snd_483_ = lean_ctor_get(v___y_481_, 1);
v_isSharedCheck_506_ = !lean_is_exclusive(v___y_481_);
if (v_isSharedCheck_506_ == 0)
{
v___x_485_ = v___y_481_;
v_isShared_486_ = v_isSharedCheck_506_;
goto v_resetjp_484_;
}
else
{
lean_inc(v_snd_483_);
lean_inc(v_fst_482_);
lean_dec(v___y_481_);
v___x_485_ = lean_box(0);
v_isShared_486_ = v_isSharedCheck_506_;
goto v_resetjp_484_;
}
v_resetjp_484_:
{
lean_object* v___f_487_; lean_object* v___f_488_; lean_object* v___x_489_; lean_object* v___x_491_; 
lean_inc_n(v_a_479_, 2);
v___f_487_ = lean_alloc_closure((void*)(lp_aesop_Aesop_cluster___redArg___lam__0), 6, 3);
lean_closure_set(v___f_487_, 0, v_inst_473_);
lean_closure_set(v___f_487_, 1, v_inst_474_);
lean_closure_set(v___f_487_, 2, v_a_479_);
lean_inc_ref(v___x_477_);
v___f_488_ = lean_alloc_closure((void*)(lp_aesop_Aesop_cluster___redArg___lam__1), 8, 5);
lean_closure_set(v___f_488_, 0, v_inst_475_);
lean_closure_set(v___f_488_, 1, v_inst_476_);
lean_closure_set(v___f_488_, 2, v_a_479_);
lean_closure_set(v___f_488_, 3, v___x_477_);
lean_closure_set(v___f_488_, 4, v___f_487_);
v___x_489_ = lean_apply_1(v_f_478_, v_a_479_);
if (v_isShared_486_ == 0)
{
v___x_491_ = v___x_485_;
goto v_reusejp_490_;
}
else
{
lean_object* v_reuseFailAlloc_505_; 
v_reuseFailAlloc_505_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_505_, 0, v_fst_482_);
lean_ctor_set(v_reuseFailAlloc_505_, 1, v_snd_483_);
v___x_491_ = v_reuseFailAlloc_505_;
goto v_reusejp_490_;
}
v_reusejp_490_:
{
size_t v_sz_492_; size_t v___x_493_; lean_object* v___x_494_; lean_object* v_fst_495_; lean_object* v_snd_496_; lean_object* v___x_498_; uint8_t v_isShared_499_; uint8_t v_isSharedCheck_504_; 
v_sz_492_ = lean_array_size(v___x_489_);
v___x_493_ = ((size_t)0ULL);
v___x_494_ = l___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop(lean_box(0), lean_box(0), lean_box(0), v___x_477_, v___x_489_, v___f_488_, v_sz_492_, v___x_493_, v___x_491_);
v_fst_495_ = lean_ctor_get(v___x_494_, 0);
v_snd_496_ = lean_ctor_get(v___x_494_, 1);
v_isSharedCheck_504_ = !lean_is_exclusive(v___x_494_);
if (v_isSharedCheck_504_ == 0)
{
v___x_498_ = v___x_494_;
v_isShared_499_ = v_isSharedCheck_504_;
goto v_resetjp_497_;
}
else
{
lean_inc(v_snd_496_);
lean_inc(v_fst_495_);
lean_dec(v___x_494_);
v___x_498_ = lean_box(0);
v_isShared_499_ = v_isSharedCheck_504_;
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
lean_object* v_reuseFailAlloc_503_; 
v_reuseFailAlloc_503_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_503_, 0, v_fst_495_);
lean_ctor_set(v_reuseFailAlloc_503_, 1, v_snd_496_);
v___x_501_ = v_reuseFailAlloc_503_;
goto v_reusejp_500_;
}
v_reusejp_500_:
{
lean_object* v___x_502_; 
v___x_502_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_502_, 0, v___x_501_);
return v___x_502_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_cluster___redArg(lean_object* v_inst_507_, lean_object* v_inst_508_, lean_object* v_inst_509_, lean_object* v_inst_510_, lean_object* v_f_511_, lean_object* v_as_512_){
_start:
{
lean_object* v___x_513_; lean_object* v___f_514_; lean_object* v_clusters_515_; lean_object* v_aOccs_516_; lean_object* v___x_517_; size_t v_sz_518_; size_t v___x_519_; lean_object* v___x_520_; lean_object* v_fst_521_; lean_object* v___x_522_; lean_object* v_fst_523_; 
v___x_513_ = ((lean_object*)(lp_aesop_Aesop_UnionFind_addArray___redArg___closed__9));
lean_inc_ref(v_inst_508_);
lean_inc_ref(v_inst_507_);
v___f_514_ = lean_alloc_closure((void*)(lp_aesop_Aesop_cluster___redArg___lam__2), 9, 6);
lean_closure_set(v___f_514_, 0, v_inst_507_);
lean_closure_set(v___f_514_, 1, v_inst_508_);
lean_closure_set(v___f_514_, 2, v_inst_509_);
lean_closure_set(v___f_514_, 3, v_inst_510_);
lean_closure_set(v___f_514_, 4, v___x_513_);
lean_closure_set(v___f_514_, 5, v_f_511_);
lean_inc_ref(v_as_512_);
v_clusters_515_ = lp_aesop_Aesop_UnionFind_ofArray___redArg(v_inst_507_, v_inst_508_, v_as_512_);
v_aOccs_516_ = lean_obj_once(&lp_aesop_Aesop_instInhabitedUnionFind_default___closed__2, &lp_aesop_Aesop_instInhabitedUnionFind_default___closed__2_once, _init_lp_aesop_Aesop_instInhabitedUnionFind_default___closed__2);
v___x_517_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_517_, 0, v_clusters_515_);
lean_ctor_set(v___x_517_, 1, v_aOccs_516_);
v_sz_518_ = lean_array_size(v_as_512_);
v___x_519_ = ((size_t)0ULL);
v___x_520_ = l___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop(lean_box(0), lean_box(0), lean_box(0), v___x_513_, v_as_512_, v___f_514_, v_sz_518_, v___x_519_, v___x_517_);
v_fst_521_ = lean_ctor_get(v___x_520_, 0);
lean_inc(v_fst_521_);
lean_dec(v___x_520_);
v___x_522_ = lp_aesop_Aesop_UnionFind_sets___redArg(v_fst_521_);
v_fst_523_ = lean_ctor_get(v___x_522_, 0);
lean_inc(v_fst_523_);
lean_dec_ref(v___x_522_);
return v_fst_523_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_cluster(lean_object* v_00_u03b1_524_, lean_object* v_00_u03b2_525_, lean_object* v_inst_526_, lean_object* v_inst_527_, lean_object* v_inst_528_, lean_object* v_inst_529_, lean_object* v_f_530_, lean_object* v_as_531_){
_start:
{
lean_object* v___x_532_; 
v___x_532_ = lp_aesop_Aesop_cluster___redArg(v_inst_526_, v_inst_527_, v_inst_528_, v_inst_529_, v_f_530_, v_as_531_);
return v___x_532_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Std_Data_HashMap_Basic(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Util_UnionFind(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize_runtime_module();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Std_Data_HashMap_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Util_UnionFind(uint8_t builtin) {
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
lean_object* initialize_Std_Data_HashMap_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Util_UnionFind(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Std_Data_HashMap_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Util_UnionFind(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Util_UnionFind(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Util_UnionFind(builtin);
}
#ifdef __cplusplus
}
#endif
