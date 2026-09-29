// Lean compiler output
// Module: Mathlib.Tactic.Order.Graph.Basic
// Imports: public import Init public meta import Init public import Mathlib.Tactic.Order.CollectFacts public meta import Mathlib.Util.AtomM
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
lean_object* l_mkPanicMessageWithDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_instInhabited(lean_object*);
lean_object* lean_panic_fn_borrowed(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
size_t lean_usize_add(size_t, size_t);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* lean_array_get_size(lean_object*);
uint64_t lean_uint64_of_nat(lean_object*);
uint64_t lean_uint64_shift_right(uint64_t, uint64_t);
uint64_t lean_uint64_xor(uint64_t, uint64_t);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_of_nat(lean_object*);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* lean_nat_div(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* lean_st_ref_get(lean_object*);
extern lean_object* l_Lean_instInhabitedExpr;
lean_object* lean_array_get(lean_object*, lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Meta_mkAppM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order_instToStringEdge___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 3, .m_data = " ⟶ "};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_instToStringEdge___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_instToStringEdge___lam__0___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_instToStringEdge___lam__0(lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_Order_instToStringEdge___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_Order_instToStringEdge___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_instToStringEdge___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_instToStringEdge___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_Order_instToStringEdge = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_instToStringEdge___closed__0_value;
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00Mathlib_Tactic_Order_Graph_addEdge_spec__0_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00Mathlib_Tactic_Order_Graph_addEdge_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00Mathlib_Tactic_Order_Graph_addEdge_spec__0_spec__1_spec__2_spec__3___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00Mathlib_Tactic_Order_Graph_addEdge_spec__0_spec__1_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00Mathlib_Tactic_Order_Graph_addEdge_spec__0_spec__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_Const_alter___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00Mathlib_Tactic_Order_Graph_addEdge_spec__0_spec__2___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_Const_alter___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00Mathlib_Tactic_Order_Graph_addEdge_spec__0_spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00Mathlib_Tactic_Order_Graph_addEdge_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_Graph_addEdge(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00Mathlib_Tactic_Order_Graph_addEdge_spec__0_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00Mathlib_Tactic_Order_Graph_addEdge_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00Mathlib_Tactic_Order_Graph_addEdge_spec__0_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00Mathlib_Tactic_Order_Graph_addEdge_spec__0_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00Mathlib_Tactic_Order_Graph_addEdge_spec__0_spec__1_spec__2_spec__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_Graph_constructLeGraph_spec__0___redArg(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_Graph_constructLeGraph_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Order_Graph_constructLeGraph___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Order_Graph_constructLeGraph___closed__0;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Order_Graph_constructLeGraph___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Order_Graph_constructLeGraph___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_Graph_constructLeGraph(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_Graph_constructLeGraph___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_Graph_constructLeGraph_spec__0(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_Graph_constructLeGraph_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__1___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__0___redArg(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_panic___at___00Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__2_spec__2_spec__3___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_panic___at___00Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__2_spec__2_spec__3___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__2_spec__2_spec__3(lean_object*);
static const lean_string_object lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__2_spec__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 43, .m_capacity = 43, .m_length = 42, .m_data = "Std.Data.DHashMap.Internal.AssocList.Basic"};
static const lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__2_spec__2___closed__0 = (const lean_object*)&lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__2_spec__2___closed__0_value;
static const lean_string_object lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__2_spec__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 37, .m_capacity = 37, .m_length = 36, .m_data = "Std.DHashMap.Internal.AssocList.get!"};
static const lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__2_spec__2___closed__1 = (const lean_object*)&lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__2_spec__2___closed__1_value;
static const lean_string_object lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__2_spec__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 33, .m_capacity = 33, .m_length = 32, .m_data = "key is not present in hash table"};
static const lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__2_spec__2___closed__2 = (const lean_object*)&lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__2_spec__2___closed__2_value;
static lean_once_cell_t lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__2_spec__2___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__2_spec__2___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__2_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__2_spec__2___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__2___boxed(lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__3___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__3___closed__0_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__3___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "le_trans"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__3___closed__1 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__3___closed__1_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__3___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__3___closed__1_value),LEAN_SCALAR_PTR_LITERAL(153, 164, 114, 182, 61, 254, 17, 252)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__3___closed__2 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__3___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "le_refl"};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS___closed__0_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS___closed__0_value),LEAN_SCALAR_PTR_LITERAL(247, 235, 88, 170, 79, 98, 97, 12)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_Graph_buildTransitiveLeProof___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_Graph_buildTransitiveLeProof___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_Graph_buildTransitiveLeProof(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_Graph_buildTransitiveLeProof___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_instToStringEdge___lam__0(lean_object* v_e_2_){
_start:
{
lean_object* v_src_3_; lean_object* v_dst_4_; lean_object* v___x_5_; lean_object* v___x_6_; lean_object* v___x_7_; lean_object* v___x_8_; lean_object* v___x_9_; 
v_src_3_ = lean_ctor_get(v_e_2_, 0);
lean_inc(v_src_3_);
v_dst_4_ = lean_ctor_get(v_e_2_, 1);
lean_inc(v_dst_4_);
lean_dec_ref(v_e_2_);
v___x_5_ = l_Nat_reprFast(v_src_3_);
v___x_6_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order_instToStringEdge___lam__0___closed__0));
v___x_7_ = lean_string_append(v___x_5_, v___x_6_);
v___x_8_ = l_Nat_reprFast(v_dst_4_);
v___x_9_ = lean_string_append(v___x_7_, v___x_8_);
lean_dec_ref(v___x_8_);
return v___x_9_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00Mathlib_Tactic_Order_Graph_addEdge_spec__0_spec__0___redArg(lean_object* v_a_12_, lean_object* v_x_13_){
_start:
{
if (lean_obj_tag(v_x_13_) == 0)
{
uint8_t v___x_14_; 
v___x_14_ = 0;
return v___x_14_;
}
else
{
lean_object* v_key_15_; lean_object* v_tail_16_; uint8_t v___x_17_; 
v_key_15_ = lean_ctor_get(v_x_13_, 0);
v_tail_16_ = lean_ctor_get(v_x_13_, 2);
v___x_17_ = lean_nat_dec_eq(v_key_15_, v_a_12_);
if (v___x_17_ == 0)
{
v_x_13_ = v_tail_16_;
goto _start;
}
else
{
return v___x_17_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00Mathlib_Tactic_Order_Graph_addEdge_spec__0_spec__0___redArg___boxed(lean_object* v_a_19_, lean_object* v_x_20_){
_start:
{
uint8_t v_res_21_; lean_object* v_r_22_; 
v_res_21_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00Mathlib_Tactic_Order_Graph_addEdge_spec__0_spec__0___redArg(v_a_19_, v_x_20_);
lean_dec(v_x_20_);
lean_dec(v_a_19_);
v_r_22_ = lean_box(v_res_21_);
return v_r_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00Mathlib_Tactic_Order_Graph_addEdge_spec__0_spec__1_spec__2_spec__3___redArg(lean_object* v_x_23_, lean_object* v_x_24_){
_start:
{
if (lean_obj_tag(v_x_24_) == 0)
{
return v_x_23_;
}
else
{
lean_object* v_key_25_; lean_object* v_value_26_; lean_object* v_tail_27_; lean_object* v___x_29_; uint8_t v_isShared_30_; uint8_t v_isSharedCheck_50_; 
v_key_25_ = lean_ctor_get(v_x_24_, 0);
v_value_26_ = lean_ctor_get(v_x_24_, 1);
v_tail_27_ = lean_ctor_get(v_x_24_, 2);
v_isSharedCheck_50_ = !lean_is_exclusive(v_x_24_);
if (v_isSharedCheck_50_ == 0)
{
v___x_29_ = v_x_24_;
v_isShared_30_ = v_isSharedCheck_50_;
goto v_resetjp_28_;
}
else
{
lean_inc(v_tail_27_);
lean_inc(v_value_26_);
lean_inc(v_key_25_);
lean_dec(v_x_24_);
v___x_29_ = lean_box(0);
v_isShared_30_ = v_isSharedCheck_50_;
goto v_resetjp_28_;
}
v_resetjp_28_:
{
lean_object* v___x_31_; uint64_t v___x_32_; uint64_t v___x_33_; uint64_t v___x_34_; uint64_t v_fold_35_; uint64_t v___x_36_; uint64_t v___x_37_; uint64_t v___x_38_; size_t v___x_39_; size_t v___x_40_; size_t v___x_41_; size_t v___x_42_; size_t v___x_43_; lean_object* v___x_44_; lean_object* v___x_46_; 
v___x_31_ = lean_array_get_size(v_x_23_);
v___x_32_ = lean_uint64_of_nat(v_key_25_);
v___x_33_ = 32ULL;
v___x_34_ = lean_uint64_shift_right(v___x_32_, v___x_33_);
v_fold_35_ = lean_uint64_xor(v___x_32_, v___x_34_);
v___x_36_ = 16ULL;
v___x_37_ = lean_uint64_shift_right(v_fold_35_, v___x_36_);
v___x_38_ = lean_uint64_xor(v_fold_35_, v___x_37_);
v___x_39_ = lean_uint64_to_usize(v___x_38_);
v___x_40_ = lean_usize_of_nat(v___x_31_);
v___x_41_ = ((size_t)1ULL);
v___x_42_ = lean_usize_sub(v___x_40_, v___x_41_);
v___x_43_ = lean_usize_land(v___x_39_, v___x_42_);
v___x_44_ = lean_array_uget_borrowed(v_x_23_, v___x_43_);
lean_inc(v___x_44_);
if (v_isShared_30_ == 0)
{
lean_ctor_set(v___x_29_, 2, v___x_44_);
v___x_46_ = v___x_29_;
goto v_reusejp_45_;
}
else
{
lean_object* v_reuseFailAlloc_49_; 
v_reuseFailAlloc_49_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_49_, 0, v_key_25_);
lean_ctor_set(v_reuseFailAlloc_49_, 1, v_value_26_);
lean_ctor_set(v_reuseFailAlloc_49_, 2, v___x_44_);
v___x_46_ = v_reuseFailAlloc_49_;
goto v_reusejp_45_;
}
v_reusejp_45_:
{
lean_object* v___x_47_; 
v___x_47_ = lean_array_uset(v_x_23_, v___x_43_, v___x_46_);
v_x_23_ = v___x_47_;
v_x_24_ = v_tail_27_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00Mathlib_Tactic_Order_Graph_addEdge_spec__0_spec__1_spec__2___redArg(lean_object* v_i_51_, lean_object* v_source_52_, lean_object* v_target_53_){
_start:
{
lean_object* v___x_54_; uint8_t v___x_55_; 
v___x_54_ = lean_array_get_size(v_source_52_);
v___x_55_ = lean_nat_dec_lt(v_i_51_, v___x_54_);
if (v___x_55_ == 0)
{
lean_dec_ref(v_source_52_);
lean_dec(v_i_51_);
return v_target_53_;
}
else
{
lean_object* v_es_56_; lean_object* v___x_57_; lean_object* v_source_58_; lean_object* v_target_59_; lean_object* v___x_60_; lean_object* v___x_61_; 
v_es_56_ = lean_array_fget(v_source_52_, v_i_51_);
v___x_57_ = lean_box(0);
v_source_58_ = lean_array_fset(v_source_52_, v_i_51_, v___x_57_);
v_target_59_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00Mathlib_Tactic_Order_Graph_addEdge_spec__0_spec__1_spec__2_spec__3___redArg(v_target_53_, v_es_56_);
v___x_60_ = lean_unsigned_to_nat(1u);
v___x_61_ = lean_nat_add(v_i_51_, v___x_60_);
lean_dec(v_i_51_);
v_i_51_ = v___x_61_;
v_source_52_ = v_source_58_;
v_target_53_ = v_target_59_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00Mathlib_Tactic_Order_Graph_addEdge_spec__0_spec__1___redArg(lean_object* v_data_63_){
_start:
{
lean_object* v___x_64_; lean_object* v___x_65_; lean_object* v_nbuckets_66_; lean_object* v___x_67_; lean_object* v___x_68_; lean_object* v___x_69_; lean_object* v___x_70_; 
v___x_64_ = lean_array_get_size(v_data_63_);
v___x_65_ = lean_unsigned_to_nat(2u);
v_nbuckets_66_ = lean_nat_mul(v___x_64_, v___x_65_);
v___x_67_ = lean_unsigned_to_nat(0u);
v___x_68_ = lean_box(0);
v___x_69_ = lean_mk_array(v_nbuckets_66_, v___x_68_);
v___x_70_ = lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00Mathlib_Tactic_Order_Graph_addEdge_spec__0_spec__1_spec__2___redArg(v___x_67_, v_data_63_, v___x_69_);
return v___x_70_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_Const_alter___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00Mathlib_Tactic_Order_Graph_addEdge_spec__0_spec__2___lam__0(lean_object* v_edge_71_, lean_object* v_x_72_){
_start:
{
if (lean_obj_tag(v_x_72_) == 0)
{
lean_object* v___x_73_; lean_object* v___x_74_; lean_object* v___x_75_; lean_object* v___x_76_; 
v___x_73_ = lean_unsigned_to_nat(1u);
v___x_74_ = lean_mk_empty_array_with_capacity(v___x_73_);
v___x_75_ = lean_array_push(v___x_74_, v_edge_71_);
v___x_76_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_76_, 0, v___x_75_);
return v___x_76_;
}
else
{
lean_object* v_val_77_; lean_object* v___x_79_; uint8_t v_isShared_80_; uint8_t v_isSharedCheck_85_; 
v_val_77_ = lean_ctor_get(v_x_72_, 0);
v_isSharedCheck_85_ = !lean_is_exclusive(v_x_72_);
if (v_isSharedCheck_85_ == 0)
{
v___x_79_ = v_x_72_;
v_isShared_80_ = v_isSharedCheck_85_;
goto v_resetjp_78_;
}
else
{
lean_inc(v_val_77_);
lean_dec(v_x_72_);
v___x_79_ = lean_box(0);
v_isShared_80_ = v_isSharedCheck_85_;
goto v_resetjp_78_;
}
v_resetjp_78_:
{
lean_object* v___x_81_; lean_object* v___x_83_; 
v___x_81_ = lean_array_push(v_val_77_, v_edge_71_);
if (v_isShared_80_ == 0)
{
lean_ctor_set(v___x_79_, 0, v___x_81_);
v___x_83_ = v___x_79_;
goto v_reusejp_82_;
}
else
{
lean_object* v_reuseFailAlloc_84_; 
v_reuseFailAlloc_84_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_84_, 0, v___x_81_);
v___x_83_ = v_reuseFailAlloc_84_;
goto v_reusejp_82_;
}
v_reusejp_82_:
{
return v___x_83_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_Const_alter___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00Mathlib_Tactic_Order_Graph_addEdge_spec__0_spec__2(lean_object* v_edge_86_, lean_object* v_a_87_, lean_object* v_x_88_){
_start:
{
if (lean_obj_tag(v_x_88_) == 0)
{
lean_object* v___x_89_; lean_object* v___x_90_; lean_object* v_val_91_; lean_object* v___x_92_; 
v___x_89_ = lean_box(0);
v___x_90_ = lp_mathlib_Std_DHashMap_Internal_AssocList_Const_alter___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00Mathlib_Tactic_Order_Graph_addEdge_spec__0_spec__2___lam__0(v_edge_86_, v___x_89_);
v_val_91_ = lean_ctor_get(v___x_90_, 0);
lean_inc(v_val_91_);
lean_dec(v___x_90_);
v___x_92_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_92_, 0, v_a_87_);
lean_ctor_set(v___x_92_, 1, v_val_91_);
lean_ctor_set(v___x_92_, 2, v_x_88_);
return v___x_92_;
}
else
{
lean_object* v_key_93_; lean_object* v_value_94_; lean_object* v_tail_95_; lean_object* v___x_97_; uint8_t v_isShared_98_; uint8_t v_isSharedCheck_110_; 
v_key_93_ = lean_ctor_get(v_x_88_, 0);
v_value_94_ = lean_ctor_get(v_x_88_, 1);
v_tail_95_ = lean_ctor_get(v_x_88_, 2);
v_isSharedCheck_110_ = !lean_is_exclusive(v_x_88_);
if (v_isSharedCheck_110_ == 0)
{
v___x_97_ = v_x_88_;
v_isShared_98_ = v_isSharedCheck_110_;
goto v_resetjp_96_;
}
else
{
lean_inc(v_tail_95_);
lean_inc(v_value_94_);
lean_inc(v_key_93_);
lean_dec(v_x_88_);
v___x_97_ = lean_box(0);
v_isShared_98_ = v_isSharedCheck_110_;
goto v_resetjp_96_;
}
v_resetjp_96_:
{
uint8_t v___x_99_; 
v___x_99_ = lean_nat_dec_eq(v_key_93_, v_a_87_);
if (v___x_99_ == 0)
{
lean_object* v_tail_100_; lean_object* v___x_102_; 
v_tail_100_ = lp_mathlib_Std_DHashMap_Internal_AssocList_Const_alter___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00Mathlib_Tactic_Order_Graph_addEdge_spec__0_spec__2(v_edge_86_, v_a_87_, v_tail_95_);
if (v_isShared_98_ == 0)
{
lean_ctor_set(v___x_97_, 2, v_tail_100_);
v___x_102_ = v___x_97_;
goto v_reusejp_101_;
}
else
{
lean_object* v_reuseFailAlloc_103_; 
v_reuseFailAlloc_103_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_103_, 0, v_key_93_);
lean_ctor_set(v_reuseFailAlloc_103_, 1, v_value_94_);
lean_ctor_set(v_reuseFailAlloc_103_, 2, v_tail_100_);
v___x_102_ = v_reuseFailAlloc_103_;
goto v_reusejp_101_;
}
v_reusejp_101_:
{
return v___x_102_;
}
}
else
{
lean_object* v___x_104_; lean_object* v___x_105_; lean_object* v_val_106_; lean_object* v___x_108_; 
lean_dec(v_key_93_);
v___x_104_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_104_, 0, v_value_94_);
v___x_105_ = lp_mathlib_Std_DHashMap_Internal_AssocList_Const_alter___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00Mathlib_Tactic_Order_Graph_addEdge_spec__0_spec__2___lam__0(v_edge_86_, v___x_104_);
v_val_106_ = lean_ctor_get(v___x_105_, 0);
lean_inc(v_val_106_);
lean_dec(v___x_105_);
if (v_isShared_98_ == 0)
{
lean_ctor_set(v___x_97_, 1, v_val_106_);
lean_ctor_set(v___x_97_, 0, v_a_87_);
v___x_108_ = v___x_97_;
goto v_reusejp_107_;
}
else
{
lean_object* v_reuseFailAlloc_109_; 
v_reuseFailAlloc_109_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_109_, 0, v_a_87_);
lean_ctor_set(v_reuseFailAlloc_109_, 1, v_val_106_);
lean_ctor_set(v_reuseFailAlloc_109_, 2, v_tail_95_);
v___x_108_ = v_reuseFailAlloc_109_;
goto v_reusejp_107_;
}
v_reusejp_107_:
{
return v___x_108_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00Mathlib_Tactic_Order_Graph_addEdge_spec__0(lean_object* v_edge_111_, lean_object* v_m_112_, lean_object* v_a_113_){
_start:
{
lean_object* v_size_114_; lean_object* v_buckets_115_; lean_object* v___x_117_; uint8_t v_isShared_118_; uint8_t v_isSharedCheck_165_; 
v_size_114_ = lean_ctor_get(v_m_112_, 0);
v_buckets_115_ = lean_ctor_get(v_m_112_, 1);
v_isSharedCheck_165_ = !lean_is_exclusive(v_m_112_);
if (v_isSharedCheck_165_ == 0)
{
v___x_117_ = v_m_112_;
v_isShared_118_ = v_isSharedCheck_165_;
goto v_resetjp_116_;
}
else
{
lean_inc(v_buckets_115_);
lean_inc(v_size_114_);
lean_dec(v_m_112_);
v___x_117_ = lean_box(0);
v_isShared_118_ = v_isSharedCheck_165_;
goto v_resetjp_116_;
}
v_resetjp_116_:
{
lean_object* v___x_119_; uint64_t v___x_120_; uint64_t v___x_121_; uint64_t v___x_122_; uint64_t v_fold_123_; uint64_t v___x_124_; uint64_t v___x_125_; uint64_t v___x_126_; size_t v___x_127_; size_t v___x_128_; size_t v___x_129_; size_t v___x_130_; size_t v___x_131_; lean_object* v_bkt_132_; uint8_t v___x_133_; 
v___x_119_ = lean_array_get_size(v_buckets_115_);
v___x_120_ = lean_uint64_of_nat(v_a_113_);
v___x_121_ = 32ULL;
v___x_122_ = lean_uint64_shift_right(v___x_120_, v___x_121_);
v_fold_123_ = lean_uint64_xor(v___x_120_, v___x_122_);
v___x_124_ = 16ULL;
v___x_125_ = lean_uint64_shift_right(v_fold_123_, v___x_124_);
v___x_126_ = lean_uint64_xor(v_fold_123_, v___x_125_);
v___x_127_ = lean_uint64_to_usize(v___x_126_);
v___x_128_ = lean_usize_of_nat(v___x_119_);
v___x_129_ = ((size_t)1ULL);
v___x_130_ = lean_usize_sub(v___x_128_, v___x_129_);
v___x_131_ = lean_usize_land(v___x_127_, v___x_130_);
v_bkt_132_ = lean_array_uget_borrowed(v_buckets_115_, v___x_131_);
v___x_133_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00Mathlib_Tactic_Order_Graph_addEdge_spec__0_spec__0___redArg(v_a_113_, v_bkt_132_);
if (v___x_133_ == 0)
{
lean_object* v___x_134_; lean_object* v___x_135_; lean_object* v___x_136_; lean_object* v_size_x27_137_; lean_object* v___x_138_; lean_object* v_buckets_x27_139_; lean_object* v___x_140_; lean_object* v___x_141_; lean_object* v___x_142_; lean_object* v___x_143_; lean_object* v___x_144_; uint8_t v___x_145_; 
v___x_134_ = lean_unsigned_to_nat(1u);
v___x_135_ = lean_mk_empty_array_with_capacity(v___x_134_);
v___x_136_ = lean_array_push(v___x_135_, v_edge_111_);
v_size_x27_137_ = lean_nat_add(v_size_114_, v___x_134_);
lean_dec(v_size_114_);
lean_inc(v_bkt_132_);
v___x_138_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_138_, 0, v_a_113_);
lean_ctor_set(v___x_138_, 1, v___x_136_);
lean_ctor_set(v___x_138_, 2, v_bkt_132_);
v_buckets_x27_139_ = lean_array_uset(v_buckets_115_, v___x_131_, v___x_138_);
v___x_140_ = lean_unsigned_to_nat(4u);
v___x_141_ = lean_nat_mul(v_size_x27_137_, v___x_140_);
v___x_142_ = lean_unsigned_to_nat(3u);
v___x_143_ = lean_nat_div(v___x_141_, v___x_142_);
lean_dec(v___x_141_);
v___x_144_ = lean_array_get_size(v_buckets_x27_139_);
v___x_145_ = lean_nat_dec_le(v___x_143_, v___x_144_);
lean_dec(v___x_143_);
if (v___x_145_ == 0)
{
lean_object* v_val_146_; lean_object* v___x_148_; 
v_val_146_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00Mathlib_Tactic_Order_Graph_addEdge_spec__0_spec__1___redArg(v_buckets_x27_139_);
if (v_isShared_118_ == 0)
{
lean_ctor_set(v___x_117_, 1, v_val_146_);
lean_ctor_set(v___x_117_, 0, v_size_x27_137_);
v___x_148_ = v___x_117_;
goto v_reusejp_147_;
}
else
{
lean_object* v_reuseFailAlloc_149_; 
v_reuseFailAlloc_149_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_149_, 0, v_size_x27_137_);
lean_ctor_set(v_reuseFailAlloc_149_, 1, v_val_146_);
v___x_148_ = v_reuseFailAlloc_149_;
goto v_reusejp_147_;
}
v_reusejp_147_:
{
return v___x_148_;
}
}
else
{
lean_object* v___x_151_; 
if (v_isShared_118_ == 0)
{
lean_ctor_set(v___x_117_, 1, v_buckets_x27_139_);
lean_ctor_set(v___x_117_, 0, v_size_x27_137_);
v___x_151_ = v___x_117_;
goto v_reusejp_150_;
}
else
{
lean_object* v_reuseFailAlloc_152_; 
v_reuseFailAlloc_152_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_152_, 0, v_size_x27_137_);
lean_ctor_set(v_reuseFailAlloc_152_, 1, v_buckets_x27_139_);
v___x_151_ = v_reuseFailAlloc_152_;
goto v_reusejp_150_;
}
v_reusejp_150_:
{
return v___x_151_;
}
}
}
else
{
lean_object* v___x_153_; lean_object* v_buckets_x27_154_; lean_object* v_bkt_x27_155_; lean_object* v___y_157_; uint8_t v___x_162_; 
lean_inc(v_bkt_132_);
v___x_153_ = lean_box(0);
v_buckets_x27_154_ = lean_array_uset(v_buckets_115_, v___x_131_, v___x_153_);
lean_inc(v_a_113_);
v_bkt_x27_155_ = lp_mathlib_Std_DHashMap_Internal_AssocList_Const_alter___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00Mathlib_Tactic_Order_Graph_addEdge_spec__0_spec__2(v_edge_111_, v_a_113_, v_bkt_132_);
v___x_162_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00Mathlib_Tactic_Order_Graph_addEdge_spec__0_spec__0___redArg(v_a_113_, v_bkt_x27_155_);
lean_dec(v_a_113_);
if (v___x_162_ == 0)
{
lean_object* v___x_163_; lean_object* v___x_164_; 
v___x_163_ = lean_unsigned_to_nat(1u);
v___x_164_ = lean_nat_sub(v_size_114_, v___x_163_);
lean_dec(v_size_114_);
v___y_157_ = v___x_164_;
goto v___jp_156_;
}
else
{
v___y_157_ = v_size_114_;
goto v___jp_156_;
}
v___jp_156_:
{
lean_object* v___x_158_; lean_object* v___x_160_; 
v___x_158_ = lean_array_uset(v_buckets_x27_154_, v___x_131_, v_bkt_x27_155_);
if (v_isShared_118_ == 0)
{
lean_ctor_set(v___x_117_, 1, v___x_158_);
lean_ctor_set(v___x_117_, 0, v___y_157_);
v___x_160_ = v___x_117_;
goto v_reusejp_159_;
}
else
{
lean_object* v_reuseFailAlloc_161_; 
v_reuseFailAlloc_161_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_161_, 0, v___y_157_);
lean_ctor_set(v_reuseFailAlloc_161_, 1, v___x_158_);
v___x_160_ = v_reuseFailAlloc_161_;
goto v_reusejp_159_;
}
v_reusejp_159_:
{
return v___x_160_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_Graph_addEdge(lean_object* v_g_166_, lean_object* v_edge_167_){
_start:
{
lean_object* v_src_168_; lean_object* v___x_169_; 
v_src_168_ = lean_ctor_get(v_edge_167_, 0);
lean_inc(v_src_168_);
v___x_169_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00Mathlib_Tactic_Order_Graph_addEdge_spec__0(v_edge_167_, v_g_166_, v_src_168_);
return v___x_169_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00Mathlib_Tactic_Order_Graph_addEdge_spec__0_spec__0(lean_object* v_00_u03b2_170_, lean_object* v_a_171_, lean_object* v_x_172_){
_start:
{
uint8_t v___x_173_; 
v___x_173_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00Mathlib_Tactic_Order_Graph_addEdge_spec__0_spec__0___redArg(v_a_171_, v_x_172_);
return v___x_173_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00Mathlib_Tactic_Order_Graph_addEdge_spec__0_spec__0___boxed(lean_object* v_00_u03b2_174_, lean_object* v_a_175_, lean_object* v_x_176_){
_start:
{
uint8_t v_res_177_; lean_object* v_r_178_; 
v_res_177_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00Mathlib_Tactic_Order_Graph_addEdge_spec__0_spec__0(v_00_u03b2_174_, v_a_175_, v_x_176_);
lean_dec(v_x_176_);
lean_dec(v_a_175_);
v_r_178_ = lean_box(v_res_177_);
return v_r_178_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00Mathlib_Tactic_Order_Graph_addEdge_spec__0_spec__1(lean_object* v_00_u03b2_179_, lean_object* v_data_180_){
_start:
{
lean_object* v___x_181_; 
v___x_181_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00Mathlib_Tactic_Order_Graph_addEdge_spec__0_spec__1___redArg(v_data_180_);
return v___x_181_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00Mathlib_Tactic_Order_Graph_addEdge_spec__0_spec__1_spec__2(lean_object* v_00_u03b2_182_, lean_object* v_i_183_, lean_object* v_source_184_, lean_object* v_target_185_){
_start:
{
lean_object* v___x_186_; 
v___x_186_ = lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00Mathlib_Tactic_Order_Graph_addEdge_spec__0_spec__1_spec__2___redArg(v_i_183_, v_source_184_, v_target_185_);
return v___x_186_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00Mathlib_Tactic_Order_Graph_addEdge_spec__0_spec__1_spec__2_spec__3(lean_object* v_00_u03b2_187_, lean_object* v_x_188_, lean_object* v_x_189_){
_start:
{
lean_object* v___x_190_; 
v___x_190_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00Mathlib_Tactic_Order_Graph_addEdge_spec__0_spec__1_spec__2_spec__3___redArg(v_x_188_, v_x_189_);
return v___x_190_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_Graph_constructLeGraph_spec__0___redArg(lean_object* v_as_191_, size_t v_sz_192_, size_t v_i_193_, lean_object* v_b_194_){
_start:
{
lean_object* v_a_197_; uint8_t v___x_201_; 
v___x_201_ = lean_usize_dec_lt(v_i_193_, v_sz_192_);
if (v___x_201_ == 0)
{
lean_object* v___x_202_; 
v___x_202_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_202_, 0, v_b_194_);
return v___x_202_;
}
else
{
lean_object* v_a_203_; 
v_a_203_ = lean_array_uget(v_as_191_, v_i_193_);
if (lean_obj_tag(v_a_203_) == 2)
{
lean_object* v_lhs_204_; lean_object* v_rhs_205_; lean_object* v_proof_206_; lean_object* v___x_208_; uint8_t v_isShared_209_; uint8_t v_isSharedCheck_214_; 
v_lhs_204_ = lean_ctor_get(v_a_203_, 0);
v_rhs_205_ = lean_ctor_get(v_a_203_, 1);
v_proof_206_ = lean_ctor_get(v_a_203_, 2);
v_isSharedCheck_214_ = !lean_is_exclusive(v_a_203_);
if (v_isSharedCheck_214_ == 0)
{
v___x_208_ = v_a_203_;
v_isShared_209_ = v_isSharedCheck_214_;
goto v_resetjp_207_;
}
else
{
lean_inc(v_proof_206_);
lean_inc(v_rhs_205_);
lean_inc(v_lhs_204_);
lean_dec(v_a_203_);
v___x_208_ = lean_box(0);
v_isShared_209_ = v_isSharedCheck_214_;
goto v_resetjp_207_;
}
v_resetjp_207_:
{
lean_object* v___x_211_; 
if (v_isShared_209_ == 0)
{
lean_ctor_set_tag(v___x_208_, 0);
v___x_211_ = v___x_208_;
goto v_reusejp_210_;
}
else
{
lean_object* v_reuseFailAlloc_213_; 
v_reuseFailAlloc_213_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_213_, 0, v_lhs_204_);
lean_ctor_set(v_reuseFailAlloc_213_, 1, v_rhs_205_);
lean_ctor_set(v_reuseFailAlloc_213_, 2, v_proof_206_);
v___x_211_ = v_reuseFailAlloc_213_;
goto v_reusejp_210_;
}
v_reusejp_210_:
{
lean_object* v___x_212_; 
v___x_212_ = lp_mathlib_Mathlib_Tactic_Order_Graph_addEdge(v_b_194_, v___x_211_);
v_a_197_ = v___x_212_;
goto v___jp_196_;
}
}
}
else
{
lean_dec(v_a_203_);
v_a_197_ = v_b_194_;
goto v___jp_196_;
}
}
v___jp_196_:
{
size_t v___x_198_; size_t v___x_199_; 
v___x_198_ = ((size_t)1ULL);
v___x_199_ = lean_usize_add(v_i_193_, v___x_198_);
v_i_193_ = v___x_199_;
v_b_194_ = v_a_197_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_Graph_constructLeGraph_spec__0___redArg___boxed(lean_object* v_as_215_, lean_object* v_sz_216_, lean_object* v_i_217_, lean_object* v_b_218_, lean_object* v___y_219_){
_start:
{
size_t v_sz_boxed_220_; size_t v_i_boxed_221_; lean_object* v_res_222_; 
v_sz_boxed_220_ = lean_unbox_usize(v_sz_216_);
lean_dec(v_sz_216_);
v_i_boxed_221_ = lean_unbox_usize(v_i_217_);
lean_dec(v_i_217_);
v_res_222_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_Graph_constructLeGraph_spec__0___redArg(v_as_215_, v_sz_boxed_220_, v_i_boxed_221_, v_b_218_);
lean_dec_ref(v_as_215_);
return v_res_222_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Order_Graph_constructLeGraph___closed__0(void){
_start:
{
lean_object* v___x_223_; lean_object* v___x_224_; lean_object* v___x_225_; 
v___x_223_ = lean_box(0);
v___x_224_ = lean_unsigned_to_nat(16u);
v___x_225_ = lean_mk_array(v___x_224_, v___x_223_);
return v___x_225_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Order_Graph_constructLeGraph___closed__1(void){
_start:
{
lean_object* v___x_226_; lean_object* v___x_227_; lean_object* v_res_228_; 
v___x_226_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Order_Graph_constructLeGraph___closed__0, &lp_mathlib_Mathlib_Tactic_Order_Graph_constructLeGraph___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_Order_Graph_constructLeGraph___closed__0);
v___x_227_ = lean_unsigned_to_nat(0u);
v_res_228_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_res_228_, 0, v___x_227_);
lean_ctor_set(v_res_228_, 1, v___x_226_);
return v_res_228_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_Graph_constructLeGraph(lean_object* v_facts_229_, lean_object* v_a_230_, lean_object* v_a_231_, lean_object* v_a_232_, lean_object* v_a_233_){
_start:
{
lean_object* v_res_235_; size_t v_sz_236_; size_t v___x_237_; lean_object* v___x_238_; 
v_res_235_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Order_Graph_constructLeGraph___closed__1, &lp_mathlib_Mathlib_Tactic_Order_Graph_constructLeGraph___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Order_Graph_constructLeGraph___closed__1);
v_sz_236_ = lean_array_size(v_facts_229_);
v___x_237_ = ((size_t)0ULL);
v___x_238_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_Graph_constructLeGraph_spec__0___redArg(v_facts_229_, v_sz_236_, v___x_237_, v_res_235_);
return v___x_238_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_Graph_constructLeGraph___boxed(lean_object* v_facts_239_, lean_object* v_a_240_, lean_object* v_a_241_, lean_object* v_a_242_, lean_object* v_a_243_, lean_object* v_a_244_){
_start:
{
lean_object* v_res_245_; 
v_res_245_ = lp_mathlib_Mathlib_Tactic_Order_Graph_constructLeGraph(v_facts_239_, v_a_240_, v_a_241_, v_a_242_, v_a_243_);
lean_dec(v_a_243_);
lean_dec_ref(v_a_242_);
lean_dec(v_a_241_);
lean_dec_ref(v_a_240_);
lean_dec_ref(v_facts_239_);
return v_res_245_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_Graph_constructLeGraph_spec__0(lean_object* v_as_246_, size_t v_sz_247_, size_t v_i_248_, lean_object* v_b_249_, lean_object* v___y_250_, lean_object* v___y_251_, lean_object* v___y_252_, lean_object* v___y_253_){
_start:
{
lean_object* v___x_255_; 
v___x_255_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_Graph_constructLeGraph_spec__0___redArg(v_as_246_, v_sz_247_, v_i_248_, v_b_249_);
return v___x_255_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_Graph_constructLeGraph_spec__0___boxed(lean_object* v_as_256_, lean_object* v_sz_257_, lean_object* v_i_258_, lean_object* v_b_259_, lean_object* v___y_260_, lean_object* v___y_261_, lean_object* v___y_262_, lean_object* v___y_263_, lean_object* v___y_264_){
_start:
{
size_t v_sz_boxed_265_; size_t v_i_boxed_266_; lean_object* v_res_267_; 
v_sz_boxed_265_ = lean_unbox_usize(v_sz_257_);
lean_dec(v_sz_257_);
v_i_boxed_266_ = lean_unbox_usize(v_i_258_);
lean_dec(v_i_258_);
v_res_267_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_Graph_constructLeGraph_spec__0(v_as_256_, v_sz_boxed_265_, v_i_boxed_266_, v_b_259_, v___y_260_, v___y_261_, v___y_262_, v___y_263_);
lean_dec(v___y_263_);
lean_dec_ref(v___y_262_);
lean_dec(v___y_261_);
lean_dec_ref(v___y_260_);
lean_dec_ref(v_as_256_);
return v_res_267_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__1___redArg(lean_object* v_m_268_, lean_object* v_a_269_){
_start:
{
lean_object* v_buckets_270_; lean_object* v___x_271_; uint64_t v___x_272_; uint64_t v___x_273_; uint64_t v___x_274_; uint64_t v_fold_275_; uint64_t v___x_276_; uint64_t v___x_277_; uint64_t v___x_278_; size_t v___x_279_; size_t v___x_280_; size_t v___x_281_; size_t v___x_282_; size_t v___x_283_; lean_object* v___x_284_; uint8_t v___x_285_; 
v_buckets_270_ = lean_ctor_get(v_m_268_, 1);
v___x_271_ = lean_array_get_size(v_buckets_270_);
v___x_272_ = lean_uint64_of_nat(v_a_269_);
v___x_273_ = 32ULL;
v___x_274_ = lean_uint64_shift_right(v___x_272_, v___x_273_);
v_fold_275_ = lean_uint64_xor(v___x_272_, v___x_274_);
v___x_276_ = 16ULL;
v___x_277_ = lean_uint64_shift_right(v_fold_275_, v___x_276_);
v___x_278_ = lean_uint64_xor(v_fold_275_, v___x_277_);
v___x_279_ = lean_uint64_to_usize(v___x_278_);
v___x_280_ = lean_usize_of_nat(v___x_271_);
v___x_281_ = ((size_t)1ULL);
v___x_282_ = lean_usize_sub(v___x_280_, v___x_281_);
v___x_283_ = lean_usize_land(v___x_279_, v___x_282_);
v___x_284_ = lean_array_uget_borrowed(v_buckets_270_, v___x_283_);
v___x_285_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00Mathlib_Tactic_Order_Graph_addEdge_spec__0_spec__0___redArg(v_a_269_, v___x_284_);
return v___x_285_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__1___redArg___boxed(lean_object* v_m_286_, lean_object* v_a_287_){
_start:
{
uint8_t v_res_288_; lean_object* v_r_289_; 
v_res_288_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__1___redArg(v_m_286_, v_a_287_);
lean_dec(v_a_287_);
lean_dec_ref(v_m_286_);
v_r_289_ = lean_box(v_res_288_);
return v_r_289_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__0___redArg(lean_object* v_m_290_, lean_object* v_a_291_, lean_object* v_b_292_){
_start:
{
lean_object* v_size_293_; lean_object* v_buckets_294_; lean_object* v___x_295_; uint64_t v___x_296_; uint64_t v___x_297_; uint64_t v___x_298_; uint64_t v_fold_299_; uint64_t v___x_300_; uint64_t v___x_301_; uint64_t v___x_302_; size_t v___x_303_; size_t v___x_304_; size_t v___x_305_; size_t v___x_306_; size_t v___x_307_; lean_object* v_bkt_308_; uint8_t v___x_309_; 
v_size_293_ = lean_ctor_get(v_m_290_, 0);
v_buckets_294_ = lean_ctor_get(v_m_290_, 1);
v___x_295_ = lean_array_get_size(v_buckets_294_);
v___x_296_ = lean_uint64_of_nat(v_a_291_);
v___x_297_ = 32ULL;
v___x_298_ = lean_uint64_shift_right(v___x_296_, v___x_297_);
v_fold_299_ = lean_uint64_xor(v___x_296_, v___x_298_);
v___x_300_ = 16ULL;
v___x_301_ = lean_uint64_shift_right(v_fold_299_, v___x_300_);
v___x_302_ = lean_uint64_xor(v_fold_299_, v___x_301_);
v___x_303_ = lean_uint64_to_usize(v___x_302_);
v___x_304_ = lean_usize_of_nat(v___x_295_);
v___x_305_ = ((size_t)1ULL);
v___x_306_ = lean_usize_sub(v___x_304_, v___x_305_);
v___x_307_ = lean_usize_land(v___x_303_, v___x_306_);
v_bkt_308_ = lean_array_uget_borrowed(v_buckets_294_, v___x_307_);
v___x_309_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00Mathlib_Tactic_Order_Graph_addEdge_spec__0_spec__0___redArg(v_a_291_, v_bkt_308_);
if (v___x_309_ == 0)
{
lean_object* v___x_311_; uint8_t v_isShared_312_; uint8_t v_isSharedCheck_330_; 
lean_inc_ref(v_buckets_294_);
lean_inc(v_size_293_);
v_isSharedCheck_330_ = !lean_is_exclusive(v_m_290_);
if (v_isSharedCheck_330_ == 0)
{
lean_object* v_unused_331_; lean_object* v_unused_332_; 
v_unused_331_ = lean_ctor_get(v_m_290_, 1);
lean_dec(v_unused_331_);
v_unused_332_ = lean_ctor_get(v_m_290_, 0);
lean_dec(v_unused_332_);
v___x_311_ = v_m_290_;
v_isShared_312_ = v_isSharedCheck_330_;
goto v_resetjp_310_;
}
else
{
lean_dec(v_m_290_);
v___x_311_ = lean_box(0);
v_isShared_312_ = v_isSharedCheck_330_;
goto v_resetjp_310_;
}
v_resetjp_310_:
{
lean_object* v___x_313_; lean_object* v_size_x27_314_; lean_object* v___x_315_; lean_object* v_buckets_x27_316_; lean_object* v___x_317_; lean_object* v___x_318_; lean_object* v___x_319_; lean_object* v___x_320_; lean_object* v___x_321_; uint8_t v___x_322_; 
v___x_313_ = lean_unsigned_to_nat(1u);
v_size_x27_314_ = lean_nat_add(v_size_293_, v___x_313_);
lean_dec(v_size_293_);
lean_inc(v_bkt_308_);
v___x_315_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_315_, 0, v_a_291_);
lean_ctor_set(v___x_315_, 1, v_b_292_);
lean_ctor_set(v___x_315_, 2, v_bkt_308_);
v_buckets_x27_316_ = lean_array_uset(v_buckets_294_, v___x_307_, v___x_315_);
v___x_317_ = lean_unsigned_to_nat(4u);
v___x_318_ = lean_nat_mul(v_size_x27_314_, v___x_317_);
v___x_319_ = lean_unsigned_to_nat(3u);
v___x_320_ = lean_nat_div(v___x_318_, v___x_319_);
lean_dec(v___x_318_);
v___x_321_ = lean_array_get_size(v_buckets_x27_316_);
v___x_322_ = lean_nat_dec_le(v___x_320_, v___x_321_);
lean_dec(v___x_320_);
if (v___x_322_ == 0)
{
lean_object* v_val_323_; lean_object* v___x_325_; 
v_val_323_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_Const_alter___at___00Mathlib_Tactic_Order_Graph_addEdge_spec__0_spec__1___redArg(v_buckets_x27_316_);
if (v_isShared_312_ == 0)
{
lean_ctor_set(v___x_311_, 1, v_val_323_);
lean_ctor_set(v___x_311_, 0, v_size_x27_314_);
v___x_325_ = v___x_311_;
goto v_reusejp_324_;
}
else
{
lean_object* v_reuseFailAlloc_326_; 
v_reuseFailAlloc_326_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_326_, 0, v_size_x27_314_);
lean_ctor_set(v_reuseFailAlloc_326_, 1, v_val_323_);
v___x_325_ = v_reuseFailAlloc_326_;
goto v_reusejp_324_;
}
v_reusejp_324_:
{
return v___x_325_;
}
}
else
{
lean_object* v___x_328_; 
if (v_isShared_312_ == 0)
{
lean_ctor_set(v___x_311_, 1, v_buckets_x27_316_);
lean_ctor_set(v___x_311_, 0, v_size_x27_314_);
v___x_328_ = v___x_311_;
goto v_reusejp_327_;
}
else
{
lean_object* v_reuseFailAlloc_329_; 
v_reuseFailAlloc_329_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_329_, 0, v_size_x27_314_);
lean_ctor_set(v_reuseFailAlloc_329_, 1, v_buckets_x27_316_);
v___x_328_ = v_reuseFailAlloc_329_;
goto v_reusejp_327_;
}
v_reusejp_327_:
{
return v___x_328_;
}
}
}
}
else
{
lean_dec(v_b_292_);
lean_dec(v_a_291_);
return v_m_290_;
}
}
}
static lean_object* _init_lp_mathlib_panic___at___00Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__2_spec__2_spec__3___closed__0(void){
_start:
{
lean_object* v___x_333_; 
v___x_333_ = l_Array_instInhabited(lean_box(0));
return v___x_333_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__2_spec__2_spec__3(lean_object* v_msg_334_){
_start:
{
lean_object* v___x_335_; lean_object* v___x_336_; 
v___x_335_ = lean_obj_once(&lp_mathlib_panic___at___00Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__2_spec__2_spec__3___closed__0, &lp_mathlib_panic___at___00Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__2_spec__2_spec__3___closed__0_once, _init_lp_mathlib_panic___at___00Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__2_spec__2_spec__3___closed__0);
v___x_336_ = lean_panic_fn_borrowed(v___x_335_, v_msg_334_);
return v___x_336_;
}
}
static lean_object* _init_lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__2_spec__2___closed__3(void){
_start:
{
lean_object* v___x_340_; lean_object* v___x_341_; lean_object* v___x_342_; lean_object* v___x_343_; lean_object* v___x_344_; lean_object* v___x_345_; 
v___x_340_ = ((lean_object*)(lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__2_spec__2___closed__2));
v___x_341_ = lean_unsigned_to_nat(11u);
v___x_342_ = lean_unsigned_to_nat(163u);
v___x_343_ = ((lean_object*)(lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__2_spec__2___closed__1));
v___x_344_ = ((lean_object*)(lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__2_spec__2___closed__0));
v___x_345_ = l_mkPanicMessageWithDecl(v___x_344_, v___x_343_, v___x_342_, v___x_341_, v___x_340_);
return v___x_345_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__2_spec__2(lean_object* v_a_346_, lean_object* v_x_347_){
_start:
{
if (lean_obj_tag(v_x_347_) == 0)
{
lean_object* v___x_348_; lean_object* v___x_349_; 
v___x_348_ = lean_obj_once(&lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__2_spec__2___closed__3, &lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__2_spec__2___closed__3_once, _init_lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__2_spec__2___closed__3);
v___x_349_ = lp_mathlib_panic___at___00Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__2_spec__2_spec__3(v___x_348_);
return v___x_349_;
}
else
{
lean_object* v_key_350_; lean_object* v_value_351_; lean_object* v_tail_352_; uint8_t v___x_353_; 
v_key_350_ = lean_ctor_get(v_x_347_, 0);
v_value_351_ = lean_ctor_get(v_x_347_, 1);
v_tail_352_ = lean_ctor_get(v_x_347_, 2);
v___x_353_ = lean_nat_dec_eq(v_key_350_, v_a_346_);
if (v___x_353_ == 0)
{
v_x_347_ = v_tail_352_;
goto _start;
}
else
{
lean_inc(v_value_351_);
return v_value_351_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__2_spec__2___boxed(lean_object* v_a_355_, lean_object* v_x_356_){
_start:
{
lean_object* v_res_357_; 
v_res_357_ = lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__2_spec__2(v_a_355_, v_x_356_);
lean_dec(v_x_356_);
lean_dec(v_a_355_);
return v_res_357_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__2(lean_object* v_m_358_, lean_object* v_a_359_){
_start:
{
lean_object* v_buckets_360_; lean_object* v___x_361_; uint64_t v___x_362_; uint64_t v___x_363_; uint64_t v___x_364_; uint64_t v_fold_365_; uint64_t v___x_366_; uint64_t v___x_367_; uint64_t v___x_368_; size_t v___x_369_; size_t v___x_370_; size_t v___x_371_; size_t v___x_372_; size_t v___x_373_; lean_object* v___x_374_; lean_object* v___x_375_; 
v_buckets_360_ = lean_ctor_get(v_m_358_, 1);
v___x_361_ = lean_array_get_size(v_buckets_360_);
v___x_362_ = lean_uint64_of_nat(v_a_359_);
v___x_363_ = 32ULL;
v___x_364_ = lean_uint64_shift_right(v___x_362_, v___x_363_);
v_fold_365_ = lean_uint64_xor(v___x_362_, v___x_364_);
v___x_366_ = 16ULL;
v___x_367_ = lean_uint64_shift_right(v_fold_365_, v___x_366_);
v___x_368_ = lean_uint64_xor(v_fold_365_, v___x_367_);
v___x_369_ = lean_uint64_to_usize(v___x_368_);
v___x_370_ = lean_usize_of_nat(v___x_361_);
v___x_371_ = ((size_t)1ULL);
v___x_372_ = lean_usize_sub(v___x_370_, v___x_371_);
v___x_373_ = lean_usize_land(v___x_369_, v___x_372_);
v___x_374_ = lean_array_uget_borrowed(v_buckets_360_, v___x_373_);
v___x_375_ = lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__2_spec__2(v_a_359_, v___x_374_);
return v___x_375_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__2___boxed(lean_object* v_m_376_, lean_object* v_a_377_){
_start:
{
lean_object* v_res_378_; 
v_res_378_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__2(v_m_376_, v_a_377_);
lean_dec(v_a_377_);
lean_dec_ref(v_m_376_);
return v_res_378_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS(lean_object* v_g_388_, lean_object* v_v_389_, lean_object* v_t_390_, lean_object* v_tExpr_391_, lean_object* v_a_392_, lean_object* v_a_393_, lean_object* v_a_394_, lean_object* v_a_395_, lean_object* v_a_396_){
_start:
{
lean_object* v___x_398_; lean_object* v___x_399_; uint8_t v___x_404_; 
v___x_398_ = lean_box(0);
lean_inc(v_v_389_);
v___x_399_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__0___redArg(v_a_392_, v_v_389_, v___x_398_);
v___x_404_ = lean_nat_dec_eq(v_v_389_, v_t_390_);
if (v___x_404_ == 0)
{
uint8_t v___x_405_; 
v___x_405_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__1___redArg(v_g_388_, v_v_389_);
if (v___x_405_ == 0)
{
lean_dec_ref(v_tExpr_391_);
lean_dec(v_v_389_);
goto v___jp_400_;
}
else
{
if (v___x_404_ == 0)
{
lean_object* v___x_406_; lean_object* v___x_407_; lean_object* v___x_408_; size_t v_sz_409_; size_t v___x_410_; lean_object* v___x_411_; 
v___x_406_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__2(v_g_388_, v_v_389_);
lean_dec(v_v_389_);
v___x_407_ = lean_box(0);
v___x_408_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__3___closed__0));
v_sz_409_ = lean_array_size(v___x_406_);
v___x_410_ = ((size_t)0ULL);
v___x_411_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__3(v_g_388_, v_t_390_, v_tExpr_391_, v___x_406_, v_sz_409_, v___x_410_, v___x_408_, v___x_399_, v_a_393_, v_a_394_, v_a_395_, v_a_396_);
lean_dec_ref(v___x_406_);
if (lean_obj_tag(v___x_411_) == 0)
{
lean_object* v_a_412_; lean_object* v___x_414_; uint8_t v_isShared_415_; uint8_t v_isSharedCheck_438_; 
v_a_412_ = lean_ctor_get(v___x_411_, 0);
v_isSharedCheck_438_ = !lean_is_exclusive(v___x_411_);
if (v_isSharedCheck_438_ == 0)
{
v___x_414_ = v___x_411_;
v_isShared_415_ = v_isSharedCheck_438_;
goto v_resetjp_413_;
}
else
{
lean_inc(v_a_412_);
lean_dec(v___x_411_);
v___x_414_ = lean_box(0);
v_isShared_415_ = v_isSharedCheck_438_;
goto v_resetjp_413_;
}
v_resetjp_413_:
{
lean_object* v_fst_416_; lean_object* v_fst_417_; lean_object* v___x_419_; uint8_t v_isShared_420_; uint8_t v_isSharedCheck_436_; 
v_fst_416_ = lean_ctor_get(v_a_412_, 0);
lean_inc(v_fst_416_);
v_fst_417_ = lean_ctor_get(v_fst_416_, 0);
v_isSharedCheck_436_ = !lean_is_exclusive(v_fst_416_);
if (v_isSharedCheck_436_ == 0)
{
lean_object* v_unused_437_; 
v_unused_437_ = lean_ctor_get(v_fst_416_, 1);
lean_dec(v_unused_437_);
v___x_419_ = v_fst_416_;
v_isShared_420_ = v_isSharedCheck_436_;
goto v_resetjp_418_;
}
else
{
lean_inc(v_fst_417_);
lean_dec(v_fst_416_);
v___x_419_ = lean_box(0);
v_isShared_420_ = v_isSharedCheck_436_;
goto v_resetjp_418_;
}
v_resetjp_418_:
{
if (lean_obj_tag(v_fst_417_) == 0)
{
lean_object* v_snd_421_; lean_object* v___x_423_; 
v_snd_421_ = lean_ctor_get(v_a_412_, 1);
lean_inc(v_snd_421_);
lean_dec(v_a_412_);
if (v_isShared_420_ == 0)
{
lean_ctor_set(v___x_419_, 1, v_snd_421_);
lean_ctor_set(v___x_419_, 0, v___x_407_);
v___x_423_ = v___x_419_;
goto v_reusejp_422_;
}
else
{
lean_object* v_reuseFailAlloc_427_; 
v_reuseFailAlloc_427_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_427_, 0, v___x_407_);
lean_ctor_set(v_reuseFailAlloc_427_, 1, v_snd_421_);
v___x_423_ = v_reuseFailAlloc_427_;
goto v_reusejp_422_;
}
v_reusejp_422_:
{
lean_object* v___x_425_; 
if (v_isShared_415_ == 0)
{
lean_ctor_set(v___x_414_, 0, v___x_423_);
v___x_425_ = v___x_414_;
goto v_reusejp_424_;
}
else
{
lean_object* v_reuseFailAlloc_426_; 
v_reuseFailAlloc_426_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_426_, 0, v___x_423_);
v___x_425_ = v_reuseFailAlloc_426_;
goto v_reusejp_424_;
}
v_reusejp_424_:
{
return v___x_425_;
}
}
}
else
{
lean_object* v_snd_428_; lean_object* v_val_429_; lean_object* v___x_431_; 
v_snd_428_ = lean_ctor_get(v_a_412_, 1);
lean_inc(v_snd_428_);
lean_dec(v_a_412_);
v_val_429_ = lean_ctor_get(v_fst_417_, 0);
lean_inc(v_val_429_);
lean_dec_ref_known(v_fst_417_, 1);
if (v_isShared_420_ == 0)
{
lean_ctor_set(v___x_419_, 1, v_snd_428_);
lean_ctor_set(v___x_419_, 0, v_val_429_);
v___x_431_ = v___x_419_;
goto v_reusejp_430_;
}
else
{
lean_object* v_reuseFailAlloc_435_; 
v_reuseFailAlloc_435_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_435_, 0, v_val_429_);
lean_ctor_set(v_reuseFailAlloc_435_, 1, v_snd_428_);
v___x_431_ = v_reuseFailAlloc_435_;
goto v_reusejp_430_;
}
v_reusejp_430_:
{
lean_object* v___x_433_; 
if (v_isShared_415_ == 0)
{
lean_ctor_set(v___x_414_, 0, v___x_431_);
v___x_433_ = v___x_414_;
goto v_reusejp_432_;
}
else
{
lean_object* v_reuseFailAlloc_434_; 
v_reuseFailAlloc_434_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_434_, 0, v___x_431_);
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
}
else
{
lean_object* v_a_439_; lean_object* v___x_441_; uint8_t v_isShared_442_; uint8_t v_isSharedCheck_446_; 
v_a_439_ = lean_ctor_get(v___x_411_, 0);
v_isSharedCheck_446_ = !lean_is_exclusive(v___x_411_);
if (v_isSharedCheck_446_ == 0)
{
v___x_441_ = v___x_411_;
v_isShared_442_ = v_isSharedCheck_446_;
goto v_resetjp_440_;
}
else
{
lean_inc(v_a_439_);
lean_dec(v___x_411_);
v___x_441_ = lean_box(0);
v_isShared_442_ = v_isSharedCheck_446_;
goto v_resetjp_440_;
}
v_resetjp_440_:
{
lean_object* v___x_444_; 
if (v_isShared_442_ == 0)
{
v___x_444_ = v___x_441_;
goto v_reusejp_443_;
}
else
{
lean_object* v_reuseFailAlloc_445_; 
v_reuseFailAlloc_445_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_445_, 0, v_a_439_);
v___x_444_ = v_reuseFailAlloc_445_;
goto v_reusejp_443_;
}
v_reusejp_443_:
{
return v___x_444_;
}
}
}
}
else
{
lean_dec_ref(v_tExpr_391_);
lean_dec(v_v_389_);
goto v___jp_400_;
}
}
}
else
{
lean_object* v___x_447_; lean_object* v___x_448_; lean_object* v___x_449_; lean_object* v___x_450_; lean_object* v___x_451_; 
lean_dec(v_v_389_);
v___x_447_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS___closed__1));
v___x_448_ = lean_unsigned_to_nat(1u);
v___x_449_ = lean_mk_empty_array_with_capacity(v___x_448_);
v___x_450_ = lean_array_push(v___x_449_, v_tExpr_391_);
v___x_451_ = l_Lean_Meta_mkAppM(v___x_447_, v___x_450_, v_a_393_, v_a_394_, v_a_395_, v_a_396_);
if (lean_obj_tag(v___x_451_) == 0)
{
lean_object* v_a_452_; lean_object* v___x_454_; uint8_t v_isShared_455_; uint8_t v_isSharedCheck_461_; 
v_a_452_ = lean_ctor_get(v___x_451_, 0);
v_isSharedCheck_461_ = !lean_is_exclusive(v___x_451_);
if (v_isSharedCheck_461_ == 0)
{
v___x_454_ = v___x_451_;
v_isShared_455_ = v_isSharedCheck_461_;
goto v_resetjp_453_;
}
else
{
lean_inc(v_a_452_);
lean_dec(v___x_451_);
v___x_454_ = lean_box(0);
v_isShared_455_ = v_isSharedCheck_461_;
goto v_resetjp_453_;
}
v_resetjp_453_:
{
lean_object* v___x_456_; lean_object* v___x_457_; lean_object* v___x_459_; 
v___x_456_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_456_, 0, v_a_452_);
v___x_457_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_457_, 0, v___x_456_);
lean_ctor_set(v___x_457_, 1, v___x_399_);
if (v_isShared_455_ == 0)
{
lean_ctor_set(v___x_454_, 0, v___x_457_);
v___x_459_ = v___x_454_;
goto v_reusejp_458_;
}
else
{
lean_object* v_reuseFailAlloc_460_; 
v_reuseFailAlloc_460_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_460_, 0, v___x_457_);
v___x_459_ = v_reuseFailAlloc_460_;
goto v_reusejp_458_;
}
v_reusejp_458_:
{
return v___x_459_;
}
}
}
else
{
lean_object* v_a_462_; lean_object* v___x_464_; uint8_t v_isShared_465_; uint8_t v_isSharedCheck_469_; 
lean_dec_ref(v___x_399_);
v_a_462_ = lean_ctor_get(v___x_451_, 0);
v_isSharedCheck_469_ = !lean_is_exclusive(v___x_451_);
if (v_isSharedCheck_469_ == 0)
{
v___x_464_ = v___x_451_;
v_isShared_465_ = v_isSharedCheck_469_;
goto v_resetjp_463_;
}
else
{
lean_inc(v_a_462_);
lean_dec(v___x_451_);
v___x_464_ = lean_box(0);
v_isShared_465_ = v_isSharedCheck_469_;
goto v_resetjp_463_;
}
v_resetjp_463_:
{
lean_object* v___x_467_; 
if (v_isShared_465_ == 0)
{
v___x_467_ = v___x_464_;
goto v_reusejp_466_;
}
else
{
lean_object* v_reuseFailAlloc_468_; 
v_reuseFailAlloc_468_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_468_, 0, v_a_462_);
v___x_467_ = v_reuseFailAlloc_468_;
goto v_reusejp_466_;
}
v_reusejp_466_:
{
return v___x_467_;
}
}
}
}
v___jp_400_:
{
lean_object* v___x_401_; lean_object* v___x_402_; lean_object* v___x_403_; 
v___x_401_ = lean_box(0);
v___x_402_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_402_, 0, v___x_401_);
lean_ctor_set(v___x_402_, 1, v___x_399_);
v___x_403_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_403_, 0, v___x_402_);
return v___x_403_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__3(lean_object* v_g_470_, lean_object* v_t_471_, lean_object* v_tExpr_472_, lean_object* v_as_473_, size_t v_sz_474_, size_t v_i_475_, lean_object* v_b_476_, lean_object* v___y_477_, lean_object* v___y_478_, lean_object* v___y_479_, lean_object* v___y_480_, lean_object* v___y_481_){
_start:
{
lean_object* v_a_484_; lean_object* v_snd_485_; uint8_t v___x_489_; 
v___x_489_ = lean_usize_dec_lt(v_i_475_, v_sz_474_);
if (v___x_489_ == 0)
{
lean_object* v___x_490_; lean_object* v___x_491_; 
lean_dec_ref(v_tExpr_472_);
v___x_490_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_490_, 0, v_b_476_);
lean_ctor_set(v___x_490_, 1, v___y_477_);
v___x_491_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_491_, 0, v___x_490_);
return v___x_491_;
}
else
{
lean_object* v_a_492_; lean_object* v_dst_493_; lean_object* v_proof_494_; lean_object* v___x_495_; lean_object* v___x_496_; uint8_t v___x_497_; 
lean_dec_ref(v_b_476_);
v_a_492_ = lean_array_uget_borrowed(v_as_473_, v_i_475_);
v_dst_493_ = lean_ctor_get(v_a_492_, 1);
v_proof_494_ = lean_ctor_get(v_a_492_, 2);
v___x_495_ = lean_box(0);
v___x_496_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__3___closed__0));
v___x_497_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__1___redArg(v___y_477_, v_dst_493_);
if (v___x_497_ == 0)
{
lean_object* v___x_498_; 
lean_inc_ref(v_tExpr_472_);
lean_inc(v_dst_493_);
v___x_498_ = lp_mathlib_Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS(v_g_470_, v_dst_493_, v_t_471_, v_tExpr_472_, v___y_477_, v___y_478_, v___y_479_, v___y_480_, v___y_481_);
if (lean_obj_tag(v___x_498_) == 0)
{
lean_object* v_a_499_; lean_object* v_fst_500_; 
v_a_499_ = lean_ctor_get(v___x_498_, 0);
lean_inc(v_a_499_);
lean_dec_ref_known(v___x_498_, 1);
v_fst_500_ = lean_ctor_get(v_a_499_, 0);
lean_inc(v_fst_500_);
if (lean_obj_tag(v_fst_500_) == 0)
{
lean_object* v_snd_501_; 
v_snd_501_ = lean_ctor_get(v_a_499_, 1);
lean_inc(v_snd_501_);
lean_dec(v_a_499_);
v_a_484_ = v___x_496_;
v_snd_485_ = v_snd_501_;
goto v___jp_483_;
}
else
{
lean_object* v_snd_502_; lean_object* v___x_504_; uint8_t v_isShared_505_; uint8_t v_isSharedCheck_541_; 
lean_dec_ref(v_tExpr_472_);
v_snd_502_ = lean_ctor_get(v_a_499_, 1);
v_isSharedCheck_541_ = !lean_is_exclusive(v_a_499_);
if (v_isSharedCheck_541_ == 0)
{
lean_object* v_unused_542_; 
v_unused_542_ = lean_ctor_get(v_a_499_, 0);
lean_dec(v_unused_542_);
v___x_504_ = v_a_499_;
v_isShared_505_ = v_isSharedCheck_541_;
goto v_resetjp_503_;
}
else
{
lean_inc(v_snd_502_);
lean_dec(v_a_499_);
v___x_504_ = lean_box(0);
v_isShared_505_ = v_isSharedCheck_541_;
goto v_resetjp_503_;
}
v_resetjp_503_:
{
lean_object* v_val_506_; lean_object* v___x_508_; uint8_t v_isShared_509_; uint8_t v_isSharedCheck_540_; 
v_val_506_ = lean_ctor_get(v_fst_500_, 0);
v_isSharedCheck_540_ = !lean_is_exclusive(v_fst_500_);
if (v_isSharedCheck_540_ == 0)
{
v___x_508_ = v_fst_500_;
v_isShared_509_ = v_isSharedCheck_540_;
goto v_resetjp_507_;
}
else
{
lean_inc(v_val_506_);
lean_dec(v_fst_500_);
v___x_508_ = lean_box(0);
v_isShared_509_ = v_isSharedCheck_540_;
goto v_resetjp_507_;
}
v_resetjp_507_:
{
lean_object* v___x_510_; lean_object* v___x_511_; lean_object* v___x_512_; lean_object* v___x_513_; lean_object* v___x_514_; lean_object* v___x_515_; 
v___x_510_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__3___closed__2));
v___x_511_ = lean_unsigned_to_nat(2u);
v___x_512_ = lean_mk_empty_array_with_capacity(v___x_511_);
lean_inc_ref(v_proof_494_);
v___x_513_ = lean_array_push(v___x_512_, v_proof_494_);
v___x_514_ = lean_array_push(v___x_513_, v_val_506_);
v___x_515_ = l_Lean_Meta_mkAppM(v___x_510_, v___x_514_, v___y_478_, v___y_479_, v___y_480_, v___y_481_);
if (lean_obj_tag(v___x_515_) == 0)
{
lean_object* v_a_516_; lean_object* v___x_518_; uint8_t v_isShared_519_; uint8_t v_isSharedCheck_531_; 
v_a_516_ = lean_ctor_get(v___x_515_, 0);
v_isSharedCheck_531_ = !lean_is_exclusive(v___x_515_);
if (v_isSharedCheck_531_ == 0)
{
v___x_518_ = v___x_515_;
v_isShared_519_ = v_isSharedCheck_531_;
goto v_resetjp_517_;
}
else
{
lean_inc(v_a_516_);
lean_dec(v___x_515_);
v___x_518_ = lean_box(0);
v_isShared_519_ = v_isSharedCheck_531_;
goto v_resetjp_517_;
}
v_resetjp_517_:
{
lean_object* v___x_521_; 
if (v_isShared_509_ == 0)
{
lean_ctor_set(v___x_508_, 0, v_a_516_);
v___x_521_ = v___x_508_;
goto v_reusejp_520_;
}
else
{
lean_object* v_reuseFailAlloc_530_; 
v_reuseFailAlloc_530_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_530_, 0, v_a_516_);
v___x_521_ = v_reuseFailAlloc_530_;
goto v_reusejp_520_;
}
v_reusejp_520_:
{
lean_object* v___x_522_; lean_object* v___x_524_; 
v___x_522_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_522_, 0, v___x_521_);
if (v_isShared_505_ == 0)
{
lean_ctor_set(v___x_504_, 1, v___x_495_);
lean_ctor_set(v___x_504_, 0, v___x_522_);
v___x_524_ = v___x_504_;
goto v_reusejp_523_;
}
else
{
lean_object* v_reuseFailAlloc_529_; 
v_reuseFailAlloc_529_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_529_, 0, v___x_522_);
lean_ctor_set(v_reuseFailAlloc_529_, 1, v___x_495_);
v___x_524_ = v_reuseFailAlloc_529_;
goto v_reusejp_523_;
}
v_reusejp_523_:
{
lean_object* v___x_525_; lean_object* v___x_527_; 
v___x_525_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_525_, 0, v___x_524_);
lean_ctor_set(v___x_525_, 1, v_snd_502_);
if (v_isShared_519_ == 0)
{
lean_ctor_set(v___x_518_, 0, v___x_525_);
v___x_527_ = v___x_518_;
goto v_reusejp_526_;
}
else
{
lean_object* v_reuseFailAlloc_528_; 
v_reuseFailAlloc_528_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_528_, 0, v___x_525_);
v___x_527_ = v_reuseFailAlloc_528_;
goto v_reusejp_526_;
}
v_reusejp_526_:
{
return v___x_527_;
}
}
}
}
}
else
{
lean_object* v_a_532_; lean_object* v___x_534_; uint8_t v_isShared_535_; uint8_t v_isSharedCheck_539_; 
lean_del_object(v___x_508_);
lean_del_object(v___x_504_);
lean_dec(v_snd_502_);
v_a_532_ = lean_ctor_get(v___x_515_, 0);
v_isSharedCheck_539_ = !lean_is_exclusive(v___x_515_);
if (v_isSharedCheck_539_ == 0)
{
v___x_534_ = v___x_515_;
v_isShared_535_ = v_isSharedCheck_539_;
goto v_resetjp_533_;
}
else
{
lean_inc(v_a_532_);
lean_dec(v___x_515_);
v___x_534_ = lean_box(0);
v_isShared_535_ = v_isSharedCheck_539_;
goto v_resetjp_533_;
}
v_resetjp_533_:
{
lean_object* v___x_537_; 
if (v_isShared_535_ == 0)
{
v___x_537_ = v___x_534_;
goto v_reusejp_536_;
}
else
{
lean_object* v_reuseFailAlloc_538_; 
v_reuseFailAlloc_538_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_538_, 0, v_a_532_);
v___x_537_ = v_reuseFailAlloc_538_;
goto v_reusejp_536_;
}
v_reusejp_536_:
{
return v___x_537_;
}
}
}
}
}
}
}
else
{
lean_object* v_a_543_; lean_object* v___x_545_; uint8_t v_isShared_546_; uint8_t v_isSharedCheck_550_; 
lean_dec_ref(v_tExpr_472_);
v_a_543_ = lean_ctor_get(v___x_498_, 0);
v_isSharedCheck_550_ = !lean_is_exclusive(v___x_498_);
if (v_isSharedCheck_550_ == 0)
{
v___x_545_ = v___x_498_;
v_isShared_546_ = v_isSharedCheck_550_;
goto v_resetjp_544_;
}
else
{
lean_inc(v_a_543_);
lean_dec(v___x_498_);
v___x_545_ = lean_box(0);
v_isShared_546_ = v_isSharedCheck_550_;
goto v_resetjp_544_;
}
v_resetjp_544_:
{
lean_object* v___x_548_; 
if (v_isShared_546_ == 0)
{
v___x_548_ = v___x_545_;
goto v_reusejp_547_;
}
else
{
lean_object* v_reuseFailAlloc_549_; 
v_reuseFailAlloc_549_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_549_, 0, v_a_543_);
v___x_548_ = v_reuseFailAlloc_549_;
goto v_reusejp_547_;
}
v_reusejp_547_:
{
return v___x_548_;
}
}
}
}
else
{
v_a_484_ = v___x_496_;
v_snd_485_ = v___y_477_;
goto v___jp_483_;
}
}
v___jp_483_:
{
size_t v___x_486_; size_t v___x_487_; 
v___x_486_ = ((size_t)1ULL);
v___x_487_ = lean_usize_add(v_i_475_, v___x_486_);
lean_inc_ref(v_a_484_);
v_i_475_ = v___x_487_;
v_b_476_ = v_a_484_;
v___y_477_ = v_snd_485_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__3___boxed(lean_object* v_g_551_, lean_object* v_t_552_, lean_object* v_tExpr_553_, lean_object* v_as_554_, lean_object* v_sz_555_, lean_object* v_i_556_, lean_object* v_b_557_, lean_object* v___y_558_, lean_object* v___y_559_, lean_object* v___y_560_, lean_object* v___y_561_, lean_object* v___y_562_, lean_object* v___y_563_){
_start:
{
size_t v_sz_boxed_564_; size_t v_i_boxed_565_; lean_object* v_res_566_; 
v_sz_boxed_564_ = lean_unbox_usize(v_sz_555_);
lean_dec(v_sz_555_);
v_i_boxed_565_ = lean_unbox_usize(v_i_556_);
lean_dec(v_i_556_);
v_res_566_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__3(v_g_551_, v_t_552_, v_tExpr_553_, v_as_554_, v_sz_boxed_564_, v_i_boxed_565_, v_b_557_, v___y_558_, v___y_559_, v___y_560_, v___y_561_, v___y_562_);
lean_dec(v___y_562_);
lean_dec_ref(v___y_561_);
lean_dec(v___y_560_);
lean_dec_ref(v___y_559_);
lean_dec_ref(v_as_554_);
lean_dec(v_t_552_);
lean_dec_ref(v_g_551_);
return v_res_566_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS___boxed(lean_object* v_g_567_, lean_object* v_v_568_, lean_object* v_t_569_, lean_object* v_tExpr_570_, lean_object* v_a_571_, lean_object* v_a_572_, lean_object* v_a_573_, lean_object* v_a_574_, lean_object* v_a_575_, lean_object* v_a_576_){
_start:
{
lean_object* v_res_577_; 
v_res_577_ = lp_mathlib_Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS(v_g_567_, v_v_568_, v_t_569_, v_tExpr_570_, v_a_571_, v_a_572_, v_a_573_, v_a_574_, v_a_575_);
lean_dec(v_a_575_);
lean_dec_ref(v_a_574_);
lean_dec(v_a_573_);
lean_dec_ref(v_a_572_);
lean_dec(v_t_569_);
lean_dec_ref(v_g_567_);
return v_res_577_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__0(lean_object* v_00_u03b2_578_, lean_object* v_m_579_, lean_object* v_a_580_, lean_object* v_b_581_){
_start:
{
lean_object* v___x_582_; 
v___x_582_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__0___redArg(v_m_579_, v_a_580_, v_b_581_);
return v___x_582_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__1(lean_object* v_00_u03b2_583_, lean_object* v_m_584_, lean_object* v_a_585_){
_start:
{
uint8_t v___x_586_; 
v___x_586_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__1___redArg(v_m_584_, v_a_585_);
return v___x_586_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__1___boxed(lean_object* v_00_u03b2_587_, lean_object* v_m_588_, lean_object* v_a_589_){
_start:
{
uint8_t v_res_590_; lean_object* v_r_591_; 
v_res_590_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS_spec__1(v_00_u03b2_587_, v_m_588_, v_a_589_);
lean_dec(v_a_589_);
lean_dec_ref(v_m_588_);
v_r_591_ = lean_box(v_res_590_);
return v_r_591_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_Graph_buildTransitiveLeProof___redArg(lean_object* v_g_592_, lean_object* v_s_593_, lean_object* v_t_594_, lean_object* v_a_595_, lean_object* v_a_596_, lean_object* v_a_597_, lean_object* v_a_598_, lean_object* v_a_599_){
_start:
{
lean_object* v___x_601_; lean_object* v___x_602_; lean_object* v_state_603_; lean_object* v___x_604_; lean_object* v___x_605_; 
v___x_601_ = lean_st_ref_get(v_a_595_);
v___x_602_ = l_Lean_instInhabitedExpr;
v_state_603_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Order_Graph_constructLeGraph___closed__1, &lp_mathlib_Mathlib_Tactic_Order_Graph_constructLeGraph___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Order_Graph_constructLeGraph___closed__1);
v___x_604_ = lean_array_get(v___x_602_, v___x_601_, v_t_594_);
lean_dec(v___x_601_);
v___x_605_ = lp_mathlib_Mathlib_Tactic_Order_Graph_buildTransitiveLeProofDFS(v_g_592_, v_s_593_, v_t_594_, v___x_604_, v_state_603_, v_a_596_, v_a_597_, v_a_598_, v_a_599_);
if (lean_obj_tag(v___x_605_) == 0)
{
lean_object* v_a_606_; lean_object* v___x_608_; uint8_t v_isShared_609_; uint8_t v_isSharedCheck_614_; 
v_a_606_ = lean_ctor_get(v___x_605_, 0);
v_isSharedCheck_614_ = !lean_is_exclusive(v___x_605_);
if (v_isSharedCheck_614_ == 0)
{
v___x_608_ = v___x_605_;
v_isShared_609_ = v_isSharedCheck_614_;
goto v_resetjp_607_;
}
else
{
lean_inc(v_a_606_);
lean_dec(v___x_605_);
v___x_608_ = lean_box(0);
v_isShared_609_ = v_isSharedCheck_614_;
goto v_resetjp_607_;
}
v_resetjp_607_:
{
lean_object* v_fst_610_; lean_object* v___x_612_; 
v_fst_610_ = lean_ctor_get(v_a_606_, 0);
lean_inc(v_fst_610_);
lean_dec(v_a_606_);
if (v_isShared_609_ == 0)
{
lean_ctor_set(v___x_608_, 0, v_fst_610_);
v___x_612_ = v___x_608_;
goto v_reusejp_611_;
}
else
{
lean_object* v_reuseFailAlloc_613_; 
v_reuseFailAlloc_613_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_613_, 0, v_fst_610_);
v___x_612_ = v_reuseFailAlloc_613_;
goto v_reusejp_611_;
}
v_reusejp_611_:
{
return v___x_612_;
}
}
}
else
{
lean_object* v_a_615_; lean_object* v___x_617_; uint8_t v_isShared_618_; uint8_t v_isSharedCheck_622_; 
v_a_615_ = lean_ctor_get(v___x_605_, 0);
v_isSharedCheck_622_ = !lean_is_exclusive(v___x_605_);
if (v_isSharedCheck_622_ == 0)
{
v___x_617_ = v___x_605_;
v_isShared_618_ = v_isSharedCheck_622_;
goto v_resetjp_616_;
}
else
{
lean_inc(v_a_615_);
lean_dec(v___x_605_);
v___x_617_ = lean_box(0);
v_isShared_618_ = v_isSharedCheck_622_;
goto v_resetjp_616_;
}
v_resetjp_616_:
{
lean_object* v___x_620_; 
if (v_isShared_618_ == 0)
{
v___x_620_ = v___x_617_;
goto v_reusejp_619_;
}
else
{
lean_object* v_reuseFailAlloc_621_; 
v_reuseFailAlloc_621_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_621_, 0, v_a_615_);
v___x_620_ = v_reuseFailAlloc_621_;
goto v_reusejp_619_;
}
v_reusejp_619_:
{
return v___x_620_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_Graph_buildTransitiveLeProof___redArg___boxed(lean_object* v_g_623_, lean_object* v_s_624_, lean_object* v_t_625_, lean_object* v_a_626_, lean_object* v_a_627_, lean_object* v_a_628_, lean_object* v_a_629_, lean_object* v_a_630_, lean_object* v_a_631_){
_start:
{
lean_object* v_res_632_; 
v_res_632_ = lp_mathlib_Mathlib_Tactic_Order_Graph_buildTransitiveLeProof___redArg(v_g_623_, v_s_624_, v_t_625_, v_a_626_, v_a_627_, v_a_628_, v_a_629_, v_a_630_);
lean_dec(v_a_630_);
lean_dec_ref(v_a_629_);
lean_dec(v_a_628_);
lean_dec_ref(v_a_627_);
lean_dec(v_a_626_);
lean_dec(v_t_625_);
lean_dec_ref(v_g_623_);
return v_res_632_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_Graph_buildTransitiveLeProof(lean_object* v_g_633_, lean_object* v_s_634_, lean_object* v_t_635_, lean_object* v_a_636_, lean_object* v_a_637_, lean_object* v_a_638_, lean_object* v_a_639_, lean_object* v_a_640_, lean_object* v_a_641_){
_start:
{
lean_object* v___x_643_; 
v___x_643_ = lp_mathlib_Mathlib_Tactic_Order_Graph_buildTransitiveLeProof___redArg(v_g_633_, v_s_634_, v_t_635_, v_a_637_, v_a_638_, v_a_639_, v_a_640_, v_a_641_);
return v___x_643_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_Graph_buildTransitiveLeProof___boxed(lean_object* v_g_644_, lean_object* v_s_645_, lean_object* v_t_646_, lean_object* v_a_647_, lean_object* v_a_648_, lean_object* v_a_649_, lean_object* v_a_650_, lean_object* v_a_651_, lean_object* v_a_652_, lean_object* v_a_653_){
_start:
{
lean_object* v_res_654_; 
v_res_654_ = lp_mathlib_Mathlib_Tactic_Order_Graph_buildTransitiveLeProof(v_g_644_, v_s_645_, v_t_646_, v_a_647_, v_a_648_, v_a_649_, v_a_650_, v_a_651_, v_a_652_);
lean_dec(v_a_652_);
lean_dec_ref(v_a_651_);
lean_dec(v_a_650_);
lean_dec_ref(v_a_649_);
lean_dec(v_a_648_);
lean_dec_ref(v_a_647_);
lean_dec(v_t_646_);
lean_dec_ref(v_g_644_);
return v_res_654_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Order_CollectFacts(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Order_Graph_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Order_CollectFacts(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Util_AtomM(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Order_Graph_Basic(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Util_AtomM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Order_CollectFacts(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Util_AtomM(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Order_Graph_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Order_CollectFacts(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Util_AtomM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Order_Graph_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Order_Graph_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Order_Graph_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
