// Lean compiler output
// Module: Mathlib.Tactic.Order.Graph.Tarjan
// Imports: public import Init public meta import Init public meta import Mathlib.Tactic.Order.Graph.Basic public import Mathlib.Tactic.Order.Graph.Basic
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
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lean_array_get_size(lean_object*);
uint64_t lean_uint64_of_nat(lean_object*);
uint64_t lean_uint64_shift_right(uint64_t, uint64_t);
uint64_t lean_uint64_xor(uint64_t, uint64_t);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_of_nat(lean_object*);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_land(size_t, size_t);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_mkPanicMessageWithDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_panic_fn_borrowed(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_array_get(lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* lean_nat_div(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_pop(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Array_instInhabited(lean_object*);
size_t lean_array_size(lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__1_spec__2___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__1_spec__2___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__5___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__5___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__1_spec__3_spec__5_spec__12___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__1_spec__3_spec__5___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__1_spec__3___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__1_spec__4___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__0_spec__0_spec__1(lean_object*);
static const lean_string_object lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__0_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 43, .m_capacity = 43, .m_length = 42, .m_data = "Std.Data.DHashMap.Internal.AssocList.Basic"};
static const lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__0_spec__0___closed__0 = (const lean_object*)&lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__0_spec__0___closed__0_value;
static const lean_string_object lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__0_spec__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 37, .m_capacity = 37, .m_length = 36, .m_data = "Std.DHashMap.Internal.AssocList.get!"};
static const lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__0_spec__0___closed__1 = (const lean_object*)&lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__0_spec__0___closed__1_value;
static const lean_string_object lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__0_spec__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 33, .m_capacity = 33, .m_length = 32, .m_data = "key is not present in hash table"};
static const lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__0_spec__0___closed__2 = (const lean_object*)&lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__0_spec__0___closed__2_value;
static lean_once_cell_t lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__0_spec__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__0_spec__0___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__0_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__0_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__4___redArg(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_panic___at___00Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__6_spec__11_spec__14___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_panic___at___00Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__6_spec__11_spec__14___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__6_spec__11_spec__14(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__6_spec__11(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__6_spec__11___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__6(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__6___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__2_spec__6___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__2_spec__6___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_erase___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__2___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_erase___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__2___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_While_0__repeatM_erased___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__3___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_While_0__repeatM_erased___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__3___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__7(lean_object*, uint8_t, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_Graph_tarjanDFS(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_Graph_tarjanDFS___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_erase___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_erase___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_While_0__repeatM_erased___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_While_0__repeatM_erased___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__4(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__5(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__5___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__1_spec__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__1_spec__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__1_spec__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__1_spec__4(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__2_spec__6(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__2_spec__6___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__1_spec__3_spec__5(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__1_spec__3_spec__5_spec__12(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_Graph_findSCCsImp_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_Graph_findSCCsImp_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_Graph_findSCCsImp_spec__1(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_Graph_findSCCsImp_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_Graph_findSCCsImp(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_Graph_findSCCsImp___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Order_Graph_findSCCs___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Order_Graph_findSCCs___closed__0;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Order_Graph_findSCCs___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Order_Graph_findSCCs___closed__1;
static const lean_array_object lp_mathlib_Mathlib_Tactic_Order_Graph_findSCCs___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic_Order_Graph_findSCCs___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_Order_Graph_findSCCs___closed__2_value;
static lean_once_cell_t lp_mathlib_Mathlib_Tactic_Order_Graph_findSCCs___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Tactic_Order_Graph_findSCCs___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_Graph_findSCCs(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_Graph_findSCCs___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__1_spec__2___redArg(lean_object* v_a_1_, lean_object* v_x_2_){
_start:
{
if (lean_obj_tag(v_x_2_) == 0)
{
uint8_t v___x_3_; 
v___x_3_ = 0;
return v___x_3_;
}
else
{
lean_object* v_key_4_; lean_object* v_tail_5_; uint8_t v___x_6_; 
v_key_4_ = lean_ctor_get(v_x_2_, 0);
v_tail_5_ = lean_ctor_get(v_x_2_, 2);
v___x_6_ = lean_nat_dec_eq(v_key_4_, v_a_1_);
if (v___x_6_ == 0)
{
v_x_2_ = v_tail_5_;
goto _start;
}
else
{
return v___x_6_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__1_spec__2___redArg___boxed(lean_object* v_a_8_, lean_object* v_x_9_){
_start:
{
uint8_t v_res_10_; lean_object* v_r_11_; 
v_res_10_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__1_spec__2___redArg(v_a_8_, v_x_9_);
lean_dec(v_x_9_);
lean_dec(v_a_8_);
v_r_11_ = lean_box(v_res_10_);
return v_r_11_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__5___redArg(lean_object* v_m_12_, lean_object* v_a_13_){
_start:
{
lean_object* v_buckets_14_; lean_object* v___x_15_; uint64_t v___x_16_; uint64_t v___x_17_; uint64_t v___x_18_; uint64_t v_fold_19_; uint64_t v___x_20_; uint64_t v___x_21_; uint64_t v___x_22_; size_t v___x_23_; size_t v___x_24_; size_t v___x_25_; size_t v___x_26_; size_t v___x_27_; lean_object* v___x_28_; uint8_t v___x_29_; 
v_buckets_14_ = lean_ctor_get(v_m_12_, 1);
v___x_15_ = lean_array_get_size(v_buckets_14_);
v___x_16_ = lean_uint64_of_nat(v_a_13_);
v___x_17_ = 32ULL;
v___x_18_ = lean_uint64_shift_right(v___x_16_, v___x_17_);
v_fold_19_ = lean_uint64_xor(v___x_16_, v___x_18_);
v___x_20_ = 16ULL;
v___x_21_ = lean_uint64_shift_right(v_fold_19_, v___x_20_);
v___x_22_ = lean_uint64_xor(v_fold_19_, v___x_21_);
v___x_23_ = lean_uint64_to_usize(v___x_22_);
v___x_24_ = lean_usize_of_nat(v___x_15_);
v___x_25_ = ((size_t)1ULL);
v___x_26_ = lean_usize_sub(v___x_24_, v___x_25_);
v___x_27_ = lean_usize_land(v___x_23_, v___x_26_);
v___x_28_ = lean_array_uget_borrowed(v_buckets_14_, v___x_27_);
v___x_29_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__1_spec__2___redArg(v_a_13_, v___x_28_);
return v___x_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__5___redArg___boxed(lean_object* v_m_30_, lean_object* v_a_31_){
_start:
{
uint8_t v_res_32_; lean_object* v_r_33_; 
v_res_32_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__5___redArg(v_m_30_, v_a_31_);
lean_dec(v_a_31_);
lean_dec_ref(v_m_30_);
v_r_33_ = lean_box(v_res_32_);
return v_r_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__1_spec__3_spec__5_spec__12___redArg(lean_object* v_x_34_, lean_object* v_x_35_){
_start:
{
if (lean_obj_tag(v_x_35_) == 0)
{
return v_x_34_;
}
else
{
lean_object* v_key_36_; lean_object* v_value_37_; lean_object* v_tail_38_; lean_object* v___x_40_; uint8_t v_isShared_41_; uint8_t v_isSharedCheck_61_; 
v_key_36_ = lean_ctor_get(v_x_35_, 0);
v_value_37_ = lean_ctor_get(v_x_35_, 1);
v_tail_38_ = lean_ctor_get(v_x_35_, 2);
v_isSharedCheck_61_ = !lean_is_exclusive(v_x_35_);
if (v_isSharedCheck_61_ == 0)
{
v___x_40_ = v_x_35_;
v_isShared_41_ = v_isSharedCheck_61_;
goto v_resetjp_39_;
}
else
{
lean_inc(v_tail_38_);
lean_inc(v_value_37_);
lean_inc(v_key_36_);
lean_dec(v_x_35_);
v___x_40_ = lean_box(0);
v_isShared_41_ = v_isSharedCheck_61_;
goto v_resetjp_39_;
}
v_resetjp_39_:
{
lean_object* v___x_42_; uint64_t v___x_43_; uint64_t v___x_44_; uint64_t v___x_45_; uint64_t v_fold_46_; uint64_t v___x_47_; uint64_t v___x_48_; uint64_t v___x_49_; size_t v___x_50_; size_t v___x_51_; size_t v___x_52_; size_t v___x_53_; size_t v___x_54_; lean_object* v___x_55_; lean_object* v___x_57_; 
v___x_42_ = lean_array_get_size(v_x_34_);
v___x_43_ = lean_uint64_of_nat(v_key_36_);
v___x_44_ = 32ULL;
v___x_45_ = lean_uint64_shift_right(v___x_43_, v___x_44_);
v_fold_46_ = lean_uint64_xor(v___x_43_, v___x_45_);
v___x_47_ = 16ULL;
v___x_48_ = lean_uint64_shift_right(v_fold_46_, v___x_47_);
v___x_49_ = lean_uint64_xor(v_fold_46_, v___x_48_);
v___x_50_ = lean_uint64_to_usize(v___x_49_);
v___x_51_ = lean_usize_of_nat(v___x_42_);
v___x_52_ = ((size_t)1ULL);
v___x_53_ = lean_usize_sub(v___x_51_, v___x_52_);
v___x_54_ = lean_usize_land(v___x_50_, v___x_53_);
v___x_55_ = lean_array_uget_borrowed(v_x_34_, v___x_54_);
lean_inc(v___x_55_);
if (v_isShared_41_ == 0)
{
lean_ctor_set(v___x_40_, 2, v___x_55_);
v___x_57_ = v___x_40_;
goto v_reusejp_56_;
}
else
{
lean_object* v_reuseFailAlloc_60_; 
v_reuseFailAlloc_60_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_60_, 0, v_key_36_);
lean_ctor_set(v_reuseFailAlloc_60_, 1, v_value_37_);
lean_ctor_set(v_reuseFailAlloc_60_, 2, v___x_55_);
v___x_57_ = v_reuseFailAlloc_60_;
goto v_reusejp_56_;
}
v_reusejp_56_:
{
lean_object* v___x_58_; 
v___x_58_ = lean_array_uset(v_x_34_, v___x_54_, v___x_57_);
v_x_34_ = v___x_58_;
v_x_35_ = v_tail_38_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__1_spec__3_spec__5___redArg(lean_object* v_i_62_, lean_object* v_source_63_, lean_object* v_target_64_){
_start:
{
lean_object* v___x_65_; uint8_t v___x_66_; 
v___x_65_ = lean_array_get_size(v_source_63_);
v___x_66_ = lean_nat_dec_lt(v_i_62_, v___x_65_);
if (v___x_66_ == 0)
{
lean_dec_ref(v_source_63_);
lean_dec(v_i_62_);
return v_target_64_;
}
else
{
lean_object* v_es_67_; lean_object* v___x_68_; lean_object* v_source_69_; lean_object* v_target_70_; lean_object* v___x_71_; lean_object* v___x_72_; 
v_es_67_ = lean_array_fget(v_source_63_, v_i_62_);
v___x_68_ = lean_box(0);
v_source_69_ = lean_array_fset(v_source_63_, v_i_62_, v___x_68_);
v_target_70_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__1_spec__3_spec__5_spec__12___redArg(v_target_64_, v_es_67_);
v___x_71_ = lean_unsigned_to_nat(1u);
v___x_72_ = lean_nat_add(v_i_62_, v___x_71_);
lean_dec(v_i_62_);
v_i_62_ = v___x_72_;
v_source_63_ = v_source_69_;
v_target_64_ = v_target_70_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__1_spec__3___redArg(lean_object* v_data_74_){
_start:
{
lean_object* v___x_75_; lean_object* v___x_76_; lean_object* v_nbuckets_77_; lean_object* v___x_78_; lean_object* v___x_79_; lean_object* v___x_80_; lean_object* v___x_81_; 
v___x_75_ = lean_array_get_size(v_data_74_);
v___x_76_ = lean_unsigned_to_nat(2u);
v_nbuckets_77_ = lean_nat_mul(v___x_75_, v___x_76_);
v___x_78_ = lean_unsigned_to_nat(0u);
v___x_79_ = lean_box(0);
v___x_80_ = lean_mk_array(v_nbuckets_77_, v___x_79_);
v___x_81_ = lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__1_spec__3_spec__5___redArg(v___x_78_, v_data_74_, v___x_80_);
return v___x_81_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__1_spec__4___redArg(lean_object* v_a_82_, lean_object* v_b_83_, lean_object* v_x_84_){
_start:
{
if (lean_obj_tag(v_x_84_) == 0)
{
lean_dec(v_b_83_);
lean_dec(v_a_82_);
return v_x_84_;
}
else
{
lean_object* v_key_85_; lean_object* v_value_86_; lean_object* v_tail_87_; lean_object* v___x_89_; uint8_t v_isShared_90_; uint8_t v_isSharedCheck_99_; 
v_key_85_ = lean_ctor_get(v_x_84_, 0);
v_value_86_ = lean_ctor_get(v_x_84_, 1);
v_tail_87_ = lean_ctor_get(v_x_84_, 2);
v_isSharedCheck_99_ = !lean_is_exclusive(v_x_84_);
if (v_isSharedCheck_99_ == 0)
{
v___x_89_ = v_x_84_;
v_isShared_90_ = v_isSharedCheck_99_;
goto v_resetjp_88_;
}
else
{
lean_inc(v_tail_87_);
lean_inc(v_value_86_);
lean_inc(v_key_85_);
lean_dec(v_x_84_);
v___x_89_ = lean_box(0);
v_isShared_90_ = v_isSharedCheck_99_;
goto v_resetjp_88_;
}
v_resetjp_88_:
{
uint8_t v___x_91_; 
v___x_91_ = lean_nat_dec_eq(v_key_85_, v_a_82_);
if (v___x_91_ == 0)
{
lean_object* v___x_92_; lean_object* v___x_94_; 
v___x_92_ = lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__1_spec__4___redArg(v_a_82_, v_b_83_, v_tail_87_);
if (v_isShared_90_ == 0)
{
lean_ctor_set(v___x_89_, 2, v___x_92_);
v___x_94_ = v___x_89_;
goto v_reusejp_93_;
}
else
{
lean_object* v_reuseFailAlloc_95_; 
v_reuseFailAlloc_95_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_95_, 0, v_key_85_);
lean_ctor_set(v_reuseFailAlloc_95_, 1, v_value_86_);
lean_ctor_set(v_reuseFailAlloc_95_, 2, v___x_92_);
v___x_94_ = v_reuseFailAlloc_95_;
goto v_reusejp_93_;
}
v_reusejp_93_:
{
return v___x_94_;
}
}
else
{
lean_object* v___x_97_; 
lean_dec(v_value_86_);
lean_dec(v_key_85_);
if (v_isShared_90_ == 0)
{
lean_ctor_set(v___x_89_, 1, v_b_83_);
lean_ctor_set(v___x_89_, 0, v_a_82_);
v___x_97_ = v___x_89_;
goto v_reusejp_96_;
}
else
{
lean_object* v_reuseFailAlloc_98_; 
v_reuseFailAlloc_98_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_98_, 0, v_a_82_);
lean_ctor_set(v_reuseFailAlloc_98_, 1, v_b_83_);
lean_ctor_set(v_reuseFailAlloc_98_, 2, v_tail_87_);
v___x_97_ = v_reuseFailAlloc_98_;
goto v_reusejp_96_;
}
v_reusejp_96_:
{
return v___x_97_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__1___redArg(lean_object* v_m_100_, lean_object* v_a_101_, lean_object* v_b_102_){
_start:
{
lean_object* v_size_103_; lean_object* v_buckets_104_; lean_object* v___x_106_; uint8_t v_isShared_107_; uint8_t v_isSharedCheck_147_; 
v_size_103_ = lean_ctor_get(v_m_100_, 0);
v_buckets_104_ = lean_ctor_get(v_m_100_, 1);
v_isSharedCheck_147_ = !lean_is_exclusive(v_m_100_);
if (v_isSharedCheck_147_ == 0)
{
v___x_106_ = v_m_100_;
v_isShared_107_ = v_isSharedCheck_147_;
goto v_resetjp_105_;
}
else
{
lean_inc(v_buckets_104_);
lean_inc(v_size_103_);
lean_dec(v_m_100_);
v___x_106_ = lean_box(0);
v_isShared_107_ = v_isSharedCheck_147_;
goto v_resetjp_105_;
}
v_resetjp_105_:
{
lean_object* v___x_108_; uint64_t v___x_109_; uint64_t v___x_110_; uint64_t v___x_111_; uint64_t v_fold_112_; uint64_t v___x_113_; uint64_t v___x_114_; uint64_t v___x_115_; size_t v___x_116_; size_t v___x_117_; size_t v___x_118_; size_t v___x_119_; size_t v___x_120_; lean_object* v_bkt_121_; uint8_t v___x_122_; 
v___x_108_ = lean_array_get_size(v_buckets_104_);
v___x_109_ = lean_uint64_of_nat(v_a_101_);
v___x_110_ = 32ULL;
v___x_111_ = lean_uint64_shift_right(v___x_109_, v___x_110_);
v_fold_112_ = lean_uint64_xor(v___x_109_, v___x_111_);
v___x_113_ = 16ULL;
v___x_114_ = lean_uint64_shift_right(v_fold_112_, v___x_113_);
v___x_115_ = lean_uint64_xor(v_fold_112_, v___x_114_);
v___x_116_ = lean_uint64_to_usize(v___x_115_);
v___x_117_ = lean_usize_of_nat(v___x_108_);
v___x_118_ = ((size_t)1ULL);
v___x_119_ = lean_usize_sub(v___x_117_, v___x_118_);
v___x_120_ = lean_usize_land(v___x_116_, v___x_119_);
v_bkt_121_ = lean_array_uget_borrowed(v_buckets_104_, v___x_120_);
v___x_122_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__1_spec__2___redArg(v_a_101_, v_bkt_121_);
if (v___x_122_ == 0)
{
lean_object* v___x_123_; lean_object* v_size_x27_124_; lean_object* v___x_125_; lean_object* v_buckets_x27_126_; lean_object* v___x_127_; lean_object* v___x_128_; lean_object* v___x_129_; lean_object* v___x_130_; lean_object* v___x_131_; uint8_t v___x_132_; 
v___x_123_ = lean_unsigned_to_nat(1u);
v_size_x27_124_ = lean_nat_add(v_size_103_, v___x_123_);
lean_dec(v_size_103_);
lean_inc(v_bkt_121_);
v___x_125_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_125_, 0, v_a_101_);
lean_ctor_set(v___x_125_, 1, v_b_102_);
lean_ctor_set(v___x_125_, 2, v_bkt_121_);
v_buckets_x27_126_ = lean_array_uset(v_buckets_104_, v___x_120_, v___x_125_);
v___x_127_ = lean_unsigned_to_nat(4u);
v___x_128_ = lean_nat_mul(v_size_x27_124_, v___x_127_);
v___x_129_ = lean_unsigned_to_nat(3u);
v___x_130_ = lean_nat_div(v___x_128_, v___x_129_);
lean_dec(v___x_128_);
v___x_131_ = lean_array_get_size(v_buckets_x27_126_);
v___x_132_ = lean_nat_dec_le(v___x_130_, v___x_131_);
lean_dec(v___x_130_);
if (v___x_132_ == 0)
{
lean_object* v_val_133_; lean_object* v___x_135_; 
v_val_133_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__1_spec__3___redArg(v_buckets_x27_126_);
if (v_isShared_107_ == 0)
{
lean_ctor_set(v___x_106_, 1, v_val_133_);
lean_ctor_set(v___x_106_, 0, v_size_x27_124_);
v___x_135_ = v___x_106_;
goto v_reusejp_134_;
}
else
{
lean_object* v_reuseFailAlloc_136_; 
v_reuseFailAlloc_136_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_136_, 0, v_size_x27_124_);
lean_ctor_set(v_reuseFailAlloc_136_, 1, v_val_133_);
v___x_135_ = v_reuseFailAlloc_136_;
goto v_reusejp_134_;
}
v_reusejp_134_:
{
return v___x_135_;
}
}
else
{
lean_object* v___x_138_; 
if (v_isShared_107_ == 0)
{
lean_ctor_set(v___x_106_, 1, v_buckets_x27_126_);
lean_ctor_set(v___x_106_, 0, v_size_x27_124_);
v___x_138_ = v___x_106_;
goto v_reusejp_137_;
}
else
{
lean_object* v_reuseFailAlloc_139_; 
v_reuseFailAlloc_139_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_139_, 0, v_size_x27_124_);
lean_ctor_set(v_reuseFailAlloc_139_, 1, v_buckets_x27_126_);
v___x_138_ = v_reuseFailAlloc_139_;
goto v_reusejp_137_;
}
v_reusejp_137_:
{
return v___x_138_;
}
}
}
else
{
lean_object* v___x_140_; lean_object* v_buckets_x27_141_; lean_object* v___x_142_; lean_object* v___x_143_; lean_object* v___x_145_; 
lean_inc(v_bkt_121_);
v___x_140_ = lean_box(0);
v_buckets_x27_141_ = lean_array_uset(v_buckets_104_, v___x_120_, v___x_140_);
v___x_142_ = lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__1_spec__4___redArg(v_a_101_, v_b_102_, v_bkt_121_);
v___x_143_ = lean_array_uset(v_buckets_x27_141_, v___x_120_, v___x_142_);
if (v_isShared_107_ == 0)
{
lean_ctor_set(v___x_106_, 1, v___x_143_);
v___x_145_ = v___x_106_;
goto v_reusejp_144_;
}
else
{
lean_object* v_reuseFailAlloc_146_; 
v_reuseFailAlloc_146_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_146_, 0, v_size_103_);
lean_ctor_set(v_reuseFailAlloc_146_, 1, v___x_143_);
v___x_145_ = v_reuseFailAlloc_146_;
goto v_reusejp_144_;
}
v_reusejp_144_:
{
return v___x_145_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__0_spec__0_spec__1(lean_object* v_msg_148_){
_start:
{
lean_object* v___x_149_; lean_object* v___x_150_; 
v___x_149_ = lean_unsigned_to_nat(0u);
v___x_150_ = lean_panic_fn_borrowed(v___x_149_, v_msg_148_);
return v___x_150_;
}
}
static lean_object* _init_lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__0_spec__0___closed__3(void){
_start:
{
lean_object* v___x_154_; lean_object* v___x_155_; lean_object* v___x_156_; lean_object* v___x_157_; lean_object* v___x_158_; lean_object* v___x_159_; 
v___x_154_ = ((lean_object*)(lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__0_spec__0___closed__2));
v___x_155_ = lean_unsigned_to_nat(11u);
v___x_156_ = lean_unsigned_to_nat(163u);
v___x_157_ = ((lean_object*)(lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__0_spec__0___closed__1));
v___x_158_ = ((lean_object*)(lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__0_spec__0___closed__0));
v___x_159_ = l_mkPanicMessageWithDecl(v___x_158_, v___x_157_, v___x_156_, v___x_155_, v___x_154_);
return v___x_159_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__0_spec__0(lean_object* v_a_160_, lean_object* v_x_161_){
_start:
{
if (lean_obj_tag(v_x_161_) == 0)
{
lean_object* v___x_162_; lean_object* v___x_163_; 
v___x_162_ = lean_obj_once(&lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__0_spec__0___closed__3, &lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__0_spec__0___closed__3_once, _init_lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__0_spec__0___closed__3);
v___x_163_ = lp_mathlib_panic___at___00Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__0_spec__0_spec__1(v___x_162_);
return v___x_163_;
}
else
{
lean_object* v_key_164_; lean_object* v_value_165_; lean_object* v_tail_166_; uint8_t v___x_167_; 
v_key_164_ = lean_ctor_get(v_x_161_, 0);
v_value_165_ = lean_ctor_get(v_x_161_, 1);
v_tail_166_ = lean_ctor_get(v_x_161_, 2);
v___x_167_ = lean_nat_dec_eq(v_key_164_, v_a_160_);
if (v___x_167_ == 0)
{
v_x_161_ = v_tail_166_;
goto _start;
}
else
{
lean_inc(v_value_165_);
return v_value_165_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__0_spec__0___boxed(lean_object* v_a_169_, lean_object* v_x_170_){
_start:
{
lean_object* v_res_171_; 
v_res_171_ = lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__0_spec__0(v_a_169_, v_x_170_);
lean_dec(v_x_170_);
lean_dec(v_a_169_);
return v_res_171_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__0(lean_object* v_m_172_, lean_object* v_a_173_){
_start:
{
lean_object* v_buckets_174_; lean_object* v___x_175_; uint64_t v___x_176_; uint64_t v___x_177_; uint64_t v___x_178_; uint64_t v_fold_179_; uint64_t v___x_180_; uint64_t v___x_181_; uint64_t v___x_182_; size_t v___x_183_; size_t v___x_184_; size_t v___x_185_; size_t v___x_186_; size_t v___x_187_; lean_object* v___x_188_; lean_object* v___x_189_; 
v_buckets_174_ = lean_ctor_get(v_m_172_, 1);
v___x_175_ = lean_array_get_size(v_buckets_174_);
v___x_176_ = lean_uint64_of_nat(v_a_173_);
v___x_177_ = 32ULL;
v___x_178_ = lean_uint64_shift_right(v___x_176_, v___x_177_);
v_fold_179_ = lean_uint64_xor(v___x_176_, v___x_178_);
v___x_180_ = 16ULL;
v___x_181_ = lean_uint64_shift_right(v_fold_179_, v___x_180_);
v___x_182_ = lean_uint64_xor(v_fold_179_, v___x_181_);
v___x_183_ = lean_uint64_to_usize(v___x_182_);
v___x_184_ = lean_usize_of_nat(v___x_175_);
v___x_185_ = ((size_t)1ULL);
v___x_186_ = lean_usize_sub(v___x_184_, v___x_185_);
v___x_187_ = lean_usize_land(v___x_183_, v___x_186_);
v___x_188_ = lean_array_uget_borrowed(v_buckets_174_, v___x_187_);
v___x_189_ = lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__0_spec__0(v_a_173_, v___x_188_);
return v___x_189_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__0___boxed(lean_object* v_m_190_, lean_object* v_a_191_){
_start:
{
lean_object* v_res_192_; 
v_res_192_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__0(v_m_190_, v_a_191_);
lean_dec(v_a_191_);
lean_dec_ref(v_m_190_);
return v_res_192_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__4___redArg(lean_object* v_m_193_, lean_object* v_a_194_, lean_object* v_b_195_){
_start:
{
lean_object* v_size_196_; lean_object* v_buckets_197_; lean_object* v___x_198_; uint64_t v___x_199_; uint64_t v___x_200_; uint64_t v___x_201_; uint64_t v_fold_202_; uint64_t v___x_203_; uint64_t v___x_204_; uint64_t v___x_205_; size_t v___x_206_; size_t v___x_207_; size_t v___x_208_; size_t v___x_209_; size_t v___x_210_; lean_object* v_bkt_211_; uint8_t v___x_212_; 
v_size_196_ = lean_ctor_get(v_m_193_, 0);
v_buckets_197_ = lean_ctor_get(v_m_193_, 1);
v___x_198_ = lean_array_get_size(v_buckets_197_);
v___x_199_ = lean_uint64_of_nat(v_a_194_);
v___x_200_ = 32ULL;
v___x_201_ = lean_uint64_shift_right(v___x_199_, v___x_200_);
v_fold_202_ = lean_uint64_xor(v___x_199_, v___x_201_);
v___x_203_ = 16ULL;
v___x_204_ = lean_uint64_shift_right(v_fold_202_, v___x_203_);
v___x_205_ = lean_uint64_xor(v_fold_202_, v___x_204_);
v___x_206_ = lean_uint64_to_usize(v___x_205_);
v___x_207_ = lean_usize_of_nat(v___x_198_);
v___x_208_ = ((size_t)1ULL);
v___x_209_ = lean_usize_sub(v___x_207_, v___x_208_);
v___x_210_ = lean_usize_land(v___x_206_, v___x_209_);
v_bkt_211_ = lean_array_uget_borrowed(v_buckets_197_, v___x_210_);
v___x_212_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__1_spec__2___redArg(v_a_194_, v_bkt_211_);
if (v___x_212_ == 0)
{
lean_object* v___x_214_; uint8_t v_isShared_215_; uint8_t v_isSharedCheck_233_; 
lean_inc_ref(v_buckets_197_);
lean_inc(v_size_196_);
v_isSharedCheck_233_ = !lean_is_exclusive(v_m_193_);
if (v_isSharedCheck_233_ == 0)
{
lean_object* v_unused_234_; lean_object* v_unused_235_; 
v_unused_234_ = lean_ctor_get(v_m_193_, 1);
lean_dec(v_unused_234_);
v_unused_235_ = lean_ctor_get(v_m_193_, 0);
lean_dec(v_unused_235_);
v___x_214_ = v_m_193_;
v_isShared_215_ = v_isSharedCheck_233_;
goto v_resetjp_213_;
}
else
{
lean_dec(v_m_193_);
v___x_214_ = lean_box(0);
v_isShared_215_ = v_isSharedCheck_233_;
goto v_resetjp_213_;
}
v_resetjp_213_:
{
lean_object* v___x_216_; lean_object* v_size_x27_217_; lean_object* v___x_218_; lean_object* v_buckets_x27_219_; lean_object* v___x_220_; lean_object* v___x_221_; lean_object* v___x_222_; lean_object* v___x_223_; lean_object* v___x_224_; uint8_t v___x_225_; 
v___x_216_ = lean_unsigned_to_nat(1u);
v_size_x27_217_ = lean_nat_add(v_size_196_, v___x_216_);
lean_dec(v_size_196_);
lean_inc(v_bkt_211_);
v___x_218_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_218_, 0, v_a_194_);
lean_ctor_set(v___x_218_, 1, v_b_195_);
lean_ctor_set(v___x_218_, 2, v_bkt_211_);
v_buckets_x27_219_ = lean_array_uset(v_buckets_197_, v___x_210_, v___x_218_);
v___x_220_ = lean_unsigned_to_nat(4u);
v___x_221_ = lean_nat_mul(v_size_x27_217_, v___x_220_);
v___x_222_ = lean_unsigned_to_nat(3u);
v___x_223_ = lean_nat_div(v___x_221_, v___x_222_);
lean_dec(v___x_221_);
v___x_224_ = lean_array_get_size(v_buckets_x27_219_);
v___x_225_ = lean_nat_dec_le(v___x_223_, v___x_224_);
lean_dec(v___x_223_);
if (v___x_225_ == 0)
{
lean_object* v_val_226_; lean_object* v___x_228_; 
v_val_226_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__1_spec__3___redArg(v_buckets_x27_219_);
if (v_isShared_215_ == 0)
{
lean_ctor_set(v___x_214_, 1, v_val_226_);
lean_ctor_set(v___x_214_, 0, v_size_x27_217_);
v___x_228_ = v___x_214_;
goto v_reusejp_227_;
}
else
{
lean_object* v_reuseFailAlloc_229_; 
v_reuseFailAlloc_229_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_229_, 0, v_size_x27_217_);
lean_ctor_set(v_reuseFailAlloc_229_, 1, v_val_226_);
v___x_228_ = v_reuseFailAlloc_229_;
goto v_reusejp_227_;
}
v_reusejp_227_:
{
return v___x_228_;
}
}
else
{
lean_object* v___x_231_; 
if (v_isShared_215_ == 0)
{
lean_ctor_set(v___x_214_, 1, v_buckets_x27_219_);
lean_ctor_set(v___x_214_, 0, v_size_x27_217_);
v___x_231_ = v___x_214_;
goto v_reusejp_230_;
}
else
{
lean_object* v_reuseFailAlloc_232_; 
v_reuseFailAlloc_232_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_232_, 0, v_size_x27_217_);
lean_ctor_set(v_reuseFailAlloc_232_, 1, v_buckets_x27_219_);
v___x_231_ = v_reuseFailAlloc_232_;
goto v_reusejp_230_;
}
v_reusejp_230_:
{
return v___x_231_;
}
}
}
}
else
{
lean_dec(v_b_195_);
lean_dec(v_a_194_);
return v_m_193_;
}
}
}
static lean_object* _init_lp_mathlib_panic___at___00Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__6_spec__11_spec__14___closed__0(void){
_start:
{
lean_object* v___x_236_; 
v___x_236_ = l_Array_instInhabited(lean_box(0));
return v___x_236_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_panic___at___00Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__6_spec__11_spec__14(lean_object* v_msg_237_){
_start:
{
lean_object* v___x_238_; lean_object* v___x_239_; 
v___x_238_ = lean_obj_once(&lp_mathlib_panic___at___00Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__6_spec__11_spec__14___closed__0, &lp_mathlib_panic___at___00Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__6_spec__11_spec__14___closed__0_once, _init_lp_mathlib_panic___at___00Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__6_spec__11_spec__14___closed__0);
v___x_239_ = lean_panic_fn_borrowed(v___x_238_, v_msg_237_);
return v___x_239_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__6_spec__11(lean_object* v_a_240_, lean_object* v_x_241_){
_start:
{
if (lean_obj_tag(v_x_241_) == 0)
{
lean_object* v___x_242_; lean_object* v___x_243_; 
v___x_242_ = lean_obj_once(&lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__0_spec__0___closed__3, &lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__0_spec__0___closed__3_once, _init_lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__0_spec__0___closed__3);
v___x_243_ = lp_mathlib_panic___at___00Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__6_spec__11_spec__14(v___x_242_);
return v___x_243_;
}
else
{
lean_object* v_key_244_; lean_object* v_value_245_; lean_object* v_tail_246_; uint8_t v___x_247_; 
v_key_244_ = lean_ctor_get(v_x_241_, 0);
v_value_245_ = lean_ctor_get(v_x_241_, 1);
v_tail_246_ = lean_ctor_get(v_x_241_, 2);
v___x_247_ = lean_nat_dec_eq(v_key_244_, v_a_240_);
if (v___x_247_ == 0)
{
v_x_241_ = v_tail_246_;
goto _start;
}
else
{
lean_inc(v_value_245_);
return v_value_245_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__6_spec__11___boxed(lean_object* v_a_249_, lean_object* v_x_250_){
_start:
{
lean_object* v_res_251_; 
v_res_251_ = lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__6_spec__11(v_a_249_, v_x_250_);
lean_dec(v_x_250_);
lean_dec(v_a_249_);
return v_res_251_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__6(lean_object* v_m_252_, lean_object* v_a_253_){
_start:
{
lean_object* v_buckets_254_; lean_object* v___x_255_; uint64_t v___x_256_; uint64_t v___x_257_; uint64_t v___x_258_; uint64_t v_fold_259_; uint64_t v___x_260_; uint64_t v___x_261_; uint64_t v___x_262_; size_t v___x_263_; size_t v___x_264_; size_t v___x_265_; size_t v___x_266_; size_t v___x_267_; lean_object* v___x_268_; lean_object* v___x_269_; 
v_buckets_254_ = lean_ctor_get(v_m_252_, 1);
v___x_255_ = lean_array_get_size(v_buckets_254_);
v___x_256_ = lean_uint64_of_nat(v_a_253_);
v___x_257_ = 32ULL;
v___x_258_ = lean_uint64_shift_right(v___x_256_, v___x_257_);
v_fold_259_ = lean_uint64_xor(v___x_256_, v___x_258_);
v___x_260_ = 16ULL;
v___x_261_ = lean_uint64_shift_right(v_fold_259_, v___x_260_);
v___x_262_ = lean_uint64_xor(v_fold_259_, v___x_261_);
v___x_263_ = lean_uint64_to_usize(v___x_262_);
v___x_264_ = lean_usize_of_nat(v___x_255_);
v___x_265_ = ((size_t)1ULL);
v___x_266_ = lean_usize_sub(v___x_264_, v___x_265_);
v___x_267_ = lean_usize_land(v___x_263_, v___x_266_);
v___x_268_ = lean_array_uget_borrowed(v_buckets_254_, v___x_267_);
v___x_269_ = lp_mathlib_Std_DHashMap_Internal_AssocList_get_x21___at___00Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__6_spec__11(v_a_253_, v___x_268_);
return v___x_269_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__6___boxed(lean_object* v_m_270_, lean_object* v_a_271_){
_start:
{
lean_object* v_res_272_; 
v_res_272_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__6(v_m_270_, v_a_271_);
lean_dec(v_a_271_);
lean_dec_ref(v_m_270_);
return v_res_272_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__2_spec__6___redArg(lean_object* v_a_273_, lean_object* v_x_274_){
_start:
{
if (lean_obj_tag(v_x_274_) == 0)
{
return v_x_274_;
}
else
{
lean_object* v_key_275_; lean_object* v_value_276_; lean_object* v_tail_277_; lean_object* v___x_279_; uint8_t v_isShared_280_; uint8_t v_isSharedCheck_286_; 
v_key_275_ = lean_ctor_get(v_x_274_, 0);
v_value_276_ = lean_ctor_get(v_x_274_, 1);
v_tail_277_ = lean_ctor_get(v_x_274_, 2);
v_isSharedCheck_286_ = !lean_is_exclusive(v_x_274_);
if (v_isSharedCheck_286_ == 0)
{
v___x_279_ = v_x_274_;
v_isShared_280_ = v_isSharedCheck_286_;
goto v_resetjp_278_;
}
else
{
lean_inc(v_tail_277_);
lean_inc(v_value_276_);
lean_inc(v_key_275_);
lean_dec(v_x_274_);
v___x_279_ = lean_box(0);
v_isShared_280_ = v_isSharedCheck_286_;
goto v_resetjp_278_;
}
v_resetjp_278_:
{
uint8_t v___x_281_; 
v___x_281_ = lean_nat_dec_eq(v_key_275_, v_a_273_);
if (v___x_281_ == 0)
{
lean_object* v___x_282_; lean_object* v___x_284_; 
v___x_282_ = lp_mathlib_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__2_spec__6___redArg(v_a_273_, v_tail_277_);
if (v_isShared_280_ == 0)
{
lean_ctor_set(v___x_279_, 2, v___x_282_);
v___x_284_ = v___x_279_;
goto v_reusejp_283_;
}
else
{
lean_object* v_reuseFailAlloc_285_; 
v_reuseFailAlloc_285_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_285_, 0, v_key_275_);
lean_ctor_set(v_reuseFailAlloc_285_, 1, v_value_276_);
lean_ctor_set(v_reuseFailAlloc_285_, 2, v___x_282_);
v___x_284_ = v_reuseFailAlloc_285_;
goto v_reusejp_283_;
}
v_reusejp_283_:
{
return v___x_284_;
}
}
else
{
lean_del_object(v___x_279_);
lean_dec(v_value_276_);
lean_dec(v_key_275_);
return v_tail_277_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__2_spec__6___redArg___boxed(lean_object* v_a_287_, lean_object* v_x_288_){
_start:
{
lean_object* v_res_289_; 
v_res_289_ = lp_mathlib_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__2_spec__6___redArg(v_a_287_, v_x_288_);
lean_dec(v_a_287_);
return v_res_289_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_erase___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__2___redArg(lean_object* v_m_290_, lean_object* v_a_291_){
_start:
{
lean_object* v_size_292_; lean_object* v_buckets_293_; lean_object* v___x_294_; uint64_t v___x_295_; uint64_t v___x_296_; uint64_t v___x_297_; uint64_t v_fold_298_; uint64_t v___x_299_; uint64_t v___x_300_; uint64_t v___x_301_; size_t v___x_302_; size_t v___x_303_; size_t v___x_304_; size_t v___x_305_; size_t v___x_306_; lean_object* v_bkt_307_; uint8_t v___x_308_; 
v_size_292_ = lean_ctor_get(v_m_290_, 0);
v_buckets_293_ = lean_ctor_get(v_m_290_, 1);
v___x_294_ = lean_array_get_size(v_buckets_293_);
v___x_295_ = lean_uint64_of_nat(v_a_291_);
v___x_296_ = 32ULL;
v___x_297_ = lean_uint64_shift_right(v___x_295_, v___x_296_);
v_fold_298_ = lean_uint64_xor(v___x_295_, v___x_297_);
v___x_299_ = 16ULL;
v___x_300_ = lean_uint64_shift_right(v_fold_298_, v___x_299_);
v___x_301_ = lean_uint64_xor(v_fold_298_, v___x_300_);
v___x_302_ = lean_uint64_to_usize(v___x_301_);
v___x_303_ = lean_usize_of_nat(v___x_294_);
v___x_304_ = ((size_t)1ULL);
v___x_305_ = lean_usize_sub(v___x_303_, v___x_304_);
v___x_306_ = lean_usize_land(v___x_302_, v___x_305_);
v_bkt_307_ = lean_array_uget_borrowed(v_buckets_293_, v___x_306_);
v___x_308_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__1_spec__2___redArg(v_a_291_, v_bkt_307_);
if (v___x_308_ == 0)
{
return v_m_290_;
}
else
{
lean_object* v___x_310_; uint8_t v_isShared_311_; uint8_t v_isSharedCheck_321_; 
lean_inc(v_bkt_307_);
lean_inc_ref(v_buckets_293_);
lean_inc(v_size_292_);
v_isSharedCheck_321_ = !lean_is_exclusive(v_m_290_);
if (v_isSharedCheck_321_ == 0)
{
lean_object* v_unused_322_; lean_object* v_unused_323_; 
v_unused_322_ = lean_ctor_get(v_m_290_, 1);
lean_dec(v_unused_322_);
v_unused_323_ = lean_ctor_get(v_m_290_, 0);
lean_dec(v_unused_323_);
v___x_310_ = v_m_290_;
v_isShared_311_ = v_isSharedCheck_321_;
goto v_resetjp_309_;
}
else
{
lean_dec(v_m_290_);
v___x_310_ = lean_box(0);
v_isShared_311_ = v_isSharedCheck_321_;
goto v_resetjp_309_;
}
v_resetjp_309_:
{
lean_object* v___x_312_; lean_object* v_buckets_x27_313_; lean_object* v___x_314_; lean_object* v___x_315_; lean_object* v___x_316_; lean_object* v___x_317_; lean_object* v___x_319_; 
v___x_312_ = lean_box(0);
v_buckets_x27_313_ = lean_array_uset(v_buckets_293_, v___x_306_, v___x_312_);
v___x_314_ = lean_unsigned_to_nat(1u);
v___x_315_ = lean_nat_sub(v_size_292_, v___x_314_);
lean_dec(v_size_292_);
v___x_316_ = lp_mathlib_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__2_spec__6___redArg(v_a_291_, v_bkt_307_);
v___x_317_ = lean_array_uset(v_buckets_x27_313_, v___x_306_, v___x_316_);
if (v_isShared_311_ == 0)
{
lean_ctor_set(v___x_310_, 1, v___x_317_);
lean_ctor_set(v___x_310_, 0, v___x_315_);
v___x_319_ = v___x_310_;
goto v_reusejp_318_;
}
else
{
lean_object* v_reuseFailAlloc_320_; 
v_reuseFailAlloc_320_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_320_, 0, v___x_315_);
lean_ctor_set(v_reuseFailAlloc_320_, 1, v___x_317_);
v___x_319_ = v_reuseFailAlloc_320_;
goto v_reusejp_318_;
}
v_reusejp_318_:
{
return v___x_319_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_erase___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__2___redArg___boxed(lean_object* v_m_324_, lean_object* v_a_325_){
_start:
{
lean_object* v_res_326_; 
v_res_326_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_erase___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__2___redArg(v_m_324_, v_a_325_);
lean_dec(v_a_325_);
return v_res_326_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_While_0__repeatM_erased___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__3___redArg(lean_object* v_v_327_, lean_object* v___y_328_){
_start:
{
lean_object* v_toDFSState_329_; lean_object* v_id_330_; lean_object* v_lowlink_331_; lean_object* v_stack_332_; lean_object* v_onStack_333_; lean_object* v_time_334_; lean_object* v___x_336_; uint8_t v_isShared_337_; uint8_t v_isSharedCheck_353_; 
v_toDFSState_329_ = lean_ctor_get(v___y_328_, 0);
v_id_330_ = lean_ctor_get(v___y_328_, 1);
v_lowlink_331_ = lean_ctor_get(v___y_328_, 2);
v_stack_332_ = lean_ctor_get(v___y_328_, 3);
v_onStack_333_ = lean_ctor_get(v___y_328_, 4);
v_time_334_ = lean_ctor_get(v___y_328_, 5);
v_isSharedCheck_353_ = !lean_is_exclusive(v___y_328_);
if (v_isSharedCheck_353_ == 0)
{
v___x_336_ = v___y_328_;
v_isShared_337_ = v_isSharedCheck_353_;
goto v_resetjp_335_;
}
else
{
lean_inc(v_time_334_);
lean_inc(v_onStack_333_);
lean_inc(v_stack_332_);
lean_inc(v_lowlink_331_);
lean_inc(v_id_330_);
lean_inc(v_toDFSState_329_);
lean_dec(v___y_328_);
v___x_336_ = lean_box(0);
v_isShared_337_ = v_isSharedCheck_353_;
goto v_resetjp_335_;
}
v_resetjp_335_:
{
lean_object* v___x_338_; lean_object* v___x_339_; lean_object* v___x_340_; lean_object* v___x_341_; lean_object* v___x_342_; lean_object* v___x_343_; lean_object* v___x_344_; lean_object* v___x_345_; lean_object* v___x_346_; lean_object* v___x_348_; 
v___x_338_ = lean_unsigned_to_nat(0u);
v___x_339_ = lean_array_get_size(v_stack_332_);
v___x_340_ = lean_unsigned_to_nat(1u);
v___x_341_ = lean_nat_sub(v___x_339_, v___x_340_);
v___x_342_ = lean_array_get(v___x_338_, v_stack_332_, v___x_341_);
lean_dec(v___x_341_);
v___x_343_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__0(v_lowlink_331_, v_v_327_);
lean_inc(v___x_342_);
v___x_344_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__1___redArg(v_lowlink_331_, v___x_342_, v___x_343_);
v___x_345_ = lean_array_pop(v_stack_332_);
v___x_346_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_erase___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__2___redArg(v_onStack_333_, v___x_342_);
if (v_isShared_337_ == 0)
{
lean_ctor_set(v___x_336_, 4, v___x_346_);
lean_ctor_set(v___x_336_, 3, v___x_345_);
lean_ctor_set(v___x_336_, 2, v___x_344_);
v___x_348_ = v___x_336_;
goto v_reusejp_347_;
}
else
{
lean_object* v_reuseFailAlloc_352_; 
v_reuseFailAlloc_352_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v_reuseFailAlloc_352_, 0, v_toDFSState_329_);
lean_ctor_set(v_reuseFailAlloc_352_, 1, v_id_330_);
lean_ctor_set(v_reuseFailAlloc_352_, 2, v___x_344_);
lean_ctor_set(v_reuseFailAlloc_352_, 3, v___x_345_);
lean_ctor_set(v_reuseFailAlloc_352_, 4, v___x_346_);
lean_ctor_set(v_reuseFailAlloc_352_, 5, v_time_334_);
v___x_348_ = v_reuseFailAlloc_352_;
goto v_reusejp_347_;
}
v_reusejp_347_:
{
uint8_t v___x_349_; 
v___x_349_ = lean_nat_dec_eq(v___x_342_, v_v_327_);
if (v___x_349_ == 0)
{
lean_dec(v___x_342_);
v___y_328_ = v___x_348_;
goto _start;
}
else
{
lean_object* v___x_351_; 
v___x_351_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_351_, 0, v___x_342_);
lean_ctor_set(v___x_351_, 1, v___x_348_);
return v___x_351_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_While_0__repeatM_erased___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__3___redArg___boxed(lean_object* v_v_354_, lean_object* v___y_355_){
_start:
{
lean_object* v_res_356_; 
v_res_356_ = lp_mathlib___private_Init_While_0__repeatM_erased___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__3___redArg(v_v_354_, v___y_355_);
lean_dec(v_v_354_);
return v_res_356_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__7(lean_object* v_v_357_, uint8_t v___x_358_, lean_object* v_g_359_, lean_object* v_as_360_, size_t v_sz_361_, size_t v_i_362_, lean_object* v_b_363_, lean_object* v___y_364_){
_start:
{
lean_object* v_a_366_; lean_object* v_snd_367_; uint8_t v___x_371_; 
v___x_371_ = lean_usize_dec_lt(v_i_362_, v_sz_361_);
if (v___x_371_ == 0)
{
lean_object* v___x_372_; 
lean_dec(v_v_357_);
v___x_372_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_372_, 0, v_b_363_);
lean_ctor_set(v___x_372_, 1, v___y_364_);
return v___x_372_;
}
else
{
lean_object* v_a_373_; lean_object* v_dst_374_; lean_object* v_toDFSState_375_; lean_object* v_id_376_; lean_object* v_lowlink_377_; lean_object* v_stack_378_; lean_object* v_onStack_379_; lean_object* v_time_380_; lean_object* v___x_381_; lean_object* v___y_383_; lean_object* v___y_384_; lean_object* v___y_385_; lean_object* v___y_386_; lean_object* v___y_387_; lean_object* v___y_388_; lean_object* v___y_389_; uint8_t v___x_397_; 
v_a_373_ = lean_array_uget_borrowed(v_as_360_, v_i_362_);
v_dst_374_ = lean_ctor_get(v_a_373_, 1);
v_toDFSState_375_ = lean_ctor_get(v___y_364_, 0);
v_id_376_ = lean_ctor_get(v___y_364_, 1);
v_lowlink_377_ = lean_ctor_get(v___y_364_, 2);
v_stack_378_ = lean_ctor_get(v___y_364_, 3);
v_onStack_379_ = lean_ctor_get(v___y_364_, 4);
v_time_380_ = lean_ctor_get(v___y_364_, 5);
v___x_381_ = lean_box(0);
v___x_397_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__5___redArg(v_toDFSState_375_, v_dst_374_);
if (v___x_397_ == 0)
{
if (v___x_358_ == 0)
{
goto v___jp_392_;
}
else
{
lean_object* v___x_398_; lean_object* v_snd_399_; lean_object* v_toDFSState_400_; lean_object* v_id_401_; lean_object* v_lowlink_402_; lean_object* v_stack_403_; lean_object* v_onStack_404_; lean_object* v_time_405_; lean_object* v___x_407_; uint8_t v_isShared_408_; uint8_t v_isSharedCheck_418_; 
lean_inc(v_dst_374_);
v___x_398_ = lp_mathlib_Mathlib_Tactic_Order_Graph_tarjanDFS(v_g_359_, v_dst_374_, v___y_364_);
v_snd_399_ = lean_ctor_get(v___x_398_, 1);
lean_inc(v_snd_399_);
lean_dec_ref(v___x_398_);
v_toDFSState_400_ = lean_ctor_get(v_snd_399_, 0);
v_id_401_ = lean_ctor_get(v_snd_399_, 1);
v_lowlink_402_ = lean_ctor_get(v_snd_399_, 2);
v_stack_403_ = lean_ctor_get(v_snd_399_, 3);
v_onStack_404_ = lean_ctor_get(v_snd_399_, 4);
v_time_405_ = lean_ctor_get(v_snd_399_, 5);
v_isSharedCheck_418_ = !lean_is_exclusive(v_snd_399_);
if (v_isSharedCheck_418_ == 0)
{
v___x_407_ = v_snd_399_;
v_isShared_408_ = v_isSharedCheck_418_;
goto v_resetjp_406_;
}
else
{
lean_inc(v_time_405_);
lean_inc(v_onStack_404_);
lean_inc(v_stack_403_);
lean_inc(v_lowlink_402_);
lean_inc(v_id_401_);
lean_inc(v_toDFSState_400_);
lean_dec(v_snd_399_);
v___x_407_ = lean_box(0);
v_isShared_408_ = v_isSharedCheck_418_;
goto v_resetjp_406_;
}
v_resetjp_406_:
{
lean_object* v___y_410_; lean_object* v___x_415_; lean_object* v___x_416_; uint8_t v___x_417_; 
v___x_415_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__0(v_lowlink_402_, v_v_357_);
v___x_416_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__0(v_lowlink_402_, v_dst_374_);
v___x_417_ = lean_nat_dec_le(v___x_415_, v___x_416_);
if (v___x_417_ == 0)
{
lean_dec(v___x_415_);
v___y_410_ = v___x_416_;
goto v___jp_409_;
}
else
{
lean_dec(v___x_416_);
v___y_410_ = v___x_415_;
goto v___jp_409_;
}
v___jp_409_:
{
lean_object* v___x_411_; lean_object* v___x_413_; 
lean_inc(v_v_357_);
v___x_411_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__1___redArg(v_lowlink_402_, v_v_357_, v___y_410_);
if (v_isShared_408_ == 0)
{
lean_ctor_set(v___x_407_, 2, v___x_411_);
v___x_413_ = v___x_407_;
goto v_reusejp_412_;
}
else
{
lean_object* v_reuseFailAlloc_414_; 
v_reuseFailAlloc_414_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v_reuseFailAlloc_414_, 0, v_toDFSState_400_);
lean_ctor_set(v_reuseFailAlloc_414_, 1, v_id_401_);
lean_ctor_set(v_reuseFailAlloc_414_, 2, v___x_411_);
lean_ctor_set(v_reuseFailAlloc_414_, 3, v_stack_403_);
lean_ctor_set(v_reuseFailAlloc_414_, 4, v_onStack_404_);
lean_ctor_set(v_reuseFailAlloc_414_, 5, v_time_405_);
v___x_413_ = v_reuseFailAlloc_414_;
goto v_reusejp_412_;
}
v_reusejp_412_:
{
v_a_366_ = v___x_381_;
v_snd_367_ = v___x_413_;
goto v___jp_365_;
}
}
}
}
}
else
{
goto v___jp_392_;
}
v___jp_382_:
{
lean_object* v___x_390_; lean_object* v___x_391_; 
lean_inc(v_v_357_);
v___x_390_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__1___redArg(v___y_387_, v_v_357_, v___y_389_);
v___x_391_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_391_, 0, v___y_388_);
lean_ctor_set(v___x_391_, 1, v___y_385_);
lean_ctor_set(v___x_391_, 2, v___x_390_);
lean_ctor_set(v___x_391_, 3, v___y_384_);
lean_ctor_set(v___x_391_, 4, v___y_383_);
lean_ctor_set(v___x_391_, 5, v___y_386_);
v_a_366_ = v___x_381_;
v_snd_367_ = v___x_391_;
goto v___jp_365_;
}
v___jp_392_:
{
uint8_t v___x_393_; 
v___x_393_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__5___redArg(v_onStack_379_, v_dst_374_);
if (v___x_393_ == 0)
{
v_a_366_ = v___x_381_;
v_snd_367_ = v___y_364_;
goto v___jp_365_;
}
else
{
lean_object* v___x_394_; lean_object* v___x_395_; uint8_t v___x_396_; 
lean_inc(v_time_380_);
lean_inc_ref(v_onStack_379_);
lean_inc_ref(v_stack_378_);
lean_inc_ref(v_lowlink_377_);
lean_inc_ref(v_id_376_);
lean_inc_ref(v_toDFSState_375_);
lean_dec_ref(v___y_364_);
v___x_394_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__0(v_lowlink_377_, v_v_357_);
v___x_395_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__0(v_id_376_, v_dst_374_);
v___x_396_ = lean_nat_dec_le(v___x_394_, v___x_395_);
if (v___x_396_ == 0)
{
lean_dec(v___x_394_);
v___y_383_ = v_onStack_379_;
v___y_384_ = v_stack_378_;
v___y_385_ = v_id_376_;
v___y_386_ = v_time_380_;
v___y_387_ = v_lowlink_377_;
v___y_388_ = v_toDFSState_375_;
v___y_389_ = v___x_395_;
goto v___jp_382_;
}
else
{
lean_dec(v___x_395_);
v___y_383_ = v_onStack_379_;
v___y_384_ = v_stack_378_;
v___y_385_ = v_id_376_;
v___y_386_ = v_time_380_;
v___y_387_ = v_lowlink_377_;
v___y_388_ = v_toDFSState_375_;
v___y_389_ = v___x_394_;
goto v___jp_382_;
}
}
}
}
v___jp_365_:
{
size_t v___x_368_; size_t v___x_369_; 
v___x_368_ = ((size_t)1ULL);
v___x_369_ = lean_usize_add(v_i_362_, v___x_368_);
v_i_362_ = v___x_369_;
v_b_363_ = v_a_366_;
v___y_364_ = v_snd_367_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_Graph_tarjanDFS(lean_object* v_g_419_, lean_object* v_v_420_, lean_object* v_a_421_){
_start:
{
lean_object* v___y_423_; lean_object* v_id_424_; lean_object* v_lowlink_425_; lean_object* v_toDFSState_442_; lean_object* v_id_443_; lean_object* v_lowlink_444_; lean_object* v_stack_445_; lean_object* v_onStack_446_; lean_object* v_time_447_; lean_object* v___x_449_; uint8_t v_isShared_450_; uint8_t v_isSharedCheck_470_; 
v_toDFSState_442_ = lean_ctor_get(v_a_421_, 0);
v_id_443_ = lean_ctor_get(v_a_421_, 1);
v_lowlink_444_ = lean_ctor_get(v_a_421_, 2);
v_stack_445_ = lean_ctor_get(v_a_421_, 3);
v_onStack_446_ = lean_ctor_get(v_a_421_, 4);
v_time_447_ = lean_ctor_get(v_a_421_, 5);
v_isSharedCheck_470_ = !lean_is_exclusive(v_a_421_);
if (v_isSharedCheck_470_ == 0)
{
v___x_449_ = v_a_421_;
v_isShared_450_ = v_isSharedCheck_470_;
goto v_resetjp_448_;
}
else
{
lean_inc(v_time_447_);
lean_inc(v_onStack_446_);
lean_inc(v_stack_445_);
lean_inc(v_lowlink_444_);
lean_inc(v_id_443_);
lean_inc(v_toDFSState_442_);
lean_dec(v_a_421_);
v___x_449_ = lean_box(0);
v_isShared_450_ = v_isSharedCheck_470_;
goto v_resetjp_448_;
}
v___jp_422_:
{
lean_object* v___x_426_; lean_object* v___x_427_; uint8_t v___x_428_; 
v___x_426_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__0(v_id_424_, v_v_420_);
lean_dec_ref(v_id_424_);
v___x_427_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__0(v_lowlink_425_, v_v_420_);
lean_dec_ref(v_lowlink_425_);
v___x_428_ = lean_nat_dec_eq(v___x_426_, v___x_427_);
lean_dec(v___x_427_);
lean_dec(v___x_426_);
if (v___x_428_ == 0)
{
lean_object* v___x_429_; lean_object* v___x_430_; 
lean_dec(v_v_420_);
v___x_429_ = lean_box(0);
v___x_430_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_430_, 0, v___x_429_);
lean_ctor_set(v___x_430_, 1, v___y_423_);
return v___x_430_;
}
else
{
lean_object* v___x_431_; lean_object* v_snd_432_; lean_object* v___x_434_; uint8_t v_isShared_435_; uint8_t v_isSharedCheck_440_; 
v___x_431_ = lp_mathlib___private_Init_While_0__repeatM_erased___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__3___redArg(v_v_420_, v___y_423_);
lean_dec(v_v_420_);
v_snd_432_ = lean_ctor_get(v___x_431_, 1);
v_isSharedCheck_440_ = !lean_is_exclusive(v___x_431_);
if (v_isSharedCheck_440_ == 0)
{
lean_object* v_unused_441_; 
v_unused_441_ = lean_ctor_get(v___x_431_, 0);
lean_dec(v_unused_441_);
v___x_434_ = v___x_431_;
v_isShared_435_ = v_isSharedCheck_440_;
goto v_resetjp_433_;
}
else
{
lean_inc(v_snd_432_);
lean_dec(v___x_431_);
v___x_434_ = lean_box(0);
v_isShared_435_ = v_isSharedCheck_440_;
goto v_resetjp_433_;
}
v_resetjp_433_:
{
lean_object* v___x_436_; lean_object* v___x_438_; 
v___x_436_ = lean_box(0);
if (v_isShared_435_ == 0)
{
lean_ctor_set(v___x_434_, 0, v___x_436_);
v___x_438_ = v___x_434_;
goto v_reusejp_437_;
}
else
{
lean_object* v_reuseFailAlloc_439_; 
v_reuseFailAlloc_439_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_439_, 0, v___x_436_);
lean_ctor_set(v_reuseFailAlloc_439_, 1, v_snd_432_);
v___x_438_ = v_reuseFailAlloc_439_;
goto v_reusejp_437_;
}
v_reusejp_437_:
{
return v___x_438_;
}
}
}
}
v_resetjp_448_:
{
lean_object* v___x_451_; lean_object* v___x_452_; lean_object* v___x_453_; lean_object* v___x_454_; lean_object* v___x_455_; lean_object* v___x_456_; lean_object* v___x_457_; lean_object* v___x_458_; lean_object* v___x_460_; 
v___x_451_ = lean_box(0);
lean_inc_n(v_v_420_, 5);
v___x_452_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__4___redArg(v_toDFSState_442_, v_v_420_, v___x_451_);
lean_inc_n(v_time_447_, 2);
v___x_453_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__1___redArg(v_id_443_, v_v_420_, v_time_447_);
v___x_454_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__1___redArg(v_lowlink_444_, v_v_420_, v_time_447_);
v___x_455_ = lean_array_push(v_stack_445_, v_v_420_);
v___x_456_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__4___redArg(v_onStack_446_, v_v_420_, v___x_451_);
v___x_457_ = lean_unsigned_to_nat(1u);
v___x_458_ = lean_nat_add(v_time_447_, v___x_457_);
lean_dec(v_time_447_);
lean_inc_ref(v___x_454_);
lean_inc_ref(v___x_453_);
if (v_isShared_450_ == 0)
{
lean_ctor_set(v___x_449_, 5, v___x_458_);
lean_ctor_set(v___x_449_, 4, v___x_456_);
lean_ctor_set(v___x_449_, 3, v___x_455_);
lean_ctor_set(v___x_449_, 2, v___x_454_);
lean_ctor_set(v___x_449_, 1, v___x_453_);
lean_ctor_set(v___x_449_, 0, v___x_452_);
v___x_460_ = v___x_449_;
goto v_reusejp_459_;
}
else
{
lean_object* v_reuseFailAlloc_469_; 
v_reuseFailAlloc_469_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v_reuseFailAlloc_469_, 0, v___x_452_);
lean_ctor_set(v_reuseFailAlloc_469_, 1, v___x_453_);
lean_ctor_set(v_reuseFailAlloc_469_, 2, v___x_454_);
lean_ctor_set(v_reuseFailAlloc_469_, 3, v___x_455_);
lean_ctor_set(v_reuseFailAlloc_469_, 4, v___x_456_);
lean_ctor_set(v_reuseFailAlloc_469_, 5, v___x_458_);
v___x_460_ = v_reuseFailAlloc_469_;
goto v_reusejp_459_;
}
v_reusejp_459_:
{
uint8_t v___x_461_; 
v___x_461_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__5___redArg(v_g_419_, v_v_420_);
if (v___x_461_ == 0)
{
v___y_423_ = v___x_460_;
v_id_424_ = v___x_453_;
v_lowlink_425_ = v___x_454_;
goto v___jp_422_;
}
else
{
lean_object* v___x_462_; size_t v_sz_463_; size_t v___x_464_; lean_object* v___x_465_; lean_object* v_snd_466_; lean_object* v_id_467_; lean_object* v_lowlink_468_; 
lean_dec_ref(v___x_454_);
lean_dec_ref(v___x_453_);
v___x_462_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_Const_get_x21___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__6(v_g_419_, v_v_420_);
v_sz_463_ = lean_array_size(v___x_462_);
v___x_464_ = ((size_t)0ULL);
lean_inc(v_v_420_);
v___x_465_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__7(v_v_420_, v___x_461_, v_g_419_, v___x_462_, v_sz_463_, v___x_464_, v___x_451_, v___x_460_);
lean_dec_ref(v___x_462_);
v_snd_466_ = lean_ctor_get(v___x_465_, 1);
lean_inc(v_snd_466_);
lean_dec_ref(v___x_465_);
v_id_467_ = lean_ctor_get(v_snd_466_, 1);
lean_inc_ref(v_id_467_);
v_lowlink_468_ = lean_ctor_get(v_snd_466_, 2);
lean_inc_ref(v_lowlink_468_);
v___y_423_ = v_snd_466_;
v_id_424_ = v_id_467_;
v_lowlink_425_ = v_lowlink_468_;
goto v___jp_422_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_Graph_tarjanDFS___boxed(lean_object* v_g_471_, lean_object* v_v_472_, lean_object* v_a_473_){
_start:
{
lean_object* v_res_474_; 
v_res_474_ = lp_mathlib_Mathlib_Tactic_Order_Graph_tarjanDFS(v_g_471_, v_v_472_, v_a_473_);
lean_dec_ref(v_g_471_);
return v_res_474_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__7___boxed(lean_object* v_v_475_, lean_object* v___x_476_, lean_object* v_g_477_, lean_object* v_as_478_, lean_object* v_sz_479_, lean_object* v_i_480_, lean_object* v_b_481_, lean_object* v___y_482_){
_start:
{
uint8_t v___x_7936__boxed_483_; size_t v_sz_boxed_484_; size_t v_i_boxed_485_; lean_object* v_res_486_; 
v___x_7936__boxed_483_ = lean_unbox(v___x_476_);
v_sz_boxed_484_ = lean_unbox_usize(v_sz_479_);
lean_dec(v_sz_479_);
v_i_boxed_485_ = lean_unbox_usize(v_i_480_);
lean_dec(v_i_480_);
v_res_486_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__7(v_v_475_, v___x_7936__boxed_483_, v_g_477_, v_as_478_, v_sz_boxed_484_, v_i_boxed_485_, v_b_481_, v___y_482_);
lean_dec_ref(v_as_478_);
lean_dec_ref(v_g_477_);
return v_res_486_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__1(lean_object* v_00_u03b2_487_, lean_object* v_m_488_, lean_object* v_a_489_, lean_object* v_b_490_){
_start:
{
lean_object* v___x_491_; 
v___x_491_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__1___redArg(v_m_488_, v_a_489_, v_b_490_);
return v___x_491_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_erase___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__2(lean_object* v_00_u03b2_492_, lean_object* v_m_493_, lean_object* v_a_494_){
_start:
{
lean_object* v___x_495_; 
v___x_495_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_erase___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__2___redArg(v_m_493_, v_a_494_);
return v___x_495_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_erase___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__2___boxed(lean_object* v_00_u03b2_496_, lean_object* v_m_497_, lean_object* v_a_498_){
_start:
{
lean_object* v_res_499_; 
v_res_499_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_erase___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__2(v_00_u03b2_496_, v_m_497_, v_a_498_);
lean_dec(v_a_498_);
return v_res_499_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_While_0__repeatM_erased___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__3(lean_object* v_v_500_, lean_object* v_inst_501_, lean_object* v_a_502_, lean_object* v___y_503_){
_start:
{
lean_object* v___x_504_; 
v___x_504_ = lp_mathlib___private_Init_While_0__repeatM_erased___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__3___redArg(v_v_500_, v___y_503_);
return v___x_504_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_While_0__repeatM_erased___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__3___boxed(lean_object* v_v_505_, lean_object* v_inst_506_, lean_object* v_a_507_, lean_object* v___y_508_){
_start:
{
lean_object* v_res_509_; 
v_res_509_ = lp_mathlib___private_Init_While_0__repeatM_erased___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__3(v_v_505_, v_inst_506_, v_a_507_, v___y_508_);
lean_dec(v_a_507_);
lean_dec(v_v_505_);
return v_res_509_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__4(lean_object* v_00_u03b2_510_, lean_object* v_m_511_, lean_object* v_a_512_, lean_object* v_b_513_){
_start:
{
lean_object* v___x_514_; 
v___x_514_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__4___redArg(v_m_511_, v_a_512_, v_b_513_);
return v___x_514_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__5(lean_object* v_00_u03b2_515_, lean_object* v_m_516_, lean_object* v_a_517_){
_start:
{
uint8_t v___x_518_; 
v___x_518_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__5___redArg(v_m_516_, v_a_517_);
return v___x_518_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__5___boxed(lean_object* v_00_u03b2_519_, lean_object* v_m_520_, lean_object* v_a_521_){
_start:
{
uint8_t v_res_522_; lean_object* v_r_523_; 
v_res_522_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__5(v_00_u03b2_519_, v_m_520_, v_a_521_);
lean_dec(v_a_521_);
lean_dec_ref(v_m_520_);
v_r_523_ = lean_box(v_res_522_);
return v_r_523_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__1_spec__2(lean_object* v_00_u03b2_524_, lean_object* v_a_525_, lean_object* v_x_526_){
_start:
{
uint8_t v___x_527_; 
v___x_527_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__1_spec__2___redArg(v_a_525_, v_x_526_);
return v___x_527_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__1_spec__2___boxed(lean_object* v_00_u03b2_528_, lean_object* v_a_529_, lean_object* v_x_530_){
_start:
{
uint8_t v_res_531_; lean_object* v_r_532_; 
v_res_531_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__1_spec__2(v_00_u03b2_528_, v_a_529_, v_x_530_);
lean_dec(v_x_530_);
lean_dec(v_a_529_);
v_r_532_ = lean_box(v_res_531_);
return v_r_532_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__1_spec__3(lean_object* v_00_u03b2_533_, lean_object* v_data_534_){
_start:
{
lean_object* v___x_535_; 
v___x_535_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__1_spec__3___redArg(v_data_534_);
return v___x_535_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__1_spec__4(lean_object* v_00_u03b2_536_, lean_object* v_a_537_, lean_object* v_b_538_, lean_object* v_x_539_){
_start:
{
lean_object* v___x_540_; 
v___x_540_ = lp_mathlib_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__1_spec__4___redArg(v_a_537_, v_b_538_, v_x_539_);
return v___x_540_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__2_spec__6(lean_object* v_00_u03b2_541_, lean_object* v_a_542_, lean_object* v_x_543_){
_start:
{
lean_object* v___x_544_; 
v___x_544_ = lp_mathlib_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__2_spec__6___redArg(v_a_542_, v_x_543_);
return v___x_544_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__2_spec__6___boxed(lean_object* v_00_u03b2_545_, lean_object* v_a_546_, lean_object* v_x_547_){
_start:
{
lean_object* v_res_548_; 
v_res_548_ = lp_mathlib_Std_DHashMap_Internal_AssocList_erase___at___00Std_DHashMap_Internal_Raw_u2080_erase___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__2_spec__6(v_00_u03b2_545_, v_a_546_, v_x_547_);
lean_dec(v_a_546_);
return v_res_548_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__1_spec__3_spec__5(lean_object* v_00_u03b2_549_, lean_object* v_i_550_, lean_object* v_source_551_, lean_object* v_target_552_){
_start:
{
lean_object* v___x_553_; 
v___x_553_ = lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__1_spec__3_spec__5___redArg(v_i_550_, v_source_551_, v_target_552_);
return v___x_553_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__1_spec__3_spec__5_spec__12(lean_object* v_00_u03b2_554_, lean_object* v_x_555_, lean_object* v_x_556_){
_start:
{
lean_object* v___x_557_; 
v___x_557_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__1_spec__3_spec__5_spec__12___redArg(v_x_555_, v_x_556_);
return v___x_557_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_Graph_findSCCsImp_spec__0(lean_object* v_g_558_, lean_object* v_a_559_, lean_object* v_a_560_, lean_object* v___y_561_){
_start:
{
if (lean_obj_tag(v_a_559_) == 0)
{
lean_object* v___x_562_; lean_object* v___x_563_; 
v___x_562_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_562_, 0, v_a_560_);
v___x_563_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_563_, 0, v___x_562_);
lean_ctor_set(v___x_563_, 1, v___y_561_);
return v___x_563_;
}
else
{
lean_object* v_key_564_; lean_object* v_tail_565_; lean_object* v_toDFSState_566_; lean_object* v___x_567_; uint8_t v___x_568_; 
v_key_564_ = lean_ctor_get(v_a_559_, 0);
lean_inc(v_key_564_);
v_tail_565_ = lean_ctor_get(v_a_559_, 2);
lean_inc(v_tail_565_);
lean_dec_ref_known(v_a_559_, 3);
v_toDFSState_566_ = lean_ctor_get(v___y_561_, 0);
v___x_567_ = lean_box(0);
v___x_568_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Mathlib_Tactic_Order_Graph_tarjanDFS_spec__5___redArg(v_toDFSState_566_, v_key_564_);
if (v___x_568_ == 0)
{
lean_object* v___x_569_; lean_object* v_snd_570_; 
v___x_569_ = lp_mathlib_Mathlib_Tactic_Order_Graph_tarjanDFS(v_g_558_, v_key_564_, v___y_561_);
v_snd_570_ = lean_ctor_get(v___x_569_, 1);
lean_inc(v_snd_570_);
lean_dec_ref(v___x_569_);
v_a_559_ = v_tail_565_;
v_a_560_ = v___x_567_;
v___y_561_ = v_snd_570_;
goto _start;
}
else
{
lean_dec(v_key_564_);
v_a_559_ = v_tail_565_;
v_a_560_ = v___x_567_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_Graph_findSCCsImp_spec__0___boxed(lean_object* v_g_573_, lean_object* v_a_574_, lean_object* v_a_575_, lean_object* v___y_576_){
_start:
{
lean_object* v_res_577_; 
v_res_577_ = lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_Graph_findSCCsImp_spec__0(v_g_573_, v_a_574_, v_a_575_, v___y_576_);
lean_dec_ref(v_g_573_);
return v_res_577_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_Graph_findSCCsImp_spec__1(lean_object* v_g_578_, lean_object* v_as_579_, size_t v_sz_580_, size_t v_i_581_, lean_object* v_b_582_, lean_object* v___y_583_){
_start:
{
uint8_t v___x_584_; 
v___x_584_ = lean_usize_dec_lt(v_i_581_, v_sz_580_);
if (v___x_584_ == 0)
{
lean_object* v___x_585_; 
v___x_585_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_585_, 0, v_b_582_);
lean_ctor_set(v___x_585_, 1, v___y_583_);
return v___x_585_;
}
else
{
lean_object* v_a_586_; lean_object* v___x_587_; lean_object* v_fst_588_; 
v_a_586_ = lean_array_uget_borrowed(v_as_579_, v_i_581_);
lean_inc(v_a_586_);
v___x_587_ = lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Mathlib_Tactic_Order_Graph_findSCCsImp_spec__0(v_g_578_, v_a_586_, v_b_582_, v___y_583_);
v_fst_588_ = lean_ctor_get(v___x_587_, 0);
lean_inc(v_fst_588_);
if (lean_obj_tag(v_fst_588_) == 0)
{
lean_object* v_snd_589_; lean_object* v___x_591_; uint8_t v_isShared_592_; uint8_t v_isSharedCheck_597_; 
v_snd_589_ = lean_ctor_get(v___x_587_, 1);
v_isSharedCheck_597_ = !lean_is_exclusive(v___x_587_);
if (v_isSharedCheck_597_ == 0)
{
lean_object* v_unused_598_; 
v_unused_598_ = lean_ctor_get(v___x_587_, 0);
lean_dec(v_unused_598_);
v___x_591_ = v___x_587_;
v_isShared_592_ = v_isSharedCheck_597_;
goto v_resetjp_590_;
}
else
{
lean_inc(v_snd_589_);
lean_dec(v___x_587_);
v___x_591_ = lean_box(0);
v_isShared_592_ = v_isSharedCheck_597_;
goto v_resetjp_590_;
}
v_resetjp_590_:
{
lean_object* v_a_593_; lean_object* v___x_595_; 
v_a_593_ = lean_ctor_get(v_fst_588_, 0);
lean_inc(v_a_593_);
lean_dec_ref_known(v_fst_588_, 1);
if (v_isShared_592_ == 0)
{
lean_ctor_set(v___x_591_, 0, v_a_593_);
v___x_595_ = v___x_591_;
goto v_reusejp_594_;
}
else
{
lean_object* v_reuseFailAlloc_596_; 
v_reuseFailAlloc_596_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_596_, 0, v_a_593_);
lean_ctor_set(v_reuseFailAlloc_596_, 1, v_snd_589_);
v___x_595_ = v_reuseFailAlloc_596_;
goto v_reusejp_594_;
}
v_reusejp_594_:
{
return v___x_595_;
}
}
}
else
{
lean_object* v_snd_599_; lean_object* v_a_600_; size_t v___x_601_; size_t v___x_602_; 
v_snd_599_ = lean_ctor_get(v___x_587_, 1);
lean_inc(v_snd_599_);
lean_dec_ref(v___x_587_);
v_a_600_ = lean_ctor_get(v_fst_588_, 0);
lean_inc(v_a_600_);
lean_dec_ref_known(v_fst_588_, 1);
v___x_601_ = ((size_t)1ULL);
v___x_602_ = lean_usize_add(v_i_581_, v___x_601_);
v_i_581_ = v___x_602_;
v_b_582_ = v_a_600_;
v___y_583_ = v_snd_599_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_Graph_findSCCsImp_spec__1___boxed(lean_object* v_g_604_, lean_object* v_as_605_, lean_object* v_sz_606_, lean_object* v_i_607_, lean_object* v_b_608_, lean_object* v___y_609_){
_start:
{
size_t v_sz_boxed_610_; size_t v_i_boxed_611_; lean_object* v_res_612_; 
v_sz_boxed_610_ = lean_unbox_usize(v_sz_606_);
lean_dec(v_sz_606_);
v_i_boxed_611_ = lean_unbox_usize(v_i_607_);
lean_dec(v_i_607_);
v_res_612_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_Graph_findSCCsImp_spec__1(v_g_604_, v_as_605_, v_sz_boxed_610_, v_i_boxed_611_, v_b_608_, v___y_609_);
lean_dec_ref(v_as_605_);
lean_dec_ref(v_g_604_);
return v_res_612_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_Graph_findSCCsImp(lean_object* v_g_613_, lean_object* v_a_614_){
_start:
{
lean_object* v_buckets_615_; lean_object* v___x_616_; size_t v_sz_617_; size_t v___x_618_; lean_object* v___x_619_; lean_object* v_snd_620_; lean_object* v___x_622_; uint8_t v_isShared_623_; uint8_t v_isSharedCheck_627_; 
v_buckets_615_ = lean_ctor_get(v_g_613_, 1);
v___x_616_ = lean_box(0);
v_sz_617_ = lean_array_size(v_buckets_615_);
v___x_618_ = ((size_t)0ULL);
v___x_619_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Mathlib_Tactic_Order_Graph_findSCCsImp_spec__1(v_g_613_, v_buckets_615_, v_sz_617_, v___x_618_, v___x_616_, v_a_614_);
v_snd_620_ = lean_ctor_get(v___x_619_, 1);
v_isSharedCheck_627_ = !lean_is_exclusive(v___x_619_);
if (v_isSharedCheck_627_ == 0)
{
lean_object* v_unused_628_; 
v_unused_628_ = lean_ctor_get(v___x_619_, 0);
lean_dec(v_unused_628_);
v___x_622_ = v___x_619_;
v_isShared_623_ = v_isSharedCheck_627_;
goto v_resetjp_621_;
}
else
{
lean_inc(v_snd_620_);
lean_dec(v___x_619_);
v___x_622_ = lean_box(0);
v_isShared_623_ = v_isSharedCheck_627_;
goto v_resetjp_621_;
}
v_resetjp_621_:
{
lean_object* v___x_625_; 
if (v_isShared_623_ == 0)
{
lean_ctor_set(v___x_622_, 0, v___x_616_);
v___x_625_ = v___x_622_;
goto v_reusejp_624_;
}
else
{
lean_object* v_reuseFailAlloc_626_; 
v_reuseFailAlloc_626_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_626_, 0, v___x_616_);
lean_ctor_set(v_reuseFailAlloc_626_, 1, v_snd_620_);
v___x_625_ = v_reuseFailAlloc_626_;
goto v_reusejp_624_;
}
v_reusejp_624_:
{
return v___x_625_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_Graph_findSCCsImp___boxed(lean_object* v_g_629_, lean_object* v_a_630_){
_start:
{
lean_object* v_res_631_; 
v_res_631_ = lp_mathlib_Mathlib_Tactic_Order_Graph_findSCCsImp(v_g_629_, v_a_630_);
lean_dec_ref(v_g_629_);
return v_res_631_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Order_Graph_findSCCs___closed__0(void){
_start:
{
lean_object* v___x_632_; lean_object* v___x_633_; lean_object* v___x_634_; 
v___x_632_ = lean_box(0);
v___x_633_ = lean_unsigned_to_nat(16u);
v___x_634_ = lean_mk_array(v___x_633_, v___x_632_);
return v___x_634_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Order_Graph_findSCCs___closed__1(void){
_start:
{
lean_object* v___x_635_; lean_object* v___x_636_; lean_object* v___x_637_; 
v___x_635_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Order_Graph_findSCCs___closed__0, &lp_mathlib_Mathlib_Tactic_Order_Graph_findSCCs___closed__0_once, _init_lp_mathlib_Mathlib_Tactic_Order_Graph_findSCCs___closed__0);
v___x_636_ = lean_unsigned_to_nat(0u);
v___x_637_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_637_, 0, v___x_636_);
lean_ctor_set(v___x_637_, 1, v___x_635_);
return v___x_637_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_Order_Graph_findSCCs___closed__3(void){
_start:
{
lean_object* v___x_640_; lean_object* v___x_641_; lean_object* v___x_642_; lean_object* v_s_643_; 
v___x_640_ = lean_unsigned_to_nat(0u);
v___x_641_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_Order_Graph_findSCCs___closed__2));
v___x_642_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Order_Graph_findSCCs___closed__1, &lp_mathlib_Mathlib_Tactic_Order_Graph_findSCCs___closed__1_once, _init_lp_mathlib_Mathlib_Tactic_Order_Graph_findSCCs___closed__1);
v_s_643_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v_s_643_, 0, v___x_642_);
lean_ctor_set(v_s_643_, 1, v___x_642_);
lean_ctor_set(v_s_643_, 2, v___x_642_);
lean_ctor_set(v_s_643_, 3, v___x_641_);
lean_ctor_set(v_s_643_, 4, v___x_642_);
lean_ctor_set(v_s_643_, 5, v___x_640_);
return v_s_643_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_Graph_findSCCs(lean_object* v_g_644_){
_start:
{
lean_object* v_s_645_; lean_object* v___x_646_; lean_object* v_snd_647_; lean_object* v_lowlink_648_; 
v_s_645_ = lean_obj_once(&lp_mathlib_Mathlib_Tactic_Order_Graph_findSCCs___closed__3, &lp_mathlib_Mathlib_Tactic_Order_Graph_findSCCs___closed__3_once, _init_lp_mathlib_Mathlib_Tactic_Order_Graph_findSCCs___closed__3);
v___x_646_ = lp_mathlib_Mathlib_Tactic_Order_Graph_findSCCsImp(v_g_644_, v_s_645_);
v_snd_647_ = lean_ctor_get(v___x_646_, 1);
lean_inc(v_snd_647_);
lean_dec_ref(v___x_646_);
v_lowlink_648_ = lean_ctor_get(v_snd_647_, 2);
lean_inc_ref(v_lowlink_648_);
lean_dec(v_snd_647_);
return v_lowlink_648_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_Order_Graph_findSCCs___boxed(lean_object* v_g_649_){
_start:
{
lean_object* v_res_650_; 
v_res_650_ = lp_mathlib_Mathlib_Tactic_Order_Graph_findSCCs(v_g_649_);
lean_dec_ref(v_g_649_);
return v_res_650_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Order_Graph_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Order_Graph_Tarjan(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Order_Graph_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Order_Graph_Basic(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Order_Graph_Tarjan(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Order_Graph_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Order_Graph_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Order_Graph_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Order_Graph_Tarjan(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Order_Graph_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Order_Graph_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Order_Graph_Tarjan(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Order_Graph_Tarjan(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Order_Graph_Tarjan(builtin);
}
#ifdef __cplusplus
}
#endif
