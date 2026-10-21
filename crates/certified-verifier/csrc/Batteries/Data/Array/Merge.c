// Lean compiler output
// Module: Batteries.Data.Array.Merge
// Imports: public import Init public meta import Init
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
lean_object* l_Id_instMonad___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Array_toSubarray___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Subarray_copy___redArg(lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__6(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__2___boxed(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t);
lean_object* l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
lean_object* l_beqOfOrd___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_merge_go___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_merge_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_merge___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_merge(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_mergeDedupWith_go___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_mergeDedupWith_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_Array_Merge_0__Array_mergeDedupWith_go_match__1_splitter___redArg(uint8_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_Array_Merge_0__Array_mergeDedupWith_go_match__1_splitter___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_Array_Merge_0__Array_mergeDedupWith_go_match__1_splitter(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_Array_Merge_0__Array_mergeDedupWith_go_match__1_splitter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_mergeDedupWith___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_mergeDedupWith(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_mergeDedup___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_mergeDedup___redArg___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_batteries_Array_mergeDedup___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries_Array_mergeDedup___redArg___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Array_mergeDedup___redArg___closed__0 = (const lean_object*)&lp_batteries_Array_mergeDedup___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_batteries_Array_mergeDedup___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_mergeDedup(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Array_mergeUnsortedDedup_go___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_mergeUnsortedDedup_go___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_batteries_Array_mergeUnsortedDedup_go___redArg___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__0, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Array_mergeUnsortedDedup_go___redArg___lam__1___closed__0 = (const lean_object*)&lp_batteries_Array_mergeUnsortedDedup_go___redArg___lam__1___closed__0_value;
static const lean_closure_object lp_batteries_Array_mergeUnsortedDedup_go___redArg___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__1___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Array_mergeUnsortedDedup_go___redArg___lam__1___closed__1 = (const lean_object*)&lp_batteries_Array_mergeUnsortedDedup_go___redArg___lam__1___closed__1_value;
static const lean_closure_object lp_batteries_Array_mergeUnsortedDedup_go___redArg___lam__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__2___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Array_mergeUnsortedDedup_go___redArg___lam__1___closed__2 = (const lean_object*)&lp_batteries_Array_mergeUnsortedDedup_go___redArg___lam__1___closed__2_value;
static const lean_closure_object lp_batteries_Array_mergeUnsortedDedup_go___redArg___lam__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__3, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Array_mergeUnsortedDedup_go___redArg___lam__1___closed__3 = (const lean_object*)&lp_batteries_Array_mergeUnsortedDedup_go___redArg___lam__1___closed__3_value;
static const lean_closure_object lp_batteries_Array_mergeUnsortedDedup_go___redArg___lam__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__4___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Array_mergeUnsortedDedup_go___redArg___lam__1___closed__4 = (const lean_object*)&lp_batteries_Array_mergeUnsortedDedup_go___redArg___lam__1___closed__4_value;
static const lean_closure_object lp_batteries_Array_mergeUnsortedDedup_go___redArg___lam__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__5___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Array_mergeUnsortedDedup_go___redArg___lam__1___closed__5 = (const lean_object*)&lp_batteries_Array_mergeUnsortedDedup_go___redArg___lam__1___closed__5_value;
static const lean_closure_object lp_batteries_Array_mergeUnsortedDedup_go___redArg___lam__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__6, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Array_mergeUnsortedDedup_go___redArg___lam__1___closed__6 = (const lean_object*)&lp_batteries_Array_mergeUnsortedDedup_go___redArg___lam__1___closed__6_value;
static const lean_ctor_object lp_batteries_Array_mergeUnsortedDedup_go___redArg___lam__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries_Array_mergeUnsortedDedup_go___redArg___lam__1___closed__0_value),((lean_object*)&lp_batteries_Array_mergeUnsortedDedup_go___redArg___lam__1___closed__1_value)}};
static const lean_object* lp_batteries_Array_mergeUnsortedDedup_go___redArg___lam__1___closed__7 = (const lean_object*)&lp_batteries_Array_mergeUnsortedDedup_go___redArg___lam__1___closed__7_value;
static const lean_ctor_object lp_batteries_Array_mergeUnsortedDedup_go___redArg___lam__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*5 + 0, .m_other = 5, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries_Array_mergeUnsortedDedup_go___redArg___lam__1___closed__7_value),((lean_object*)&lp_batteries_Array_mergeUnsortedDedup_go___redArg___lam__1___closed__2_value),((lean_object*)&lp_batteries_Array_mergeUnsortedDedup_go___redArg___lam__1___closed__3_value),((lean_object*)&lp_batteries_Array_mergeUnsortedDedup_go___redArg___lam__1___closed__4_value),((lean_object*)&lp_batteries_Array_mergeUnsortedDedup_go___redArg___lam__1___closed__5_value)}};
static const lean_object* lp_batteries_Array_mergeUnsortedDedup_go___redArg___lam__1___closed__8 = (const lean_object*)&lp_batteries_Array_mergeUnsortedDedup_go___redArg___lam__1___closed__8_value;
static const lean_ctor_object lp_batteries_Array_mergeUnsortedDedup_go___redArg___lam__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries_Array_mergeUnsortedDedup_go___redArg___lam__1___closed__8_value),((lean_object*)&lp_batteries_Array_mergeUnsortedDedup_go___redArg___lam__1___closed__6_value)}};
static const lean_object* lp_batteries_Array_mergeUnsortedDedup_go___redArg___lam__1___closed__9 = (const lean_object*)&lp_batteries_Array_mergeUnsortedDedup_go___redArg___lam__1___closed__9_value;
LEAN_EXPORT lean_object* lp_batteries_Array_mergeUnsortedDedup_go___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_mergeUnsortedDedup_go___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_mergeUnsortedDedup_go___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_mergeUnsortedDedup_go(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_mergeUnsortedDedup___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_mergeUnsortedDedup___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_mergeUnsortedDedup___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_mergeUnsortedDedup(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_mergeAdjacentDups_go___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_mergeAdjacentDups_go___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_mergeAdjacentDups_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_mergeAdjacentDups_go___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_mergeAdjacentDups___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_mergeAdjacentDups___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_mergeAdjacentDups(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_mergeAdjacentDups___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_dedupSorted___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_dedupSorted___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_dedupSorted(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_dedupSorted___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Array_sortDedup___redArg___lam__0(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_sortDedup___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_sortDedup___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_sortDedup(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_merge_go___redArg(lean_object* v_lt_1_, lean_object* v_xs_2_, lean_object* v_ys_3_, lean_object* v_acc_4_, lean_object* v_i_5_, lean_object* v_j_6_){
_start:
{
lean_object* v___x_7_; uint8_t v___x_8_; 
v___x_7_ = lean_array_get_size(v_xs_2_);
v___x_8_ = lean_nat_dec_le(v___x_7_, v_i_5_);
if (v___x_8_ == 0)
{
lean_object* v___x_9_; uint8_t v___x_10_; 
v___x_9_ = lean_array_get_size(v_ys_3_);
v___x_10_ = lean_nat_dec_le(v___x_9_, v_j_6_);
if (v___x_10_ == 0)
{
lean_object* v_x_11_; lean_object* v_y_12_; lean_object* v___x_13_; uint8_t v___x_14_; 
v_x_11_ = lean_array_fget_borrowed(v_xs_2_, v_i_5_);
v_y_12_ = lean_array_fget_borrowed(v_ys_3_, v_j_6_);
lean_inc_ref(v_lt_1_);
lean_inc(v_y_12_);
lean_inc(v_x_11_);
v___x_13_ = lean_apply_2(v_lt_1_, v_x_11_, v_y_12_);
v___x_14_ = lean_unbox(v___x_13_);
if (v___x_14_ == 0)
{
lean_object* v___x_15_; lean_object* v___x_16_; lean_object* v___x_17_; 
lean_inc(v_y_12_);
v___x_15_ = lean_array_push(v_acc_4_, v_y_12_);
v___x_16_ = lean_unsigned_to_nat(1u);
v___x_17_ = lean_nat_add(v_j_6_, v___x_16_);
lean_dec(v_j_6_);
v_acc_4_ = v___x_15_;
v_j_6_ = v___x_17_;
goto _start;
}
else
{
lean_object* v___x_19_; lean_object* v___x_20_; lean_object* v___x_21_; 
lean_inc(v_x_11_);
v___x_19_ = lean_array_push(v_acc_4_, v_x_11_);
v___x_20_ = lean_unsigned_to_nat(1u);
v___x_21_ = lean_nat_add(v_i_5_, v___x_20_);
lean_dec(v_i_5_);
v_acc_4_ = v___x_19_;
v_i_5_ = v___x_21_;
goto _start;
}
}
else
{
lean_object* v___x_23_; lean_object* v___x_24_; lean_object* v___x_25_; 
lean_dec(v_j_6_);
lean_dec_ref(v_ys_3_);
lean_dec_ref(v_lt_1_);
v___x_23_ = l_Array_toSubarray___redArg(v_xs_2_, v_i_5_, v___x_7_);
v___x_24_ = l_Subarray_copy___redArg(v___x_23_);
v___x_25_ = l_Array_append___redArg(v_acc_4_, v___x_24_);
lean_dec_ref(v___x_24_);
return v___x_25_;
}
}
else
{
lean_object* v___x_26_; lean_object* v___x_27_; lean_object* v___x_28_; lean_object* v___x_29_; 
lean_dec(v_i_5_);
lean_dec_ref(v_xs_2_);
lean_dec_ref(v_lt_1_);
v___x_26_ = lean_array_get_size(v_ys_3_);
v___x_27_ = l_Array_toSubarray___redArg(v_ys_3_, v_j_6_, v___x_26_);
v___x_28_ = l_Subarray_copy___redArg(v___x_27_);
v___x_29_ = l_Array_append___redArg(v_acc_4_, v___x_28_);
lean_dec_ref(v___x_28_);
return v___x_29_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_merge_go(lean_object* v_00_u03b1_30_, lean_object* v_lt_31_, lean_object* v_xs_32_, lean_object* v_ys_33_, lean_object* v_acc_34_, lean_object* v_i_35_, lean_object* v_j_36_){
_start:
{
lean_object* v___x_37_; 
v___x_37_ = lp_batteries_Array_merge_go___redArg(v_lt_31_, v_xs_32_, v_ys_33_, v_acc_34_, v_i_35_, v_j_36_);
return v___x_37_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_merge___redArg(lean_object* v_lt_38_, lean_object* v_xs_39_, lean_object* v_ys_40_){
_start:
{
lean_object* v___x_41_; lean_object* v___x_42_; lean_object* v___x_43_; lean_object* v___x_44_; lean_object* v___x_45_; lean_object* v___x_46_; 
v___x_41_ = lean_array_get_size(v_xs_39_);
v___x_42_ = lean_array_get_size(v_ys_40_);
v___x_43_ = lean_nat_add(v___x_41_, v___x_42_);
v___x_44_ = lean_mk_empty_array_with_capacity(v___x_43_);
lean_dec(v___x_43_);
v___x_45_ = lean_unsigned_to_nat(0u);
v___x_46_ = lp_batteries_Array_merge_go___redArg(v_lt_38_, v_xs_39_, v_ys_40_, v___x_44_, v___x_45_, v___x_45_);
return v___x_46_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_merge(lean_object* v_00_u03b1_47_, lean_object* v_lt_48_, lean_object* v_xs_49_, lean_object* v_ys_50_){
_start:
{
lean_object* v___x_51_; 
v___x_51_ = lp_batteries_Array_merge___redArg(v_lt_48_, v_xs_49_, v_ys_50_);
return v___x_51_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_mergeDedupWith_go___redArg(lean_object* v_ord_52_, lean_object* v_xs_53_, lean_object* v_ys_54_, lean_object* v_merge_55_, lean_object* v_acc_56_, lean_object* v_i_57_, lean_object* v_j_58_){
_start:
{
lean_object* v___x_59_; uint8_t v___x_60_; 
v___x_59_ = lean_array_get_size(v_xs_53_);
v___x_60_ = lean_nat_dec_le(v___x_59_, v_i_57_);
if (v___x_60_ == 0)
{
lean_object* v___x_61_; uint8_t v___x_62_; 
v___x_61_ = lean_array_get_size(v_ys_54_);
v___x_62_ = lean_nat_dec_le(v___x_61_, v_j_58_);
if (v___x_62_ == 0)
{
lean_object* v_x_63_; lean_object* v_y_64_; lean_object* v___x_65_; uint8_t v___x_66_; 
v_x_63_ = lean_array_fget_borrowed(v_xs_53_, v_i_57_);
v_y_64_ = lean_array_fget_borrowed(v_ys_54_, v_j_58_);
lean_inc_ref(v_ord_52_);
lean_inc(v_y_64_);
lean_inc(v_x_63_);
v___x_65_ = lean_apply_2(v_ord_52_, v_x_63_, v_y_64_);
v___x_66_ = lean_unbox(v___x_65_);
switch(v___x_66_)
{
case 0:
{
lean_object* v___x_67_; lean_object* v___x_68_; lean_object* v___x_69_; 
lean_inc(v_x_63_);
v___x_67_ = lean_array_push(v_acc_56_, v_x_63_);
v___x_68_ = lean_unsigned_to_nat(1u);
v___x_69_ = lean_nat_add(v_i_57_, v___x_68_);
lean_dec(v_i_57_);
v_acc_56_ = v___x_67_;
v_i_57_ = v___x_69_;
goto _start;
}
case 1:
{
lean_object* v___x_71_; lean_object* v___x_72_; lean_object* v___x_73_; lean_object* v___x_74_; lean_object* v___x_75_; 
lean_inc(v_merge_55_);
lean_inc(v_y_64_);
lean_inc(v_x_63_);
v___x_71_ = lean_apply_2(v_merge_55_, v_x_63_, v_y_64_);
v___x_72_ = lean_array_push(v_acc_56_, v___x_71_);
v___x_73_ = lean_unsigned_to_nat(1u);
v___x_74_ = lean_nat_add(v_i_57_, v___x_73_);
lean_dec(v_i_57_);
v___x_75_ = lean_nat_add(v_j_58_, v___x_73_);
lean_dec(v_j_58_);
v_acc_56_ = v___x_72_;
v_i_57_ = v___x_74_;
v_j_58_ = v___x_75_;
goto _start;
}
default: 
{
lean_object* v___x_77_; lean_object* v___x_78_; lean_object* v___x_79_; 
lean_inc(v_y_64_);
v___x_77_ = lean_array_push(v_acc_56_, v_y_64_);
v___x_78_ = lean_unsigned_to_nat(1u);
v___x_79_ = lean_nat_add(v_j_58_, v___x_78_);
lean_dec(v_j_58_);
v_acc_56_ = v___x_77_;
v_j_58_ = v___x_79_;
goto _start;
}
}
}
else
{
lean_object* v___x_81_; lean_object* v___x_82_; lean_object* v___x_83_; 
lean_dec(v_j_58_);
lean_dec(v_merge_55_);
lean_dec_ref(v_ys_54_);
lean_dec_ref(v_ord_52_);
v___x_81_ = l_Array_toSubarray___redArg(v_xs_53_, v_i_57_, v___x_59_);
v___x_82_ = l_Subarray_copy___redArg(v___x_81_);
v___x_83_ = l_Array_append___redArg(v_acc_56_, v___x_82_);
lean_dec_ref(v___x_82_);
return v___x_83_;
}
}
else
{
lean_object* v___x_84_; lean_object* v___x_85_; lean_object* v___x_86_; lean_object* v___x_87_; 
lean_dec(v_i_57_);
lean_dec(v_merge_55_);
lean_dec_ref(v_xs_53_);
lean_dec_ref(v_ord_52_);
v___x_84_ = lean_array_get_size(v_ys_54_);
v___x_85_ = l_Array_toSubarray___redArg(v_ys_54_, v_j_58_, v___x_84_);
v___x_86_ = l_Subarray_copy___redArg(v___x_85_);
v___x_87_ = l_Array_append___redArg(v_acc_56_, v___x_86_);
lean_dec_ref(v___x_86_);
return v___x_87_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_mergeDedupWith_go(lean_object* v_00_u03b1_88_, lean_object* v_ord_89_, lean_object* v_xs_90_, lean_object* v_ys_91_, lean_object* v_merge_92_, lean_object* v_acc_93_, lean_object* v_i_94_, lean_object* v_j_95_){
_start:
{
lean_object* v___x_96_; 
v___x_96_ = lp_batteries_Array_mergeDedupWith_go___redArg(v_ord_89_, v_xs_90_, v_ys_91_, v_merge_92_, v_acc_93_, v_i_94_, v_j_95_);
return v___x_96_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_Array_Merge_0__Array_mergeDedupWith_go_match__1_splitter___redArg(uint8_t v_x_97_, lean_object* v_h__1_98_, lean_object* v_h__2_99_, lean_object* v_h__3_100_){
_start:
{
switch(v_x_97_)
{
case 0:
{
lean_object* v___x_101_; lean_object* v___x_102_; 
lean_dec(v_h__3_100_);
lean_dec(v_h__2_99_);
v___x_101_ = lean_box(0);
v___x_102_ = lean_apply_1(v_h__1_98_, v___x_101_);
return v___x_102_;
}
case 1:
{
lean_object* v___x_103_; lean_object* v___x_104_; 
lean_dec(v_h__2_99_);
lean_dec(v_h__1_98_);
v___x_103_ = lean_box(0);
v___x_104_ = lean_apply_1(v_h__3_100_, v___x_103_);
return v___x_104_;
}
default: 
{
lean_object* v___x_105_; lean_object* v___x_106_; 
lean_dec(v_h__3_100_);
lean_dec(v_h__1_98_);
v___x_105_ = lean_box(0);
v___x_106_ = lean_apply_1(v_h__2_99_, v___x_105_);
return v___x_106_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_Array_Merge_0__Array_mergeDedupWith_go_match__1_splitter___redArg___boxed(lean_object* v_x_107_, lean_object* v_h__1_108_, lean_object* v_h__2_109_, lean_object* v_h__3_110_){
_start:
{
uint8_t v_x_33__boxed_111_; lean_object* v_res_112_; 
v_x_33__boxed_111_ = lean_unbox(v_x_107_);
v_res_112_ = lp_batteries___private_Batteries_Data_Array_Merge_0__Array_mergeDedupWith_go_match__1_splitter___redArg(v_x_33__boxed_111_, v_h__1_108_, v_h__2_109_, v_h__3_110_);
return v_res_112_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_Array_Merge_0__Array_mergeDedupWith_go_match__1_splitter(lean_object* v_motive_113_, uint8_t v_x_114_, lean_object* v_h__1_115_, lean_object* v_h__2_116_, lean_object* v_h__3_117_){
_start:
{
switch(v_x_114_)
{
case 0:
{
lean_object* v___x_118_; lean_object* v___x_119_; 
lean_dec(v_h__3_117_);
lean_dec(v_h__2_116_);
v___x_118_ = lean_box(0);
v___x_119_ = lean_apply_1(v_h__1_115_, v___x_118_);
return v___x_119_;
}
case 1:
{
lean_object* v___x_120_; lean_object* v___x_121_; 
lean_dec(v_h__2_116_);
lean_dec(v_h__1_115_);
v___x_120_ = lean_box(0);
v___x_121_ = lean_apply_1(v_h__3_117_, v___x_120_);
return v___x_121_;
}
default: 
{
lean_object* v___x_122_; lean_object* v___x_123_; 
lean_dec(v_h__3_117_);
lean_dec(v_h__1_115_);
v___x_122_ = lean_box(0);
v___x_123_ = lean_apply_1(v_h__2_116_, v___x_122_);
return v___x_123_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_Array_Merge_0__Array_mergeDedupWith_go_match__1_splitter___boxed(lean_object* v_motive_124_, lean_object* v_x_125_, lean_object* v_h__1_126_, lean_object* v_h__2_127_, lean_object* v_h__3_128_){
_start:
{
uint8_t v_x_48__boxed_129_; lean_object* v_res_130_; 
v_x_48__boxed_129_ = lean_unbox(v_x_125_);
v_res_130_ = lp_batteries___private_Batteries_Data_Array_Merge_0__Array_mergeDedupWith_go_match__1_splitter(v_motive_124_, v_x_48__boxed_129_, v_h__1_126_, v_h__2_127_, v_h__3_128_);
return v_res_130_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_mergeDedupWith___redArg(lean_object* v_ord_131_, lean_object* v_xs_132_, lean_object* v_ys_133_, lean_object* v_merge_134_){
_start:
{
lean_object* v___x_135_; lean_object* v___x_136_; lean_object* v___x_137_; lean_object* v___x_138_; lean_object* v___x_139_; lean_object* v___x_140_; 
v___x_135_ = lean_array_get_size(v_xs_132_);
v___x_136_ = lean_array_get_size(v_ys_133_);
v___x_137_ = lean_nat_add(v___x_135_, v___x_136_);
v___x_138_ = lean_mk_empty_array_with_capacity(v___x_137_);
lean_dec(v___x_137_);
v___x_139_ = lean_unsigned_to_nat(0u);
v___x_140_ = lp_batteries_Array_mergeDedupWith_go___redArg(v_ord_131_, v_xs_132_, v_ys_133_, v_merge_134_, v___x_138_, v___x_139_, v___x_139_);
return v___x_140_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_mergeDedupWith(lean_object* v_00_u03b1_141_, lean_object* v_ord_142_, lean_object* v_xs_143_, lean_object* v_ys_144_, lean_object* v_merge_145_){
_start:
{
lean_object* v___x_146_; 
v___x_146_ = lp_batteries_Array_mergeDedupWith___redArg(v_ord_142_, v_xs_143_, v_ys_144_, v_merge_145_);
return v___x_146_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_mergeDedup___redArg___lam__0(lean_object* v_x_147_, lean_object* v_x_148_){
_start:
{
lean_inc(v_x_147_);
return v_x_147_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_mergeDedup___redArg___lam__0___boxed(lean_object* v_x_149_, lean_object* v_x_150_){
_start:
{
lean_object* v_res_151_; 
v_res_151_ = lp_batteries_Array_mergeDedup___redArg___lam__0(v_x_149_, v_x_150_);
lean_dec(v_x_150_);
lean_dec(v_x_149_);
return v_res_151_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_mergeDedup___redArg(lean_object* v_ord_153_, lean_object* v_xs_154_, lean_object* v_ys_155_){
_start:
{
lean_object* v___f_156_; lean_object* v___x_157_; 
v___f_156_ = ((lean_object*)(lp_batteries_Array_mergeDedup___redArg___closed__0));
v___x_157_ = lp_batteries_Array_mergeDedupWith___redArg(v_ord_153_, v_xs_154_, v_ys_155_, v___f_156_);
return v___x_157_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_mergeDedup(lean_object* v_00_u03b1_158_, lean_object* v_ord_159_, lean_object* v_xs_160_, lean_object* v_ys_161_){
_start:
{
lean_object* v___f_162_; lean_object* v___x_163_; 
v___f_162_ = ((lean_object*)(lp_batteries_Array_mergeDedup___redArg___closed__0));
v___x_163_ = lp_batteries_Array_mergeDedupWith___redArg(v_ord_159_, v_xs_160_, v_ys_161_, v___f_162_);
return v___x_163_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Array_mergeUnsortedDedup_go___redArg___lam__0(lean_object* v_eq_164_, lean_object* v_x2_165_, lean_object* v_x_166_){
_start:
{
lean_object* v___x_167_; uint8_t v___x_168_; 
v___x_167_ = lean_apply_2(v_eq_164_, v_x_166_, v_x2_165_);
v___x_168_ = lean_unbox(v___x_167_);
return v___x_168_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_mergeUnsortedDedup_go___redArg___lam__0___boxed(lean_object* v_eq_169_, lean_object* v_x2_170_, lean_object* v_x_171_){
_start:
{
uint8_t v_res_172_; lean_object* v_r_173_; 
v_res_172_ = lp_batteries_Array_mergeUnsortedDedup_go___redArg___lam__0(v_eq_169_, v_x2_170_, v_x_171_);
v_r_173_ = lean_box(v_res_172_);
return v_r_173_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_mergeUnsortedDedup_go___redArg___lam__1(lean_object* v___x_193_, lean_object* v_xsSize_194_, lean_object* v_eq_195_, lean_object* v_x1_196_, lean_object* v_x2_197_){
_start:
{
lean_object* v___x_198_; uint8_t v___x_199_; 
v___x_198_ = ((lean_object*)(lp_batteries_Array_mergeUnsortedDedup_go___redArg___lam__1___closed__9));
v___x_199_ = lean_nat_dec_lt(v___x_193_, v_xsSize_194_);
if (v___x_199_ == 0)
{
lean_object* v___x_200_; 
lean_dec_ref(v_eq_195_);
lean_dec(v_xsSize_194_);
v___x_200_ = lean_array_push(v_x1_196_, v_x2_197_);
return v___x_200_;
}
else
{
lean_object* v___f_201_; lean_object* v___y_203_; lean_object* v___x_211_; uint8_t v___x_212_; 
lean_inc(v_x2_197_);
v___f_201_ = lean_alloc_closure((void*)(lp_batteries_Array_mergeUnsortedDedup_go___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_201_, 0, v_eq_195_);
lean_closure_set(v___f_201_, 1, v_x2_197_);
v___x_211_ = lean_array_get_size(v_x1_196_);
v___x_212_ = lean_nat_dec_le(v_xsSize_194_, v___x_211_);
if (v___x_212_ == 0)
{
lean_dec(v_xsSize_194_);
v___y_203_ = v___x_211_;
goto v___jp_202_;
}
else
{
v___y_203_ = v_xsSize_194_;
goto v___jp_202_;
}
v___jp_202_:
{
uint8_t v___x_204_; 
v___x_204_ = lean_nat_dec_lt(v___x_193_, v___y_203_);
if (v___x_204_ == 0)
{
lean_object* v___x_205_; 
lean_dec(v___y_203_);
lean_dec_ref(v___f_201_);
v___x_205_ = lean_array_push(v_x1_196_, v_x2_197_);
return v___x_205_;
}
else
{
size_t v___x_206_; size_t v___x_207_; lean_object* v___x_208_; uint8_t v___x_209_; 
v___x_206_ = ((size_t)0ULL);
v___x_207_ = lean_usize_of_nat(v___y_203_);
lean_dec(v___y_203_);
lean_inc_ref(v_x1_196_);
v___x_208_ = l___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any(lean_box(0), lean_box(0), v___x_198_, v___f_201_, v_x1_196_, v___x_206_, v___x_207_);
v___x_209_ = lean_unbox(v___x_208_);
lean_dec(v___x_208_);
if (v___x_209_ == 0)
{
lean_object* v___x_210_; 
v___x_210_ = lean_array_push(v_x1_196_, v_x2_197_);
return v___x_210_;
}
else
{
lean_dec(v_x2_197_);
return v_x1_196_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_mergeUnsortedDedup_go___redArg___lam__1___boxed(lean_object* v___x_213_, lean_object* v_xsSize_214_, lean_object* v_eq_215_, lean_object* v_x1_216_, lean_object* v_x2_217_){
_start:
{
lean_object* v_res_218_; 
v_res_218_ = lp_batteries_Array_mergeUnsortedDedup_go___redArg___lam__1(v___x_213_, v_xsSize_214_, v_eq_215_, v_x1_216_, v_x2_217_);
lean_dec(v___x_213_);
return v_res_218_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_mergeUnsortedDedup_go___redArg(lean_object* v_eq_219_, lean_object* v_xs_220_, lean_object* v_ys_221_){
_start:
{
lean_object* v___x_222_; lean_object* v___x_223_; lean_object* v___x_224_; uint8_t v___x_225_; 
v___x_222_ = lean_unsigned_to_nat(0u);
v___x_223_ = lean_array_get_size(v_ys_221_);
v___x_224_ = ((lean_object*)(lp_batteries_Array_mergeUnsortedDedup_go___redArg___lam__1___closed__9));
v___x_225_ = lean_nat_dec_lt(v___x_222_, v___x_223_);
if (v___x_225_ == 0)
{
lean_dec_ref(v_ys_221_);
lean_dec_ref(v_eq_219_);
return v_xs_220_;
}
else
{
lean_object* v_xsSize_226_; lean_object* v___f_227_; uint8_t v___x_228_; 
v_xsSize_226_ = lean_array_get_size(v_xs_220_);
v___f_227_ = lean_alloc_closure((void*)(lp_batteries_Array_mergeUnsortedDedup_go___redArg___lam__1___boxed), 5, 3);
lean_closure_set(v___f_227_, 0, v___x_222_);
lean_closure_set(v___f_227_, 1, v_xsSize_226_);
lean_closure_set(v___f_227_, 2, v_eq_219_);
v___x_228_ = lean_nat_dec_le(v___x_223_, v___x_223_);
if (v___x_228_ == 0)
{
if (v___x_225_ == 0)
{
lean_dec_ref(v___f_227_);
lean_dec_ref(v_ys_221_);
return v_xs_220_;
}
else
{
size_t v___x_229_; size_t v___x_230_; lean_object* v___x_231_; 
v___x_229_ = ((size_t)0ULL);
v___x_230_ = lean_usize_of_nat(v___x_223_);
v___x_231_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_224_, v___f_227_, v_ys_221_, v___x_229_, v___x_230_, v_xs_220_);
return v___x_231_;
}
}
else
{
size_t v___x_232_; size_t v___x_233_; lean_object* v___x_234_; 
v___x_232_ = ((size_t)0ULL);
v___x_233_ = lean_usize_of_nat(v___x_223_);
v___x_234_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_224_, v___f_227_, v_ys_221_, v___x_232_, v___x_233_, v_xs_220_);
return v___x_234_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_mergeUnsortedDedup_go(lean_object* v_00_u03b1_235_, lean_object* v_eq_236_, lean_object* v_xs_237_, lean_object* v_ys_238_){
_start:
{
lean_object* v___x_239_; lean_object* v___x_240_; lean_object* v___x_241_; uint8_t v___x_242_; 
v___x_239_ = lean_unsigned_to_nat(0u);
v___x_240_ = lean_array_get_size(v_ys_238_);
v___x_241_ = ((lean_object*)(lp_batteries_Array_mergeUnsortedDedup_go___redArg___lam__1___closed__9));
v___x_242_ = lean_nat_dec_lt(v___x_239_, v___x_240_);
if (v___x_242_ == 0)
{
lean_dec_ref(v_ys_238_);
lean_dec_ref(v_eq_236_);
return v_xs_237_;
}
else
{
lean_object* v_xsSize_243_; lean_object* v___f_244_; uint8_t v___x_245_; 
v_xsSize_243_ = lean_array_get_size(v_xs_237_);
v___f_244_ = lean_alloc_closure((void*)(lp_batteries_Array_mergeUnsortedDedup_go___redArg___lam__1___boxed), 5, 3);
lean_closure_set(v___f_244_, 0, v___x_239_);
lean_closure_set(v___f_244_, 1, v_xsSize_243_);
lean_closure_set(v___f_244_, 2, v_eq_236_);
v___x_245_ = lean_nat_dec_le(v___x_240_, v___x_240_);
if (v___x_245_ == 0)
{
if (v___x_242_ == 0)
{
lean_dec_ref(v___f_244_);
lean_dec_ref(v_ys_238_);
return v_xs_237_;
}
else
{
size_t v___x_246_; size_t v___x_247_; lean_object* v___x_248_; 
v___x_246_ = ((size_t)0ULL);
v___x_247_ = lean_usize_of_nat(v___x_240_);
v___x_248_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_241_, v___f_244_, v_ys_238_, v___x_246_, v___x_247_, v_xs_237_);
return v___x_248_;
}
}
else
{
size_t v___x_249_; size_t v___x_250_; lean_object* v___x_251_; 
v___x_249_ = ((size_t)0ULL);
v___x_250_ = lean_usize_of_nat(v___x_240_);
v___x_251_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_241_, v___f_244_, v_ys_238_, v___x_249_, v___x_250_, v_xs_237_);
return v___x_251_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_mergeUnsortedDedup___redArg___lam__1(lean_object* v___x_252_, lean_object* v___x_253_, lean_object* v_eq_254_, lean_object* v_x1_255_, lean_object* v_x2_256_){
_start:
{
lean_object* v___x_257_; uint8_t v___x_258_; 
v___x_257_ = ((lean_object*)(lp_batteries_Array_mergeUnsortedDedup_go___redArg___lam__1___closed__9));
v___x_258_ = lean_nat_dec_lt(v___x_252_, v___x_253_);
if (v___x_258_ == 0)
{
lean_object* v___x_259_; 
lean_dec_ref(v_eq_254_);
lean_dec(v___x_253_);
v___x_259_ = lean_array_push(v_x1_255_, v_x2_256_);
return v___x_259_;
}
else
{
lean_object* v___f_260_; lean_object* v___y_262_; lean_object* v___x_270_; uint8_t v___x_271_; 
lean_inc(v_x2_256_);
v___f_260_ = lean_alloc_closure((void*)(lp_batteries_Array_mergeUnsortedDedup_go___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_260_, 0, v_eq_254_);
lean_closure_set(v___f_260_, 1, v_x2_256_);
v___x_270_ = lean_array_get_size(v_x1_255_);
v___x_271_ = lean_nat_dec_le(v___x_253_, v___x_270_);
if (v___x_271_ == 0)
{
lean_dec(v___x_253_);
v___y_262_ = v___x_270_;
goto v___jp_261_;
}
else
{
v___y_262_ = v___x_253_;
goto v___jp_261_;
}
v___jp_261_:
{
uint8_t v___x_263_; 
v___x_263_ = lean_nat_dec_lt(v___x_252_, v___y_262_);
if (v___x_263_ == 0)
{
lean_object* v___x_264_; 
lean_dec(v___y_262_);
lean_dec_ref(v___f_260_);
v___x_264_ = lean_array_push(v_x1_255_, v_x2_256_);
return v___x_264_;
}
else
{
size_t v___x_265_; size_t v___x_266_; lean_object* v___x_267_; uint8_t v___x_268_; 
v___x_265_ = ((size_t)0ULL);
v___x_266_ = lean_usize_of_nat(v___y_262_);
lean_dec(v___y_262_);
lean_inc_ref(v_x1_255_);
v___x_267_ = l___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any(lean_box(0), lean_box(0), v___x_257_, v___f_260_, v_x1_255_, v___x_265_, v___x_266_);
v___x_268_ = lean_unbox(v___x_267_);
lean_dec(v___x_267_);
if (v___x_268_ == 0)
{
lean_object* v___x_269_; 
v___x_269_ = lean_array_push(v_x1_255_, v_x2_256_);
return v___x_269_;
}
else
{
lean_dec(v_x2_256_);
return v_x1_255_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_mergeUnsortedDedup___redArg___lam__1___boxed(lean_object* v___x_272_, lean_object* v___x_273_, lean_object* v_eq_274_, lean_object* v_x1_275_, lean_object* v_x2_276_){
_start:
{
lean_object* v_res_277_; 
v_res_277_ = lp_batteries_Array_mergeUnsortedDedup___redArg___lam__1(v___x_272_, v___x_273_, v_eq_274_, v_x1_275_, v_x2_276_);
lean_dec(v___x_272_);
return v_res_277_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_mergeUnsortedDedup___redArg(lean_object* v_eq_278_, lean_object* v_xs_279_, lean_object* v_ys_280_){
_start:
{
lean_object* v___x_281_; lean_object* v___x_282_; uint8_t v___x_283_; 
v___x_281_ = lean_array_get_size(v_xs_279_);
v___x_282_ = lean_array_get_size(v_ys_280_);
v___x_283_ = lean_nat_dec_lt(v___x_281_, v___x_282_);
if (v___x_283_ == 0)
{
lean_object* v___x_284_; lean_object* v___x_285_; uint8_t v___x_286_; 
v___x_284_ = lean_unsigned_to_nat(0u);
v___x_285_ = ((lean_object*)(lp_batteries_Array_mergeUnsortedDedup_go___redArg___lam__1___closed__9));
v___x_286_ = lean_nat_dec_lt(v___x_284_, v___x_282_);
if (v___x_286_ == 0)
{
lean_dec_ref(v_ys_280_);
lean_dec_ref(v_eq_278_);
return v_xs_279_;
}
else
{
lean_object* v___f_287_; uint8_t v___x_288_; 
v___f_287_ = lean_alloc_closure((void*)(lp_batteries_Array_mergeUnsortedDedup___redArg___lam__1___boxed), 5, 3);
lean_closure_set(v___f_287_, 0, v___x_284_);
lean_closure_set(v___f_287_, 1, v___x_281_);
lean_closure_set(v___f_287_, 2, v_eq_278_);
v___x_288_ = lean_nat_dec_le(v___x_282_, v___x_282_);
if (v___x_288_ == 0)
{
if (v___x_286_ == 0)
{
lean_dec_ref(v___f_287_);
lean_dec_ref(v_ys_280_);
return v_xs_279_;
}
else
{
size_t v___x_289_; size_t v___x_290_; lean_object* v___x_291_; 
v___x_289_ = ((size_t)0ULL);
v___x_290_ = lean_usize_of_nat(v___x_282_);
v___x_291_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_285_, v___f_287_, v_ys_280_, v___x_289_, v___x_290_, v_xs_279_);
return v___x_291_;
}
}
else
{
size_t v___x_292_; size_t v___x_293_; lean_object* v___x_294_; 
v___x_292_ = ((size_t)0ULL);
v___x_293_ = lean_usize_of_nat(v___x_282_);
v___x_294_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_285_, v___f_287_, v_ys_280_, v___x_292_, v___x_293_, v_xs_279_);
return v___x_294_;
}
}
}
else
{
lean_object* v___x_295_; lean_object* v___x_296_; uint8_t v___x_297_; 
v___x_295_ = lean_unsigned_to_nat(0u);
v___x_296_ = ((lean_object*)(lp_batteries_Array_mergeUnsortedDedup_go___redArg___lam__1___closed__9));
v___x_297_ = lean_nat_dec_lt(v___x_295_, v___x_281_);
if (v___x_297_ == 0)
{
lean_dec_ref(v_xs_279_);
lean_dec_ref(v_eq_278_);
return v_ys_280_;
}
else
{
lean_object* v___f_298_; uint8_t v___x_299_; 
v___f_298_ = lean_alloc_closure((void*)(lp_batteries_Array_mergeUnsortedDedup___redArg___lam__1___boxed), 5, 3);
lean_closure_set(v___f_298_, 0, v___x_295_);
lean_closure_set(v___f_298_, 1, v___x_282_);
lean_closure_set(v___f_298_, 2, v_eq_278_);
v___x_299_ = lean_nat_dec_le(v___x_281_, v___x_281_);
if (v___x_299_ == 0)
{
if (v___x_297_ == 0)
{
lean_dec_ref(v___f_298_);
lean_dec_ref(v_xs_279_);
return v_ys_280_;
}
else
{
size_t v___x_300_; size_t v___x_301_; lean_object* v___x_302_; 
v___x_300_ = ((size_t)0ULL);
v___x_301_ = lean_usize_of_nat(v___x_281_);
v___x_302_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_296_, v___f_298_, v_xs_279_, v___x_300_, v___x_301_, v_ys_280_);
return v___x_302_;
}
}
else
{
size_t v___x_303_; size_t v___x_304_; lean_object* v___x_305_; 
v___x_303_ = ((size_t)0ULL);
v___x_304_ = lean_usize_of_nat(v___x_281_);
v___x_305_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_296_, v___f_298_, v_xs_279_, v___x_303_, v___x_304_, v_ys_280_);
return v___x_305_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_mergeUnsortedDedup(lean_object* v_00_u03b1_306_, lean_object* v_eq_307_, lean_object* v_xs_308_, lean_object* v_ys_309_){
_start:
{
lean_object* v___x_310_; 
v___x_310_ = lp_batteries_Array_mergeUnsortedDedup___redArg(v_eq_307_, v_xs_308_, v_ys_309_);
return v___x_310_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_mergeAdjacentDups_go___redArg(lean_object* v_eq_311_, lean_object* v_f_312_, lean_object* v_xs_313_, lean_object* v_acc_314_, lean_object* v_i_315_, lean_object* v_hd_316_){
_start:
{
lean_object* v___x_317_; uint8_t v___x_318_; 
v___x_317_ = lean_array_get_size(v_xs_313_);
v___x_318_ = lean_nat_dec_lt(v_i_315_, v___x_317_);
if (v___x_318_ == 0)
{
lean_object* v___x_319_; 
lean_dec(v_i_315_);
lean_dec(v_f_312_);
lean_dec_ref(v_eq_311_);
v___x_319_ = lean_array_push(v_acc_314_, v_hd_316_);
return v___x_319_;
}
else
{
lean_object* v_x_320_; lean_object* v___x_321_; uint8_t v___x_322_; 
v_x_320_ = lean_array_fget_borrowed(v_xs_313_, v_i_315_);
lean_inc_ref(v_eq_311_);
lean_inc(v_hd_316_);
lean_inc(v_x_320_);
v___x_321_ = lean_apply_2(v_eq_311_, v_x_320_, v_hd_316_);
v___x_322_ = lean_unbox(v___x_321_);
if (v___x_322_ == 0)
{
lean_object* v___x_323_; lean_object* v___x_324_; lean_object* v___x_325_; 
v___x_323_ = lean_array_push(v_acc_314_, v_hd_316_);
v___x_324_ = lean_unsigned_to_nat(1u);
v___x_325_ = lean_nat_add(v_i_315_, v___x_324_);
lean_dec(v_i_315_);
lean_inc(v_x_320_);
v_acc_314_ = v___x_323_;
v_i_315_ = v___x_325_;
v_hd_316_ = v_x_320_;
goto _start;
}
else
{
lean_object* v___x_327_; lean_object* v___x_328_; lean_object* v___x_329_; 
v___x_327_ = lean_unsigned_to_nat(1u);
v___x_328_ = lean_nat_add(v_i_315_, v___x_327_);
lean_dec(v_i_315_);
lean_inc(v_f_312_);
lean_inc(v_x_320_);
v___x_329_ = lean_apply_2(v_f_312_, v_hd_316_, v_x_320_);
v_i_315_ = v___x_328_;
v_hd_316_ = v___x_329_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_mergeAdjacentDups_go___redArg___boxed(lean_object* v_eq_331_, lean_object* v_f_332_, lean_object* v_xs_333_, lean_object* v_acc_334_, lean_object* v_i_335_, lean_object* v_hd_336_){
_start:
{
lean_object* v_res_337_; 
v_res_337_ = lp_batteries_Array_mergeAdjacentDups_go___redArg(v_eq_331_, v_f_332_, v_xs_333_, v_acc_334_, v_i_335_, v_hd_336_);
lean_dec_ref(v_xs_333_);
return v_res_337_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_mergeAdjacentDups_go(lean_object* v_00_u03b1_338_, lean_object* v_eq_339_, lean_object* v_f_340_, lean_object* v_xs_341_, lean_object* v_acc_342_, lean_object* v_i_343_, lean_object* v_hd_344_){
_start:
{
lean_object* v___x_345_; 
v___x_345_ = lp_batteries_Array_mergeAdjacentDups_go___redArg(v_eq_339_, v_f_340_, v_xs_341_, v_acc_342_, v_i_343_, v_hd_344_);
return v___x_345_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_mergeAdjacentDups_go___boxed(lean_object* v_00_u03b1_346_, lean_object* v_eq_347_, lean_object* v_f_348_, lean_object* v_xs_349_, lean_object* v_acc_350_, lean_object* v_i_351_, lean_object* v_hd_352_){
_start:
{
lean_object* v_res_353_; 
v_res_353_ = lp_batteries_Array_mergeAdjacentDups_go(v_00_u03b1_346_, v_eq_347_, v_f_348_, v_xs_349_, v_acc_350_, v_i_351_, v_hd_352_);
lean_dec_ref(v_xs_349_);
return v_res_353_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_mergeAdjacentDups___redArg(lean_object* v_eq_354_, lean_object* v_f_355_, lean_object* v_xs_356_){
_start:
{
lean_object* v___x_357_; lean_object* v___x_358_; uint8_t v___x_359_; 
v___x_357_ = lean_unsigned_to_nat(0u);
v___x_358_ = lean_array_get_size(v_xs_356_);
v___x_359_ = lean_nat_dec_lt(v___x_357_, v___x_358_);
if (v___x_359_ == 0)
{
lean_dec(v_f_355_);
lean_dec_ref(v_eq_354_);
lean_inc_ref(v_xs_356_);
return v_xs_356_;
}
else
{
lean_object* v___x_360_; lean_object* v___x_361_; lean_object* v___x_362_; lean_object* v___x_363_; 
v___x_360_ = lean_mk_empty_array_with_capacity(v___x_358_);
v___x_361_ = lean_unsigned_to_nat(1u);
v___x_362_ = lean_array_fget_borrowed(v_xs_356_, v___x_357_);
lean_inc(v___x_362_);
v___x_363_ = lp_batteries_Array_mergeAdjacentDups_go___redArg(v_eq_354_, v_f_355_, v_xs_356_, v___x_360_, v___x_361_, v___x_362_);
return v___x_363_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_mergeAdjacentDups___redArg___boxed(lean_object* v_eq_364_, lean_object* v_f_365_, lean_object* v_xs_366_){
_start:
{
lean_object* v_res_367_; 
v_res_367_ = lp_batteries_Array_mergeAdjacentDups___redArg(v_eq_364_, v_f_365_, v_xs_366_);
lean_dec_ref(v_xs_366_);
return v_res_367_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_mergeAdjacentDups(lean_object* v_00_u03b1_368_, lean_object* v_eq_369_, lean_object* v_f_370_, lean_object* v_xs_371_){
_start:
{
lean_object* v___x_372_; 
v___x_372_ = lp_batteries_Array_mergeAdjacentDups___redArg(v_eq_369_, v_f_370_, v_xs_371_);
return v___x_372_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_mergeAdjacentDups___boxed(lean_object* v_00_u03b1_373_, lean_object* v_eq_374_, lean_object* v_f_375_, lean_object* v_xs_376_){
_start:
{
lean_object* v_res_377_; 
v_res_377_ = lp_batteries_Array_mergeAdjacentDups(v_00_u03b1_373_, v_eq_374_, v_f_375_, v_xs_376_);
lean_dec_ref(v_xs_376_);
return v_res_377_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_dedupSorted___redArg(lean_object* v_eq_378_, lean_object* v_xs_379_){
_start:
{
lean_object* v___f_380_; lean_object* v___x_381_; 
v___f_380_ = ((lean_object*)(lp_batteries_Array_mergeDedup___redArg___closed__0));
v___x_381_ = lp_batteries_Array_mergeAdjacentDups___redArg(v_eq_378_, v___f_380_, v_xs_379_);
return v___x_381_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_dedupSorted___redArg___boxed(lean_object* v_eq_382_, lean_object* v_xs_383_){
_start:
{
lean_object* v_res_384_; 
v_res_384_ = lp_batteries_Array_dedupSorted___redArg(v_eq_382_, v_xs_383_);
lean_dec_ref(v_xs_383_);
return v_res_384_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_dedupSorted(lean_object* v_00_u03b1_385_, lean_object* v_eq_386_, lean_object* v_xs_387_){
_start:
{
lean_object* v___x_388_; 
v___x_388_ = lp_batteries_Array_dedupSorted___redArg(v_eq_386_, v_xs_387_);
return v___x_388_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_dedupSorted___boxed(lean_object* v_00_u03b1_389_, lean_object* v_eq_390_, lean_object* v_xs_391_){
_start:
{
lean_object* v_res_392_; 
v_res_392_ = lp_batteries_Array_dedupSorted(v_00_u03b1_389_, v_eq_390_, v_xs_391_);
lean_dec_ref(v_xs_391_);
return v_res_392_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Array_sortDedup___redArg___lam__0(lean_object* v_ord_393_, uint8_t v___x_394_, lean_object* v_x1_395_, lean_object* v_x2_396_){
_start:
{
lean_object* v___x_397_; uint8_t v___x_398_; 
v___x_397_ = lean_apply_2(v_ord_393_, v_x1_395_, v_x2_396_);
v___x_398_ = lean_unbox(v___x_397_);
if (v___x_398_ == 0)
{
uint8_t v___x_399_; 
v___x_399_ = 1;
return v___x_399_;
}
else
{
return v___x_394_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_sortDedup___redArg___lam__0___boxed(lean_object* v_ord_400_, lean_object* v___x_401_, lean_object* v_x1_402_, lean_object* v_x2_403_){
_start:
{
uint8_t v___x_72__boxed_404_; uint8_t v_res_405_; lean_object* v_r_406_; 
v___x_72__boxed_404_ = lean_unbox(v___x_401_);
v_res_405_ = lp_batteries_Array_sortDedup___redArg___lam__0(v_ord_400_, v___x_72__boxed_404_, v_x1_402_, v_x2_403_);
v_r_406_ = lean_box(v_res_405_);
return v_r_406_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_sortDedup___redArg(lean_object* v_ord_407_, lean_object* v_xs_408_){
_start:
{
lean_object* v_this_409_; lean_object* v___x_410_; lean_object* v___x_411_; uint8_t v___x_412_; 
lean_inc_ref(v_ord_407_);
v_this_409_ = lean_alloc_closure((void*)(l_beqOfOrd___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v_this_409_, 0, v_ord_407_);
v___x_410_ = lean_array_get_size(v_xs_408_);
v___x_411_ = lean_unsigned_to_nat(0u);
v___x_412_ = lean_nat_dec_eq(v___x_410_, v___x_411_);
if (v___x_412_ == 0)
{
lean_object* v___x_413_; lean_object* v___f_414_; lean_object* v___y_416_; lean_object* v___y_417_; lean_object* v___x_420_; lean_object* v___x_421_; lean_object* v___y_423_; uint8_t v___x_425_; 
v___x_413_ = lean_box(v___x_412_);
v___f_414_ = lean_alloc_closure((void*)(lp_batteries_Array_sortDedup___redArg___lam__0___boxed), 4, 2);
lean_closure_set(v___f_414_, 0, v_ord_407_);
lean_closure_set(v___f_414_, 1, v___x_413_);
v___x_420_ = lean_unsigned_to_nat(1u);
v___x_421_ = lean_nat_sub(v___x_410_, v___x_420_);
v___x_425_ = lean_nat_dec_le(v___x_411_, v___x_421_);
if (v___x_425_ == 0)
{
lean_inc(v___x_421_);
v___y_423_ = v___x_421_;
goto v___jp_422_;
}
else
{
v___y_423_ = v___x_411_;
goto v___jp_422_;
}
v___jp_415_:
{
lean_object* v___x_418_; lean_object* v___x_419_; 
v___x_418_ = l___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort(lean_box(0), v___f_414_, v___x_410_, v_xs_408_, v___y_416_, v___y_417_, lean_box(0), lean_box(0), lean_box(0));
lean_dec(v___y_417_);
v___x_419_ = lp_batteries_Array_dedupSorted___redArg(v_this_409_, v___x_418_);
lean_dec_ref(v___x_418_);
return v___x_419_;
}
v___jp_422_:
{
uint8_t v___x_424_; 
v___x_424_ = lean_nat_dec_le(v___y_423_, v___x_421_);
if (v___x_424_ == 0)
{
lean_dec(v___x_421_);
lean_inc(v___y_423_);
v___y_416_ = v___y_423_;
v___y_417_ = v___y_423_;
goto v___jp_415_;
}
else
{
v___y_416_ = v___y_423_;
v___y_417_ = v___x_421_;
goto v___jp_415_;
}
}
}
else
{
lean_object* v___x_426_; 
lean_dec_ref(v_ord_407_);
v___x_426_ = lp_batteries_Array_dedupSorted___redArg(v_this_409_, v_xs_408_);
lean_dec_ref(v_xs_408_);
return v___x_426_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_sortDedup(lean_object* v_00_u03b1_427_, lean_object* v_ord_428_, lean_object* v_xs_429_){
_start:
{
lean_object* v___x_430_; 
v___x_430_ = lp_batteries_Array_sortDedup___redArg(v_ord_428_, v_xs_429_);
return v___x_430_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_batteries_Batteries_Data_Array_Merge(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize_runtime_module();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_batteries_Batteries_Data_Array_Merge(uint8_t builtin) {
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
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_batteries_Batteries_Data_Array_Merge(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Data_Array_Merge(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_batteries_Batteries_Data_Array_Merge(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_batteries_Batteries_Data_Array_Merge(builtin);
}
#ifdef __cplusplus
}
#endif
