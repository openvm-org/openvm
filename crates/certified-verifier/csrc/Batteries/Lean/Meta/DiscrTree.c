// Lean compiler output
// Module: Batteries.Lean.Meta.DiscrTree
// Imports: public import Init public meta import Init public import Lean.Meta.DiscrTree public import Batteries.Data.Array.Merge public import Batteries.Lean.Meta.Expr public import Batteries.Lean.PersistentHashMap
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
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* l_Lean_Meta_DiscrTree_Key_ctorIdx(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
uint8_t lean_string_compare(lean_object*, lean_object*);
uint8_t l___private_Lean_Data_Name_0__Lean_Name_quickCmpImpl(lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_Array_toSubarray___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Subarray_copy___redArg(lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Meta_DiscrTree_Key_hash___boxed(lean_object*);
lean_object* l_Lean_Meta_DiscrTree_instBEqKey_beq___boxed(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_find_x3f___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_insert___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_PersistentHashMap_foldl___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Lean_Meta_DiscrTree_Key_cmp(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_DiscrTree_Key_cmp___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_batteries_Lean_Meta_DiscrTree_Key_instOrd__batteries___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries_Lean_Meta_DiscrTree_Key_cmp___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Lean_Meta_DiscrTree_Key_instOrd__batteries___closed__0 = (const lean_object*)&lp_batteries_Lean_Meta_DiscrTree_Key_instOrd__batteries___closed__0_value;
LEAN_EXPORT const lean_object* lp_batteries_Lean_Meta_DiscrTree_Key_instOrd__batteries = (const lean_object*)&lp_batteries_Lean_Meta_DiscrTree_Key_instOrd__batteries___closed__0_value;
LEAN_EXPORT lean_object* lp_batteries_Array_mergeDedupWith_go___at___00Array_mergeDedupWith___at___00Lean_Meta_DiscrTree_Trie_mergePreservingDuplicates_mergeChildren_spec__0_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_mergeDedupWith___at___00Lean_Meta_DiscrTree_Trie_mergePreservingDuplicates_mergeChildren_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_DiscrTree_Trie_mergePreservingDuplicates___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_DiscrTree_Trie_mergePreservingDuplicates_mergeChildren___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_DiscrTree_Trie_mergePreservingDuplicates_mergeChildren___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_DiscrTree_Trie_mergePreservingDuplicates(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_DiscrTree_Trie_mergePreservingDuplicates_mergeChildren(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_mergeDedupWith___at___00Lean_Meta_DiscrTree_Trie_mergePreservingDuplicates_mergeChildren_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_mergeDedupWith_go___at___00Array_mergeDedupWith___at___00Lean_Meta_DiscrTree_Trie_mergePreservingDuplicates_mergeChildren_spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_DiscrTree_mergePreservingDuplicates___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_batteries_Lean_Meta_DiscrTree_mergePreservingDuplicates___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_DiscrTree_instBEqKey_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Lean_Meta_DiscrTree_mergePreservingDuplicates___redArg___closed__0 = (const lean_object*)&lp_batteries_Lean_Meta_DiscrTree_mergePreservingDuplicates___redArg___closed__0_value;
static const lean_closure_object lp_batteries_Lean_Meta_DiscrTree_mergePreservingDuplicates___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_DiscrTree_Key_hash___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Lean_Meta_DiscrTree_mergePreservingDuplicates___redArg___closed__1 = (const lean_object*)&lp_batteries_Lean_Meta_DiscrTree_mergePreservingDuplicates___redArg___closed__1_value;
static const lean_closure_object lp_batteries_Lean_Meta_DiscrTree_mergePreservingDuplicates___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries_Lean_Meta_DiscrTree_mergePreservingDuplicates___redArg___lam__0, .m_arity = 5, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_batteries_Lean_Meta_DiscrTree_mergePreservingDuplicates___redArg___closed__0_value),((lean_object*)&lp_batteries_Lean_Meta_DiscrTree_mergePreservingDuplicates___redArg___closed__1_value)} };
static const lean_object* lp_batteries_Lean_Meta_DiscrTree_mergePreservingDuplicates___redArg___closed__2 = (const lean_object*)&lp_batteries_Lean_Meta_DiscrTree_mergePreservingDuplicates___redArg___closed__2_value;
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_DiscrTree_mergePreservingDuplicates___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_DiscrTree_mergePreservingDuplicates(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Lean_Meta_DiscrTree_Key_cmp(lean_object* v_x_1_, lean_object* v_x_2_){
_start:
{
lean_object* v_k_u2081_4_; lean_object* v_k_u2082_5_; 
switch(lean_obj_tag(v_x_1_))
{
case 2:
{
if (lean_obj_tag(v_x_2_) == 2)
{
lean_object* v_a_13_; 
v_a_13_ = lean_ctor_get(v_x_1_, 0);
if (lean_obj_tag(v_a_13_) == 0)
{
lean_object* v_a_14_; 
v_a_14_ = lean_ctor_get(v_x_2_, 0);
if (lean_obj_tag(v_a_14_) == 0)
{
lean_object* v_val_15_; lean_object* v_val_16_; uint8_t v___x_17_; 
v_val_15_ = lean_ctor_get(v_a_13_, 0);
v_val_16_ = lean_ctor_get(v_a_14_, 0);
v___x_17_ = lean_nat_dec_lt(v_val_15_, v_val_16_);
if (v___x_17_ == 0)
{
uint8_t v___x_18_; 
v___x_18_ = lean_nat_dec_eq(v_val_15_, v_val_16_);
if (v___x_18_ == 0)
{
uint8_t v___x_19_; 
v___x_19_ = 2;
return v___x_19_;
}
else
{
uint8_t v___x_20_; 
v___x_20_ = 1;
return v___x_20_;
}
}
else
{
uint8_t v___x_21_; 
v___x_21_ = 0;
return v___x_21_;
}
}
else
{
uint8_t v___x_22_; 
v___x_22_ = 0;
return v___x_22_;
}
}
else
{
lean_object* v_a_23_; 
v_a_23_ = lean_ctor_get(v_x_2_, 0);
if (lean_obj_tag(v_a_23_) == 0)
{
uint8_t v___x_24_; 
v___x_24_ = 2;
return v___x_24_;
}
else
{
lean_object* v_val_25_; lean_object* v_val_26_; uint8_t v___x_27_; 
v_val_25_ = lean_ctor_get(v_a_13_, 0);
v_val_26_ = lean_ctor_get(v_a_23_, 0);
v___x_27_ = lean_string_compare(v_val_25_, v_val_26_);
return v___x_27_;
}
}
}
else
{
v_k_u2081_4_ = v_x_1_;
v_k_u2082_5_ = v_x_2_;
goto v___jp_3_;
}
}
case 3:
{
if (lean_obj_tag(v_x_2_) == 3)
{
lean_object* v_a_28_; lean_object* v_a_29_; lean_object* v_a_30_; lean_object* v_a_31_; uint8_t v___x_32_; 
v_a_28_ = lean_ctor_get(v_x_1_, 0);
v_a_29_ = lean_ctor_get(v_x_1_, 1);
v_a_30_ = lean_ctor_get(v_x_2_, 0);
v_a_31_ = lean_ctor_get(v_x_2_, 1);
v___x_32_ = l___private_Lean_Data_Name_0__Lean_Name_quickCmpImpl(v_a_28_, v_a_30_);
if (v___x_32_ == 1)
{
uint8_t v___x_33_; 
v___x_33_ = lean_nat_dec_lt(v_a_29_, v_a_31_);
if (v___x_33_ == 0)
{
uint8_t v___x_34_; 
v___x_34_ = lean_nat_dec_eq(v_a_29_, v_a_31_);
if (v___x_34_ == 0)
{
uint8_t v___x_35_; 
v___x_35_ = 2;
return v___x_35_;
}
else
{
return v___x_32_;
}
}
else
{
uint8_t v___x_36_; 
v___x_36_ = 0;
return v___x_36_;
}
}
else
{
return v___x_32_;
}
}
else
{
v_k_u2081_4_ = v_x_1_;
v_k_u2082_5_ = v_x_2_;
goto v___jp_3_;
}
}
case 4:
{
if (lean_obj_tag(v_x_2_) == 4)
{
lean_object* v_a_37_; lean_object* v_a_38_; lean_object* v_a_39_; lean_object* v_a_40_; uint8_t v___x_41_; 
v_a_37_ = lean_ctor_get(v_x_1_, 0);
v_a_38_ = lean_ctor_get(v_x_1_, 1);
v_a_39_ = lean_ctor_get(v_x_2_, 0);
v_a_40_ = lean_ctor_get(v_x_2_, 1);
v___x_41_ = l___private_Lean_Data_Name_0__Lean_Name_quickCmpImpl(v_a_37_, v_a_39_);
if (v___x_41_ == 1)
{
uint8_t v___x_42_; 
v___x_42_ = lean_nat_dec_lt(v_a_38_, v_a_40_);
if (v___x_42_ == 0)
{
uint8_t v___x_43_; 
v___x_43_ = lean_nat_dec_eq(v_a_38_, v_a_40_);
if (v___x_43_ == 0)
{
uint8_t v___x_44_; 
v___x_44_ = 2;
return v___x_44_;
}
else
{
return v___x_41_;
}
}
else
{
uint8_t v___x_45_; 
v___x_45_ = 0;
return v___x_45_;
}
}
else
{
return v___x_41_;
}
}
else
{
v_k_u2081_4_ = v_x_1_;
v_k_u2082_5_ = v_x_2_;
goto v___jp_3_;
}
}
case 6:
{
if (lean_obj_tag(v_x_2_) == 6)
{
lean_object* v_a_46_; lean_object* v_a_47_; lean_object* v_a_48_; lean_object* v_a_49_; lean_object* v_a_50_; lean_object* v_a_51_; uint8_t v___x_52_; 
v_a_46_ = lean_ctor_get(v_x_1_, 0);
v_a_47_ = lean_ctor_get(v_x_1_, 1);
v_a_48_ = lean_ctor_get(v_x_1_, 2);
v_a_49_ = lean_ctor_get(v_x_2_, 0);
v_a_50_ = lean_ctor_get(v_x_2_, 1);
v_a_51_ = lean_ctor_get(v_x_2_, 2);
v___x_52_ = l___private_Lean_Data_Name_0__Lean_Name_quickCmpImpl(v_a_46_, v_a_49_);
if (v___x_52_ == 1)
{
uint8_t v___x_53_; 
v___x_53_ = lean_nat_dec_lt(v_a_47_, v_a_50_);
if (v___x_53_ == 0)
{
uint8_t v___x_54_; 
v___x_54_ = lean_nat_dec_eq(v_a_47_, v_a_50_);
if (v___x_54_ == 0)
{
uint8_t v___x_55_; 
v___x_55_ = 2;
return v___x_55_;
}
else
{
uint8_t v___x_56_; 
v___x_56_ = lean_nat_dec_lt(v_a_48_, v_a_51_);
if (v___x_56_ == 0)
{
uint8_t v___x_57_; 
v___x_57_ = lean_nat_dec_eq(v_a_48_, v_a_51_);
if (v___x_57_ == 0)
{
uint8_t v___x_58_; 
v___x_58_ = 2;
return v___x_58_;
}
else
{
return v___x_52_;
}
}
else
{
uint8_t v___x_59_; 
v___x_59_ = 0;
return v___x_59_;
}
}
}
else
{
uint8_t v___x_60_; 
v___x_60_ = 0;
return v___x_60_;
}
}
else
{
return v___x_52_;
}
}
else
{
v_k_u2081_4_ = v_x_1_;
v_k_u2082_5_ = v_x_2_;
goto v___jp_3_;
}
}
default: 
{
v_k_u2081_4_ = v_x_1_;
v_k_u2082_5_ = v_x_2_;
goto v___jp_3_;
}
}
v___jp_3_:
{
lean_object* v___x_6_; lean_object* v___x_7_; uint8_t v___x_8_; 
v___x_6_ = l_Lean_Meta_DiscrTree_Key_ctorIdx(v_k_u2081_4_);
v___x_7_ = l_Lean_Meta_DiscrTree_Key_ctorIdx(v_k_u2082_5_);
v___x_8_ = lean_nat_dec_lt(v___x_6_, v___x_7_);
if (v___x_8_ == 0)
{
uint8_t v___x_9_; 
v___x_9_ = lean_nat_dec_eq(v___x_6_, v___x_7_);
lean_dec(v___x_7_);
lean_dec(v___x_6_);
if (v___x_9_ == 0)
{
uint8_t v___x_10_; 
v___x_10_ = 2;
return v___x_10_;
}
else
{
uint8_t v___x_11_; 
v___x_11_ = 1;
return v___x_11_;
}
}
else
{
uint8_t v___x_12_; 
lean_dec(v___x_7_);
lean_dec(v___x_6_);
v___x_12_ = 0;
return v___x_12_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_DiscrTree_Key_cmp___boxed(lean_object* v_x_61_, lean_object* v_x_62_){
_start:
{
uint8_t v_res_63_; lean_object* v_r_64_; 
v_res_63_ = lp_batteries_Lean_Meta_DiscrTree_Key_cmp(v_x_61_, v_x_62_);
lean_dec(v_x_62_);
lean_dec(v_x_61_);
v_r_64_ = lean_box(v_res_63_);
return v_r_64_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_mergeDedupWith_go___at___00Array_mergeDedupWith___at___00Lean_Meta_DiscrTree_Trie_mergePreservingDuplicates_mergeChildren_spec__0_spec__1___redArg(lean_object* v_xs_67_, lean_object* v_ys_68_, lean_object* v_merge_69_, lean_object* v_acc_70_, lean_object* v_i_71_, lean_object* v_j_72_){
_start:
{
lean_object* v___x_73_; uint8_t v___x_74_; 
v___x_73_ = lean_array_get_size(v_xs_67_);
v___x_74_ = lean_nat_dec_le(v___x_73_, v_i_71_);
if (v___x_74_ == 0)
{
lean_object* v___x_75_; uint8_t v___x_76_; 
v___x_75_ = lean_array_get_size(v_ys_68_);
v___x_76_ = lean_nat_dec_le(v___x_75_, v_j_72_);
if (v___x_76_ == 0)
{
lean_object* v_x_77_; lean_object* v_fst_78_; lean_object* v_y_79_; lean_object* v_fst_80_; uint8_t v___x_81_; 
v_x_77_ = lean_array_fget_borrowed(v_xs_67_, v_i_71_);
v_fst_78_ = lean_ctor_get(v_x_77_, 0);
v_y_79_ = lean_array_fget_borrowed(v_ys_68_, v_j_72_);
v_fst_80_ = lean_ctor_get(v_y_79_, 0);
v___x_81_ = lp_batteries_Lean_Meta_DiscrTree_Key_cmp(v_fst_78_, v_fst_80_);
switch(v___x_81_)
{
case 0:
{
lean_object* v___x_82_; lean_object* v___x_83_; lean_object* v___x_84_; 
lean_inc(v_x_77_);
v___x_82_ = lean_array_push(v_acc_70_, v_x_77_);
v___x_83_ = lean_unsigned_to_nat(1u);
v___x_84_ = lean_nat_add(v_i_71_, v___x_83_);
lean_dec(v_i_71_);
v_acc_70_ = v___x_82_;
v_i_71_ = v___x_84_;
goto _start;
}
case 1:
{
lean_object* v___x_86_; lean_object* v___x_87_; lean_object* v___x_88_; lean_object* v___x_89_; lean_object* v___x_90_; 
lean_inc_ref(v_merge_69_);
lean_inc(v_y_79_);
lean_inc(v_x_77_);
v___x_86_ = lean_apply_2(v_merge_69_, v_x_77_, v_y_79_);
v___x_87_ = lean_array_push(v_acc_70_, v___x_86_);
v___x_88_ = lean_unsigned_to_nat(1u);
v___x_89_ = lean_nat_add(v_i_71_, v___x_88_);
lean_dec(v_i_71_);
v___x_90_ = lean_nat_add(v_j_72_, v___x_88_);
lean_dec(v_j_72_);
v_acc_70_ = v___x_87_;
v_i_71_ = v___x_89_;
v_j_72_ = v___x_90_;
goto _start;
}
default: 
{
lean_object* v___x_92_; lean_object* v___x_93_; lean_object* v___x_94_; 
lean_inc(v_y_79_);
v___x_92_ = lean_array_push(v_acc_70_, v_y_79_);
v___x_93_ = lean_unsigned_to_nat(1u);
v___x_94_ = lean_nat_add(v_j_72_, v___x_93_);
lean_dec(v_j_72_);
v_acc_70_ = v___x_92_;
v_j_72_ = v___x_94_;
goto _start;
}
}
}
else
{
lean_object* v___x_96_; lean_object* v___x_97_; lean_object* v___x_98_; 
lean_dec(v_j_72_);
lean_dec_ref(v_merge_69_);
lean_dec_ref(v_ys_68_);
v___x_96_ = l_Array_toSubarray___redArg(v_xs_67_, v_i_71_, v___x_73_);
v___x_97_ = l_Subarray_copy___redArg(v___x_96_);
v___x_98_ = l_Array_append___redArg(v_acc_70_, v___x_97_);
lean_dec_ref(v___x_97_);
return v___x_98_;
}
}
else
{
lean_object* v___x_99_; lean_object* v___x_100_; lean_object* v___x_101_; lean_object* v___x_102_; 
lean_dec(v_i_71_);
lean_dec_ref(v_merge_69_);
lean_dec_ref(v_xs_67_);
v___x_99_ = lean_array_get_size(v_ys_68_);
v___x_100_ = l_Array_toSubarray___redArg(v_ys_68_, v_j_72_, v___x_99_);
v___x_101_ = l_Subarray_copy___redArg(v___x_100_);
v___x_102_ = l_Array_append___redArg(v_acc_70_, v___x_101_);
lean_dec_ref(v___x_101_);
return v___x_102_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_mergeDedupWith___at___00Lean_Meta_DiscrTree_Trie_mergePreservingDuplicates_mergeChildren_spec__0___redArg(lean_object* v_xs_103_, lean_object* v_ys_104_, lean_object* v_merge_105_){
_start:
{
lean_object* v___x_106_; lean_object* v___x_107_; lean_object* v___x_108_; lean_object* v___x_109_; lean_object* v___x_110_; lean_object* v___x_111_; 
v___x_106_ = lean_array_get_size(v_xs_103_);
v___x_107_ = lean_array_get_size(v_ys_104_);
v___x_108_ = lean_nat_add(v___x_106_, v___x_107_);
v___x_109_ = lean_mk_empty_array_with_capacity(v___x_108_);
lean_dec(v___x_108_);
v___x_110_ = lean_unsigned_to_nat(0u);
v___x_111_ = lp_batteries_Array_mergeDedupWith_go___at___00Array_mergeDedupWith___at___00Lean_Meta_DiscrTree_Trie_mergePreservingDuplicates_mergeChildren_spec__0_spec__1___redArg(v_xs_103_, v_ys_104_, v_merge_105_, v___x_109_, v___x_110_, v___x_110_);
return v___x_111_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_DiscrTree_Trie_mergePreservingDuplicates___redArg(lean_object* v_x_112_, lean_object* v_x_113_){
_start:
{
lean_object* v_vs_114_; lean_object* v_children_115_; lean_object* v_vs_116_; lean_object* v_children_117_; lean_object* v___x_119_; uint8_t v_isShared_120_; uint8_t v_isSharedCheck_126_; 
v_vs_114_ = lean_ctor_get(v_x_112_, 0);
lean_inc_ref(v_vs_114_);
v_children_115_ = lean_ctor_get(v_x_112_, 1);
lean_inc_ref(v_children_115_);
lean_dec_ref(v_x_112_);
v_vs_116_ = lean_ctor_get(v_x_113_, 0);
v_children_117_ = lean_ctor_get(v_x_113_, 1);
v_isSharedCheck_126_ = !lean_is_exclusive(v_x_113_);
if (v_isSharedCheck_126_ == 0)
{
v___x_119_ = v_x_113_;
v_isShared_120_ = v_isSharedCheck_126_;
goto v_resetjp_118_;
}
else
{
lean_inc(v_children_117_);
lean_inc(v_vs_116_);
lean_dec(v_x_113_);
v___x_119_ = lean_box(0);
v_isShared_120_ = v_isSharedCheck_126_;
goto v_resetjp_118_;
}
v_resetjp_118_:
{
lean_object* v___x_121_; lean_object* v___x_122_; lean_object* v___x_124_; 
v___x_121_ = l_Array_append___redArg(v_vs_114_, v_vs_116_);
lean_dec_ref(v_vs_116_);
v___x_122_ = lp_batteries_Lean_Meta_DiscrTree_Trie_mergePreservingDuplicates_mergeChildren___redArg(v_children_115_, v_children_117_);
if (v_isShared_120_ == 0)
{
lean_ctor_set(v___x_119_, 1, v___x_122_);
lean_ctor_set(v___x_119_, 0, v___x_121_);
v___x_124_ = v___x_119_;
goto v_reusejp_123_;
}
else
{
lean_object* v_reuseFailAlloc_125_; 
v_reuseFailAlloc_125_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_125_, 0, v___x_121_);
lean_ctor_set(v_reuseFailAlloc_125_, 1, v___x_122_);
v___x_124_ = v_reuseFailAlloc_125_;
goto v_reusejp_123_;
}
v_reusejp_123_:
{
return v___x_124_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_DiscrTree_Trie_mergePreservingDuplicates_mergeChildren___redArg___lam__0(lean_object* v_x_127_, lean_object* v_x_128_){
_start:
{
lean_object* v_fst_129_; lean_object* v_snd_130_; lean_object* v_snd_131_; lean_object* v___x_133_; uint8_t v_isShared_134_; uint8_t v_isSharedCheck_139_; 
v_fst_129_ = lean_ctor_get(v_x_127_, 0);
lean_inc(v_fst_129_);
v_snd_130_ = lean_ctor_get(v_x_127_, 1);
lean_inc(v_snd_130_);
lean_dec_ref(v_x_127_);
v_snd_131_ = lean_ctor_get(v_x_128_, 1);
v_isSharedCheck_139_ = !lean_is_exclusive(v_x_128_);
if (v_isSharedCheck_139_ == 0)
{
lean_object* v_unused_140_; 
v_unused_140_ = lean_ctor_get(v_x_128_, 0);
lean_dec(v_unused_140_);
v___x_133_ = v_x_128_;
v_isShared_134_ = v_isSharedCheck_139_;
goto v_resetjp_132_;
}
else
{
lean_inc(v_snd_131_);
lean_dec(v_x_128_);
v___x_133_ = lean_box(0);
v_isShared_134_ = v_isSharedCheck_139_;
goto v_resetjp_132_;
}
v_resetjp_132_:
{
lean_object* v___x_135_; lean_object* v___x_137_; 
v___x_135_ = lp_batteries_Lean_Meta_DiscrTree_Trie_mergePreservingDuplicates___redArg(v_snd_130_, v_snd_131_);
if (v_isShared_134_ == 0)
{
lean_ctor_set(v___x_133_, 1, v___x_135_);
lean_ctor_set(v___x_133_, 0, v_fst_129_);
v___x_137_ = v___x_133_;
goto v_reusejp_136_;
}
else
{
lean_object* v_reuseFailAlloc_138_; 
v_reuseFailAlloc_138_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_138_, 0, v_fst_129_);
lean_ctor_set(v_reuseFailAlloc_138_, 1, v___x_135_);
v___x_137_ = v_reuseFailAlloc_138_;
goto v_reusejp_136_;
}
v_reusejp_136_:
{
return v___x_137_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_DiscrTree_Trie_mergePreservingDuplicates_mergeChildren___redArg(lean_object* v_cs_u2081_141_, lean_object* v_cs_u2082_142_){
_start:
{
lean_object* v___f_143_; lean_object* v___x_144_; 
v___f_143_ = lean_alloc_closure((void*)(lp_batteries_Lean_Meta_DiscrTree_Trie_mergePreservingDuplicates_mergeChildren___redArg___lam__0), 2, 0);
v___x_144_ = lp_batteries_Array_mergeDedupWith___at___00Lean_Meta_DiscrTree_Trie_mergePreservingDuplicates_mergeChildren_spec__0___redArg(v_cs_u2081_141_, v_cs_u2082_142_, v___f_143_);
return v___x_144_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_DiscrTree_Trie_mergePreservingDuplicates(lean_object* v_00_u03b1_145_, lean_object* v_x_146_, lean_object* v_x_147_){
_start:
{
lean_object* v___x_148_; 
v___x_148_ = lp_batteries_Lean_Meta_DiscrTree_Trie_mergePreservingDuplicates___redArg(v_x_146_, v_x_147_);
return v___x_148_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_DiscrTree_Trie_mergePreservingDuplicates_mergeChildren(lean_object* v_00_u03b1_149_, lean_object* v_cs_u2081_150_, lean_object* v_cs_u2082_151_){
_start:
{
lean_object* v___x_152_; 
v___x_152_ = lp_batteries_Lean_Meta_DiscrTree_Trie_mergePreservingDuplicates_mergeChildren___redArg(v_cs_u2081_150_, v_cs_u2082_151_);
return v___x_152_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_mergeDedupWith___at___00Lean_Meta_DiscrTree_Trie_mergePreservingDuplicates_mergeChildren_spec__0(lean_object* v_00_u03b1_153_, lean_object* v_xs_154_, lean_object* v_ys_155_, lean_object* v_merge_156_){
_start:
{
lean_object* v___x_157_; 
v___x_157_ = lp_batteries_Array_mergeDedupWith___at___00Lean_Meta_DiscrTree_Trie_mergePreservingDuplicates_mergeChildren_spec__0___redArg(v_xs_154_, v_ys_155_, v_merge_156_);
return v___x_157_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_mergeDedupWith_go___at___00Array_mergeDedupWith___at___00Lean_Meta_DiscrTree_Trie_mergePreservingDuplicates_mergeChildren_spec__0_spec__1(lean_object* v_00_u03b1_158_, lean_object* v_xs_159_, lean_object* v_ys_160_, lean_object* v_merge_161_, lean_object* v_acc_162_, lean_object* v_i_163_, lean_object* v_j_164_){
_start:
{
lean_object* v___x_165_; 
v___x_165_ = lp_batteries_Array_mergeDedupWith_go___at___00Array_mergeDedupWith___at___00Lean_Meta_DiscrTree_Trie_mergePreservingDuplicates_mergeChildren_spec__0_spec__1___redArg(v_xs_159_, v_ys_160_, v_merge_161_, v_acc_162_, v_i_163_, v_j_164_);
return v___x_165_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_DiscrTree_mergePreservingDuplicates___redArg___lam__0(lean_object* v___x_166_, lean_object* v___x_167_, lean_object* v_map_168_, lean_object* v_k_169_, lean_object* v_v_u2082_170_){
_start:
{
lean_object* v___x_171_; 
lean_inc(v_k_169_);
lean_inc_ref(v___x_167_);
lean_inc_ref(v___x_166_);
v___x_171_ = l_Lean_PersistentHashMap_find_x3f___redArg(v___x_166_, v___x_167_, v_map_168_, v_k_169_);
if (lean_obj_tag(v___x_171_) == 0)
{
lean_object* v___x_172_; 
v___x_172_ = l_Lean_PersistentHashMap_insert___redArg(v___x_166_, v___x_167_, v_map_168_, v_k_169_, v_v_u2082_170_);
return v___x_172_;
}
else
{
lean_object* v_val_173_; lean_object* v___x_174_; lean_object* v___x_175_; 
v_val_173_ = lean_ctor_get(v___x_171_, 0);
lean_inc(v_val_173_);
lean_dec_ref_known(v___x_171_, 1);
v___x_174_ = lp_batteries_Lean_Meta_DiscrTree_Trie_mergePreservingDuplicates___redArg(v_val_173_, v_v_u2082_170_);
v___x_175_ = l_Lean_PersistentHashMap_insert___redArg(v___x_166_, v___x_167_, v_map_168_, v_k_169_, v___x_174_);
return v___x_175_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_DiscrTree_mergePreservingDuplicates___redArg(lean_object* v_t_181_, lean_object* v_u_182_){
_start:
{
lean_object* v___f_183_; lean_object* v___x_184_; 
v___f_183_ = ((lean_object*)(lp_batteries_Lean_Meta_DiscrTree_mergePreservingDuplicates___redArg___closed__2));
v___x_184_ = l_Lean_PersistentHashMap_foldl___redArg(v_u_182_, v___f_183_, v_t_181_);
return v___x_184_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Meta_DiscrTree_mergePreservingDuplicates(lean_object* v_00_u03b1_185_, lean_object* v_t_186_, lean_object* v_u_187_){
_start:
{
lean_object* v___f_188_; lean_object* v___x_189_; 
v___f_188_ = ((lean_object*)(lp_batteries_Lean_Meta_DiscrTree_mergePreservingDuplicates___redArg___closed__2));
v___x_189_ = l_Lean_PersistentHashMap_foldl___redArg(v_u_187_, v___f_188_, v_t_186_);
return v___x_189_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_DiscrTree(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Data_Array_Merge(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Lean_Meta_Expr(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Lean_PersistentHashMap(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_batteries_Batteries_Lean_Meta_DiscrTree(uint8_t builtin) {
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
res = runtime_initialize_batteries_Batteries_Data_Array_Merge(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Lean_Meta_Expr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Lean_PersistentHashMap(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_batteries_Batteries_Lean_Meta_DiscrTree(uint8_t builtin) {
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
lean_object* initialize_batteries_Batteries_Data_Array_Merge(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Lean_Meta_Expr(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Lean_PersistentHashMap(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_batteries_Batteries_Lean_Meta_DiscrTree(uint8_t builtin) {
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
res = initialize_batteries_Batteries_Data_Array_Merge(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Lean_Meta_Expr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Lean_PersistentHashMap(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Lean_Meta_DiscrTree(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_batteries_Batteries_Lean_Meta_DiscrTree(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_batteries_Batteries_Lean_Meta_DiscrTree(builtin);
}
#ifdef __cplusplus
}
#endif
