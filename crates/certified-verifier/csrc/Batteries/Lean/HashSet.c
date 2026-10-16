// Lean compiler output
// Module: Batteries.Lean.HashSet
// Imports: public import Init public meta import Init public import Std.Data.HashSet.Basic
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
lean_object* l_Id_instMonad___lam__2___boxed(lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__6(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Std_DHashMap_Internal_Raw_u2080_contains___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_HashSet_anyM___redArg___lam__0(lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_batteries_Std_HashSet_anyM___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_HashSet_anyM___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_HashSet_anyM___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_HashSet_anyM___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_HashSet_anyM___redArg___lam__3(lean_object*, lean_object*);
static const lean_ctor_object lp_batteries_Std_HashSet_anyM___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_batteries_Std_HashSet_anyM___redArg___closed__0 = (const lean_object*)&lp_batteries_Std_HashSet_anyM___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_batteries_Std_HashSet_anyM___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_HashSet_anyM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_HashSet_anyM___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_HashSet_allM___redArg___lam__0(lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_batteries_Std_HashSet_allM___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_HashSet_allM___redArg___lam__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_HashSet_allM___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_HashSet_allM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_HashSet_allM___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_HashSet_instBEq__batteries___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_HashSet_instBEq__batteries___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_HashSet_instBEq__batteries___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_HashSet_instBEq__batteries___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_HashSet_instBEq__batteries___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_batteries_Std_HashSet_instBEq__batteries___redArg___lam__4___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__0, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Std_HashSet_instBEq__batteries___redArg___lam__4___closed__0 = (const lean_object*)&lp_batteries_Std_HashSet_instBEq__batteries___redArg___lam__4___closed__0_value;
static const lean_closure_object lp_batteries_Std_HashSet_instBEq__batteries___redArg___lam__4___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__1___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Std_HashSet_instBEq__batteries___redArg___lam__4___closed__1 = (const lean_object*)&lp_batteries_Std_HashSet_instBEq__batteries___redArg___lam__4___closed__1_value;
static const lean_closure_object lp_batteries_Std_HashSet_instBEq__batteries___redArg___lam__4___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__2___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Std_HashSet_instBEq__batteries___redArg___lam__4___closed__2 = (const lean_object*)&lp_batteries_Std_HashSet_instBEq__batteries___redArg___lam__4___closed__2_value;
static const lean_closure_object lp_batteries_Std_HashSet_instBEq__batteries___redArg___lam__4___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__3, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Std_HashSet_instBEq__batteries___redArg___lam__4___closed__3 = (const lean_object*)&lp_batteries_Std_HashSet_instBEq__batteries___redArg___lam__4___closed__3_value;
static const lean_closure_object lp_batteries_Std_HashSet_instBEq__batteries___redArg___lam__4___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__4___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Std_HashSet_instBEq__batteries___redArg___lam__4___closed__4 = (const lean_object*)&lp_batteries_Std_HashSet_instBEq__batteries___redArg___lam__4___closed__4_value;
static const lean_closure_object lp_batteries_Std_HashSet_instBEq__batteries___redArg___lam__4___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__5___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Std_HashSet_instBEq__batteries___redArg___lam__4___closed__5 = (const lean_object*)&lp_batteries_Std_HashSet_instBEq__batteries___redArg___lam__4___closed__5_value;
static const lean_closure_object lp_batteries_Std_HashSet_instBEq__batteries___redArg___lam__4___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__6, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Std_HashSet_instBEq__batteries___redArg___lam__4___closed__6 = (const lean_object*)&lp_batteries_Std_HashSet_instBEq__batteries___redArg___lam__4___closed__6_value;
static const lean_ctor_object lp_batteries_Std_HashSet_instBEq__batteries___redArg___lam__4___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries_Std_HashSet_instBEq__batteries___redArg___lam__4___closed__0_value),((lean_object*)&lp_batteries_Std_HashSet_instBEq__batteries___redArg___lam__4___closed__1_value)}};
static const lean_object* lp_batteries_Std_HashSet_instBEq__batteries___redArg___lam__4___closed__7 = (const lean_object*)&lp_batteries_Std_HashSet_instBEq__batteries___redArg___lam__4___closed__7_value;
static const lean_ctor_object lp_batteries_Std_HashSet_instBEq__batteries___redArg___lam__4___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*5 + 0, .m_other = 5, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries_Std_HashSet_instBEq__batteries___redArg___lam__4___closed__7_value),((lean_object*)&lp_batteries_Std_HashSet_instBEq__batteries___redArg___lam__4___closed__2_value),((lean_object*)&lp_batteries_Std_HashSet_instBEq__batteries___redArg___lam__4___closed__3_value),((lean_object*)&lp_batteries_Std_HashSet_instBEq__batteries___redArg___lam__4___closed__4_value),((lean_object*)&lp_batteries_Std_HashSet_instBEq__batteries___redArg___lam__4___closed__5_value)}};
static const lean_object* lp_batteries_Std_HashSet_instBEq__batteries___redArg___lam__4___closed__8 = (const lean_object*)&lp_batteries_Std_HashSet_instBEq__batteries___redArg___lam__4___closed__8_value;
static const lean_ctor_object lp_batteries_Std_HashSet_instBEq__batteries___redArg___lam__4___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries_Std_HashSet_instBEq__batteries___redArg___lam__4___closed__8_value),((lean_object*)&lp_batteries_Std_HashSet_instBEq__batteries___redArg___lam__4___closed__6_value)}};
static const lean_object* lp_batteries_Std_HashSet_instBEq__batteries___redArg___lam__4___closed__9 = (const lean_object*)&lp_batteries_Std_HashSet_instBEq__batteries___redArg___lam__4___closed__9_value;
LEAN_EXPORT uint8_t lp_batteries_Std_HashSet_instBEq__batteries___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_HashSet_instBEq__batteries___redArg___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_HashSet_instBEq__batteries___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_HashSet_instBEq__batteries(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_HashSet_anyM___redArg___lam__0(lean_object* v___x_1_, lean_object* v_toPure_2_, lean_object* v___x_3_, uint8_t v_____do__lift_4_){
_start:
{
if (v_____do__lift_4_ == 0)
{
lean_object* v___x_5_; lean_object* v___x_6_; 
v___x_5_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_5_, 0, v___x_1_);
v___x_6_ = lean_apply_2(v_toPure_2_, lean_box(0), v___x_5_);
return v___x_6_;
}
else
{
lean_object* v___x_7_; lean_object* v___x_8_; lean_object* v___x_9_; lean_object* v___x_10_; lean_object* v___x_11_; 
lean_dec_ref(v___x_1_);
v___x_7_ = lean_box(v_____do__lift_4_);
v___x_8_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_8_, 0, v___x_7_);
v___x_9_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_9_, 0, v___x_8_);
lean_ctor_set(v___x_9_, 1, v___x_3_);
v___x_10_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_10_, 0, v___x_9_);
v___x_11_ = lean_apply_2(v_toPure_2_, lean_box(0), v___x_10_);
return v___x_11_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_HashSet_anyM___redArg___lam__0___boxed(lean_object* v___x_12_, lean_object* v_toPure_13_, lean_object* v___x_14_, lean_object* v_____do__lift_15_){
_start:
{
uint8_t v_____do__lift_168__boxed_16_; lean_object* v_res_17_; 
v_____do__lift_168__boxed_16_ = lean_unbox(v_____do__lift_15_);
v_res_17_ = lp_batteries_Std_HashSet_anyM___redArg___lam__0(v___x_12_, v_toPure_13_, v___x_14_, v_____do__lift_168__boxed_16_);
return v_res_17_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_HashSet_anyM___redArg___lam__1(lean_object* v_f_18_, lean_object* v_toBind_19_, lean_object* v___f_20_, lean_object* v_a_21_, lean_object* v_x_22_, lean_object* v_acc_23_){
_start:
{
lean_object* v___x_24_; lean_object* v___x_25_; 
v___x_24_ = lean_apply_1(v_f_18_, v_a_21_);
v___x_25_ = lean_apply_4(v_toBind_19_, lean_box(0), lean_box(0), v___x_24_, v___f_20_);
return v___x_25_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_HashSet_anyM___redArg___lam__1___boxed(lean_object* v_f_26_, lean_object* v_toBind_27_, lean_object* v___f_28_, lean_object* v_a_29_, lean_object* v_x_30_, lean_object* v_acc_31_){
_start:
{
lean_object* v_res_32_; 
v_res_32_ = lp_batteries_Std_HashSet_anyM___redArg___lam__1(v_f_26_, v_toBind_27_, v___f_28_, v_a_29_, v_x_30_, v_acc_31_);
lean_dec_ref(v_acc_31_);
return v_res_32_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_HashSet_anyM___redArg___lam__2(lean_object* v_inst_33_, lean_object* v___f_34_, lean_object* v_a_35_, lean_object* v_x_36_, lean_object* v___y_37_){
_start:
{
lean_object* v___x_38_; 
v___x_38_ = l___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go(lean_box(0), lean_box(0), lean_box(0), lean_box(0), v_inst_33_, v___f_34_, v_a_35_, v___y_37_);
return v___x_38_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_HashSet_anyM___redArg___lam__3(lean_object* v_toPure_39_, lean_object* v_____s_40_){
_start:
{
lean_object* v_fst_41_; 
v_fst_41_ = lean_ctor_get(v_____s_40_, 0);
lean_inc(v_fst_41_);
lean_dec_ref(v_____s_40_);
if (lean_obj_tag(v_fst_41_) == 0)
{
uint8_t v___x_42_; lean_object* v___x_43_; lean_object* v___x_44_; 
v___x_42_ = 0;
v___x_43_ = lean_box(v___x_42_);
v___x_44_ = lean_apply_2(v_toPure_39_, lean_box(0), v___x_43_);
return v___x_44_;
}
else
{
lean_object* v_val_45_; lean_object* v___x_46_; 
v_val_45_ = lean_ctor_get(v_fst_41_, 0);
lean_inc(v_val_45_);
lean_dec_ref_known(v_fst_41_, 1);
v___x_46_ = lean_apply_2(v_toPure_39_, lean_box(0), v_val_45_);
return v___x_46_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_HashSet_anyM___redArg(lean_object* v_inst_50_, lean_object* v_s_51_, lean_object* v_f_52_){
_start:
{
lean_object* v_toApplicative_53_; lean_object* v_toBind_54_; lean_object* v_toPure_55_; lean_object* v_buckets_56_; lean_object* v___x_57_; lean_object* v___x_58_; lean_object* v___f_59_; lean_object* v___f_60_; lean_object* v___f_61_; lean_object* v___f_62_; size_t v_sz_63_; size_t v___x_64_; lean_object* v___x_65_; lean_object* v___x_66_; 
v_toApplicative_53_ = lean_ctor_get(v_inst_50_, 0);
v_toBind_54_ = lean_ctor_get(v_inst_50_, 1);
lean_inc_n(v_toBind_54_, 2);
v_toPure_55_ = lean_ctor_get(v_toApplicative_53_, 1);
v_buckets_56_ = lean_ctor_get(v_s_51_, 1);
lean_inc_ref(v_buckets_56_);
lean_dec_ref(v_s_51_);
v___x_57_ = lean_box(0);
v___x_58_ = ((lean_object*)(lp_batteries_Std_HashSet_anyM___redArg___closed__0));
lean_inc_n(v_toPure_55_, 2);
v___f_59_ = lean_alloc_closure((void*)(lp_batteries_Std_HashSet_anyM___redArg___lam__0___boxed), 4, 3);
lean_closure_set(v___f_59_, 0, v___x_58_);
lean_closure_set(v___f_59_, 1, v_toPure_55_);
lean_closure_set(v___f_59_, 2, v___x_57_);
v___f_60_ = lean_alloc_closure((void*)(lp_batteries_Std_HashSet_anyM___redArg___lam__1___boxed), 6, 3);
lean_closure_set(v___f_60_, 0, v_f_52_);
lean_closure_set(v___f_60_, 1, v_toBind_54_);
lean_closure_set(v___f_60_, 2, v___f_59_);
lean_inc_ref(v_inst_50_);
v___f_61_ = lean_alloc_closure((void*)(lp_batteries_Std_HashSet_anyM___redArg___lam__2), 5, 2);
lean_closure_set(v___f_61_, 0, v_inst_50_);
lean_closure_set(v___f_61_, 1, v___f_60_);
v___f_62_ = lean_alloc_closure((void*)(lp_batteries_Std_HashSet_anyM___redArg___lam__3), 2, 1);
lean_closure_set(v___f_62_, 0, v_toPure_55_);
v_sz_63_ = lean_array_size(v_buckets_56_);
v___x_64_ = ((size_t)0ULL);
v___x_65_ = l___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop(lean_box(0), lean_box(0), lean_box(0), v_inst_50_, v_buckets_56_, v___f_61_, v_sz_63_, v___x_64_, v___x_58_);
v___x_66_ = lean_apply_4(v_toBind_54_, lean_box(0), lean_box(0), v___x_65_, v___f_62_);
return v___x_66_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_HashSet_anyM(lean_object* v_00_u03b1_67_, lean_object* v_inst_68_, lean_object* v_inst_69_, lean_object* v_m_70_, lean_object* v_inst_71_, lean_object* v_s_72_, lean_object* v_f_73_){
_start:
{
lean_object* v___x_74_; 
v___x_74_ = lp_batteries_Std_HashSet_anyM___redArg(v_inst_71_, v_s_72_, v_f_73_);
return v___x_74_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_HashSet_anyM___boxed(lean_object* v_00_u03b1_75_, lean_object* v_inst_76_, lean_object* v_inst_77_, lean_object* v_m_78_, lean_object* v_inst_79_, lean_object* v_s_80_, lean_object* v_f_81_){
_start:
{
lean_object* v_res_82_; 
v_res_82_ = lp_batteries_Std_HashSet_anyM(v_00_u03b1_75_, v_inst_76_, v_inst_77_, v_m_78_, v_inst_79_, v_s_80_, v_f_81_);
lean_dec_ref(v_inst_77_);
lean_dec_ref(v_inst_76_);
return v_res_82_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_HashSet_allM___redArg___lam__0(lean_object* v___x_83_, lean_object* v_toPure_84_, lean_object* v___x_85_, uint8_t v_____do__lift_86_){
_start:
{
if (v_____do__lift_86_ == 0)
{
lean_object* v___x_87_; lean_object* v___x_88_; lean_object* v___x_89_; lean_object* v___x_90_; lean_object* v___x_91_; 
lean_dec_ref(v___x_85_);
v___x_87_ = lean_box(v_____do__lift_86_);
v___x_88_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_88_, 0, v___x_87_);
v___x_89_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_89_, 0, v___x_88_);
lean_ctor_set(v___x_89_, 1, v___x_83_);
v___x_90_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_90_, 0, v___x_89_);
v___x_91_ = lean_apply_2(v_toPure_84_, lean_box(0), v___x_90_);
return v___x_91_;
}
else
{
lean_object* v___x_92_; lean_object* v___x_93_; 
v___x_92_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_92_, 0, v___x_85_);
v___x_93_ = lean_apply_2(v_toPure_84_, lean_box(0), v___x_92_);
return v___x_93_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_HashSet_allM___redArg___lam__0___boxed(lean_object* v___x_94_, lean_object* v_toPure_95_, lean_object* v___x_96_, lean_object* v_____do__lift_97_){
_start:
{
uint8_t v_____do__lift_200__boxed_98_; lean_object* v_res_99_; 
v_____do__lift_200__boxed_98_ = lean_unbox(v_____do__lift_97_);
v_res_99_ = lp_batteries_Std_HashSet_allM___redArg___lam__0(v___x_94_, v_toPure_95_, v___x_96_, v_____do__lift_200__boxed_98_);
return v_res_99_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_HashSet_allM___redArg___lam__3(lean_object* v_toPure_100_, lean_object* v_____s_101_){
_start:
{
lean_object* v_fst_102_; 
v_fst_102_ = lean_ctor_get(v_____s_101_, 0);
lean_inc(v_fst_102_);
lean_dec_ref(v_____s_101_);
if (lean_obj_tag(v_fst_102_) == 0)
{
uint8_t v___x_103_; lean_object* v___x_104_; lean_object* v___x_105_; 
v___x_103_ = 1;
v___x_104_ = lean_box(v___x_103_);
v___x_105_ = lean_apply_2(v_toPure_100_, lean_box(0), v___x_104_);
return v___x_105_;
}
else
{
lean_object* v_val_106_; lean_object* v___x_107_; 
v_val_106_ = lean_ctor_get(v_fst_102_, 0);
lean_inc(v_val_106_);
lean_dec_ref_known(v_fst_102_, 1);
v___x_107_ = lean_apply_2(v_toPure_100_, lean_box(0), v_val_106_);
return v___x_107_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_HashSet_allM___redArg(lean_object* v_inst_108_, lean_object* v_s_109_, lean_object* v_f_110_){
_start:
{
lean_object* v_toApplicative_111_; lean_object* v_toBind_112_; lean_object* v_toPure_113_; lean_object* v_buckets_114_; lean_object* v___x_115_; lean_object* v___x_116_; lean_object* v___f_117_; lean_object* v___f_118_; lean_object* v___f_119_; lean_object* v___f_120_; size_t v_sz_121_; size_t v___x_122_; lean_object* v___x_123_; lean_object* v___x_124_; 
v_toApplicative_111_ = lean_ctor_get(v_inst_108_, 0);
v_toBind_112_ = lean_ctor_get(v_inst_108_, 1);
lean_inc_n(v_toBind_112_, 2);
v_toPure_113_ = lean_ctor_get(v_toApplicative_111_, 1);
v_buckets_114_ = lean_ctor_get(v_s_109_, 1);
lean_inc_ref(v_buckets_114_);
lean_dec_ref(v_s_109_);
v___x_115_ = lean_box(0);
v___x_116_ = ((lean_object*)(lp_batteries_Std_HashSet_anyM___redArg___closed__0));
lean_inc_n(v_toPure_113_, 2);
v___f_117_ = lean_alloc_closure((void*)(lp_batteries_Std_HashSet_allM___redArg___lam__0___boxed), 4, 3);
lean_closure_set(v___f_117_, 0, v___x_115_);
lean_closure_set(v___f_117_, 1, v_toPure_113_);
lean_closure_set(v___f_117_, 2, v___x_116_);
v___f_118_ = lean_alloc_closure((void*)(lp_batteries_Std_HashSet_anyM___redArg___lam__1___boxed), 6, 3);
lean_closure_set(v___f_118_, 0, v_f_110_);
lean_closure_set(v___f_118_, 1, v_toBind_112_);
lean_closure_set(v___f_118_, 2, v___f_117_);
lean_inc_ref(v_inst_108_);
v___f_119_ = lean_alloc_closure((void*)(lp_batteries_Std_HashSet_anyM___redArg___lam__2), 5, 2);
lean_closure_set(v___f_119_, 0, v_inst_108_);
lean_closure_set(v___f_119_, 1, v___f_118_);
v___f_120_ = lean_alloc_closure((void*)(lp_batteries_Std_HashSet_allM___redArg___lam__3), 2, 1);
lean_closure_set(v___f_120_, 0, v_toPure_113_);
v_sz_121_ = lean_array_size(v_buckets_114_);
v___x_122_ = ((size_t)0ULL);
v___x_123_ = l___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop(lean_box(0), lean_box(0), lean_box(0), v_inst_108_, v_buckets_114_, v___f_119_, v_sz_121_, v___x_122_, v___x_116_);
v___x_124_ = lean_apply_4(v_toBind_112_, lean_box(0), lean_box(0), v___x_123_, v___f_120_);
return v___x_124_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_HashSet_allM(lean_object* v_00_u03b1_125_, lean_object* v_inst_126_, lean_object* v_inst_127_, lean_object* v_m_128_, lean_object* v_inst_129_, lean_object* v_s_130_, lean_object* v_f_131_){
_start:
{
lean_object* v___x_132_; 
v___x_132_ = lp_batteries_Std_HashSet_allM___redArg(v_inst_129_, v_s_130_, v_f_131_);
return v___x_132_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_HashSet_allM___boxed(lean_object* v_00_u03b1_133_, lean_object* v_inst_134_, lean_object* v_inst_135_, lean_object* v_m_136_, lean_object* v_inst_137_, lean_object* v_s_138_, lean_object* v_f_139_){
_start:
{
lean_object* v_res_140_; 
v_res_140_ = lp_batteries_Std_HashSet_allM(v_00_u03b1_133_, v_inst_134_, v_inst_135_, v_m_136_, v_inst_137_, v_s_138_, v_f_139_);
lean_dec_ref(v_inst_135_);
lean_dec_ref(v_inst_134_);
return v_res_140_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_HashSet_instBEq__batteries___redArg___lam__0(lean_object* v_inst_141_, lean_object* v_inst_142_, lean_object* v_t_143_, lean_object* v___x_144_, lean_object* v___x_145_, lean_object* v_a_146_, lean_object* v_b_147_, lean_object* v_acc_148_){
_start:
{
uint8_t v___x_149_; 
v___x_149_ = l_Std_DHashMap_Internal_Raw_u2080_contains___redArg(v_inst_141_, v_inst_142_, v_t_143_, v_a_146_);
if (v___x_149_ == 0)
{
lean_object* v___x_150_; lean_object* v___x_151_; lean_object* v___x_152_; lean_object* v___x_153_; 
lean_dec_ref(v___x_145_);
v___x_150_ = lean_box(v___x_149_);
v___x_151_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_151_, 0, v___x_150_);
v___x_152_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_152_, 0, v___x_151_);
lean_ctor_set(v___x_152_, 1, v___x_144_);
v___x_153_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_153_, 0, v___x_152_);
return v___x_153_;
}
else
{
lean_object* v___x_154_; 
v___x_154_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_154_, 0, v___x_145_);
return v___x_154_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_HashSet_instBEq__batteries___redArg___lam__0___boxed(lean_object* v_inst_155_, lean_object* v_inst_156_, lean_object* v_t_157_, lean_object* v___x_158_, lean_object* v___x_159_, lean_object* v_a_160_, lean_object* v_b_161_, lean_object* v_acc_162_){
_start:
{
lean_object* v_res_163_; 
v_res_163_ = lp_batteries_Std_HashSet_instBEq__batteries___redArg___lam__0(v_inst_155_, v_inst_156_, v_t_157_, v___x_158_, v___x_159_, v_a_160_, v_b_161_, v_acc_162_);
lean_dec_ref(v_acc_162_);
lean_dec_ref(v_t_157_);
return v_res_163_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_HashSet_instBEq__batteries___redArg___lam__1(lean_object* v___x_164_, lean_object* v___f_165_, lean_object* v_a_166_, lean_object* v_x_167_, lean_object* v___y_168_){
_start:
{
lean_object* v___x_169_; 
v___x_169_ = l___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go(lean_box(0), lean_box(0), lean_box(0), lean_box(0), v___x_164_, v___f_165_, v_a_166_, v___y_168_);
return v___x_169_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_HashSet_instBEq__batteries___redArg___lam__2(lean_object* v_inst_170_, lean_object* v_inst_171_, lean_object* v_s_172_, lean_object* v___x_173_, lean_object* v___x_174_, lean_object* v_a_175_, lean_object* v_b_176_, lean_object* v_acc_177_){
_start:
{
uint8_t v___x_178_; 
v___x_178_ = l_Std_DHashMap_Internal_Raw_u2080_contains___redArg(v_inst_170_, v_inst_171_, v_s_172_, v_a_175_);
if (v___x_178_ == 0)
{
lean_object* v___x_179_; lean_object* v___x_180_; lean_object* v___x_181_; lean_object* v___x_182_; 
lean_dec_ref(v___x_174_);
v___x_179_ = lean_box(v___x_178_);
v___x_180_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_180_, 0, v___x_179_);
v___x_181_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_181_, 0, v___x_180_);
lean_ctor_set(v___x_181_, 1, v___x_173_);
v___x_182_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_182_, 0, v___x_181_);
return v___x_182_;
}
else
{
lean_object* v___x_183_; 
v___x_183_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_183_, 0, v___x_174_);
return v___x_183_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_HashSet_instBEq__batteries___redArg___lam__2___boxed(lean_object* v_inst_184_, lean_object* v_inst_185_, lean_object* v_s_186_, lean_object* v___x_187_, lean_object* v___x_188_, lean_object* v_a_189_, lean_object* v_b_190_, lean_object* v_acc_191_){
_start:
{
lean_object* v_res_192_; 
v_res_192_ = lp_batteries_Std_HashSet_instBEq__batteries___redArg___lam__2(v_inst_184_, v_inst_185_, v_s_186_, v___x_187_, v___x_188_, v_a_189_, v_b_190_, v_acc_191_);
lean_dec_ref(v_acc_191_);
lean_dec_ref(v_s_186_);
return v_res_192_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Std_HashSet_instBEq__batteries___redArg___lam__4(lean_object* v_inst_212_, lean_object* v_inst_213_, lean_object* v_s_214_, lean_object* v_t_215_){
_start:
{
lean_object* v___x_230_; lean_object* v_buckets_231_; lean_object* v___x_232_; lean_object* v___x_233_; lean_object* v___f_234_; lean_object* v___f_235_; size_t v_sz_236_; size_t v___x_237_; lean_object* v___x_238_; lean_object* v_fst_239_; 
v___x_230_ = ((lean_object*)(lp_batteries_Std_HashSet_instBEq__batteries___redArg___lam__4___closed__9));
v_buckets_231_ = lean_ctor_get(v_s_214_, 1);
v___x_232_ = lean_box(0);
v___x_233_ = ((lean_object*)(lp_batteries_Std_HashSet_anyM___redArg___closed__0));
lean_inc_ref(v_t_215_);
lean_inc_ref(v_inst_213_);
lean_inc_ref(v_inst_212_);
v___f_234_ = lean_alloc_closure((void*)(lp_batteries_Std_HashSet_instBEq__batteries___redArg___lam__0___boxed), 8, 5);
lean_closure_set(v___f_234_, 0, v_inst_212_);
lean_closure_set(v___f_234_, 1, v_inst_213_);
lean_closure_set(v___f_234_, 2, v_t_215_);
lean_closure_set(v___f_234_, 3, v___x_232_);
lean_closure_set(v___f_234_, 4, v___x_233_);
v___f_235_ = lean_alloc_closure((void*)(lp_batteries_Std_HashSet_instBEq__batteries___redArg___lam__1), 5, 2);
lean_closure_set(v___f_235_, 0, v___x_230_);
lean_closure_set(v___f_235_, 1, v___f_234_);
v_sz_236_ = lean_array_size(v_buckets_231_);
v___x_237_ = ((size_t)0ULL);
lean_inc_ref(v_buckets_231_);
v___x_238_ = l___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop(lean_box(0), lean_box(0), lean_box(0), v___x_230_, v_buckets_231_, v___f_235_, v_sz_236_, v___x_237_, v___x_233_);
v_fst_239_ = lean_ctor_get(v___x_238_, 0);
lean_inc(v_fst_239_);
lean_dec(v___x_238_);
if (lean_obj_tag(v_fst_239_) == 0)
{
goto v___jp_216_;
}
else
{
lean_object* v_val_240_; uint8_t v___x_241_; 
v_val_240_ = lean_ctor_get(v_fst_239_, 0);
lean_inc(v_val_240_);
lean_dec_ref_known(v_fst_239_, 1);
v___x_241_ = lean_unbox(v_val_240_);
if (v___x_241_ == 0)
{
uint8_t v___x_242_; 
lean_dec_ref(v_t_215_);
lean_dec_ref(v_s_214_);
lean_dec_ref(v_inst_213_);
lean_dec_ref(v_inst_212_);
v___x_242_ = lean_unbox(v_val_240_);
lean_dec(v_val_240_);
return v___x_242_;
}
else
{
lean_dec(v_val_240_);
goto v___jp_216_;
}
}
v___jp_216_:
{
lean_object* v___x_217_; lean_object* v_buckets_218_; lean_object* v___x_219_; lean_object* v___x_220_; lean_object* v___f_221_; lean_object* v___f_222_; size_t v_sz_223_; size_t v___x_224_; lean_object* v___x_225_; lean_object* v_fst_226_; 
v___x_217_ = ((lean_object*)(lp_batteries_Std_HashSet_instBEq__batteries___redArg___lam__4___closed__9));
v_buckets_218_ = lean_ctor_get(v_t_215_, 1);
lean_inc_ref(v_buckets_218_);
lean_dec_ref(v_t_215_);
v___x_219_ = lean_box(0);
v___x_220_ = ((lean_object*)(lp_batteries_Std_HashSet_anyM___redArg___closed__0));
v___f_221_ = lean_alloc_closure((void*)(lp_batteries_Std_HashSet_instBEq__batteries___redArg___lam__2___boxed), 8, 5);
lean_closure_set(v___f_221_, 0, v_inst_212_);
lean_closure_set(v___f_221_, 1, v_inst_213_);
lean_closure_set(v___f_221_, 2, v_s_214_);
lean_closure_set(v___f_221_, 3, v___x_219_);
lean_closure_set(v___f_221_, 4, v___x_220_);
v___f_222_ = lean_alloc_closure((void*)(lp_batteries_Std_HashSet_instBEq__batteries___redArg___lam__1), 5, 2);
lean_closure_set(v___f_222_, 0, v___x_217_);
lean_closure_set(v___f_222_, 1, v___f_221_);
v_sz_223_ = lean_array_size(v_buckets_218_);
v___x_224_ = ((size_t)0ULL);
v___x_225_ = l___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop(lean_box(0), lean_box(0), lean_box(0), v___x_217_, v_buckets_218_, v___f_222_, v_sz_223_, v___x_224_, v___x_220_);
v_fst_226_ = lean_ctor_get(v___x_225_, 0);
lean_inc(v_fst_226_);
lean_dec(v___x_225_);
if (lean_obj_tag(v_fst_226_) == 0)
{
uint8_t v___x_227_; 
v___x_227_ = 1;
return v___x_227_;
}
else
{
lean_object* v_val_228_; uint8_t v___x_229_; 
v_val_228_ = lean_ctor_get(v_fst_226_, 0);
lean_inc(v_val_228_);
lean_dec_ref_known(v_fst_226_, 1);
v___x_229_ = lean_unbox(v_val_228_);
lean_dec(v_val_228_);
return v___x_229_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_HashSet_instBEq__batteries___redArg___lam__4___boxed(lean_object* v_inst_243_, lean_object* v_inst_244_, lean_object* v_s_245_, lean_object* v_t_246_){
_start:
{
uint8_t v_res_247_; lean_object* v_r_248_; 
v_res_247_ = lp_batteries_Std_HashSet_instBEq__batteries___redArg___lam__4(v_inst_243_, v_inst_244_, v_s_245_, v_t_246_);
v_r_248_ = lean_box(v_res_247_);
return v_r_248_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_HashSet_instBEq__batteries___redArg(lean_object* v_inst_249_, lean_object* v_inst_250_){
_start:
{
lean_object* v___f_251_; 
v___f_251_ = lean_alloc_closure((void*)(lp_batteries_Std_HashSet_instBEq__batteries___redArg___lam__4___boxed), 4, 2);
lean_closure_set(v___f_251_, 0, v_inst_249_);
lean_closure_set(v___f_251_, 1, v_inst_250_);
return v___f_251_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_HashSet_instBEq__batteries(lean_object* v_00_u03b1_252_, lean_object* v_inst_253_, lean_object* v_inst_254_){
_start:
{
lean_object* v___f_255_; 
v___f_255_ = lean_alloc_closure((void*)(lp_batteries_Std_HashSet_instBEq__batteries___redArg___lam__4___boxed), 4, 2);
lean_closure_set(v___f_255_, 0, v_inst_253_);
lean_closure_set(v___f_255_, 1, v_inst_254_);
return v___f_255_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Std_Data_HashSet_Basic(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_batteries_Batteries_Lean_HashSet(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize_runtime_module();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Std_Data_HashSet_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_batteries_Batteries_Lean_HashSet(uint8_t builtin) {
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
lean_object* initialize_Std_Data_HashSet_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_batteries_Batteries_Lean_HashSet(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Std_Data_HashSet_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Lean_HashSet(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_batteries_Batteries_Lean_HashSet(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_batteries_Batteries_Lean_HashSet(builtin);
}
#ifdef __cplusplus
}
#endif
