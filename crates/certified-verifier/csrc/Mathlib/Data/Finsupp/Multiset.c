// Lean compiler output
// Module: Mathlib.Data.Finsupp.Multiset
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Order.Group.Finset public import Mathlib.Data.Finsupp.Basic public import Mathlib.Data.Sym.Basic public import Mathlib.Order.Preorder.Finsupp
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
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* l_List_appendTR___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_map___redArg(lean_object*, lean_object*);
lean_object* l_List_foldrTR___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_count___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_List_dedup___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nsmulRec___at___00Finsupp_toMultiset_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nsmulRec___at___00Finsupp_toMultiset_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_toMultiset___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_toMultiset___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_sum___at___00Finsupp_toMultiset_spec__1___redArg___lam__0(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Multiset_sum___at___00Finset_sum___at___00Finsupp_sum___at___00Finsupp_toMultiset_spec__1_spec__1_spec__2___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_List_appendTR___redArg, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Multiset_sum___at___00Finset_sum___at___00Finsupp_sum___at___00Finsupp_toMultiset_spec__1_spec__1_spec__2___redArg___closed__0 = (const lean_object*)&lp_mathlib_Multiset_sum___at___00Finset_sum___at___00Finsupp_sum___at___00Finsupp_toMultiset_spec__1_spec__1_spec__2___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Multiset_sum___at___00Finset_sum___at___00Finsupp_sum___at___00Finsupp_toMultiset_spec__1_spec__1_spec__2___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_sum___at___00Finsupp_sum___at___00Finsupp_toMultiset_spec__1_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_sum___at___00Finsupp_toMultiset_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_toMultiset___lam__1(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Finsupp_toMultiset___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Finsupp_toMultiset___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Finsupp_toMultiset___closed__0 = (const lean_object*)&lp_mathlib_Finsupp_toMultiset___closed__0_value;
static const lean_closure_object lp_mathlib_Finsupp_toMultiset___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Finsupp_toMultiset___lam__1, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_Finsupp_toMultiset___closed__0_value)} };
static const lean_object* lp_mathlib_Finsupp_toMultiset___closed__1 = (const lean_object*)&lp_mathlib_Finsupp_toMultiset___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_toMultiset(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nsmulRec___at___00Finsupp_toMultiset_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nsmulRec___at___00Finsupp_toMultiset_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_sum___at___00Finsupp_toMultiset_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_sum___at___00Finset_sum___at___00Finsupp_sum___at___00Finsupp_toMultiset_spec__1_spec__1_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_sum___at___00Finsupp_sum___at___00Finsupp_toMultiset_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_toFinsupp___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_toFinsupp___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_toFinsupp___redArg___lam__2(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Multiset_toFinsupp___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Multiset_toFinsupp___redArg___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Multiset_toFinsupp___redArg___closed__0 = (const lean_object*)&lp_mathlib_Multiset_toFinsupp___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Multiset_toFinsupp___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_toFinsupp(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_instWellFoundedRelationNat(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_nsmulRec___at___00Finsupp_toMultiset_spec__0___redArg(lean_object* v_x_1_, lean_object* v_x_2_){
_start:
{
lean_object* v_zero_3_; uint8_t v_isZero_4_; 
v_zero_3_ = lean_unsigned_to_nat(0u);
v_isZero_4_ = lean_nat_dec_eq(v_x_1_, v_zero_3_);
if (v_isZero_4_ == 1)
{
lean_object* v___x_5_; 
lean_dec(v_x_2_);
v___x_5_ = lean_box(0);
return v___x_5_;
}
else
{
lean_object* v_one_6_; lean_object* v_n_7_; lean_object* v___x_8_; lean_object* v___x_9_; 
v_one_6_ = lean_unsigned_to_nat(1u);
v_n_7_ = lean_nat_sub(v_x_1_, v_one_6_);
lean_inc(v_x_2_);
v___x_8_ = lp_mathlib_nsmulRec___at___00Finsupp_toMultiset_spec__0___redArg(v_n_7_, v_x_2_);
lean_dec(v_n_7_);
v___x_9_ = l_List_appendTR___redArg(v___x_8_, v_x_2_);
return v___x_9_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_nsmulRec___at___00Finsupp_toMultiset_spec__0___redArg___boxed(lean_object* v_x_10_, lean_object* v_x_11_){
_start:
{
lean_object* v_res_12_; 
v_res_12_ = lp_mathlib_nsmulRec___at___00Finsupp_toMultiset_spec__0___redArg(v_x_10_, v_x_11_);
lean_dec(v_x_10_);
return v_res_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_toMultiset___lam__0(lean_object* v_a_13_, lean_object* v_n_14_){
_start:
{
lean_object* v___x_15_; lean_object* v___x_16_; lean_object* v___x_17_; 
v___x_15_ = lean_box(0);
v___x_16_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_16_, 0, v_a_13_);
lean_ctor_set(v___x_16_, 1, v___x_15_);
v___x_17_ = lp_mathlib_nsmulRec___at___00Finsupp_toMultiset_spec__0___redArg(v_n_14_, v___x_16_);
return v___x_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_toMultiset___lam__0___boxed(lean_object* v_a_18_, lean_object* v_n_19_){
_start:
{
lean_object* v_res_20_; 
v_res_20_ = lp_mathlib_Finsupp_toMultiset___lam__0(v_a_18_, v_n_19_);
lean_dec(v_n_19_);
return v_res_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_sum___at___00Finsupp_toMultiset_spec__1___redArg___lam__0(lean_object* v_toFun_21_, lean_object* v_g_22_, lean_object* v_a_23_){
_start:
{
lean_object* v___x_24_; lean_object* v___x_25_; 
lean_inc(v_a_23_);
v___x_24_ = lean_apply_1(v_toFun_21_, v_a_23_);
v___x_25_ = lean_apply_2(v_g_22_, v_a_23_, v___x_24_);
return v___x_25_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_sum___at___00Finset_sum___at___00Finsupp_sum___at___00Finsupp_toMultiset_spec__1_spec__1_spec__2___redArg(lean_object* v_s_27_){
_start:
{
lean_object* v___f_28_; lean_object* v___x_29_; lean_object* v___x_30_; 
v___f_28_ = ((lean_object*)(lp_mathlib_Multiset_sum___at___00Finset_sum___at___00Finsupp_sum___at___00Finsupp_toMultiset_spec__1_spec__1_spec__2___redArg___closed__0));
v___x_29_ = lean_box(0);
v___x_30_ = l_List_foldrTR___redArg(v___f_28_, v___x_29_, v_s_27_);
return v___x_30_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_sum___at___00Finsupp_sum___at___00Finsupp_toMultiset_spec__1_spec__1___redArg(lean_object* v_s_31_, lean_object* v_f_32_){
_start:
{
lean_object* v___x_33_; lean_object* v___x_34_; 
v___x_33_ = lp_mathlib_Multiset_map___redArg(v_f_32_, v_s_31_);
v___x_34_ = lp_mathlib_Multiset_sum___at___00Finset_sum___at___00Finsupp_sum___at___00Finsupp_toMultiset_spec__1_spec__1_spec__2___redArg(v___x_33_);
return v___x_34_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_sum___at___00Finsupp_toMultiset_spec__1___redArg(lean_object* v_f_35_, lean_object* v_g_36_){
_start:
{
lean_object* v_support_37_; lean_object* v_toFun_38_; lean_object* v___f_39_; lean_object* v___x_40_; 
v_support_37_ = lean_ctor_get(v_f_35_, 0);
lean_inc(v_support_37_);
v_toFun_38_ = lean_ctor_get(v_f_35_, 1);
lean_inc(v_toFun_38_);
lean_dec_ref(v_f_35_);
v___f_39_ = lean_alloc_closure((void*)(lp_mathlib_Finsupp_sum___at___00Finsupp_toMultiset_spec__1___redArg___lam__0), 3, 2);
lean_closure_set(v___f_39_, 0, v_toFun_38_);
lean_closure_set(v___f_39_, 1, v_g_36_);
v___x_40_ = lp_mathlib_Finset_sum___at___00Finsupp_sum___at___00Finsupp_toMultiset_spec__1_spec__1___redArg(v_support_37_, v___f_39_);
return v___x_40_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_toMultiset___lam__1(lean_object* v___f_41_, lean_object* v_f_42_){
_start:
{
lean_object* v___x_43_; 
v___x_43_ = lp_mathlib_Finsupp_sum___at___00Finsupp_toMultiset_spec__1___redArg(v_f_42_, v___f_41_);
return v___x_43_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_toMultiset(lean_object* v_00_u03b1_47_){
_start:
{
lean_object* v___f_48_; 
v___f_48_ = ((lean_object*)(lp_mathlib_Finsupp_toMultiset___closed__1));
return v___f_48_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nsmulRec___at___00Finsupp_toMultiset_spec__0(lean_object* v_00_u03b1_49_, lean_object* v_x_50_, lean_object* v_x_51_){
_start:
{
lean_object* v___x_52_; 
v___x_52_ = lp_mathlib_nsmulRec___at___00Finsupp_toMultiset_spec__0___redArg(v_x_50_, v_x_51_);
return v___x_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_nsmulRec___at___00Finsupp_toMultiset_spec__0___boxed(lean_object* v_00_u03b1_53_, lean_object* v_x_54_, lean_object* v_x_55_){
_start:
{
lean_object* v_res_56_; 
v_res_56_ = lp_mathlib_nsmulRec___at___00Finsupp_toMultiset_spec__0(v_00_u03b1_53_, v_x_54_, v_x_55_);
lean_dec(v_x_54_);
return v_res_56_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_sum___at___00Finsupp_toMultiset_spec__1(lean_object* v_00_u03b1_57_, lean_object* v_00_u03b1_58_, lean_object* v_f_59_, lean_object* v_g_60_){
_start:
{
lean_object* v___x_61_; 
v___x_61_ = lp_mathlib_Finsupp_sum___at___00Finsupp_toMultiset_spec__1___redArg(v_f_59_, v_g_60_);
return v___x_61_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_sum___at___00Finset_sum___at___00Finsupp_sum___at___00Finsupp_toMultiset_spec__1_spec__1_spec__2(lean_object* v_00_u03b1_62_, lean_object* v_s_63_){
_start:
{
lean_object* v___x_64_; 
v___x_64_ = lp_mathlib_Multiset_sum___at___00Finset_sum___at___00Finsupp_sum___at___00Finsupp_toMultiset_spec__1_spec__1_spec__2___redArg(v_s_63_);
return v___x_64_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_sum___at___00Finsupp_sum___at___00Finsupp_toMultiset_spec__1_spec__1(lean_object* v_00_u03b1_65_, lean_object* v_00_u03b9_66_, lean_object* v_s_67_, lean_object* v_f_68_){
_start:
{
lean_object* v___x_69_; 
v___x_69_ = lp_mathlib_Finset_sum___at___00Finsupp_sum___at___00Finsupp_toMultiset_spec__1_spec__1___redArg(v_s_67_, v_f_68_);
return v___x_69_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_toFinsupp___redArg___lam__0(lean_object* v_f_70_){
_start:
{
lean_object* v___x_39__overap_71_; lean_object* v___x_72_; 
v___x_39__overap_71_ = lp_mathlib_Finsupp_toMultiset(lean_box(0));
v___x_72_ = lean_apply_1(v___x_39__overap_71_, v_f_70_);
return v___x_72_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_toFinsupp___redArg___lam__1(lean_object* v_inst_73_, lean_object* v_s_74_, lean_object* v_a_75_){
_start:
{
lean_object* v___x_76_; 
v___x_76_ = lp_mathlib_Multiset_count___redArg(v_inst_73_, v_a_75_, v_s_74_);
return v___x_76_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_toFinsupp___redArg___lam__2(lean_object* v_inst_77_, lean_object* v_s_78_){
_start:
{
lean_object* v___f_79_; lean_object* v___x_80_; lean_object* v___x_81_; 
lean_inc(v_s_78_);
lean_inc_ref(v_inst_77_);
v___f_79_ = lean_alloc_closure((void*)(lp_mathlib_Multiset_toFinsupp___redArg___lam__1), 3, 2);
lean_closure_set(v___f_79_, 0, v_inst_77_);
lean_closure_set(v___f_79_, 1, v_s_78_);
v___x_80_ = lp_mathlib_List_dedup___redArg(v_inst_77_, v_s_78_);
v___x_81_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_81_, 0, v___x_80_);
lean_ctor_set(v___x_81_, 1, v___f_79_);
return v___x_81_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_toFinsupp___redArg(lean_object* v_inst_83_){
_start:
{
lean_object* v___f_84_; lean_object* v___f_85_; lean_object* v___x_86_; 
v___f_84_ = ((lean_object*)(lp_mathlib_Multiset_toFinsupp___redArg___closed__0));
v___f_85_ = lean_alloc_closure((void*)(lp_mathlib_Multiset_toFinsupp___redArg___lam__2), 2, 1);
lean_closure_set(v___f_85_, 0, v_inst_83_);
v___x_86_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_86_, 0, v___f_85_);
lean_ctor_set(v___x_86_, 1, v___f_84_);
return v___x_86_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_toFinsupp(lean_object* v_00_u03b1_87_, lean_object* v_inst_88_){
_start:
{
lean_object* v___x_89_; 
v___x_89_ = lp_mathlib_Multiset_toFinsupp___redArg(v_inst_88_);
return v___x_89_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_instWellFoundedRelationNat(lean_object* v_00_u03b9_90_){
_start:
{
lean_object* v___x_91_; 
v___x_91_ = lean_box(0);
return v___x_91_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Group_Finset(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finsupp_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Sym_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Preorder_Finsupp(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Finsupp_Multiset(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Group_Finset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finsupp_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Sym_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Preorder_Finsupp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Finsupp_Multiset(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Group_Finset(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finsupp_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Sym_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Preorder_Finsupp(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Finsupp_Multiset(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Group_Finset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finsupp_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Sym_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Preorder_Finsupp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finsupp_Multiset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Finsupp_Multiset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Finsupp_Multiset(builtin);
}
#ifdef __cplusplus
}
#endif
