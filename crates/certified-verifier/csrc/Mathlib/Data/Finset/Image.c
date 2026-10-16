// Lean compiler output
// Module: Mathlib.Data.Finset.Image
// Imports: public import Init public meta import Init public import Mathlib.Algebra.NeZero public import Mathlib.Data.Finset.Attach public import Mathlib.Data.Finset.Disjoint public import Mathlib.Data.Finset.Erase public import Mathlib.Data.Finset.Filter public import Mathlib.Data.Finset.Range public import Mathlib.Data.Finset.Lattice.Lemmas public import Mathlib.Data.Finset.SDiff public import Mathlib.Data.Fintype.Defs
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
lean_object* lp_mathlib_Equiv_subtypeEquiv___redArg(lean_object*);
lean_object* lp_mathlib_Multiset_map___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_toEmbedding___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_Subtype_impEmbedding___lam__0___boxed(lean_object*);
lean_object* lp_mathlib_Multiset_attach___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_subtypeEquivProp(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_filter___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_filterMap___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_List_dedup___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_map___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_map___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_map(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_mapEmbedding___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_mapEmbedding(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_image___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_image(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_filterMap___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_filterMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_subtype___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_subtype___redArg___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Finset_subtype___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Finset_subtype___redArg___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Finset_subtype___redArg___closed__0 = (const lean_object*)&lp_mathlib_Finset_subtype___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Finset_subtype___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_subtype(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_imageFinset___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_imageFinset(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_imageFinset___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_finsetCongr___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_finsetCongr___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_finsetCongr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_finsetCongr(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_finsetSubtypeComm___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_finsetSubtypeComm___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_finsetSubtypeComm___lam__1(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Equiv_finsetSubtypeComm___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Subtype_impEmbedding___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_finsetSubtypeComm___lam__2___closed__0 = (const lean_object*)&lp_mathlib_Equiv_finsetSubtypeComm___lam__2___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_finsetSubtypeComm___lam__2(lean_object*);
static const lean_closure_object lp_mathlib_Equiv_finsetSubtypeComm___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_finsetSubtypeComm___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_finsetSubtypeComm___closed__0 = (const lean_object*)&lp_mathlib_Equiv_finsetSubtypeComm___closed__0_value;
static const lean_closure_object lp_mathlib_Equiv_finsetSubtypeComm___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_finsetSubtypeComm___lam__1, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_Equiv_finsetSubtypeComm___closed__0_value)} };
static const lean_object* lp_mathlib_Equiv_finsetSubtypeComm___closed__1 = (const lean_object*)&lp_mathlib_Equiv_finsetSubtypeComm___closed__1_value;
static const lean_closure_object lp_mathlib_Equiv_finsetSubtypeComm___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Equiv_finsetSubtypeComm___lam__2, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Equiv_finsetSubtypeComm___closed__2 = (const lean_object*)&lp_mathlib_Equiv_finsetSubtypeComm___closed__2_value;
static const lean_ctor_object lp_mathlib_Equiv_finsetSubtypeComm___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Equiv_finsetSubtypeComm___closed__1_value),((lean_object*)&lp_mathlib_Equiv_finsetSubtypeComm___closed__2_value)}};
static const lean_object* lp_mathlib_Equiv_finsetSubtypeComm___closed__3 = (const lean_object*)&lp_mathlib_Equiv_finsetSubtypeComm___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_finsetSubtypeComm(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Finset_equivOfEq___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finset_equivOfEq___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Finset_equivOfEq(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_equivOfEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Finset_congr(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Finset_congr___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_map___redArg___lam__0(lean_object* v_f_1_, lean_object* v___y_2_){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lean_apply_1(v_f_1_, v___y_2_);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_map___redArg(lean_object* v_f_4_, lean_object* v_s_5_){
_start:
{
lean_object* v___f_6_; lean_object* v___x_7_; 
v___f_6_ = lean_alloc_closure((void*)(lp_mathlib_Finset_map___redArg___lam__0), 2, 1);
lean_closure_set(v___f_6_, 0, v_f_4_);
v___x_7_ = lp_mathlib_Multiset_map___redArg(v___f_6_, v_s_5_);
return v___x_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_map(lean_object* v_00_u03b1_8_, lean_object* v_00_u03b2_9_, lean_object* v_f_10_, lean_object* v_s_11_){
_start:
{
lean_object* v___x_12_; 
v___x_12_ = lp_mathlib_Finset_map___redArg(v_f_10_, v_s_11_);
return v___x_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_mapEmbedding___redArg(lean_object* v_f_13_){
_start:
{
lean_object* v___x_14_; 
v___x_14_ = lean_alloc_closure((void*)(lp_mathlib_Finset_map), 4, 3);
lean_closure_set(v___x_14_, 0, lean_box(0));
lean_closure_set(v___x_14_, 1, lean_box(0));
lean_closure_set(v___x_14_, 2, v_f_13_);
return v___x_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_mapEmbedding(lean_object* v_00_u03b1_15_, lean_object* v_00_u03b2_16_, lean_object* v_f_17_){
_start:
{
lean_object* v___x_18_; 
v___x_18_ = lean_alloc_closure((void*)(lp_mathlib_Finset_map), 4, 3);
lean_closure_set(v___x_18_, 0, lean_box(0));
lean_closure_set(v___x_18_, 1, lean_box(0));
lean_closure_set(v___x_18_, 2, v_f_17_);
return v___x_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_image___redArg(lean_object* v_inst_19_, lean_object* v_f_20_, lean_object* v_s_21_){
_start:
{
lean_object* v___x_22_; lean_object* v___x_23_; 
v___x_22_ = lp_mathlib_Multiset_map___redArg(v_f_20_, v_s_21_);
v___x_23_ = lp_mathlib_List_dedup___redArg(v_inst_19_, v___x_22_);
return v___x_23_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_image(lean_object* v_00_u03b1_24_, lean_object* v_00_u03b2_25_, lean_object* v_inst_26_, lean_object* v_f_27_, lean_object* v_s_28_){
_start:
{
lean_object* v___x_29_; 
v___x_29_ = lp_mathlib_Finset_image___redArg(v_inst_26_, v_f_27_, v_s_28_);
return v___x_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_filterMap___redArg(lean_object* v_f_30_, lean_object* v_s_31_){
_start:
{
lean_object* v___x_32_; 
v___x_32_ = lp_mathlib_Multiset_filterMap___redArg(v_f_30_, v_s_31_);
return v___x_32_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_filterMap(lean_object* v_00_u03b1_33_, lean_object* v_00_u03b2_34_, lean_object* v_f_35_, lean_object* v_s_36_, lean_object* v_f__inj_37_){
_start:
{
lean_object* v___x_38_; 
v___x_38_ = lp_mathlib_Multiset_filterMap___redArg(v_f_35_, v_s_36_);
return v___x_38_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_subtype___redArg___lam__0(lean_object* v_x_39_){
_start:
{
lean_inc(v_x_39_);
return v_x_39_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_subtype___redArg___lam__0___boxed(lean_object* v_x_40_){
_start:
{
lean_object* v_res_41_; 
v_res_41_ = lp_mathlib_Finset_subtype___redArg___lam__0(v_x_40_);
lean_dec(v_x_40_);
return v_res_41_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_subtype___redArg(lean_object* v_inst_43_, lean_object* v_s_44_){
_start:
{
lean_object* v___f_45_; lean_object* v___x_46_; lean_object* v___x_47_; lean_object* v___x_48_; 
v___f_45_ = ((lean_object*)(lp_mathlib_Finset_subtype___redArg___closed__0));
v___x_46_ = lp_mathlib_Multiset_filter___redArg(v_inst_43_, v_s_44_);
v___x_47_ = lp_mathlib_Multiset_attach___redArg(v___x_46_);
v___x_48_ = lp_mathlib_Finset_map___redArg(v___f_45_, v___x_47_);
return v___x_48_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_subtype(lean_object* v_00_u03b1_49_, lean_object* v_p_50_, lean_object* v_inst_51_, lean_object* v_s_52_){
_start:
{
lean_object* v___x_53_; 
v___x_53_ = lp_mathlib_Finset_subtype___redArg(v_inst_51_, v_s_52_);
return v___x_53_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_imageFinset___redArg(lean_object* v_e_54_){
_start:
{
lean_object* v___x_55_; 
v___x_55_ = lp_mathlib_Equiv_subtypeEquiv___redArg(v_e_54_);
return v___x_55_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_imageFinset(lean_object* v_00_u03b1_56_, lean_object* v_00_u03b2_57_, lean_object* v_inst_58_, lean_object* v_e_59_, lean_object* v_s_60_){
_start:
{
lean_object* v___x_61_; 
v___x_61_ = lp_mathlib_Equiv_subtypeEquiv___redArg(v_e_59_);
return v___x_61_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_imageFinset___boxed(lean_object* v_00_u03b1_62_, lean_object* v_00_u03b2_63_, lean_object* v_inst_64_, lean_object* v_e_65_, lean_object* v_s_66_){
_start:
{
lean_object* v_res_67_; 
v_res_67_ = lp_mathlib_Equiv_imageFinset(v_00_u03b1_62_, v_00_u03b2_63_, v_inst_64_, v_e_65_, v_s_66_);
lean_dec(v_s_66_);
lean_dec_ref(v_inst_64_);
return v_res_67_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_finsetCongr___redArg___lam__0(lean_object* v_e_68_, lean_object* v_s_69_){
_start:
{
lean_object* v___f_70_; lean_object* v___x_71_; 
v___f_70_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_toEmbedding___redArg___lam__0), 2, 1);
lean_closure_set(v___f_70_, 0, v_e_68_);
v___x_71_ = lp_mathlib_Finset_map___redArg(v___f_70_, v_s_69_);
return v___x_71_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_finsetCongr___redArg___lam__1(lean_object* v_e_72_, lean_object* v_s_73_){
_start:
{
lean_object* v___x_74_; lean_object* v___f_75_; lean_object* v___x_76_; 
v___x_74_ = lp_mathlib_Equiv_symm___redArg(v_e_72_);
v___f_75_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_toEmbedding___redArg___lam__0), 2, 1);
lean_closure_set(v___f_75_, 0, v___x_74_);
v___x_76_ = lp_mathlib_Finset_map___redArg(v___f_75_, v_s_73_);
return v___x_76_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_finsetCongr___redArg(lean_object* v_e_77_){
_start:
{
lean_object* v___f_78_; lean_object* v___f_79_; lean_object* v___x_80_; 
lean_inc_ref(v_e_77_);
v___f_78_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_finsetCongr___redArg___lam__0), 2, 1);
lean_closure_set(v___f_78_, 0, v_e_77_);
v___f_79_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_finsetCongr___redArg___lam__1), 2, 1);
lean_closure_set(v___f_79_, 0, v_e_77_);
v___x_80_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_80_, 0, v___f_78_);
lean_ctor_set(v___x_80_, 1, v___f_79_);
return v___x_80_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_finsetCongr(lean_object* v_00_u03b1_81_, lean_object* v_00_u03b2_82_, lean_object* v_e_83_){
_start:
{
lean_object* v___x_84_; 
v___x_84_ = lp_mathlib_Equiv_finsetCongr___redArg(v_e_83_);
return v___x_84_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_finsetSubtypeComm___lam__0(lean_object* v_a_85_){
_start:
{
lean_inc(v_a_85_);
return v_a_85_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_finsetSubtypeComm___lam__0___boxed(lean_object* v_a_86_){
_start:
{
lean_object* v_res_87_; 
v_res_87_ = lp_mathlib_Equiv_finsetSubtypeComm___lam__0(v_a_86_);
lean_dec(v_a_86_);
return v_res_87_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_finsetSubtypeComm___lam__1(lean_object* v___f_88_, lean_object* v_s_89_){
_start:
{
lean_object* v___x_90_; 
v___x_90_ = lp_mathlib_Finset_map___redArg(v___f_88_, v_s_89_);
return v___x_90_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_finsetSubtypeComm___lam__2(lean_object* v_s_92_){
_start:
{
lean_object* v___f_93_; lean_object* v___x_94_; lean_object* v___x_95_; 
v___f_93_ = ((lean_object*)(lp_mathlib_Equiv_finsetSubtypeComm___lam__2___closed__0));
v___x_94_ = lp_mathlib_Multiset_attach___redArg(v_s_92_);
v___x_95_ = lp_mathlib_Finset_map___redArg(v___f_93_, v___x_94_);
return v___x_95_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_finsetSubtypeComm(lean_object* v_00_u03b1_103_, lean_object* v_p_104_){
_start:
{
lean_object* v___x_105_; 
v___x_105_ = ((lean_object*)(lp_mathlib_Equiv_finsetSubtypeComm___closed__3));
return v___x_105_;
}
}
static lean_object* _init_lp_mathlib_Finset_equivOfEq___closed__0(void){
_start:
{
lean_object* v___x_106_; 
v___x_106_ = lp_mathlib_Equiv_subtypeEquivProp(lean_box(0), lean_box(0), lean_box(0), lean_box(0));
return v___x_106_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_equivOfEq(lean_object* v_00_u03b1_107_, lean_object* v_s_108_, lean_object* v_t_109_, lean_object* v_h_110_){
_start:
{
lean_object* v___x_111_; 
v___x_111_ = lean_obj_once(&lp_mathlib_Finset_equivOfEq___closed__0, &lp_mathlib_Finset_equivOfEq___closed__0_once, _init_lp_mathlib_Finset_equivOfEq___closed__0);
return v___x_111_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_equivOfEq___boxed(lean_object* v_00_u03b1_112_, lean_object* v_s_113_, lean_object* v_t_114_, lean_object* v_h_115_){
_start:
{
lean_object* v_res_116_; 
v_res_116_ = lp_mathlib_Finset_equivOfEq(v_00_u03b1_112_, v_s_113_, v_t_114_, v_h_115_);
lean_dec(v_t_114_);
lean_dec(v_s_113_);
return v_res_116_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Finset_congr(lean_object* v_00_u03b1_117_, lean_object* v_s_118_, lean_object* v_t_119_, lean_object* v_h_120_){
_start:
{
lean_object* v___x_121_; 
v___x_121_ = lean_obj_once(&lp_mathlib_Finset_equivOfEq___closed__0, &lp_mathlib_Finset_equivOfEq___closed__0_once, _init_lp_mathlib_Finset_equivOfEq___closed__0);
return v___x_121_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Finset_congr___boxed(lean_object* v_00_u03b1_122_, lean_object* v_s_123_, lean_object* v_t_124_, lean_object* v_h_125_){
_start:
{
lean_object* v_res_126_; 
v_res_126_ = lp_mathlib_Equiv_Finset_congr(v_00_u03b1_122_, v_s_123_, v_t_124_, v_h_125_);
lean_dec(v_t_124_);
lean_dec(v_s_123_);
return v_res_126_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_NeZero(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Attach(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Disjoint(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Erase(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Filter(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Range(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Lattice_Lemmas(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_SDiff(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Fintype_Defs(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Image(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_NeZero(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Attach(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Disjoint(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Erase(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Filter(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Range(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Lattice_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_SDiff(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fintype_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Finset_Image(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_NeZero(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Attach(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Disjoint(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Erase(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Filter(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Range(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Lattice_Lemmas(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_SDiff(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Finset_Image(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_NeZero(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Attach(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Disjoint(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Erase(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Filter(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Range(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Lattice_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_SDiff(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Image(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Finset_Image(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Finset_Image(builtin);
}
#ifdef __cplusplus
}
#endif
