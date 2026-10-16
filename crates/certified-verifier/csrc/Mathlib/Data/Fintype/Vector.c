// Lean compiler output
// Module: Mathlib.Data.Fintype.Vector
// Imports: public import Init public meta import Init public import Mathlib.Basic.Finite.Prod public import Mathlib.Data.Fintype.Pi public import Mathlib.Data.Sym.Basic
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
uint8_t l_List_decidablePerm___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_subtypeQuotientEquivQuotientSubtype___at___00Sym_symEquivSym_x27_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_Quotient_mk_x27_x27___boxed(lean_object*, lean_object*, lean_object*);
lean_object* l_List_finRange(lean_object*);
lean_object* lp_mathlib_Equiv_vectorEquivFin___redArg(lean_object*);
lean_object* l_instDecidableEqFin___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Fintype_piFinset___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Finset_map___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Finset_image___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_instFintype___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_instFintype___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_instFintype___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_instFintype___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_instFintype(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Sym_Sym_x27_instFintype___aux__1___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_Sym_x27_instFintype___aux__1___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Sym_Sym_x27_instFintype___aux__1___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Quotient_mk_x27_x27___boxed, .m_arity = 3, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Sym_Sym_x27_instFintype___aux__1___redArg___closed__0 = (const lean_object*)&lp_mathlib_Sym_Sym_x27_instFintype___aux__1___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Sym_Sym_x27_instFintype___aux__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_Sym_x27_instFintype___aux__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_Sym_x27_instFintype___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_Sym_x27_instFintype(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_instFintype___redArg___lam__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_instFintype___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym_instFintype(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_instFintype___redArg___lam__0(lean_object* v_inst_1_, lean_object* v_x_2_){
_start:
{
lean_inc(v_inst_1_);
return v_inst_1_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_instFintype___redArg___lam__0___boxed(lean_object* v_inst_3_, lean_object* v_x_4_){
_start:
{
lean_object* v_res_5_; 
v_res_5_ = lp_mathlib_List_Vector_instFintype___redArg___lam__0(v_inst_3_, v_x_4_);
lean_dec(v_x_4_);
lean_dec(v_inst_3_);
return v_res_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_instFintype___redArg___lam__1(lean_object* v___x_6_, lean_object* v___y_7_){
_start:
{
lean_object* v_toFun_8_; lean_object* v___x_9_; 
v_toFun_8_ = lean_ctor_get(v___x_6_, 0);
lean_inc(v_toFun_8_);
lean_dec_ref(v___x_6_);
v___x_9_ = lean_apply_1(v_toFun_8_, v___y_7_);
return v___x_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_instFintype___redArg(lean_object* v_inst_10_, lean_object* v_n_11_){
_start:
{
lean_object* v___f_12_; lean_object* v___x_13_; lean_object* v___x_14_; lean_object* v___x_15_; lean_object* v___f_16_; lean_object* v___x_17_; lean_object* v___x_18_; lean_object* v___x_19_; 
v___f_12_ = lean_alloc_closure((void*)(lp_mathlib_List_Vector_instFintype___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_12_, 0, v_inst_10_);
lean_inc_n(v_n_11_, 2);
v___x_13_ = l_List_finRange(v_n_11_);
v___x_14_ = lp_mathlib_Equiv_vectorEquivFin___redArg(v_n_11_);
v___x_15_ = lp_mathlib_Equiv_symm___redArg(v___x_14_);
v___f_16_ = lean_alloc_closure((void*)(lp_mathlib_List_Vector_instFintype___redArg___lam__1), 2, 1);
lean_closure_set(v___f_16_, 0, v___x_15_);
v___x_17_ = lean_alloc_closure((void*)(l_instDecidableEqFin___boxed), 3, 1);
lean_closure_set(v___x_17_, 0, v_n_11_);
v___x_18_ = lp_mathlib_Fintype_piFinset___redArg(v___x_17_, v___x_13_, v___f_12_);
v___x_19_ = lp_mathlib_Finset_map___redArg(v___f_16_, v___x_18_);
return v___x_19_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Vector_instFintype(lean_object* v_00_u03b1_20_, lean_object* v_inst_21_, lean_object* v_n_22_){
_start:
{
lean_object* v___x_23_; 
v___x_23_ = lp_mathlib_List_Vector_instFintype___redArg(v_inst_21_, v_n_22_);
return v___x_23_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Sym_Sym_x27_instFintype___aux__1___redArg___lam__0(lean_object* v_inst_24_, lean_object* v_a_25_, lean_object* v_b_26_){
_start:
{
uint8_t v___x_27_; 
v___x_27_ = l_List_decidablePerm___redArg(v_inst_24_, v_a_25_, v_b_26_);
return v___x_27_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_Sym_x27_instFintype___aux__1___redArg___lam__0___boxed(lean_object* v_inst_28_, lean_object* v_a_29_, lean_object* v_b_30_){
_start:
{
uint8_t v_res_31_; lean_object* v_r_32_; 
v_res_31_ = lp_mathlib_Sym_Sym_x27_instFintype___aux__1___redArg___lam__0(v_inst_28_, v_a_29_, v_b_30_);
v_r_32_ = lean_box(v_res_31_);
return v_r_32_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_Sym_x27_instFintype___aux__1___redArg(lean_object* v_inst_35_, lean_object* v_inst_36_, lean_object* v_n_37_){
_start:
{
lean_object* v___f_38_; lean_object* v___f_39_; lean_object* v___x_40_; lean_object* v___x_41_; lean_object* v___x_42_; lean_object* v___x_43_; lean_object* v___f_44_; lean_object* v___x_45_; lean_object* v___x_46_; lean_object* v___x_47_; lean_object* v___x_48_; 
v___f_38_ = lean_alloc_closure((void*)(lp_mathlib_Sym_Sym_x27_instFintype___aux__1___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_38_, 0, v_inst_35_);
v___f_39_ = lean_alloc_closure((void*)(lp_mathlib_List_Vector_instFintype___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_39_, 0, v_inst_36_);
v___x_40_ = ((lean_object*)(lp_mathlib_Sym_Sym_x27_instFintype___aux__1___redArg___closed__0));
lean_inc_n(v_n_37_, 2);
v___x_41_ = l_List_finRange(v_n_37_);
v___x_42_ = lp_mathlib_Equiv_vectorEquivFin___redArg(v_n_37_);
v___x_43_ = lp_mathlib_Equiv_symm___redArg(v___x_42_);
v___f_44_ = lean_alloc_closure((void*)(lp_mathlib_List_Vector_instFintype___redArg___lam__1), 2, 1);
lean_closure_set(v___f_44_, 0, v___x_43_);
v___x_45_ = lean_alloc_closure((void*)(l_instDecidableEqFin___boxed), 3, 1);
lean_closure_set(v___x_45_, 0, v_n_37_);
v___x_46_ = lp_mathlib_Fintype_piFinset___redArg(v___x_45_, v___x_41_, v___f_39_);
v___x_47_ = lp_mathlib_Finset_map___redArg(v___f_44_, v___x_46_);
v___x_48_ = lp_mathlib_Finset_image___redArg(v___f_38_, v___x_40_, v___x_47_);
return v___x_48_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_Sym_x27_instFintype___aux__1(lean_object* v_00_u03b1_49_, lean_object* v_inst_50_, lean_object* v_inst_51_, lean_object* v_n_52_){
_start:
{
lean_object* v___f_53_; lean_object* v___f_54_; lean_object* v___x_55_; lean_object* v___x_56_; lean_object* v___x_57_; lean_object* v___x_58_; lean_object* v___f_59_; lean_object* v___x_60_; lean_object* v___x_61_; lean_object* v___x_62_; lean_object* v___x_63_; 
v___f_53_ = lean_alloc_closure((void*)(lp_mathlib_Sym_Sym_x27_instFintype___aux__1___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_53_, 0, v_inst_50_);
v___f_54_ = lean_alloc_closure((void*)(lp_mathlib_List_Vector_instFintype___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_54_, 0, v_inst_51_);
v___x_55_ = ((lean_object*)(lp_mathlib_Sym_Sym_x27_instFintype___aux__1___redArg___closed__0));
lean_inc_n(v_n_52_, 2);
v___x_56_ = l_List_finRange(v_n_52_);
v___x_57_ = lp_mathlib_Equiv_vectorEquivFin___redArg(v_n_52_);
v___x_58_ = lp_mathlib_Equiv_symm___redArg(v___x_57_);
v___f_59_ = lean_alloc_closure((void*)(lp_mathlib_List_Vector_instFintype___redArg___lam__1), 2, 1);
lean_closure_set(v___f_59_, 0, v___x_58_);
v___x_60_ = lean_alloc_closure((void*)(l_instDecidableEqFin___boxed), 3, 1);
lean_closure_set(v___x_60_, 0, v_n_52_);
v___x_61_ = lp_mathlib_Fintype_piFinset___redArg(v___x_60_, v___x_56_, v___f_54_);
v___x_62_ = lp_mathlib_Finset_map___redArg(v___f_59_, v___x_61_);
v___x_63_ = lp_mathlib_Finset_image___redArg(v___f_53_, v___x_55_, v___x_62_);
return v___x_63_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_Sym_x27_instFintype___redArg(lean_object* v_inst_64_, lean_object* v_inst_65_, lean_object* v_n_66_){
_start:
{
lean_object* v___f_67_; lean_object* v___f_68_; lean_object* v___x_69_; lean_object* v___x_70_; lean_object* v___x_71_; lean_object* v___x_72_; lean_object* v___f_73_; lean_object* v___x_74_; lean_object* v___x_75_; lean_object* v___x_76_; lean_object* v___x_77_; 
v___f_67_ = lean_alloc_closure((void*)(lp_mathlib_Sym_Sym_x27_instFintype___aux__1___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_67_, 0, v_inst_64_);
v___f_68_ = lean_alloc_closure((void*)(lp_mathlib_List_Vector_instFintype___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_68_, 0, v_inst_65_);
v___x_69_ = ((lean_object*)(lp_mathlib_Sym_Sym_x27_instFintype___aux__1___redArg___closed__0));
lean_inc_n(v_n_66_, 2);
v___x_70_ = l_List_finRange(v_n_66_);
v___x_71_ = lp_mathlib_Equiv_vectorEquivFin___redArg(v_n_66_);
v___x_72_ = lp_mathlib_Equiv_symm___redArg(v___x_71_);
v___f_73_ = lean_alloc_closure((void*)(lp_mathlib_List_Vector_instFintype___redArg___lam__1), 2, 1);
lean_closure_set(v___f_73_, 0, v___x_72_);
v___x_74_ = lean_alloc_closure((void*)(l_instDecidableEqFin___boxed), 3, 1);
lean_closure_set(v___x_74_, 0, v_n_66_);
v___x_75_ = lp_mathlib_Fintype_piFinset___redArg(v___x_74_, v___x_70_, v___f_68_);
v___x_76_ = lp_mathlib_Finset_map___redArg(v___f_73_, v___x_75_);
v___x_77_ = lp_mathlib_Finset_image___redArg(v___f_67_, v___x_69_, v___x_76_);
return v___x_77_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_Sym_x27_instFintype(lean_object* v_00_u03b1_78_, lean_object* v_inst_79_, lean_object* v_inst_80_, lean_object* v_n_81_){
_start:
{
lean_object* v___x_82_; 
v___x_82_ = lp_mathlib_Sym_Sym_x27_instFintype___redArg(v_inst_79_, v_inst_80_, v_n_81_);
return v___x_82_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_instFintype___redArg___lam__2(lean_object* v___x_83_, lean_object* v___y_84_){
_start:
{
lean_object* v_toFun_85_; lean_object* v___x_86_; 
v_toFun_85_ = lean_ctor_get(v___x_83_, 0);
lean_inc(v_toFun_85_);
lean_dec_ref(v___x_83_);
v___x_86_ = lean_apply_1(v_toFun_85_, v___y_84_);
return v___x_86_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_instFintype___redArg(lean_object* v_inst_87_, lean_object* v_inst_88_, lean_object* v_n_89_){
_start:
{
lean_object* v___f_90_; lean_object* v___f_91_; lean_object* v___x_92_; lean_object* v___x_93_; lean_object* v___f_94_; lean_object* v___x_95_; lean_object* v___x_96_; lean_object* v___x_97_; lean_object* v___x_98_; lean_object* v___f_99_; lean_object* v___x_100_; lean_object* v___x_101_; lean_object* v___x_102_; lean_object* v___x_103_; lean_object* v___x_104_; 
v___f_90_ = lean_alloc_closure((void*)(lp_mathlib_List_Vector_instFintype___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_90_, 0, v_inst_88_);
v___f_91_ = lean_alloc_closure((void*)(lp_mathlib_Sym_Sym_x27_instFintype___aux__1___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_91_, 0, v_inst_87_);
v___x_92_ = lp_mathlib_Equiv_subtypeQuotientEquivQuotientSubtype___at___00Sym_symEquivSym_x27_spec__0(lean_box(0), v_n_89_, lean_box(0), lean_box(0), lean_box(0), lean_box(0));
v___x_93_ = lp_mathlib_Equiv_symm___redArg(v___x_92_);
v___f_94_ = lean_alloc_closure((void*)(lp_mathlib_Sym_instFintype___redArg___lam__2), 2, 1);
lean_closure_set(v___f_94_, 0, v___x_93_);
v___x_95_ = ((lean_object*)(lp_mathlib_Sym_Sym_x27_instFintype___aux__1___redArg___closed__0));
lean_inc_n(v_n_89_, 2);
v___x_96_ = l_List_finRange(v_n_89_);
v___x_97_ = lp_mathlib_Equiv_vectorEquivFin___redArg(v_n_89_);
v___x_98_ = lp_mathlib_Equiv_symm___redArg(v___x_97_);
v___f_99_ = lean_alloc_closure((void*)(lp_mathlib_List_Vector_instFintype___redArg___lam__1), 2, 1);
lean_closure_set(v___f_99_, 0, v___x_98_);
v___x_100_ = lean_alloc_closure((void*)(l_instDecidableEqFin___boxed), 3, 1);
lean_closure_set(v___x_100_, 0, v_n_89_);
v___x_101_ = lp_mathlib_Fintype_piFinset___redArg(v___x_100_, v___x_96_, v___f_90_);
v___x_102_ = lp_mathlib_Finset_map___redArg(v___f_99_, v___x_101_);
v___x_103_ = lp_mathlib_Finset_image___redArg(v___f_91_, v___x_95_, v___x_102_);
v___x_104_ = lp_mathlib_Finset_map___redArg(v___f_94_, v___x_103_);
return v___x_104_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym_instFintype(lean_object* v_00_u03b1_105_, lean_object* v_inst_106_, lean_object* v_inst_107_, lean_object* v_n_108_){
_start:
{
lean_object* v___x_109_; 
v___x_109_ = lp_mathlib_Sym_instFintype___redArg(v_inst_106_, v_inst_107_, v_n_108_);
return v___x_109_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Basic_Finite_Prod(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Fintype_Pi(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Sym_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Fintype_Vector(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Basic_Finite_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fintype_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Sym_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Fintype_Vector(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Basic_Finite_Prod(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_Pi(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Sym_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Fintype_Vector(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Basic_Finite_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Sym_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fintype_Vector(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Fintype_Vector(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Fintype_Vector(builtin);
}
#ifdef __cplusplus
}
#endif
