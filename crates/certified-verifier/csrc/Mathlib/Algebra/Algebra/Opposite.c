// Lean compiler output
// Module: Mathlib.Algebra.Algebra.Opposite
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Algebra.Equiv public import Mathlib.Algebra.Module.Opposite public import Mathlib.Algebra.Ring.Opposite
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
lean_object* lp_mathlib_MulOpposite_instSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_RingHom_toOpposite___redArg(lean_object*);
lean_object* lp_mathlib_instDistribOfSemiring___redArg(lean_object*);
lean_object* lp_mathlib_Semiring_toNonUnitalSemiring___redArg(lean_object*);
lean_object* lp_mathlib_RingEquiv_op___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_RingEquiv_unop___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_RingEquiv_moduleEndSelf___redArg(lean_object*);
lean_object* lp_mathlib_MulOpposite_unop___boxed(lean_object*, lean_object*);
lean_object* l_Function_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_MulOpposite_op___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_MulOpposite_instSemiring___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_refl(lean_object*);
lean_object* lp_mathlib_MulEquiv_opOp(lean_object*, lean_object*);
lean_object* lp_mathlib_AlgEquiv_equivCongr___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_trans___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Semiring_toNonAssocSemiring___redArg(lean_object*);
lean_object* lp_mathlib_RingHom_op___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_RingHom_unop___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_RingEquiv_moduleEndSelfOp___redArg(lean_object*);
lean_object* lp_mathlib_MulOpposite_opEquiv(lean_object*);
lean_object* lp_mathlib_AlgEquiv_arrowCongr___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instAlgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instAlgebra(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instAlgebra___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_opOp___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_opOp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_opOp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_fromOpposite___redArg___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_AlgHom_fromOpposite___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_MulOpposite_unop___boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_AlgHom_fromOpposite___redArg___closed__0 = (const lean_object*)&lp_mathlib_AlgHom_fromOpposite___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_fromOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_fromOpposite(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_fromOpposite___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_AlgHom_toOpposite___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_MulOpposite_op___boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_AlgHom_toOpposite___redArg___closed__0 = (const lean_object*)&lp_mathlib_AlgHom_toOpposite___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_toOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_toOpposite(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_toOpposite___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_op___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_op___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_op___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_op(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_op___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_unop___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_unop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_unop___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_AlgHom_opComm___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AlgHom_opComm___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_opComm___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_opComm(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_opComm___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_op___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_op___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_op___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_op(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_op___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_unop___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_unop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_unop___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_opComm___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_opComm(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_opComm___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_moduleEndSelf___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_moduleEndSelf(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_moduleEndSelf___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_moduleEndSelfOp___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_moduleEndSelfOp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_moduleEndSelfOp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_AlgEquiv_toOpposite___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AlgEquiv_toOpposite___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_toOpposite(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_toOpposite___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instAlgebra___redArg(lean_object* v_inst_1_){
_start:
{
lean_object* v_toSMul_2_; lean_object* v_algebraMap_3_; lean_object* v___x_5_; uint8_t v_isShared_6_; uint8_t v_isSharedCheck_12_; 
v_toSMul_2_ = lean_ctor_get(v_inst_1_, 0);
v_algebraMap_3_ = lean_ctor_get(v_inst_1_, 1);
v_isSharedCheck_12_ = !lean_is_exclusive(v_inst_1_);
if (v_isSharedCheck_12_ == 0)
{
v___x_5_ = v_inst_1_;
v_isShared_6_ = v_isSharedCheck_12_;
goto v_resetjp_4_;
}
else
{
lean_inc(v_algebraMap_3_);
lean_inc(v_toSMul_2_);
lean_dec(v_inst_1_);
v___x_5_ = lean_box(0);
v_isShared_6_ = v_isSharedCheck_12_;
goto v_resetjp_4_;
}
v_resetjp_4_:
{
lean_object* v___f_7_; lean_object* v___x_8_; lean_object* v___x_10_; 
v___f_7_ = lean_alloc_closure((void*)(lp_mathlib_MulOpposite_instSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_7_, 0, v_toSMul_2_);
v___x_8_ = lp_mathlib_RingHom_toOpposite___redArg(v_algebraMap_3_);
if (v_isShared_6_ == 0)
{
lean_ctor_set(v___x_5_, 1, v___x_8_);
lean_ctor_set(v___x_5_, 0, v___f_7_);
v___x_10_ = v___x_5_;
goto v_reusejp_9_;
}
else
{
lean_object* v_reuseFailAlloc_11_; 
v_reuseFailAlloc_11_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_11_, 0, v___f_7_);
lean_ctor_set(v_reuseFailAlloc_11_, 1, v___x_8_);
v___x_10_ = v_reuseFailAlloc_11_;
goto v_reusejp_9_;
}
v_reusejp_9_:
{
return v___x_10_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instAlgebra(lean_object* v_R_13_, lean_object* v_A_14_, lean_object* v_inst_15_, lean_object* v_inst_16_, lean_object* v_inst_17_){
_start:
{
lean_object* v___x_18_; 
v___x_18_ = lp_mathlib_MulOpposite_instAlgebra___redArg(v_inst_17_);
return v___x_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulOpposite_instAlgebra___boxed(lean_object* v_R_19_, lean_object* v_A_20_, lean_object* v_inst_21_, lean_object* v_inst_22_, lean_object* v_inst_23_){
_start:
{
lean_object* v_res_24_; 
v_res_24_ = lp_mathlib_MulOpposite_instAlgebra(v_R_19_, v_A_20_, v_inst_21_, v_inst_22_, v_inst_23_);
lean_dec_ref(v_inst_22_);
lean_dec_ref(v_inst_21_);
return v_res_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_opOp___redArg(lean_object* v_inst_25_){
_start:
{
lean_object* v___x_26_; lean_object* v_toMul_27_; lean_object* v___x_28_; 
v___x_26_ = lp_mathlib_instDistribOfSemiring___redArg(v_inst_25_);
v_toMul_27_ = lean_ctor_get(v___x_26_, 0);
lean_inc(v_toMul_27_);
lean_dec_ref(v___x_26_);
v___x_28_ = lp_mathlib_MulEquiv_opOp(lean_box(0), v_toMul_27_);
lean_dec(v_toMul_27_);
return v___x_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_opOp(lean_object* v_R_29_, lean_object* v_A_30_, lean_object* v_inst_31_, lean_object* v_inst_32_, lean_object* v_inst_33_){
_start:
{
lean_object* v___x_34_; 
v___x_34_ = lp_mathlib_AlgEquiv_opOp___redArg(v_inst_32_);
return v___x_34_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_opOp___boxed(lean_object* v_R_35_, lean_object* v_A_36_, lean_object* v_inst_37_, lean_object* v_inst_38_, lean_object* v_inst_39_){
_start:
{
lean_object* v_res_40_; 
v_res_40_ = lp_mathlib_AlgEquiv_opOp(v_R_35_, v_A_36_, v_inst_37_, v_inst_38_, v_inst_39_);
lean_dec_ref(v_inst_39_);
lean_dec_ref(v_inst_37_);
return v_res_40_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_fromOpposite___redArg___lam__0(lean_object* v_f_41_, lean_object* v___y_42_){
_start:
{
lean_object* v___x_43_; 
v___x_43_ = lean_apply_1(v_f_41_, v___y_42_);
return v___x_43_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_fromOpposite___redArg(lean_object* v_f_45_){
_start:
{
lean_object* v___f_46_; lean_object* v___x_47_; lean_object* v___x_48_; 
v___f_46_ = lean_alloc_closure((void*)(lp_mathlib_AlgHom_fromOpposite___redArg___lam__0), 2, 1);
lean_closure_set(v___f_46_, 0, v_f_45_);
v___x_47_ = ((lean_object*)(lp_mathlib_AlgHom_fromOpposite___redArg___closed__0));
v___x_48_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_48_, 0, lean_box(0));
lean_closure_set(v___x_48_, 1, lean_box(0));
lean_closure_set(v___x_48_, 2, lean_box(0));
lean_closure_set(v___x_48_, 3, v___f_46_);
lean_closure_set(v___x_48_, 4, v___x_47_);
return v___x_48_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_fromOpposite(lean_object* v_R_49_, lean_object* v_A_50_, lean_object* v_B_51_, lean_object* v_inst_52_, lean_object* v_inst_53_, lean_object* v_inst_54_, lean_object* v_inst_55_, lean_object* v_inst_56_, lean_object* v_f_57_, lean_object* v_hf_58_){
_start:
{
lean_object* v___x_59_; 
v___x_59_ = lp_mathlib_AlgHom_fromOpposite___redArg(v_f_57_);
return v___x_59_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_fromOpposite___boxed(lean_object* v_R_60_, lean_object* v_A_61_, lean_object* v_B_62_, lean_object* v_inst_63_, lean_object* v_inst_64_, lean_object* v_inst_65_, lean_object* v_inst_66_, lean_object* v_inst_67_, lean_object* v_f_68_, lean_object* v_hf_69_){
_start:
{
lean_object* v_res_70_; 
v_res_70_ = lp_mathlib_AlgHom_fromOpposite(v_R_60_, v_A_61_, v_B_62_, v_inst_63_, v_inst_64_, v_inst_65_, v_inst_66_, v_inst_67_, v_f_68_, v_hf_69_);
lean_dec_ref(v_inst_67_);
lean_dec_ref(v_inst_66_);
lean_dec_ref(v_inst_65_);
lean_dec_ref(v_inst_64_);
lean_dec_ref(v_inst_63_);
return v_res_70_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_toOpposite___redArg(lean_object* v_f_72_){
_start:
{
lean_object* v___f_73_; lean_object* v___x_74_; lean_object* v___x_75_; 
v___f_73_ = lean_alloc_closure((void*)(lp_mathlib_AlgHom_fromOpposite___redArg___lam__0), 2, 1);
lean_closure_set(v___f_73_, 0, v_f_72_);
v___x_74_ = ((lean_object*)(lp_mathlib_AlgHom_toOpposite___redArg___closed__0));
v___x_75_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_75_, 0, lean_box(0));
lean_closure_set(v___x_75_, 1, lean_box(0));
lean_closure_set(v___x_75_, 2, lean_box(0));
lean_closure_set(v___x_75_, 3, v___x_74_);
lean_closure_set(v___x_75_, 4, v___f_73_);
return v___x_75_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_toOpposite(lean_object* v_R_76_, lean_object* v_A_77_, lean_object* v_B_78_, lean_object* v_inst_79_, lean_object* v_inst_80_, lean_object* v_inst_81_, lean_object* v_inst_82_, lean_object* v_inst_83_, lean_object* v_f_84_, lean_object* v_hf_85_){
_start:
{
lean_object* v___x_86_; 
v___x_86_ = lp_mathlib_AlgHom_toOpposite___redArg(v_f_84_);
return v___x_86_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_toOpposite___boxed(lean_object* v_R_87_, lean_object* v_A_88_, lean_object* v_B_89_, lean_object* v_inst_90_, lean_object* v_inst_91_, lean_object* v_inst_92_, lean_object* v_inst_93_, lean_object* v_inst_94_, lean_object* v_f_95_, lean_object* v_hf_96_){
_start:
{
lean_object* v_res_97_; 
v_res_97_ = lp_mathlib_AlgHom_toOpposite(v_R_87_, v_A_88_, v_B_89_, v_inst_90_, v_inst_91_, v_inst_92_, v_inst_93_, v_inst_94_, v_f_95_, v_hf_96_);
lean_dec_ref(v_inst_94_);
lean_dec_ref(v_inst_93_);
lean_dec_ref(v_inst_92_);
lean_dec_ref(v_inst_91_);
lean_dec_ref(v_inst_90_);
return v_res_97_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_op___redArg___lam__0(lean_object* v___x_98_, lean_object* v___x_99_, lean_object* v_f_100_, lean_object* v___y_101_){
_start:
{
lean_object* v___x_102_; lean_object* v_toFun_103_; lean_object* v___x_104_; 
v___x_102_ = lp_mathlib_RingHom_op___redArg(v___x_98_, v___x_99_);
v_toFun_103_ = lean_ctor_get(v___x_102_, 0);
lean_inc(v_toFun_103_);
lean_dec_ref(v___x_102_);
v___x_104_ = lean_apply_2(v_toFun_103_, v_f_100_, v___y_101_);
return v___x_104_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_op___redArg___lam__1(lean_object* v___x_105_, lean_object* v___x_106_, lean_object* v_f_107_, lean_object* v___y_108_){
_start:
{
lean_object* v___x_109_; lean_object* v_toFun_110_; lean_object* v___x_111_; 
v___x_109_ = lp_mathlib_RingHom_unop___redArg(v___x_105_, v___x_106_);
v_toFun_110_ = lean_ctor_get(v___x_109_, 0);
lean_inc(v_toFun_110_);
lean_dec_ref(v___x_109_);
v___x_111_ = lean_apply_2(v_toFun_110_, v_f_107_, v___y_108_);
return v___x_111_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_op___redArg(lean_object* v_inst_112_, lean_object* v_inst_113_){
_start:
{
lean_object* v___x_114_; lean_object* v___x_115_; lean_object* v___f_116_; lean_object* v___f_117_; lean_object* v___x_118_; 
v___x_114_ = lp_mathlib_Semiring_toNonAssocSemiring___redArg(v_inst_112_);
v___x_115_ = lp_mathlib_Semiring_toNonAssocSemiring___redArg(v_inst_113_);
lean_inc_ref(v___x_115_);
lean_inc_ref(v___x_114_);
v___f_116_ = lean_alloc_closure((void*)(lp_mathlib_AlgHom_op___redArg___lam__0), 4, 2);
lean_closure_set(v___f_116_, 0, v___x_114_);
lean_closure_set(v___f_116_, 1, v___x_115_);
v___f_117_ = lean_alloc_closure((void*)(lp_mathlib_AlgHom_op___redArg___lam__1), 4, 2);
lean_closure_set(v___f_117_, 0, v___x_114_);
lean_closure_set(v___f_117_, 1, v___x_115_);
v___x_118_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_118_, 0, v___f_116_);
lean_ctor_set(v___x_118_, 1, v___f_117_);
return v___x_118_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_op(lean_object* v_R_119_, lean_object* v_A_120_, lean_object* v_B_121_, lean_object* v_inst_122_, lean_object* v_inst_123_, lean_object* v_inst_124_, lean_object* v_inst_125_, lean_object* v_inst_126_){
_start:
{
lean_object* v___x_127_; 
v___x_127_ = lp_mathlib_AlgHom_op___redArg(v_inst_123_, v_inst_124_);
return v___x_127_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_op___boxed(lean_object* v_R_128_, lean_object* v_A_129_, lean_object* v_B_130_, lean_object* v_inst_131_, lean_object* v_inst_132_, lean_object* v_inst_133_, lean_object* v_inst_134_, lean_object* v_inst_135_){
_start:
{
lean_object* v_res_136_; 
v_res_136_ = lp_mathlib_AlgHom_op(v_R_128_, v_A_129_, v_B_130_, v_inst_131_, v_inst_132_, v_inst_133_, v_inst_134_, v_inst_135_);
lean_dec_ref(v_inst_135_);
lean_dec_ref(v_inst_134_);
lean_dec_ref(v_inst_131_);
return v_res_136_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_unop___redArg(lean_object* v_inst_137_, lean_object* v_inst_138_){
_start:
{
lean_object* v___x_139_; lean_object* v___x_140_; 
v___x_139_ = lp_mathlib_AlgHom_op___redArg(v_inst_137_, v_inst_138_);
v___x_140_ = lp_mathlib_Equiv_symm___redArg(v___x_139_);
return v___x_140_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_unop(lean_object* v_R_141_, lean_object* v_A_142_, lean_object* v_B_143_, lean_object* v_inst_144_, lean_object* v_inst_145_, lean_object* v_inst_146_, lean_object* v_inst_147_, lean_object* v_inst_148_){
_start:
{
lean_object* v___x_149_; lean_object* v___x_150_; 
v___x_149_ = lp_mathlib_AlgHom_op___redArg(v_inst_145_, v_inst_146_);
v___x_150_ = lp_mathlib_Equiv_symm___redArg(v___x_149_);
return v___x_150_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_unop___boxed(lean_object* v_R_151_, lean_object* v_A_152_, lean_object* v_B_153_, lean_object* v_inst_154_, lean_object* v_inst_155_, lean_object* v_inst_156_, lean_object* v_inst_157_, lean_object* v_inst_158_){
_start:
{
lean_object* v_res_159_; 
v_res_159_ = lp_mathlib_AlgHom_unop(v_R_151_, v_A_152_, v_B_153_, v_inst_154_, v_inst_155_, v_inst_156_, v_inst_157_, v_inst_158_);
lean_dec_ref(v_inst_158_);
lean_dec_ref(v_inst_157_);
lean_dec_ref(v_inst_154_);
return v_res_159_;
}
}
static lean_object* _init_lp_mathlib_AlgHom_opComm___redArg___closed__0(void){
_start:
{
lean_object* v___x_160_; 
v___x_160_ = lp_mathlib_Equiv_refl(lean_box(0));
return v___x_160_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_opComm___redArg(lean_object* v_inst_161_, lean_object* v_inst_162_){
_start:
{
lean_object* v___x_163_; lean_object* v___x_164_; lean_object* v___x_165_; lean_object* v___x_166_; lean_object* v___x_167_; lean_object* v___x_168_; lean_object* v___x_169_; 
lean_inc_ref(v_inst_162_);
v___x_163_ = lp_mathlib_MulOpposite_instSemiring___redArg(v_inst_162_);
v___x_164_ = lp_mathlib_AlgHom_op___redArg(v_inst_161_, v___x_163_);
v___x_165_ = lean_obj_once(&lp_mathlib_AlgHom_opComm___redArg___closed__0, &lp_mathlib_AlgHom_opComm___redArg___closed__0_once, _init_lp_mathlib_AlgHom_opComm___redArg___closed__0);
v___x_166_ = lp_mathlib_AlgEquiv_opOp___redArg(v_inst_162_);
v___x_167_ = lp_mathlib_Equiv_symm___redArg(v___x_166_);
v___x_168_ = lp_mathlib_AlgEquiv_arrowCongr___redArg(v___x_165_, v___x_167_);
v___x_169_ = lp_mathlib_Equiv_trans___redArg(v___x_164_, v___x_168_);
return v___x_169_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_opComm(lean_object* v_R_170_, lean_object* v_A_171_, lean_object* v_B_172_, lean_object* v_inst_173_, lean_object* v_inst_174_, lean_object* v_inst_175_, lean_object* v_inst_176_, lean_object* v_inst_177_){
_start:
{
lean_object* v___x_178_; 
v___x_178_ = lp_mathlib_AlgHom_opComm___redArg(v_inst_174_, v_inst_175_);
return v___x_178_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_opComm___boxed(lean_object* v_R_179_, lean_object* v_A_180_, lean_object* v_B_181_, lean_object* v_inst_182_, lean_object* v_inst_183_, lean_object* v_inst_184_, lean_object* v_inst_185_, lean_object* v_inst_186_){
_start:
{
lean_object* v_res_187_; 
v_res_187_ = lp_mathlib_AlgHom_opComm(v_R_179_, v_A_180_, v_B_181_, v_inst_182_, v_inst_183_, v_inst_184_, v_inst_185_, v_inst_186_);
lean_dec_ref(v_inst_186_);
lean_dec_ref(v_inst_185_);
lean_dec_ref(v_inst_182_);
return v_res_187_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_op___redArg___lam__0(lean_object* v_toAdd_188_, lean_object* v_toAdd_189_, lean_object* v_f_190_){
_start:
{
lean_object* v___x_191_; lean_object* v_toFun_192_; lean_object* v___x_193_; 
v___x_191_ = lp_mathlib_RingEquiv_op___redArg(v_toAdd_188_, v_toAdd_189_);
v_toFun_192_ = lean_ctor_get(v___x_191_, 0);
lean_inc(v_toFun_192_);
lean_dec_ref(v___x_191_);
v___x_193_ = lean_apply_1(v_toFun_192_, v_f_190_);
return v___x_193_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_op___redArg___lam__1(lean_object* v_toAdd_194_, lean_object* v_toAdd_195_, lean_object* v_f_196_){
_start:
{
lean_object* v___x_197_; lean_object* v_toFun_198_; lean_object* v___x_199_; 
v___x_197_ = lp_mathlib_RingEquiv_unop___redArg(v_toAdd_194_, v_toAdd_195_);
v_toFun_198_ = lean_ctor_get(v___x_197_, 0);
lean_inc(v_toFun_198_);
lean_dec_ref(v___x_197_);
v___x_199_ = lean_apply_1(v_toFun_198_, v_f_196_);
return v___x_199_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_op___redArg(lean_object* v_inst_200_, lean_object* v_inst_201_){
_start:
{
lean_object* v___x_202_; lean_object* v_toAdd_203_; lean_object* v___x_204_; lean_object* v_toAdd_205_; lean_object* v___x_206_; lean_object* v_toAddCommMonoid_207_; lean_object* v_toAdd_208_; lean_object* v___x_209_; lean_object* v_toAddCommMonoid_210_; lean_object* v___x_212_; uint8_t v_isShared_213_; uint8_t v_isSharedCheck_220_; 
lean_inc_ref(v_inst_200_);
v___x_202_ = lp_mathlib_instDistribOfSemiring___redArg(v_inst_200_);
v_toAdd_203_ = lean_ctor_get(v___x_202_, 1);
lean_inc(v_toAdd_203_);
lean_dec_ref(v___x_202_);
lean_inc_ref(v_inst_201_);
v___x_204_ = lp_mathlib_instDistribOfSemiring___redArg(v_inst_201_);
v_toAdd_205_ = lean_ctor_get(v___x_204_, 1);
lean_inc(v_toAdd_205_);
lean_dec_ref(v___x_204_);
v___x_206_ = lp_mathlib_Semiring_toNonUnitalSemiring___redArg(v_inst_200_);
lean_dec_ref(v_inst_200_);
v_toAddCommMonoid_207_ = lean_ctor_get(v___x_206_, 0);
lean_inc_ref(v_toAddCommMonoid_207_);
lean_dec_ref(v___x_206_);
v_toAdd_208_ = lean_ctor_get(v_toAddCommMonoid_207_, 1);
lean_inc(v_toAdd_208_);
lean_dec_ref(v_toAddCommMonoid_207_);
v___x_209_ = lp_mathlib_Semiring_toNonUnitalSemiring___redArg(v_inst_201_);
lean_dec_ref(v_inst_201_);
v_toAddCommMonoid_210_ = lean_ctor_get(v___x_209_, 0);
v_isSharedCheck_220_ = !lean_is_exclusive(v___x_209_);
if (v_isSharedCheck_220_ == 0)
{
lean_object* v_unused_221_; 
v_unused_221_ = lean_ctor_get(v___x_209_, 1);
lean_dec(v_unused_221_);
v___x_212_ = v___x_209_;
v_isShared_213_ = v_isSharedCheck_220_;
goto v_resetjp_211_;
}
else
{
lean_inc(v_toAddCommMonoid_210_);
lean_dec(v___x_209_);
v___x_212_ = lean_box(0);
v_isShared_213_ = v_isSharedCheck_220_;
goto v_resetjp_211_;
}
v_resetjp_211_:
{
lean_object* v_toAdd_214_; lean_object* v___f_215_; lean_object* v___f_216_; lean_object* v___x_218_; 
v_toAdd_214_ = lean_ctor_get(v_toAddCommMonoid_210_, 1);
lean_inc(v_toAdd_214_);
lean_dec_ref(v_toAddCommMonoid_210_);
v___f_215_ = lean_alloc_closure((void*)(lp_mathlib_AlgEquiv_op___redArg___lam__0), 3, 2);
lean_closure_set(v___f_215_, 0, v_toAdd_203_);
lean_closure_set(v___f_215_, 1, v_toAdd_205_);
v___f_216_ = lean_alloc_closure((void*)(lp_mathlib_AlgEquiv_op___redArg___lam__1), 3, 2);
lean_closure_set(v___f_216_, 0, v_toAdd_208_);
lean_closure_set(v___f_216_, 1, v_toAdd_214_);
if (v_isShared_213_ == 0)
{
lean_ctor_set(v___x_212_, 1, v___f_216_);
lean_ctor_set(v___x_212_, 0, v___f_215_);
v___x_218_ = v___x_212_;
goto v_reusejp_217_;
}
else
{
lean_object* v_reuseFailAlloc_219_; 
v_reuseFailAlloc_219_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_219_, 0, v___f_215_);
lean_ctor_set(v_reuseFailAlloc_219_, 1, v___f_216_);
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
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_op(lean_object* v_R_222_, lean_object* v_A_223_, lean_object* v_B_224_, lean_object* v_inst_225_, lean_object* v_inst_226_, lean_object* v_inst_227_, lean_object* v_inst_228_, lean_object* v_inst_229_){
_start:
{
lean_object* v___x_230_; 
v___x_230_ = lp_mathlib_AlgEquiv_op___redArg(v_inst_226_, v_inst_227_);
return v___x_230_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_op___boxed(lean_object* v_R_231_, lean_object* v_A_232_, lean_object* v_B_233_, lean_object* v_inst_234_, lean_object* v_inst_235_, lean_object* v_inst_236_, lean_object* v_inst_237_, lean_object* v_inst_238_){
_start:
{
lean_object* v_res_239_; 
v_res_239_ = lp_mathlib_AlgEquiv_op(v_R_231_, v_A_232_, v_B_233_, v_inst_234_, v_inst_235_, v_inst_236_, v_inst_237_, v_inst_238_);
lean_dec_ref(v_inst_238_);
lean_dec_ref(v_inst_237_);
lean_dec_ref(v_inst_234_);
return v_res_239_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_unop___redArg(lean_object* v_inst_240_, lean_object* v_inst_241_){
_start:
{
lean_object* v___x_242_; lean_object* v___x_243_; 
v___x_242_ = lp_mathlib_AlgEquiv_op___redArg(v_inst_240_, v_inst_241_);
v___x_243_ = lp_mathlib_Equiv_symm___redArg(v___x_242_);
return v___x_243_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_unop(lean_object* v_R_244_, lean_object* v_A_245_, lean_object* v_B_246_, lean_object* v_inst_247_, lean_object* v_inst_248_, lean_object* v_inst_249_, lean_object* v_inst_250_, lean_object* v_inst_251_){
_start:
{
lean_object* v___x_252_; lean_object* v___x_253_; 
v___x_252_ = lp_mathlib_AlgEquiv_op___redArg(v_inst_248_, v_inst_249_);
v___x_253_ = lp_mathlib_Equiv_symm___redArg(v___x_252_);
return v___x_253_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_unop___boxed(lean_object* v_R_254_, lean_object* v_A_255_, lean_object* v_B_256_, lean_object* v_inst_257_, lean_object* v_inst_258_, lean_object* v_inst_259_, lean_object* v_inst_260_, lean_object* v_inst_261_){
_start:
{
lean_object* v_res_262_; 
v_res_262_ = lp_mathlib_AlgEquiv_unop(v_R_254_, v_A_255_, v_B_256_, v_inst_257_, v_inst_258_, v_inst_259_, v_inst_260_, v_inst_261_);
lean_dec_ref(v_inst_261_);
lean_dec_ref(v_inst_260_);
lean_dec_ref(v_inst_257_);
return v_res_262_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_opComm___redArg(lean_object* v_inst_263_, lean_object* v_inst_264_){
_start:
{
lean_object* v___x_265_; lean_object* v___x_266_; lean_object* v___x_267_; lean_object* v___x_268_; lean_object* v___x_269_; lean_object* v___x_270_; lean_object* v___x_271_; 
lean_inc_ref(v_inst_264_);
v___x_265_ = lp_mathlib_MulOpposite_instSemiring___redArg(v_inst_264_);
v___x_266_ = lp_mathlib_AlgEquiv_op___redArg(v_inst_263_, v___x_265_);
v___x_267_ = lean_obj_once(&lp_mathlib_AlgHom_opComm___redArg___closed__0, &lp_mathlib_AlgHom_opComm___redArg___closed__0_once, _init_lp_mathlib_AlgHom_opComm___redArg___closed__0);
v___x_268_ = lp_mathlib_AlgEquiv_opOp___redArg(v_inst_264_);
v___x_269_ = lp_mathlib_Equiv_symm___redArg(v___x_268_);
v___x_270_ = lp_mathlib_AlgEquiv_equivCongr___redArg(v___x_267_, v___x_269_);
v___x_271_ = lp_mathlib_Equiv_trans___redArg(v___x_266_, v___x_270_);
return v___x_271_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_opComm(lean_object* v_R_272_, lean_object* v_A_273_, lean_object* v_B_274_, lean_object* v_inst_275_, lean_object* v_inst_276_, lean_object* v_inst_277_, lean_object* v_inst_278_, lean_object* v_inst_279_){
_start:
{
lean_object* v___x_280_; 
v___x_280_ = lp_mathlib_AlgEquiv_opComm___redArg(v_inst_276_, v_inst_277_);
return v___x_280_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_opComm___boxed(lean_object* v_R_281_, lean_object* v_A_282_, lean_object* v_B_283_, lean_object* v_inst_284_, lean_object* v_inst_285_, lean_object* v_inst_286_, lean_object* v_inst_287_, lean_object* v_inst_288_){
_start:
{
lean_object* v_res_289_; 
v_res_289_ = lp_mathlib_AlgEquiv_opComm(v_R_281_, v_A_282_, v_B_283_, v_inst_284_, v_inst_285_, v_inst_286_, v_inst_287_, v_inst_288_);
lean_dec_ref(v_inst_288_);
lean_dec_ref(v_inst_287_);
lean_dec_ref(v_inst_284_);
return v_res_289_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_moduleEndSelf___redArg(lean_object* v_inst_290_){
_start:
{
lean_object* v___x_291_; 
v___x_291_ = lp_mathlib_RingEquiv_moduleEndSelf___redArg(v_inst_290_);
return v___x_291_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_moduleEndSelf(lean_object* v_R_292_, lean_object* v_A_293_, lean_object* v_inst_294_, lean_object* v_inst_295_, lean_object* v_inst_296_){
_start:
{
lean_object* v___x_297_; 
v___x_297_ = lp_mathlib_RingEquiv_moduleEndSelf___redArg(v_inst_295_);
return v___x_297_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_moduleEndSelf___boxed(lean_object* v_R_298_, lean_object* v_A_299_, lean_object* v_inst_300_, lean_object* v_inst_301_, lean_object* v_inst_302_){
_start:
{
lean_object* v_res_303_; 
v_res_303_ = lp_mathlib_AlgEquiv_moduleEndSelf(v_R_298_, v_A_299_, v_inst_300_, v_inst_301_, v_inst_302_);
lean_dec_ref(v_inst_302_);
lean_dec_ref(v_inst_300_);
return v_res_303_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_moduleEndSelfOp___redArg(lean_object* v_inst_304_){
_start:
{
lean_object* v___x_305_; 
v___x_305_ = lp_mathlib_RingEquiv_moduleEndSelfOp___redArg(v_inst_304_);
return v___x_305_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_moduleEndSelfOp(lean_object* v_R_306_, lean_object* v_A_307_, lean_object* v_inst_308_, lean_object* v_inst_309_, lean_object* v_inst_310_){
_start:
{
lean_object* v___x_311_; 
v___x_311_ = lp_mathlib_RingEquiv_moduleEndSelfOp___redArg(v_inst_309_);
return v___x_311_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_moduleEndSelfOp___boxed(lean_object* v_R_312_, lean_object* v_A_313_, lean_object* v_inst_314_, lean_object* v_inst_315_, lean_object* v_inst_316_){
_start:
{
lean_object* v_res_317_; 
v_res_317_ = lp_mathlib_AlgEquiv_moduleEndSelfOp(v_R_312_, v_A_313_, v_inst_314_, v_inst_315_, v_inst_316_);
lean_dec_ref(v_inst_316_);
lean_dec_ref(v_inst_314_);
return v_res_317_;
}
}
static lean_object* _init_lp_mathlib_AlgEquiv_toOpposite___closed__0(void){
_start:
{
lean_object* v___x_318_; 
v___x_318_ = lp_mathlib_MulOpposite_opEquiv(lean_box(0));
return v___x_318_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_toOpposite(lean_object* v_R_319_, lean_object* v_A_320_, lean_object* v_inst_321_, lean_object* v_inst_322_, lean_object* v_inst_323_){
_start:
{
lean_object* v___x_324_; 
v___x_324_ = lean_obj_once(&lp_mathlib_AlgEquiv_toOpposite___closed__0, &lp_mathlib_AlgEquiv_toOpposite___closed__0_once, _init_lp_mathlib_AlgEquiv_toOpposite___closed__0);
return v___x_324_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_toOpposite___boxed(lean_object* v_R_325_, lean_object* v_A_326_, lean_object* v_inst_327_, lean_object* v_inst_328_, lean_object* v_inst_329_){
_start:
{
lean_object* v_res_330_; 
v_res_330_ = lp_mathlib_AlgEquiv_toOpposite(v_R_325_, v_A_326_, v_inst_327_, v_inst_328_, v_inst_329_);
lean_dec_ref(v_inst_329_);
lean_dec_ref(v_inst_328_);
lean_dec_ref(v_inst_327_);
return v_res_330_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Equiv(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_Opposite(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Opposite(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Opposite(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Equiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Algebra_Opposite(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Algebra_Equiv(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Opposite(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Opposite(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Algebra_Opposite(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Algebra_Equiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Algebra_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Algebra_Opposite(builtin);
}
#ifdef __cplusplus
}
#endif
