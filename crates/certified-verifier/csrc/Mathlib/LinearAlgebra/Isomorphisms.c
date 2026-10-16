// Lean compiler output
// Module: Mathlib.LinearAlgebra.Isomorphisms
// Imports: public import Init public meta import Init public import Mathlib.LinearAlgebra.Quotient.Basic public import Mathlib.LinearAlgebra.Quotient.Card
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
lean_object* lp_mathlib_LinearMap_id___lam__0___boxed(lean_object*);
lean_object* lp_mathlib_Submodule_mapQ___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Submodule_liftQ___redArg(lean_object*);
lean_object* lp_mathlib_Submodule_Quotient_addCommGroup___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Submodule_Quotient_instSMul_x27___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Submodule_Quotient_mk___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddSubgroupClass_toAddGroup___redArg(lean_object*);
lean_object* lp_mathlib_Submodule_smul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Submodule_inclusion(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearMap_comp___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Submodule_quotEquivOfEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearEquiv_trans___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_subToSupQuotient___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_subToSupQuotient(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_quotientInfToSupQuotient___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_quotientInfToSupQuotient(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Submodule_quotientQuotientEquivQuotientAux___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_LinearMap_id___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Submodule_quotientQuotientEquivQuotientAux___redArg___closed__0 = (const lean_object*)&lp_mathlib_Submodule_quotientQuotientEquivQuotientAux___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Submodule_quotientQuotientEquivQuotientAux___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_quotientQuotientEquivQuotientAux(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_quotientQuotientEquivQuotient___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_quotientQuotientEquivQuotient___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_quotientQuotientEquivQuotient___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_quotientQuotientEquivQuotient(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_quotientQuotientEquivQuotientSup___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_quotientQuotientEquivQuotientSup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_subToSupQuotient___redArg(lean_object* v_inst_1_, lean_object* v_inst_2_, lean_object* v_inst_3_, lean_object* v_p_4_){
_start:
{
lean_object* v_toSemiring_5_; lean_object* v_toAddMonoid_6_; lean_object* v___x_7_; lean_object* v___x_8_; lean_object* v___f_9_; lean_object* v___x_10_; lean_object* v___x_11_; lean_object* v___f_12_; 
v_toSemiring_5_ = lean_ctor_get(v_inst_1_, 0);
lean_inc_ref(v_toSemiring_5_);
v_toAddMonoid_6_ = lean_ctor_get(v_inst_2_, 0);
lean_inc_ref(v_toAddMonoid_6_);
v___x_7_ = lean_box(0);
v___x_8_ = lp_mathlib_AddSubgroupClass_toAddGroup___redArg(v_inst_2_);
lean_inc(v_inst_3_);
v___f_9_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_9_, 0, v_inst_3_);
v___x_10_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_Quotient_mk___boxed), 7, 6);
lean_closure_set(v___x_10_, 0, lean_box(0));
lean_closure_set(v___x_10_, 1, lean_box(0));
lean_closure_set(v___x_10_, 2, v_inst_1_);
lean_closure_set(v___x_10_, 3, v___x_8_);
lean_closure_set(v___x_10_, 4, v___f_9_);
lean_closure_set(v___x_10_, 5, v___x_7_);
v___x_11_ = lp_mathlib_Submodule_inclusion(lean_box(0), lean_box(0), v_toSemiring_5_, v_toAddMonoid_6_, v_inst_3_, v_p_4_, v___x_7_, lean_box(0));
lean_dec(v_inst_3_);
lean_dec_ref(v_toAddMonoid_6_);
lean_dec_ref(v_toSemiring_5_);
v___f_12_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_12_, 0, v___x_11_);
lean_closure_set(v___f_12_, 1, v___x_10_);
return v___f_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_subToSupQuotient(lean_object* v_R_13_, lean_object* v_M_14_, lean_object* v_inst_15_, lean_object* v_inst_16_, lean_object* v_inst_17_, lean_object* v_p_18_, lean_object* v_p_x27_19_){
_start:
{
lean_object* v_toSemiring_20_; lean_object* v_toAddMonoid_21_; lean_object* v___x_22_; lean_object* v___x_23_; lean_object* v___f_24_; lean_object* v___x_25_; lean_object* v___x_26_; lean_object* v___f_27_; 
v_toSemiring_20_ = lean_ctor_get(v_inst_15_, 0);
lean_inc_ref(v_toSemiring_20_);
v_toAddMonoid_21_ = lean_ctor_get(v_inst_16_, 0);
lean_inc_ref(v_toAddMonoid_21_);
v___x_22_ = lean_box(0);
v___x_23_ = lp_mathlib_AddSubgroupClass_toAddGroup___redArg(v_inst_16_);
lean_inc(v_inst_17_);
v___f_24_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_24_, 0, v_inst_17_);
v___x_25_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_Quotient_mk___boxed), 7, 6);
lean_closure_set(v___x_25_, 0, lean_box(0));
lean_closure_set(v___x_25_, 1, lean_box(0));
lean_closure_set(v___x_25_, 2, v_inst_15_);
lean_closure_set(v___x_25_, 3, v___x_23_);
lean_closure_set(v___x_25_, 4, v___f_24_);
lean_closure_set(v___x_25_, 5, v___x_22_);
v___x_26_ = lp_mathlib_Submodule_inclusion(lean_box(0), lean_box(0), v_toSemiring_20_, v_toAddMonoid_21_, v_inst_17_, v_p_18_, v___x_22_, lean_box(0));
lean_dec(v_inst_17_);
lean_dec_ref(v_toAddMonoid_21_);
lean_dec_ref(v_toSemiring_20_);
v___f_27_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_27_, 0, v___x_26_);
lean_closure_set(v___f_27_, 1, v___x_25_);
return v___f_27_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_quotientInfToSupQuotient___redArg(lean_object* v_inst_28_, lean_object* v_inst_29_, lean_object* v_inst_30_, lean_object* v_p_31_){
_start:
{
lean_object* v_toSemiring_32_; lean_object* v_toAddMonoid_33_; lean_object* v___x_34_; lean_object* v___x_35_; lean_object* v___f_36_; lean_object* v___x_37_; lean_object* v___x_38_; lean_object* v___f_39_; lean_object* v___x_40_; 
v_toSemiring_32_ = lean_ctor_get(v_inst_28_, 0);
lean_inc_ref(v_toSemiring_32_);
v_toAddMonoid_33_ = lean_ctor_get(v_inst_29_, 0);
lean_inc_ref(v_toAddMonoid_33_);
v___x_34_ = lean_box(0);
v___x_35_ = lp_mathlib_AddSubgroupClass_toAddGroup___redArg(v_inst_29_);
lean_inc(v_inst_30_);
v___f_36_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_smul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_36_, 0, v_inst_30_);
v___x_37_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_Quotient_mk___boxed), 7, 6);
lean_closure_set(v___x_37_, 0, lean_box(0));
lean_closure_set(v___x_37_, 1, lean_box(0));
lean_closure_set(v___x_37_, 2, v_inst_28_);
lean_closure_set(v___x_37_, 3, v___x_35_);
lean_closure_set(v___x_37_, 4, v___f_36_);
lean_closure_set(v___x_37_, 5, v___x_34_);
v___x_38_ = lp_mathlib_Submodule_inclusion(lean_box(0), lean_box(0), v_toSemiring_32_, v_toAddMonoid_33_, v_inst_30_, v_p_31_, v___x_34_, lean_box(0));
lean_dec(v_inst_30_);
lean_dec_ref(v_toAddMonoid_33_);
lean_dec_ref(v_toSemiring_32_);
v___f_39_ = lean_alloc_closure((void*)(lp_mathlib_LinearMap_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_39_, 0, v___x_38_);
lean_closure_set(v___f_39_, 1, v___x_37_);
v___x_40_ = lp_mathlib_Submodule_liftQ___redArg(v___f_39_);
return v___x_40_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_quotientInfToSupQuotient(lean_object* v_R_41_, lean_object* v_M_42_, lean_object* v_inst_43_, lean_object* v_inst_44_, lean_object* v_inst_45_, lean_object* v_p_46_, lean_object* v_p_x27_47_){
_start:
{
lean_object* v___x_48_; 
v___x_48_ = lp_mathlib_LinearMap_quotientInfToSupQuotient___redArg(v_inst_43_, v_inst_44_, v_inst_45_, v_p_46_);
return v___x_48_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_quotientQuotientEquivQuotientAux___redArg(lean_object* v_inst_50_, lean_object* v_inst_51_, lean_object* v_inst_52_, lean_object* v_T_53_){
_start:
{
lean_object* v___f_54_; lean_object* v___x_55_; lean_object* v___x_56_; 
v___f_54_ = ((lean_object*)(lp_mathlib_Submodule_quotientQuotientEquivQuotientAux___redArg___closed__0));
v___x_55_ = lp_mathlib_Submodule_mapQ___redArg(v_inst_50_, v_inst_51_, v_inst_52_, v_T_53_, v___f_54_);
v___x_56_ = lp_mathlib_Submodule_liftQ___redArg(v___x_55_);
return v___x_56_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_quotientQuotientEquivQuotientAux(lean_object* v_R_57_, lean_object* v_M_58_, lean_object* v_inst_59_, lean_object* v_inst_60_, lean_object* v_inst_61_, lean_object* v_S_62_, lean_object* v_T_63_, lean_object* v_h_64_){
_start:
{
lean_object* v___x_65_; 
v___x_65_ = lp_mathlib_Submodule_quotientQuotientEquivQuotientAux___redArg(v_inst_59_, v_inst_60_, v_inst_61_, v_T_63_);
return v___x_65_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_quotientQuotientEquivQuotient___redArg___lam__0(lean_object* v_inst_66_, lean_object* v_inst_67_, lean_object* v_inst_68_, lean_object* v_T_69_, lean_object* v___y_70_){
_start:
{
lean_object* v___x_84__overap_71_; lean_object* v___x_72_; 
v___x_84__overap_71_ = lp_mathlib_Submodule_quotientQuotientEquivQuotientAux___redArg(v_inst_66_, v_inst_67_, v_inst_68_, v_T_69_);
v___x_72_ = lean_apply_1(v___x_84__overap_71_, v___y_70_);
return v___x_72_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_quotientQuotientEquivQuotient___redArg___lam__1(lean_object* v_inst_73_, lean_object* v___x_74_, lean_object* v___f_75_, lean_object* v___x_76_, lean_object* v___x_77_, lean_object* v___y_78_){
_start:
{
lean_object* v___x_91__overap_79_; lean_object* v___x_80_; 
v___x_91__overap_79_ = lp_mathlib_Submodule_mapQ___redArg(v_inst_73_, v___x_74_, v___f_75_, v___x_76_, v___x_77_);
v___x_80_ = lean_apply_1(v___x_91__overap_79_, v___y_78_);
return v___x_80_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_quotientQuotientEquivQuotient___redArg(lean_object* v_inst_81_, lean_object* v_inst_82_, lean_object* v_inst_83_, lean_object* v_S_84_, lean_object* v_T_85_){
_start:
{
lean_object* v___f_86_; lean_object* v___x_87_; lean_object* v___f_88_; lean_object* v___x_89_; lean_object* v___x_90_; lean_object* v___f_91_; lean_object* v___x_92_; 
lean_inc_n(v_inst_83_, 3);
lean_inc_ref_n(v_inst_82_, 2);
lean_inc_ref_n(v_inst_81_, 3);
v___f_86_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_quotientQuotientEquivQuotient___redArg___lam__0), 5, 4);
lean_closure_set(v___f_86_, 0, v_inst_81_);
lean_closure_set(v___f_86_, 1, v_inst_82_);
lean_closure_set(v___f_86_, 2, v_inst_83_);
lean_closure_set(v___f_86_, 3, v_T_85_);
v___x_87_ = lp_mathlib_Submodule_Quotient_addCommGroup___redArg(v_inst_81_, v_inst_82_, v_inst_83_, v_S_84_);
v___f_88_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_Quotient_instSMul_x27___redArg___lam__0), 3, 1);
lean_closure_set(v___f_88_, 0, v_inst_83_);
v___x_89_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_Quotient_mk___boxed), 7, 6);
lean_closure_set(v___x_89_, 0, lean_box(0));
lean_closure_set(v___x_89_, 1, lean_box(0));
lean_closure_set(v___x_89_, 2, v_inst_81_);
lean_closure_set(v___x_89_, 3, v_inst_82_);
lean_closure_set(v___x_89_, 4, v_inst_83_);
lean_closure_set(v___x_89_, 5, v_S_84_);
v___x_90_ = lean_box(0);
v___f_91_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_quotientQuotientEquivQuotient___redArg___lam__1), 6, 5);
lean_closure_set(v___f_91_, 0, v_inst_81_);
lean_closure_set(v___f_91_, 1, v___x_87_);
lean_closure_set(v___f_91_, 2, v___f_88_);
lean_closure_set(v___f_91_, 3, v___x_90_);
lean_closure_set(v___f_91_, 4, v___x_89_);
v___x_92_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_92_, 0, v___f_86_);
lean_ctor_set(v___x_92_, 1, v___f_91_);
return v___x_92_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_quotientQuotientEquivQuotient(lean_object* v_R_93_, lean_object* v_M_94_, lean_object* v_inst_95_, lean_object* v_inst_96_, lean_object* v_inst_97_, lean_object* v_S_98_, lean_object* v_T_99_, lean_object* v_h_100_){
_start:
{
lean_object* v___x_101_; 
v___x_101_ = lp_mathlib_Submodule_quotientQuotientEquivQuotient___redArg(v_inst_95_, v_inst_96_, v_inst_97_, v_S_98_, v_T_99_);
return v___x_101_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_quotientQuotientEquivQuotientSup___redArg(lean_object* v_inst_102_, lean_object* v_inst_103_, lean_object* v_inst_104_, lean_object* v_S_105_){
_start:
{
lean_object* v___x_106_; lean_object* v___f_107_; lean_object* v___x_108_; lean_object* v___x_109_; lean_object* v___x_110_; lean_object* v___x_111_; 
lean_inc_n(v_inst_104_, 2);
lean_inc_ref(v_inst_103_);
lean_inc_ref(v_inst_102_);
v___x_106_ = lp_mathlib_Submodule_Quotient_addCommGroup___redArg(v_inst_102_, v_inst_103_, v_inst_104_, v_S_105_);
v___f_107_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_Quotient_instSMul_x27___redArg___lam__0), 3, 1);
lean_closure_set(v___f_107_, 0, v_inst_104_);
v___x_108_ = lean_box(0);
v___x_109_ = lp_mathlib_Submodule_quotEquivOfEq(lean_box(0), lean_box(0), v_inst_102_, v___x_106_, v___f_107_, v___x_108_, v___x_108_, lean_box(0));
lean_dec_ref(v___f_107_);
lean_dec_ref(v___x_106_);
v___x_110_ = lp_mathlib_Submodule_quotientQuotientEquivQuotient___redArg(v_inst_102_, v_inst_103_, v_inst_104_, v_S_105_, v___x_108_);
v___x_111_ = lp_mathlib_LinearEquiv_trans___redArg(v___x_109_, v___x_110_);
return v___x_111_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_quotientQuotientEquivQuotientSup(lean_object* v_R_112_, lean_object* v_M_113_, lean_object* v_inst_114_, lean_object* v_inst_115_, lean_object* v_inst_116_, lean_object* v_S_117_, lean_object* v_T_118_){
_start:
{
lean_object* v___x_119_; 
v___x_119_ = lp_mathlib_Submodule_quotientQuotientEquivQuotientSup___redArg(v_inst_114_, v_inst_115_, v_inst_116_, v_S_117_);
return v___x_119_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_LinearAlgebra_Quotient_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_LinearAlgebra_Quotient_Card(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_LinearAlgebra_Isomorphisms(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_LinearAlgebra_Quotient_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_LinearAlgebra_Quotient_Card(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_LinearAlgebra_Isomorphisms(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Quotient_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Quotient_Card(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Isomorphisms(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Quotient_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Quotient_Card(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_LinearAlgebra_Isomorphisms(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_LinearAlgebra_Isomorphisms(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_LinearAlgebra_Isomorphisms(builtin);
}
#ifdef __cplusplus
}
#endif
