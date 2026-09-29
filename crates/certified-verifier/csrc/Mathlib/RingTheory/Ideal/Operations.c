// Lean compiler output
// Module: Mathlib.RingTheory.Ideal.Operations
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Algebra.Operations public import Mathlib.Algebra.Module.BigOperators public import Mathlib.Data.Fintype.Lattice public import Mathlib.Algebra.Group.Subgroup.ZPowers.Basic public import Mathlib.RingTheory.Coprime.Lemmas public import Mathlib.RingTheory.Ideal.Basic public import Mathlib.RingTheory.NonUnitalSubsemiring.Basic public import Mathlib.Tactic.Order
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
lean_object* lp_mathlib_Submodule_mapHom___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_NonUnitalRingHom_id___lam__0___boxed(lean_object*);
lean_object* lp_mathlib_Submodule_instSMul___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Semiring_toModule___redArg(lean_object*);
lean_object* lp_mathlib_Submodule_map___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Algebra_id___redArg(lean_object*);
lean_object* lp_mathlib_Submodule_instIdemCommSemiring___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ideal_radical(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ideal_radical___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ideal_radicalInfTopHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ideal_radicalInfTopHom(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ideal_instIdemCommSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ideal_instIdemCommSemiring(lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Ideal_uniqueUnits___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Ideal_uniqueUnits___closed__0 = (const lean_object*)&lp_mathlib_Ideal_uniqueUnits___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Ideal_uniqueUnits(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ideal_uniqueUnits___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_moduleSubmodule___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_moduleSubmodule(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_moduleSubmodule___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Submodule_algebraIdeal___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_NonUnitalRingHom_id___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Submodule_algebraIdeal___redArg___closed__0 = (const lean_object*)&lp_mathlib_Submodule_algebraIdeal___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Submodule_algebraIdeal___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_algebraIdeal(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_mapAlgHom___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_mapAlgHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_mapAlgEquiv___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_mapAlgEquiv___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_mapAlgEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ideal_radical(lean_object* v_R_1_, lean_object* v_inst_2_, lean_object* v_I_3_){
_start:
{
lean_object* v___x_4_; 
v___x_4_ = lean_box(0);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ideal_radical___boxed(lean_object* v_R_5_, lean_object* v_inst_6_, lean_object* v_I_7_){
_start:
{
lean_object* v_res_8_; 
v_res_8_ = lp_mathlib_Ideal_radical(v_R_5_, v_inst_6_, v_I_7_);
lean_dec_ref(v_inst_6_);
return v_res_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ideal_radicalInfTopHom___redArg(lean_object* v_inst_9_){
_start:
{
lean_object* v___x_10_; 
v___x_10_ = lean_alloc_closure((void*)(lp_mathlib_Ideal_radical___boxed), 3, 2);
lean_closure_set(v___x_10_, 0, lean_box(0));
lean_closure_set(v___x_10_, 1, v_inst_9_);
return v___x_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ideal_radicalInfTopHom(lean_object* v_R_11_, lean_object* v_inst_12_){
_start:
{
lean_object* v___x_13_; 
v___x_13_ = lean_alloc_closure((void*)(lp_mathlib_Ideal_radical___boxed), 3, 2);
lean_closure_set(v___x_13_, 0, lean_box(0));
lean_closure_set(v___x_13_, 1, v_inst_12_);
return v___x_13_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ideal_instIdemCommSemiring___redArg(lean_object* v_inst_14_){
_start:
{
lean_object* v___x_15_; lean_object* v___x_16_; 
lean_inc_ref_n(v_inst_14_, 2);
v___x_15_ = lp_mathlib_Algebra_id___redArg(v_inst_14_);
v___x_16_ = lp_mathlib_Submodule_instIdemCommSemiring___redArg(v_inst_14_, v_inst_14_, v___x_15_);
lean_dec_ref(v___x_15_);
lean_dec_ref(v_inst_14_);
return v___x_16_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ideal_instIdemCommSemiring(lean_object* v_R_17_, lean_object* v_inst_18_){
_start:
{
lean_object* v___x_19_; 
v___x_19_ = lp_mathlib_Ideal_instIdemCommSemiring___redArg(v_inst_18_);
return v___x_19_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ideal_uniqueUnits(lean_object* v_R_22_, lean_object* v_inst_23_){
_start:
{
lean_object* v___x_24_; 
v___x_24_ = ((lean_object*)(lp_mathlib_Ideal_uniqueUnits___closed__0));
return v___x_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ideal_uniqueUnits___boxed(lean_object* v_R_25_, lean_object* v_inst_26_){
_start:
{
lean_object* v_res_27_; 
v_res_27_ = lp_mathlib_Ideal_uniqueUnits(v_R_25_, v_inst_26_);
lean_dec_ref(v_inst_26_);
return v_res_27_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_moduleSubmodule___redArg(lean_object* v_inst_28_){
_start:
{
lean_object* v___f_29_; 
v___f_29_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_instSMul___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_29_, 0, v_inst_28_);
return v___f_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_moduleSubmodule(lean_object* v_R_30_, lean_object* v_inst_31_, lean_object* v_M_32_, lean_object* v_inst_33_, lean_object* v_inst_34_){
_start:
{
lean_object* v___f_35_; 
v___f_35_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_instSMul___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_35_, 0, v_inst_33_);
return v___f_35_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_moduleSubmodule___boxed(lean_object* v_R_36_, lean_object* v_inst_37_, lean_object* v_M_38_, lean_object* v_inst_39_, lean_object* v_inst_40_){
_start:
{
lean_object* v_res_41_; 
v_res_41_ = lp_mathlib_Submodule_moduleSubmodule(v_R_36_, v_inst_37_, v_M_38_, v_inst_39_, v_inst_40_);
lean_dec(v_inst_40_);
lean_dec_ref(v_inst_37_);
return v_res_41_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_algebraIdeal___redArg(lean_object* v_inst_43_, lean_object* v_inst_44_, lean_object* v_inst_45_){
_start:
{
lean_object* v_toAddCommMonoid_46_; lean_object* v_toSMul_47_; lean_object* v_algebraMap_48_; lean_object* v___x_50_; uint8_t v_isShared_51_; uint8_t v_isSharedCheck_60_; 
v_toAddCommMonoid_46_ = lean_ctor_get(v_inst_44_, 0);
lean_inc_ref(v_toAddCommMonoid_46_);
lean_dec_ref(v_inst_44_);
v_toSMul_47_ = lean_ctor_get(v_inst_45_, 0);
v_algebraMap_48_ = lean_ctor_get(v_inst_45_, 1);
v_isSharedCheck_60_ = !lean_is_exclusive(v_inst_45_);
if (v_isSharedCheck_60_ == 0)
{
v___x_50_ = v_inst_45_;
v_isShared_51_ = v_isSharedCheck_60_;
goto v_resetjp_49_;
}
else
{
lean_inc(v_algebraMap_48_);
lean_inc(v_toSMul_47_);
lean_dec(v_inst_45_);
v___x_50_ = lean_box(0);
v_isShared_51_ = v_isSharedCheck_60_;
goto v_resetjp_49_;
}
v_resetjp_49_:
{
lean_object* v_toAddCommMonoid_52_; lean_object* v___f_53_; lean_object* v___x_54_; lean_object* v___f_55_; lean_object* v___x_56_; lean_object* v___x_58_; 
v_toAddCommMonoid_52_ = lean_ctor_get(v_inst_43_, 0);
lean_inc_ref(v_toAddCommMonoid_52_);
lean_inc_ref(v_toAddCommMonoid_46_);
v___f_53_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_instSMul___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_53_, 0, v_toAddCommMonoid_46_);
v___x_54_ = lp_mathlib_Semiring_toModule___redArg(v_inst_43_);
v___f_55_ = ((lean_object*)(lp_mathlib_Submodule_algebraIdeal___redArg___closed__0));
lean_inc_ref(v_inst_43_);
v___x_56_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_map___boxed), 14, 13);
lean_closure_set(v___x_56_, 0, lean_box(0));
lean_closure_set(v___x_56_, 1, lean_box(0));
lean_closure_set(v___x_56_, 2, lean_box(0));
lean_closure_set(v___x_56_, 3, lean_box(0));
lean_closure_set(v___x_56_, 4, v_inst_43_);
lean_closure_set(v___x_56_, 5, v_inst_43_);
lean_closure_set(v___x_56_, 6, v_toAddCommMonoid_52_);
lean_closure_set(v___x_56_, 7, v_toAddCommMonoid_46_);
lean_closure_set(v___x_56_, 8, v___x_54_);
lean_closure_set(v___x_56_, 9, v_toSMul_47_);
lean_closure_set(v___x_56_, 10, v___f_55_);
lean_closure_set(v___x_56_, 11, lean_box(0));
lean_closure_set(v___x_56_, 12, v_algebraMap_48_);
if (v_isShared_51_ == 0)
{
lean_ctor_set(v___x_50_, 1, v___x_56_);
lean_ctor_set(v___x_50_, 0, v___f_53_);
v___x_58_ = v___x_50_;
goto v_reusejp_57_;
}
else
{
lean_object* v_reuseFailAlloc_59_; 
v_reuseFailAlloc_59_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_59_, 0, v___f_53_);
lean_ctor_set(v_reuseFailAlloc_59_, 1, v___x_56_);
v___x_58_ = v_reuseFailAlloc_59_;
goto v_reusejp_57_;
}
v_reusejp_57_:
{
return v___x_58_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_algebraIdeal(lean_object* v_R_61_, lean_object* v_inst_62_, lean_object* v_A_63_, lean_object* v_inst_64_, lean_object* v_inst_65_){
_start:
{
lean_object* v___x_66_; 
v___x_66_ = lp_mathlib_Submodule_algebraIdeal___redArg(v_inst_62_, v_inst_64_, v_inst_65_);
return v___x_66_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_mapAlgHom___redArg(lean_object* v_inst_67_, lean_object* v_inst_68_, lean_object* v_inst_69_, lean_object* v_inst_70_, lean_object* v_inst_71_, lean_object* v_f_72_){
_start:
{
lean_object* v___x_73_; 
v___x_73_ = lp_mathlib_Submodule_mapHom___redArg(v_inst_67_, v_inst_68_, v_inst_70_, v_inst_69_, v_inst_71_, v_f_72_);
return v___x_73_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_mapAlgHom(lean_object* v_R_74_, lean_object* v_inst_75_, lean_object* v_A_76_, lean_object* v_B_77_, lean_object* v_inst_78_, lean_object* v_inst_79_, lean_object* v_inst_80_, lean_object* v_inst_81_, lean_object* v_f_82_){
_start:
{
lean_object* v___x_83_; 
v___x_83_ = lp_mathlib_Submodule_mapHom___redArg(v_inst_75_, v_inst_78_, v_inst_80_, v_inst_79_, v_inst_81_, v_f_82_);
return v___x_83_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_mapAlgEquiv___redArg___lam__0(lean_object* v_inst_84_, lean_object* v_inst_85_, lean_object* v_inst_86_, lean_object* v_inst_87_, lean_object* v_inst_88_, lean_object* v_toFun_89_, lean_object* v___y_90_){
_start:
{
lean_object* v___x_59__overap_91_; lean_object* v___x_92_; 
v___x_59__overap_91_ = lp_mathlib_Submodule_mapHom___redArg(v_inst_84_, v_inst_85_, v_inst_86_, v_inst_87_, v_inst_88_, v_toFun_89_);
v___x_92_ = lean_apply_1(v___x_59__overap_91_, v___y_90_);
return v___x_92_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_mapAlgEquiv___redArg(lean_object* v_inst_93_, lean_object* v_inst_94_, lean_object* v_inst_95_, lean_object* v_inst_96_, lean_object* v_inst_97_, lean_object* v_f_98_){
_start:
{
lean_object* v_toFun_99_; lean_object* v___x_100_; lean_object* v_toFun_101_; lean_object* v___x_103_; uint8_t v_isShared_104_; uint8_t v_isSharedCheck_110_; 
v_toFun_99_ = lean_ctor_get(v_f_98_, 0);
lean_inc(v_toFun_99_);
v___x_100_ = lp_mathlib_Equiv_symm___redArg(v_f_98_);
v_toFun_101_ = lean_ctor_get(v___x_100_, 0);
v_isSharedCheck_110_ = !lean_is_exclusive(v___x_100_);
if (v_isSharedCheck_110_ == 0)
{
lean_object* v_unused_111_; 
v_unused_111_ = lean_ctor_get(v___x_100_, 1);
lean_dec(v_unused_111_);
v___x_103_ = v___x_100_;
v_isShared_104_ = v_isSharedCheck_110_;
goto v_resetjp_102_;
}
else
{
lean_inc(v_toFun_101_);
lean_dec(v___x_100_);
v___x_103_ = lean_box(0);
v_isShared_104_ = v_isSharedCheck_110_;
goto v_resetjp_102_;
}
v_resetjp_102_:
{
lean_object* v___x_105_; lean_object* v___f_106_; lean_object* v___x_108_; 
lean_inc_ref(v_inst_97_);
lean_inc_ref(v_inst_95_);
lean_inc_ref(v_inst_96_);
lean_inc_ref(v_inst_94_);
lean_inc_ref(v_inst_93_);
v___x_105_ = lp_mathlib_Submodule_mapHom___redArg(v_inst_93_, v_inst_94_, v_inst_96_, v_inst_95_, v_inst_97_, v_toFun_99_);
v___f_106_ = lean_alloc_closure((void*)(lp_mathlib_Submodule_mapAlgEquiv___redArg___lam__0), 7, 6);
lean_closure_set(v___f_106_, 0, v_inst_93_);
lean_closure_set(v___f_106_, 1, v_inst_95_);
lean_closure_set(v___f_106_, 2, v_inst_97_);
lean_closure_set(v___f_106_, 3, v_inst_94_);
lean_closure_set(v___f_106_, 4, v_inst_96_);
lean_closure_set(v___f_106_, 5, v_toFun_101_);
if (v_isShared_104_ == 0)
{
lean_ctor_set(v___x_103_, 1, v___f_106_);
lean_ctor_set(v___x_103_, 0, v___x_105_);
v___x_108_ = v___x_103_;
goto v_reusejp_107_;
}
else
{
lean_object* v_reuseFailAlloc_109_; 
v_reuseFailAlloc_109_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_109_, 0, v___x_105_);
lean_ctor_set(v_reuseFailAlloc_109_, 1, v___f_106_);
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
LEAN_EXPORT lean_object* lp_mathlib_Submodule_mapAlgEquiv(lean_object* v_R_112_, lean_object* v_inst_113_, lean_object* v_A_114_, lean_object* v_B_115_, lean_object* v_inst_116_, lean_object* v_inst_117_, lean_object* v_inst_118_, lean_object* v_inst_119_, lean_object* v_f_120_){
_start:
{
lean_object* v___x_121_; 
v___x_121_ = lp_mathlib_Submodule_mapAlgEquiv___redArg(v_inst_113_, v_inst_116_, v_inst_117_, v_inst_118_, v_inst_119_, v_f_120_);
return v___x_121_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Operations(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_BigOperators(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Fintype_Lattice(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_ZPowers_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_RingTheory_Coprime_Lemmas(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_RingTheory_Ideal_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_RingTheory_NonUnitalSubsemiring_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Order(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_RingTheory_Ideal_Operations(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Operations(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_BigOperators(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fintype_Lattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_ZPowers_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_RingTheory_Coprime_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_RingTheory_Ideal_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_RingTheory_NonUnitalSubsemiring_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Order(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_RingTheory_Ideal_Operations(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Algebra_Operations(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_BigOperators(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_Lattice(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Subgroup_ZPowers_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Coprime_Lemmas(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Ideal_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_NonUnitalSubsemiring_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Order(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_RingTheory_Ideal_Operations(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Algebra_Operations(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_BigOperators(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_Lattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Subgroup_ZPowers_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Coprime_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Ideal_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_NonUnitalSubsemiring_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Order(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_RingTheory_Ideal_Operations(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_RingTheory_Ideal_Operations(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_RingTheory_Ideal_Operations(builtin);
}
#ifdef __cplusplus
}
#endif
