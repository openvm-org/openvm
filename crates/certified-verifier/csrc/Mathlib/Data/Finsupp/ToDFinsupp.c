// Lean compiler output
// Module: Mathlib.Data.Finsupp.ToDFinsupp
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Module.Equiv.Defs public import Mathlib.Data.DFinsupp.Module public import Mathlib.Data.Finsupp.SMul
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
lean_object* lp_mathlib_DFinsupp_support___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_toDFinsupp___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_toDFinsupp___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_toDFinsupp(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_toDFinsupp___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_DFinsupp_toFinsupp___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_toFinsupp___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_toFinsupp___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_toFinsupp___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_toFinsupp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_toFinsupp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_finsuppEquivDFinsupp___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_finsuppEquivDFinsupp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_finsuppAddEquivDFinsupp___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_finsuppAddEquivDFinsupp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_finsuppLequivDFinsupp___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_finsuppLequivDFinsupp___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_finsuppLequivDFinsupp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_finsuppLequivDFinsupp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_toDFinsupp___redArg___lam__0(lean_object* v_toFun_1_, lean_object* v___y_2_){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lean_apply_1(v_toFun_1_, v___y_2_);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_toDFinsupp___redArg(lean_object* v_f_4_){
_start:
{
lean_object* v_support_5_; lean_object* v_toFun_6_; lean_object* v___x_8_; uint8_t v_isShared_9_; uint8_t v_isSharedCheck_14_; 
v_support_5_ = lean_ctor_get(v_f_4_, 0);
v_toFun_6_ = lean_ctor_get(v_f_4_, 1);
v_isSharedCheck_14_ = !lean_is_exclusive(v_f_4_);
if (v_isSharedCheck_14_ == 0)
{
v___x_8_ = v_f_4_;
v_isShared_9_ = v_isSharedCheck_14_;
goto v_resetjp_7_;
}
else
{
lean_inc(v_toFun_6_);
lean_inc(v_support_5_);
lean_dec(v_f_4_);
v___x_8_ = lean_box(0);
v_isShared_9_ = v_isSharedCheck_14_;
goto v_resetjp_7_;
}
v_resetjp_7_:
{
lean_object* v___f_10_; lean_object* v___x_12_; 
v___f_10_ = lean_alloc_closure((void*)(lp_mathlib_Finsupp_toDFinsupp___redArg___lam__0), 2, 1);
lean_closure_set(v___f_10_, 0, v_toFun_6_);
if (v_isShared_9_ == 0)
{
lean_ctor_set(v___x_8_, 1, v_support_5_);
lean_ctor_set(v___x_8_, 0, v___f_10_);
v___x_12_ = v___x_8_;
goto v_reusejp_11_;
}
else
{
lean_object* v_reuseFailAlloc_13_; 
v_reuseFailAlloc_13_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_13_, 0, v___f_10_);
lean_ctor_set(v_reuseFailAlloc_13_, 1, v_support_5_);
v___x_12_ = v_reuseFailAlloc_13_;
goto v_reusejp_11_;
}
v_reusejp_11_:
{
return v___x_12_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_toDFinsupp(lean_object* v_00_u03b9_15_, lean_object* v_M_16_, lean_object* v_inst_17_, lean_object* v_f_18_){
_start:
{
lean_object* v___x_19_; 
v___x_19_ = lp_mathlib_Finsupp_toDFinsupp___redArg(v_f_18_);
return v___x_19_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_toDFinsupp___boxed(lean_object* v_00_u03b9_20_, lean_object* v_M_21_, lean_object* v_inst_22_, lean_object* v_f_23_){
_start:
{
lean_object* v_res_24_; 
v_res_24_ = lp_mathlib_Finsupp_toDFinsupp(v_00_u03b9_20_, v_M_21_, v_inst_22_, v_f_23_);
lean_dec(v_inst_22_);
return v_res_24_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_DFinsupp_toFinsupp___redArg___lam__0(lean_object* v_inst_25_, lean_object* v_i_26_, lean_object* v___y_27_){
_start:
{
lean_object* v___x_28_; uint8_t v___x_29_; 
v___x_28_ = lean_apply_1(v_inst_25_, v___y_27_);
v___x_29_ = lean_unbox(v___x_28_);
return v___x_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_toFinsupp___redArg___lam__0___boxed(lean_object* v_inst_30_, lean_object* v_i_31_, lean_object* v___y_32_){
_start:
{
uint8_t v_res_33_; lean_object* v_r_34_; 
v_res_33_ = lp_mathlib_DFinsupp_toFinsupp___redArg___lam__0(v_inst_30_, v_i_31_, v___y_32_);
lean_dec(v_i_31_);
v_r_34_ = lean_box(v_res_33_);
return v_r_34_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_toFinsupp___redArg___lam__1(lean_object* v_f_35_, lean_object* v___y_36_){
_start:
{
lean_object* v_toFun_37_; lean_object* v___x_38_; 
v_toFun_37_ = lean_ctor_get(v_f_35_, 0);
lean_inc(v_toFun_37_);
lean_dec_ref(v_f_35_);
v___x_38_ = lean_apply_1(v_toFun_37_, v___y_36_);
return v___x_38_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_toFinsupp___redArg(lean_object* v_inst_39_, lean_object* v_inst_40_, lean_object* v_f_41_){
_start:
{
lean_object* v___f_42_; lean_object* v___f_43_; lean_object* v___x_44_; lean_object* v___x_45_; 
v___f_42_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_toFinsupp___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_42_, 0, v_inst_40_);
lean_inc_ref(v_f_41_);
v___f_43_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_toFinsupp___redArg___lam__1), 2, 1);
lean_closure_set(v___f_43_, 0, v_f_41_);
v___x_44_ = lp_mathlib_DFinsupp_support___redArg(v_inst_39_, v___f_42_, v_f_41_);
v___x_45_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_45_, 0, v___x_44_);
lean_ctor_set(v___x_45_, 1, v___f_43_);
return v___x_45_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_toFinsupp(lean_object* v_00_u03b9_46_, lean_object* v_M_47_, lean_object* v_inst_48_, lean_object* v_inst_49_, lean_object* v_inst_50_, lean_object* v_f_51_){
_start:
{
lean_object* v___x_52_; 
v___x_52_ = lp_mathlib_DFinsupp_toFinsupp___redArg(v_inst_48_, v_inst_50_, v_f_51_);
return v___x_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_toFinsupp___boxed(lean_object* v_00_u03b9_53_, lean_object* v_M_54_, lean_object* v_inst_55_, lean_object* v_inst_56_, lean_object* v_inst_57_, lean_object* v_f_58_){
_start:
{
lean_object* v_res_59_; 
v_res_59_ = lp_mathlib_DFinsupp_toFinsupp(v_00_u03b9_53_, v_M_54_, v_inst_55_, v_inst_56_, v_inst_57_, v_f_58_);
lean_dec(v_inst_56_);
return v_res_59_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_finsuppEquivDFinsupp___redArg(lean_object* v_inst_60_, lean_object* v_inst_61_, lean_object* v_inst_62_){
_start:
{
lean_object* v___x_63_; lean_object* v___x_64_; lean_object* v___x_65_; 
lean_inc(v_inst_61_);
v___x_63_ = lean_alloc_closure((void*)(lp_mathlib_Finsupp_toDFinsupp___boxed), 4, 3);
lean_closure_set(v___x_63_, 0, lean_box(0));
lean_closure_set(v___x_63_, 1, lean_box(0));
lean_closure_set(v___x_63_, 2, v_inst_61_);
v___x_64_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_toFinsupp___boxed), 6, 5);
lean_closure_set(v___x_64_, 0, lean_box(0));
lean_closure_set(v___x_64_, 1, lean_box(0));
lean_closure_set(v___x_64_, 2, v_inst_60_);
lean_closure_set(v___x_64_, 3, v_inst_61_);
lean_closure_set(v___x_64_, 4, v_inst_62_);
v___x_65_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_65_, 0, v___x_63_);
lean_ctor_set(v___x_65_, 1, v___x_64_);
return v___x_65_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_finsuppEquivDFinsupp(lean_object* v_00_u03b9_66_, lean_object* v_M_67_, lean_object* v_inst_68_, lean_object* v_inst_69_, lean_object* v_inst_70_){
_start:
{
lean_object* v___x_71_; 
v___x_71_ = lp_mathlib_finsuppEquivDFinsupp___redArg(v_inst_68_, v_inst_69_, v_inst_70_);
return v___x_71_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_finsuppAddEquivDFinsupp___redArg(lean_object* v_inst_72_, lean_object* v_inst_73_, lean_object* v_inst_74_){
_start:
{
lean_object* v___x_75_; lean_object* v_toZero_76_; lean_object* v___x_78_; uint8_t v_isShared_79_; uint8_t v_isSharedCheck_85_; 
v___x_75_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v_inst_73_);
v_toZero_76_ = lean_ctor_get(v___x_75_, 0);
v_isSharedCheck_85_ = !lean_is_exclusive(v___x_75_);
if (v_isSharedCheck_85_ == 0)
{
lean_object* v_unused_86_; 
v_unused_86_ = lean_ctor_get(v___x_75_, 1);
lean_dec(v_unused_86_);
v___x_78_ = v___x_75_;
v_isShared_79_ = v_isSharedCheck_85_;
goto v_resetjp_77_;
}
else
{
lean_inc(v_toZero_76_);
lean_dec(v___x_75_);
v___x_78_ = lean_box(0);
v_isShared_79_ = v_isSharedCheck_85_;
goto v_resetjp_77_;
}
v_resetjp_77_:
{
lean_object* v___x_80_; lean_object* v___x_81_; lean_object* v___x_83_; 
lean_inc(v_toZero_76_);
v___x_80_ = lean_alloc_closure((void*)(lp_mathlib_Finsupp_toDFinsupp___boxed), 4, 3);
lean_closure_set(v___x_80_, 0, lean_box(0));
lean_closure_set(v___x_80_, 1, lean_box(0));
lean_closure_set(v___x_80_, 2, v_toZero_76_);
v___x_81_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_toFinsupp___boxed), 6, 5);
lean_closure_set(v___x_81_, 0, lean_box(0));
lean_closure_set(v___x_81_, 1, lean_box(0));
lean_closure_set(v___x_81_, 2, v_inst_72_);
lean_closure_set(v___x_81_, 3, v_toZero_76_);
lean_closure_set(v___x_81_, 4, v_inst_74_);
if (v_isShared_79_ == 0)
{
lean_ctor_set(v___x_78_, 1, v___x_81_);
lean_ctor_set(v___x_78_, 0, v___x_80_);
v___x_83_ = v___x_78_;
goto v_reusejp_82_;
}
else
{
lean_object* v_reuseFailAlloc_84_; 
v_reuseFailAlloc_84_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_84_, 0, v___x_80_);
lean_ctor_set(v_reuseFailAlloc_84_, 1, v___x_81_);
v___x_83_ = v_reuseFailAlloc_84_;
goto v_reusejp_82_;
}
v_reusejp_82_:
{
return v___x_83_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_finsuppAddEquivDFinsupp(lean_object* v_00_u03b9_87_, lean_object* v_M_88_, lean_object* v_inst_89_, lean_object* v_inst_90_, lean_object* v_inst_91_){
_start:
{
lean_object* v___x_92_; 
v___x_92_ = lp_mathlib_finsuppAddEquivDFinsupp___redArg(v_inst_89_, v_inst_90_, v_inst_91_);
return v___x_92_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_finsuppLequivDFinsupp___redArg(lean_object* v_inst_93_, lean_object* v_inst_94_, lean_object* v_inst_95_){
_start:
{
lean_object* v___x_96_; lean_object* v___x_97_; lean_object* v_toZero_98_; lean_object* v___x_100_; uint8_t v_isShared_101_; uint8_t v_isSharedCheck_107_; 
v___x_96_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_94_);
v___x_97_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_96_);
v_toZero_98_ = lean_ctor_get(v___x_97_, 0);
v_isSharedCheck_107_ = !lean_is_exclusive(v___x_97_);
if (v_isSharedCheck_107_ == 0)
{
lean_object* v_unused_108_; 
v_unused_108_ = lean_ctor_get(v___x_97_, 1);
lean_dec(v_unused_108_);
v___x_100_ = v___x_97_;
v_isShared_101_ = v_isSharedCheck_107_;
goto v_resetjp_99_;
}
else
{
lean_inc(v_toZero_98_);
lean_dec(v___x_97_);
v___x_100_ = lean_box(0);
v_isShared_101_ = v_isSharedCheck_107_;
goto v_resetjp_99_;
}
v_resetjp_99_:
{
lean_object* v___x_102_; lean_object* v___x_103_; lean_object* v___x_105_; 
lean_inc(v_toZero_98_);
v___x_102_ = lean_alloc_closure((void*)(lp_mathlib_Finsupp_toDFinsupp___boxed), 4, 3);
lean_closure_set(v___x_102_, 0, lean_box(0));
lean_closure_set(v___x_102_, 1, lean_box(0));
lean_closure_set(v___x_102_, 2, v_toZero_98_);
v___x_103_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_toFinsupp___boxed), 6, 5);
lean_closure_set(v___x_103_, 0, lean_box(0));
lean_closure_set(v___x_103_, 1, lean_box(0));
lean_closure_set(v___x_103_, 2, v_inst_93_);
lean_closure_set(v___x_103_, 3, v_toZero_98_);
lean_closure_set(v___x_103_, 4, v_inst_95_);
if (v_isShared_101_ == 0)
{
lean_ctor_set(v___x_100_, 1, v___x_103_);
lean_ctor_set(v___x_100_, 0, v___x_102_);
v___x_105_ = v___x_100_;
goto v_reusejp_104_;
}
else
{
lean_object* v_reuseFailAlloc_106_; 
v_reuseFailAlloc_106_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_106_, 0, v___x_102_);
lean_ctor_set(v_reuseFailAlloc_106_, 1, v___x_103_);
v___x_105_ = v_reuseFailAlloc_106_;
goto v_reusejp_104_;
}
v_reusejp_104_:
{
return v___x_105_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_finsuppLequivDFinsupp___redArg___boxed(lean_object* v_inst_109_, lean_object* v_inst_110_, lean_object* v_inst_111_){
_start:
{
lean_object* v_res_112_; 
v_res_112_ = lp_mathlib_finsuppLequivDFinsupp___redArg(v_inst_109_, v_inst_110_, v_inst_111_);
lean_dec_ref(v_inst_110_);
return v_res_112_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_finsuppLequivDFinsupp(lean_object* v_00_u03b9_113_, lean_object* v_R_114_, lean_object* v_M_115_, lean_object* v_inst_116_, lean_object* v_inst_117_, lean_object* v_inst_118_, lean_object* v_inst_119_, lean_object* v_inst_120_){
_start:
{
lean_object* v___x_121_; 
v___x_121_ = lp_mathlib_finsuppLequivDFinsupp___redArg(v_inst_116_, v_inst_118_, v_inst_119_);
return v___x_121_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_finsuppLequivDFinsupp___boxed(lean_object* v_00_u03b9_122_, lean_object* v_R_123_, lean_object* v_M_124_, lean_object* v_inst_125_, lean_object* v_inst_126_, lean_object* v_inst_127_, lean_object* v_inst_128_, lean_object* v_inst_129_){
_start:
{
lean_object* v_res_130_; 
v_res_130_ = lp_mathlib_finsuppLequivDFinsupp(v_00_u03b9_122_, v_R_123_, v_M_124_, v_inst_125_, v_inst_126_, v_inst_127_, v_inst_128_, v_inst_129_);
lean_dec(v_inst_129_);
lean_dec_ref(v_inst_127_);
lean_dec_ref(v_inst_126_);
return v_res_130_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Module_Equiv_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_DFinsupp_Module(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finsupp_SMul(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Finsupp_ToDFinsupp(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Module_Equiv_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_DFinsupp_Module(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finsupp_SMul(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Finsupp_ToDFinsupp(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Equiv_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_DFinsupp_Module(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finsupp_SMul(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Finsupp_ToDFinsupp(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Equiv_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_DFinsupp_Module(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finsupp_SMul(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finsupp_ToDFinsupp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Finsupp_ToDFinsupp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Finsupp_ToDFinsupp(builtin);
}
#ifdef __cplusplus
}
#endif
