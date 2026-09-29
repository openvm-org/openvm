// Lean compiler output
// Module: Mathlib.RingTheory.Localization.Basic
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Algebra.Tower public import Mathlib.Algebra.Field.IsField public import Mathlib.Algebra.GroupWithZero.NonZeroDivisors public import Mathlib.Basic.Finite.Prod public import Mathlib.GroupTheory.MonoidLocalization.MonoidWithZero public import Mathlib.RingTheory.Localization.Defs public import Mathlib.RingTheory.OreLocalization.Ring
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
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_extendScalarsOfIsLocalization___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_extendScalarsOfIsLocalization___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_extendScalarsOfIsLocalization(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_extendScalarsOfIsLocalization___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_extendScalarsOfIsLocalization___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_extendScalarsOfIsLocalization(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_extendScalarsOfIsLocalization___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_extendScalarsOfIsLocalization___redArg(lean_object* v_f_1_){
_start:
{
lean_inc(v_f_1_);
return v_f_1_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_extendScalarsOfIsLocalization___redArg___boxed(lean_object* v_f_2_){
_start:
{
lean_object* v_res_3_; 
v_res_3_ = lp_mathlib_AlgHom_extendScalarsOfIsLocalization___redArg(v_f_2_);
lean_dec(v_f_2_);
return v_res_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_extendScalarsOfIsLocalization(lean_object* v_R_4_, lean_object* v_A_5_, lean_object* v_B_6_, lean_object* v_inst_7_, lean_object* v_inst_8_, lean_object* v_inst_9_, lean_object* v_S_10_, lean_object* v_inst_11_, lean_object* v_inst_12_, lean_object* v_M_13_, lean_object* v_inst_14_, lean_object* v_inst_15_, lean_object* v_inst_16_, lean_object* v_inst_17_, lean_object* v_inst_18_, lean_object* v_inst_19_, lean_object* v_inst_20_, lean_object* v_f_21_){
_start:
{
lean_inc(v_f_21_);
return v_f_21_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgHom_extendScalarsOfIsLocalization___boxed(lean_object** _args){
lean_object* v_R_22_ = _args[0];
lean_object* v_A_23_ = _args[1];
lean_object* v_B_24_ = _args[2];
lean_object* v_inst_25_ = _args[3];
lean_object* v_inst_26_ = _args[4];
lean_object* v_inst_27_ = _args[5];
lean_object* v_S_28_ = _args[6];
lean_object* v_inst_29_ = _args[7];
lean_object* v_inst_30_ = _args[8];
lean_object* v_M_31_ = _args[9];
lean_object* v_inst_32_ = _args[10];
lean_object* v_inst_33_ = _args[11];
lean_object* v_inst_34_ = _args[12];
lean_object* v_inst_35_ = _args[13];
lean_object* v_inst_36_ = _args[14];
lean_object* v_inst_37_ = _args[15];
lean_object* v_inst_38_ = _args[16];
lean_object* v_f_39_ = _args[17];
_start:
{
lean_object* v_res_40_; 
v_res_40_ = lp_mathlib_AlgHom_extendScalarsOfIsLocalization(v_R_22_, v_A_23_, v_B_24_, v_inst_25_, v_inst_26_, v_inst_27_, v_S_28_, v_inst_29_, v_inst_30_, v_M_31_, v_inst_32_, v_inst_33_, v_inst_34_, v_inst_35_, v_inst_36_, v_inst_37_, v_inst_38_, v_f_39_);
lean_dec(v_f_39_);
lean_dec_ref(v_inst_37_);
lean_dec_ref(v_inst_36_);
lean_dec_ref(v_inst_34_);
lean_dec_ref(v_inst_33_);
lean_dec_ref(v_inst_30_);
lean_dec_ref(v_inst_29_);
lean_dec_ref(v_inst_27_);
lean_dec_ref(v_inst_26_);
lean_dec_ref(v_inst_25_);
return v_res_40_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_extendScalarsOfIsLocalization___redArg(lean_object* v_f_41_){
_start:
{
lean_object* v_toFun_42_; lean_object* v_invFun_43_; lean_object* v___x_45_; uint8_t v_isShared_46_; uint8_t v_isSharedCheck_50_; 
v_toFun_42_ = lean_ctor_get(v_f_41_, 0);
v_invFun_43_ = lean_ctor_get(v_f_41_, 1);
v_isSharedCheck_50_ = !lean_is_exclusive(v_f_41_);
if (v_isSharedCheck_50_ == 0)
{
v___x_45_ = v_f_41_;
v_isShared_46_ = v_isSharedCheck_50_;
goto v_resetjp_44_;
}
else
{
lean_inc(v_invFun_43_);
lean_inc(v_toFun_42_);
lean_dec(v_f_41_);
v___x_45_ = lean_box(0);
v_isShared_46_ = v_isSharedCheck_50_;
goto v_resetjp_44_;
}
v_resetjp_44_:
{
lean_object* v___x_48_; 
if (v_isShared_46_ == 0)
{
v___x_48_ = v___x_45_;
goto v_reusejp_47_;
}
else
{
lean_object* v_reuseFailAlloc_49_; 
v_reuseFailAlloc_49_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_49_, 0, v_toFun_42_);
lean_ctor_set(v_reuseFailAlloc_49_, 1, v_invFun_43_);
v___x_48_ = v_reuseFailAlloc_49_;
goto v_reusejp_47_;
}
v_reusejp_47_:
{
return v___x_48_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_extendScalarsOfIsLocalization(lean_object* v_R_51_, lean_object* v_A_52_, lean_object* v_B_53_, lean_object* v_inst_54_, lean_object* v_inst_55_, lean_object* v_inst_56_, lean_object* v_S_57_, lean_object* v_inst_58_, lean_object* v_inst_59_, lean_object* v_M_60_, lean_object* v_inst_61_, lean_object* v_inst_62_, lean_object* v_inst_63_, lean_object* v_inst_64_, lean_object* v_inst_65_, lean_object* v_inst_66_, lean_object* v_inst_67_, lean_object* v_f_68_){
_start:
{
lean_object* v___x_69_; 
v___x_69_ = lp_mathlib_AlgEquiv_extendScalarsOfIsLocalization___redArg(v_f_68_);
return v___x_69_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AlgEquiv_extendScalarsOfIsLocalization___boxed(lean_object** _args){
lean_object* v_R_70_ = _args[0];
lean_object* v_A_71_ = _args[1];
lean_object* v_B_72_ = _args[2];
lean_object* v_inst_73_ = _args[3];
lean_object* v_inst_74_ = _args[4];
lean_object* v_inst_75_ = _args[5];
lean_object* v_S_76_ = _args[6];
lean_object* v_inst_77_ = _args[7];
lean_object* v_inst_78_ = _args[8];
lean_object* v_M_79_ = _args[9];
lean_object* v_inst_80_ = _args[10];
lean_object* v_inst_81_ = _args[11];
lean_object* v_inst_82_ = _args[12];
lean_object* v_inst_83_ = _args[13];
lean_object* v_inst_84_ = _args[14];
lean_object* v_inst_85_ = _args[15];
lean_object* v_inst_86_ = _args[16];
lean_object* v_f_87_ = _args[17];
_start:
{
lean_object* v_res_88_; 
v_res_88_ = lp_mathlib_AlgEquiv_extendScalarsOfIsLocalization(v_R_70_, v_A_71_, v_B_72_, v_inst_73_, v_inst_74_, v_inst_75_, v_S_76_, v_inst_77_, v_inst_78_, v_M_79_, v_inst_80_, v_inst_81_, v_inst_82_, v_inst_83_, v_inst_84_, v_inst_85_, v_inst_86_, v_f_87_);
lean_dec_ref(v_inst_85_);
lean_dec_ref(v_inst_84_);
lean_dec_ref(v_inst_82_);
lean_dec_ref(v_inst_81_);
lean_dec_ref(v_inst_78_);
lean_dec_ref(v_inst_77_);
lean_dec_ref(v_inst_75_);
lean_dec_ref(v_inst_74_);
lean_dec_ref(v_inst_73_);
return v_res_88_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Tower(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Field_IsField(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_NonZeroDivisors(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Basic_Finite_Prod(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_MonoidLocalization_MonoidWithZero(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_RingTheory_Localization_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_RingTheory_OreLocalization_Ring(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_RingTheory_Localization_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Algebra_Tower(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Field_IsField(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_NonZeroDivisors(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Basic_Finite_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_MonoidLocalization_MonoidWithZero(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_RingTheory_Localization_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_RingTheory_OreLocalization_Ring(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_RingTheory_Localization_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Algebra_Tower(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Field_IsField(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_NonZeroDivisors(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Basic_Finite_Prod(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_MonoidLocalization_MonoidWithZero(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Localization_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_OreLocalization_Ring(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_RingTheory_Localization_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Algebra_Tower(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Field_IsField(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_NonZeroDivisors(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Basic_Finite_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_MonoidLocalization_MonoidWithZero(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Localization_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_OreLocalization_Ring(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_RingTheory_Localization_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_RingTheory_Localization_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_RingTheory_Localization_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
