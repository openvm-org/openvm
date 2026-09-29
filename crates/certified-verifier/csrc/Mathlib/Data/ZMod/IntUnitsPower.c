// Lean compiler output
// Module: Mathlib.Data.ZMod.IntUnitsPower
// Imports: public import Init public meta import Init public import Mathlib.Algebra.GroupWithZero.Divisibility public import Mathlib.Data.Int.Order.Units public import Mathlib.Data.ZMod.Basic
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
lean_object* lp_mathlib_Additive_ofMul(lean_object*);
lean_object* lp_mathlib_Additive_toMul(lean_object*);
lean_object* lp_mathlib_ZMod_val(lean_object*, lean_object*);
lean_object* l_Int_pow(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_instSMulZModOfNatNatAdditiveUnitsInt___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_instSMulZModOfNatNatAdditiveUnitsInt___lam__0___closed__0;
static lean_once_cell_t lp_mathlib_instSMulZModOfNatNatAdditiveUnitsInt___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_instSMulZModOfNatNatAdditiveUnitsInt___lam__0___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_instSMulZModOfNatNatAdditiveUnitsInt___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instSMulZModOfNatNatAdditiveUnitsInt___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_instSMulZModOfNatNatAdditiveUnitsInt___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_instSMulZModOfNatNatAdditiveUnitsInt___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_instSMulZModOfNatNatAdditiveUnitsInt___closed__0 = (const lean_object*)&lp_mathlib_instSMulZModOfNatNatAdditiveUnitsInt___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_instSMulZModOfNatNatAdditiveUnitsInt = (const lean_object*)&lp_mathlib_instSMulZModOfNatNatAdditiveUnitsInt___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_instModuleZModOfNatNatAdditiveUnitsInt___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instModuleZModOfNatNatAdditiveUnitsInt___lam__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_instModuleZModOfNatNatAdditiveUnitsInt___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_instModuleZModOfNatNatAdditiveUnitsInt___lam__0___boxed, .m_arity = 3, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(2) << 1) | 1))} };
static const lean_object* lp_mathlib_instModuleZModOfNatNatAdditiveUnitsInt___closed__0 = (const lean_object*)&lp_mathlib_instModuleZModOfNatNatAdditiveUnitsInt___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_instModuleZModOfNatNatAdditiveUnitsInt = (const lean_object*)&lp_mathlib_instModuleZModOfNatNatAdditiveUnitsInt___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Int_instUnitsPow___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_instUnitsPow___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_instUnitsPow(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_instUnitsPow___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_instSMulZModOfNatNatAdditiveUnitsInt___lam__0___closed__0(void){
_start:
{
lean_object* v___x_1_; 
v___x_1_ = lp_mathlib_Additive_toMul(lean_box(0));
return v___x_1_;
}
}
static lean_object* _init_lp_mathlib_instSMulZModOfNatNatAdditiveUnitsInt___lam__0___closed__1(void){
_start:
{
lean_object* v___x_2_; 
v___x_2_ = lp_mathlib_Additive_ofMul(lean_box(0));
return v___x_2_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instSMulZModOfNatNatAdditiveUnitsInt___lam__0(lean_object* v_z_3_, lean_object* v_au_4_){
_start:
{
lean_object* v___x_5_; lean_object* v_toFun_6_; lean_object* v___x_7_; lean_object* v_val_8_; lean_object* v_inv_9_; lean_object* v___x_11_; uint8_t v_isShared_12_; uint8_t v_isSharedCheck_23_; 
v___x_5_ = lean_obj_once(&lp_mathlib_instSMulZModOfNatNatAdditiveUnitsInt___lam__0___closed__0, &lp_mathlib_instSMulZModOfNatNatAdditiveUnitsInt___lam__0___closed__0_once, _init_lp_mathlib_instSMulZModOfNatNatAdditiveUnitsInt___lam__0___closed__0);
v_toFun_6_ = lean_ctor_get(v___x_5_, 0);
lean_inc(v_toFun_6_);
v___x_7_ = lean_apply_1(v_toFun_6_, v_au_4_);
v_val_8_ = lean_ctor_get(v___x_7_, 0);
v_inv_9_ = lean_ctor_get(v___x_7_, 1);
v_isSharedCheck_23_ = !lean_is_exclusive(v___x_7_);
if (v_isSharedCheck_23_ == 0)
{
v___x_11_ = v___x_7_;
v_isShared_12_ = v_isSharedCheck_23_;
goto v_resetjp_10_;
}
else
{
lean_inc(v_inv_9_);
lean_inc(v_val_8_);
lean_dec(v___x_7_);
v___x_11_ = lean_box(0);
v_isShared_12_ = v_isSharedCheck_23_;
goto v_resetjp_10_;
}
v_resetjp_10_:
{
lean_object* v___x_13_; lean_object* v_toFun_14_; lean_object* v___x_15_; lean_object* v___x_16_; lean_object* v___x_17_; lean_object* v___x_18_; lean_object* v___x_20_; 
v___x_13_ = lean_obj_once(&lp_mathlib_instSMulZModOfNatNatAdditiveUnitsInt___lam__0___closed__1, &lp_mathlib_instSMulZModOfNatNatAdditiveUnitsInt___lam__0___closed__1_once, _init_lp_mathlib_instSMulZModOfNatNatAdditiveUnitsInt___lam__0___closed__1);
v_toFun_14_ = lean_ctor_get(v___x_13_, 0);
v___x_15_ = lean_unsigned_to_nat(2u);
v___x_16_ = lp_mathlib_ZMod_val(v___x_15_, v_z_3_);
v___x_17_ = l_Int_pow(v_val_8_, v___x_16_);
lean_dec(v_val_8_);
v___x_18_ = l_Int_pow(v_inv_9_, v___x_16_);
lean_dec(v___x_16_);
lean_dec(v_inv_9_);
if (v_isShared_12_ == 0)
{
lean_ctor_set(v___x_11_, 1, v___x_18_);
lean_ctor_set(v___x_11_, 0, v___x_17_);
v___x_20_ = v___x_11_;
goto v_reusejp_19_;
}
else
{
lean_object* v_reuseFailAlloc_22_; 
v_reuseFailAlloc_22_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_22_, 0, v___x_17_);
lean_ctor_set(v_reuseFailAlloc_22_, 1, v___x_18_);
v___x_20_ = v_reuseFailAlloc_22_;
goto v_reusejp_19_;
}
v_reusejp_19_:
{
lean_object* v___x_21_; 
lean_inc(v_toFun_14_);
v___x_21_ = lean_apply_1(v_toFun_14_, v___x_20_);
return v___x_21_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_instSMulZModOfNatNatAdditiveUnitsInt___lam__0___boxed(lean_object* v_z_24_, lean_object* v_au_25_){
_start:
{
lean_object* v_res_26_; 
v_res_26_ = lp_mathlib_instSMulZModOfNatNatAdditiveUnitsInt___lam__0(v_z_24_, v_au_25_);
lean_dec(v_z_24_);
return v_res_26_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instModuleZModOfNatNatAdditiveUnitsInt___lam__0(lean_object* v___x_29_, lean_object* v_z_30_, lean_object* v_au_31_){
_start:
{
lean_object* v___x_32_; lean_object* v_toFun_33_; lean_object* v___x_34_; lean_object* v_val_35_; lean_object* v_inv_36_; lean_object* v___x_38_; uint8_t v_isShared_39_; uint8_t v_isSharedCheck_49_; 
v___x_32_ = lean_obj_once(&lp_mathlib_instSMulZModOfNatNatAdditiveUnitsInt___lam__0___closed__0, &lp_mathlib_instSMulZModOfNatNatAdditiveUnitsInt___lam__0___closed__0_once, _init_lp_mathlib_instSMulZModOfNatNatAdditiveUnitsInt___lam__0___closed__0);
v_toFun_33_ = lean_ctor_get(v___x_32_, 0);
lean_inc(v_toFun_33_);
v___x_34_ = lean_apply_1(v_toFun_33_, v_au_31_);
v_val_35_ = lean_ctor_get(v___x_34_, 0);
v_inv_36_ = lean_ctor_get(v___x_34_, 1);
v_isSharedCheck_49_ = !lean_is_exclusive(v___x_34_);
if (v_isSharedCheck_49_ == 0)
{
v___x_38_ = v___x_34_;
v_isShared_39_ = v_isSharedCheck_49_;
goto v_resetjp_37_;
}
else
{
lean_inc(v_inv_36_);
lean_inc(v_val_35_);
lean_dec(v___x_34_);
v___x_38_ = lean_box(0);
v_isShared_39_ = v_isSharedCheck_49_;
goto v_resetjp_37_;
}
v_resetjp_37_:
{
lean_object* v___x_40_; lean_object* v_toFun_41_; lean_object* v___x_42_; lean_object* v___x_43_; lean_object* v___x_44_; lean_object* v___x_46_; 
v___x_40_ = lean_obj_once(&lp_mathlib_instSMulZModOfNatNatAdditiveUnitsInt___lam__0___closed__1, &lp_mathlib_instSMulZModOfNatNatAdditiveUnitsInt___lam__0___closed__1_once, _init_lp_mathlib_instSMulZModOfNatNatAdditiveUnitsInt___lam__0___closed__1);
v_toFun_41_ = lean_ctor_get(v___x_40_, 0);
v___x_42_ = lp_mathlib_ZMod_val(v___x_29_, v_z_30_);
v___x_43_ = l_Int_pow(v_val_35_, v___x_42_);
lean_dec(v_val_35_);
v___x_44_ = l_Int_pow(v_inv_36_, v___x_42_);
lean_dec(v___x_42_);
lean_dec(v_inv_36_);
if (v_isShared_39_ == 0)
{
lean_ctor_set(v___x_38_, 1, v___x_44_);
lean_ctor_set(v___x_38_, 0, v___x_43_);
v___x_46_ = v___x_38_;
goto v_reusejp_45_;
}
else
{
lean_object* v_reuseFailAlloc_48_; 
v_reuseFailAlloc_48_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_48_, 0, v___x_43_);
lean_ctor_set(v_reuseFailAlloc_48_, 1, v___x_44_);
v___x_46_ = v_reuseFailAlloc_48_;
goto v_reusejp_45_;
}
v_reusejp_45_:
{
lean_object* v___x_47_; 
lean_inc(v_toFun_41_);
v___x_47_ = lean_apply_1(v_toFun_41_, v___x_46_);
return v___x_47_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_instModuleZModOfNatNatAdditiveUnitsInt___lam__0___boxed(lean_object* v___x_50_, lean_object* v_z_51_, lean_object* v_au_52_){
_start:
{
lean_object* v_res_53_; 
v_res_53_ = lp_mathlib_instModuleZModOfNatNatAdditiveUnitsInt___lam__0(v___x_50_, v_z_51_, v_au_52_);
lean_dec(v_z_51_);
lean_dec(v___x_50_);
return v_res_53_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_instUnitsPow___redArg___lam__0(lean_object* v_inst_57_, lean_object* v_u_58_, lean_object* v_r_59_){
_start:
{
lean_object* v___x_60_; lean_object* v_toFun_61_; lean_object* v___x_62_; lean_object* v_toFun_63_; lean_object* v___x_64_; lean_object* v___x_65_; lean_object* v___x_66_; 
v___x_60_ = lean_obj_once(&lp_mathlib_instSMulZModOfNatNatAdditiveUnitsInt___lam__0___closed__1, &lp_mathlib_instSMulZModOfNatNatAdditiveUnitsInt___lam__0___closed__1_once, _init_lp_mathlib_instSMulZModOfNatNatAdditiveUnitsInt___lam__0___closed__1);
v_toFun_61_ = lean_ctor_get(v___x_60_, 0);
v___x_62_ = lean_obj_once(&lp_mathlib_instSMulZModOfNatNatAdditiveUnitsInt___lam__0___closed__0, &lp_mathlib_instSMulZModOfNatNatAdditiveUnitsInt___lam__0___closed__0_once, _init_lp_mathlib_instSMulZModOfNatNatAdditiveUnitsInt___lam__0___closed__0);
v_toFun_63_ = lean_ctor_get(v___x_62_, 0);
lean_inc(v_toFun_61_);
v___x_64_ = lean_apply_1(v_toFun_61_, v_u_58_);
v___x_65_ = lean_apply_2(v_inst_57_, v_r_59_, v___x_64_);
lean_inc(v_toFun_63_);
v___x_66_ = lean_apply_1(v_toFun_63_, v___x_65_);
return v___x_66_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_instUnitsPow___redArg(lean_object* v_inst_67_){
_start:
{
lean_object* v___f_68_; 
v___f_68_ = lean_alloc_closure((void*)(lp_mathlib_Int_instUnitsPow___redArg___lam__0), 3, 1);
lean_closure_set(v___f_68_, 0, v_inst_67_);
return v___f_68_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_instUnitsPow(lean_object* v_R_69_, lean_object* v_inst_70_, lean_object* v_inst_71_){
_start:
{
lean_object* v___f_72_; 
v___f_72_ = lean_alloc_closure((void*)(lp_mathlib_Int_instUnitsPow___redArg___lam__0), 3, 1);
lean_closure_set(v___f_72_, 0, v_inst_71_);
return v___f_72_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_instUnitsPow___boxed(lean_object* v_R_73_, lean_object* v_inst_74_, lean_object* v_inst_75_){
_start:
{
lean_object* v_res_76_; 
v_res_76_ = lp_mathlib_Int_instUnitsPow(v_R_73_, v_inst_74_, v_inst_75_);
lean_dec_ref(v_inst_74_);
return v_res_76_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Divisibility(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Int_Order_Units(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_ZMod_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_ZMod_IntUnitsPower(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Divisibility(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Int_Order_Units(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_ZMod_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_ZMod_IntUnitsPower(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Divisibility(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Int_Order_Units(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_ZMod_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_ZMod_IntUnitsPower(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Divisibility(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Int_Order_Units(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_ZMod_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_ZMod_IntUnitsPower(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_ZMod_IntUnitsPower(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_ZMod_IntUnitsPower(builtin);
}
#ifdef __cplusplus
}
#endif
