// Lean compiler output
// Module: Mathlib.Order.ModularLattice
// Imports: public import Init public meta import Init public import Mathlib.Data.Set.Monotone public import Mathlib.Order.Cover public import Mathlib.Order.LatticeIntervals public import Mathlib.Order.GaloisConnection.Defs
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
lean_object* lp_mathlib_Equiv_subtypeEquivProp(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_trans___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_infIccOrderIsoIccSup___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_infIccOrderIsoIccSup___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_infIccOrderIsoIccSup___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_infIccOrderIsoIccSup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_infIccOrderIsoIccSup_x27___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_infIccOrderIsoIccSup_x27___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_infIccOrderIsoIccSup_x27___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_infIccOrderIsoIccSup_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_infIooOrderIsoIooSup___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_infIooOrderIsoIooSup___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_infIooOrderIsoIooSup___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_infIooOrderIsoIooSup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_infIooOrderIsoIooSup_x27___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_infIooOrderIsoIooSup_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsCompl_IicOrderIsoIci___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsCompl_IicOrderIsoIci(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsCompl_IicOrderIsoIci___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_infIccOrderIsoIccSup___redArg___lam__0(lean_object* v_inf_1_, lean_object* v_a_2_, lean_object* v_x_3_){
_start:
{
lean_object* v___x_4_; 
v___x_4_ = lean_apply_2(v_inf_1_, v_a_2_, v_x_3_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_infIccOrderIsoIccSup___redArg___lam__1(lean_object* v_toSemilatticeSup_5_, lean_object* v_b_6_, lean_object* v_x_7_){
_start:
{
lean_object* v_sup_8_; lean_object* v___x_9_; 
v_sup_8_ = lean_ctor_get(v_toSemilatticeSup_5_, 1);
lean_inc(v_sup_8_);
lean_dec_ref(v_toSemilatticeSup_5_);
v___x_9_ = lean_apply_2(v_sup_8_, v_x_7_, v_b_6_);
return v___x_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_infIccOrderIsoIccSup___redArg(lean_object* v_inst_10_, lean_object* v_a_11_, lean_object* v_b_12_){
_start:
{
lean_object* v_toSemilatticeSup_13_; lean_object* v_inf_14_; lean_object* v___x_16_; uint8_t v_isShared_17_; uint8_t v_isSharedCheck_23_; 
v_toSemilatticeSup_13_ = lean_ctor_get(v_inst_10_, 0);
v_inf_14_ = lean_ctor_get(v_inst_10_, 1);
v_isSharedCheck_23_ = !lean_is_exclusive(v_inst_10_);
if (v_isSharedCheck_23_ == 0)
{
v___x_16_ = v_inst_10_;
v_isShared_17_ = v_isSharedCheck_23_;
goto v_resetjp_15_;
}
else
{
lean_inc(v_inf_14_);
lean_inc(v_toSemilatticeSup_13_);
lean_dec(v_inst_10_);
v___x_16_ = lean_box(0);
v_isShared_17_ = v_isSharedCheck_23_;
goto v_resetjp_15_;
}
v_resetjp_15_:
{
lean_object* v___f_18_; lean_object* v___f_19_; lean_object* v___x_21_; 
v___f_18_ = lean_alloc_closure((void*)(lp_mathlib_infIccOrderIsoIccSup___redArg___lam__0), 3, 2);
lean_closure_set(v___f_18_, 0, v_inf_14_);
lean_closure_set(v___f_18_, 1, v_a_11_);
v___f_19_ = lean_alloc_closure((void*)(lp_mathlib_infIccOrderIsoIccSup___redArg___lam__1), 3, 2);
lean_closure_set(v___f_19_, 0, v_toSemilatticeSup_13_);
lean_closure_set(v___f_19_, 1, v_b_12_);
if (v_isShared_17_ == 0)
{
lean_ctor_set(v___x_16_, 1, v___f_18_);
lean_ctor_set(v___x_16_, 0, v___f_19_);
v___x_21_ = v___x_16_;
goto v_reusejp_20_;
}
else
{
lean_object* v_reuseFailAlloc_22_; 
v_reuseFailAlloc_22_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_22_, 0, v___f_19_);
lean_ctor_set(v_reuseFailAlloc_22_, 1, v___f_18_);
v___x_21_ = v_reuseFailAlloc_22_;
goto v_reusejp_20_;
}
v_reusejp_20_:
{
return v___x_21_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_infIccOrderIsoIccSup(lean_object* v_00_u03b1_24_, lean_object* v_inst_25_, lean_object* v_inst_26_, lean_object* v_a_27_, lean_object* v_b_28_){
_start:
{
lean_object* v___x_29_; 
v___x_29_ = lp_mathlib_infIccOrderIsoIccSup___redArg(v_inst_25_, v_a_27_, v_b_28_);
return v___x_29_;
}
}
static lean_object* _init_lp_mathlib_infIccOrderIsoIccSup_x27___redArg___closed__0(void){
_start:
{
lean_object* v___x_30_; 
v___x_30_ = lp_mathlib_Equiv_subtypeEquivProp(lean_box(0), lean_box(0), lean_box(0), lean_box(0));
return v___x_30_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_infIccOrderIsoIccSup_x27___redArg(lean_object* v_inst_31_, lean_object* v_a_32_, lean_object* v_b_33_){
_start:
{
lean_object* v___x_34_; lean_object* v___x_35_; lean_object* v___x_36_; lean_object* v___x_37_; 
v___x_34_ = lean_obj_once(&lp_mathlib_infIccOrderIsoIccSup_x27___redArg___closed__0, &lp_mathlib_infIccOrderIsoIccSup_x27___redArg___closed__0_once, _init_lp_mathlib_infIccOrderIsoIccSup_x27___redArg___closed__0);
v___x_35_ = lp_mathlib_infIccOrderIsoIccSup___redArg(v_inst_31_, v_b_33_, v_a_32_);
v___x_36_ = lp_mathlib_Equiv_trans___redArg(v___x_35_, v___x_34_);
v___x_37_ = lp_mathlib_Equiv_trans___redArg(v___x_34_, v___x_36_);
return v___x_37_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_infIccOrderIsoIccSup_x27(lean_object* v_00_u03b1_38_, lean_object* v_inst_39_, lean_object* v_inst_40_, lean_object* v_a_41_, lean_object* v_b_42_){
_start:
{
lean_object* v___x_43_; 
v___x_43_ = lp_mathlib_infIccOrderIsoIccSup_x27___redArg(v_inst_39_, v_a_41_, v_b_42_);
return v___x_43_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_infIooOrderIsoIooSup___redArg___lam__0(lean_object* v_inf_44_, lean_object* v_a_45_, lean_object* v_c_46_){
_start:
{
lean_object* v___x_47_; 
v___x_47_ = lean_apply_2(v_inf_44_, v_a_45_, v_c_46_);
return v___x_47_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_infIooOrderIsoIooSup___redArg___lam__1(lean_object* v_toSemilatticeSup_48_, lean_object* v_b_49_, lean_object* v_c_50_){
_start:
{
lean_object* v_sup_51_; lean_object* v___x_52_; 
v_sup_51_ = lean_ctor_get(v_toSemilatticeSup_48_, 1);
lean_inc(v_sup_51_);
lean_dec_ref(v_toSemilatticeSup_48_);
v___x_52_ = lean_apply_2(v_sup_51_, v_c_50_, v_b_49_);
return v___x_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_infIooOrderIsoIooSup___redArg(lean_object* v_inst_53_, lean_object* v_a_54_, lean_object* v_b_55_){
_start:
{
lean_object* v_toSemilatticeSup_56_; lean_object* v_inf_57_; lean_object* v___x_59_; uint8_t v_isShared_60_; uint8_t v_isSharedCheck_66_; 
v_toSemilatticeSup_56_ = lean_ctor_get(v_inst_53_, 0);
v_inf_57_ = lean_ctor_get(v_inst_53_, 1);
v_isSharedCheck_66_ = !lean_is_exclusive(v_inst_53_);
if (v_isSharedCheck_66_ == 0)
{
v___x_59_ = v_inst_53_;
v_isShared_60_ = v_isSharedCheck_66_;
goto v_resetjp_58_;
}
else
{
lean_inc(v_inf_57_);
lean_inc(v_toSemilatticeSup_56_);
lean_dec(v_inst_53_);
v___x_59_ = lean_box(0);
v_isShared_60_ = v_isSharedCheck_66_;
goto v_resetjp_58_;
}
v_resetjp_58_:
{
lean_object* v___f_61_; lean_object* v___f_62_; lean_object* v___x_64_; 
v___f_61_ = lean_alloc_closure((void*)(lp_mathlib_infIooOrderIsoIooSup___redArg___lam__0), 3, 2);
lean_closure_set(v___f_61_, 0, v_inf_57_);
lean_closure_set(v___f_61_, 1, v_a_54_);
v___f_62_ = lean_alloc_closure((void*)(lp_mathlib_infIooOrderIsoIooSup___redArg___lam__1), 3, 2);
lean_closure_set(v___f_62_, 0, v_toSemilatticeSup_56_);
lean_closure_set(v___f_62_, 1, v_b_55_);
if (v_isShared_60_ == 0)
{
lean_ctor_set(v___x_59_, 1, v___f_61_);
lean_ctor_set(v___x_59_, 0, v___f_62_);
v___x_64_ = v___x_59_;
goto v_reusejp_63_;
}
else
{
lean_object* v_reuseFailAlloc_65_; 
v_reuseFailAlloc_65_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_65_, 0, v___f_62_);
lean_ctor_set(v_reuseFailAlloc_65_, 1, v___f_61_);
v___x_64_ = v_reuseFailAlloc_65_;
goto v_reusejp_63_;
}
v_reusejp_63_:
{
return v___x_64_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_infIooOrderIsoIooSup(lean_object* v_00_u03b1_67_, lean_object* v_inst_68_, lean_object* v_inst_69_, lean_object* v_a_70_, lean_object* v_b_71_){
_start:
{
lean_object* v___x_72_; 
v___x_72_ = lp_mathlib_infIooOrderIsoIooSup___redArg(v_inst_68_, v_a_70_, v_b_71_);
return v___x_72_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_infIooOrderIsoIooSup_x27___redArg(lean_object* v_inst_73_, lean_object* v_a_74_, lean_object* v_b_75_){
_start:
{
lean_object* v___x_76_; lean_object* v___x_77_; lean_object* v___x_78_; lean_object* v___x_79_; 
v___x_76_ = lean_obj_once(&lp_mathlib_infIccOrderIsoIccSup_x27___redArg___closed__0, &lp_mathlib_infIccOrderIsoIccSup_x27___redArg___closed__0_once, _init_lp_mathlib_infIccOrderIsoIccSup_x27___redArg___closed__0);
v___x_77_ = lp_mathlib_infIooOrderIsoIooSup___redArg(v_inst_73_, v_b_75_, v_a_74_);
v___x_78_ = lp_mathlib_Equiv_trans___redArg(v___x_77_, v___x_76_);
v___x_79_ = lp_mathlib_Equiv_trans___redArg(v___x_76_, v___x_78_);
return v___x_79_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_infIooOrderIsoIooSup_x27(lean_object* v_00_u03b1_80_, lean_object* v_inst_81_, lean_object* v_inst_82_, lean_object* v_a_83_, lean_object* v_b_84_){
_start:
{
lean_object* v___x_85_; 
v___x_85_ = lp_mathlib_infIooOrderIsoIooSup_x27___redArg(v_inst_81_, v_a_83_, v_b_84_);
return v___x_85_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsCompl_IicOrderIsoIci___redArg(lean_object* v_inst_86_, lean_object* v_a_87_, lean_object* v_b_88_){
_start:
{
lean_object* v___x_89_; lean_object* v___x_90_; lean_object* v___x_91_; lean_object* v___x_92_; 
v___x_89_ = lean_obj_once(&lp_mathlib_infIccOrderIsoIccSup_x27___redArg___closed__0, &lp_mathlib_infIccOrderIsoIccSup_x27___redArg___closed__0_once, _init_lp_mathlib_infIccOrderIsoIccSup_x27___redArg___closed__0);
v___x_90_ = lp_mathlib_infIccOrderIsoIccSup___redArg(v_inst_86_, v_a_87_, v_b_88_);
v___x_91_ = lp_mathlib_Equiv_trans___redArg(v___x_90_, v___x_89_);
v___x_92_ = lp_mathlib_Equiv_trans___redArg(v___x_89_, v___x_91_);
return v___x_92_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsCompl_IicOrderIsoIci(lean_object* v_00_u03b1_93_, lean_object* v_inst_94_, lean_object* v_inst_95_, lean_object* v_inst_96_, lean_object* v_a_97_, lean_object* v_b_98_, lean_object* v_h_99_){
_start:
{
lean_object* v___x_100_; 
v___x_100_ = lp_mathlib_IsCompl_IicOrderIsoIci___redArg(v_inst_94_, v_a_97_, v_b_98_);
return v___x_100_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsCompl_IicOrderIsoIci___boxed(lean_object* v_00_u03b1_101_, lean_object* v_inst_102_, lean_object* v_inst_103_, lean_object* v_inst_104_, lean_object* v_a_105_, lean_object* v_b_106_, lean_object* v_h_107_){
_start:
{
lean_object* v_res_108_; 
v_res_108_ = lp_mathlib_IsCompl_IicOrderIsoIci(v_00_u03b1_101_, v_inst_102_, v_inst_103_, v_inst_104_, v_a_105_, v_b_106_, v_h_107_);
lean_dec_ref(v_inst_103_);
return v_res_108_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Monotone(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Cover(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_LatticeIntervals(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_GaloisConnection_Defs(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_ModularLattice(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Monotone(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Cover(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_LatticeIntervals(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_GaloisConnection_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_ModularLattice(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Set_Monotone(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Cover(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_LatticeIntervals(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_GaloisConnection_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_ModularLattice(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Monotone(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Cover(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_LatticeIntervals(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_GaloisConnection_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_ModularLattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_ModularLattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_ModularLattice(builtin);
}
#ifdef __cplusplus
}
#endif
