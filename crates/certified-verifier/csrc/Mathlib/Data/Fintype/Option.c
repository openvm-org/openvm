// Lean compiler output
// Module: Mathlib.Data.Fintype.Option
// Imports: public import Init public meta import Init public import Mathlib.Data.Fintype.EquivFin public import Mathlib.Data.Finset.Option
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
lean_object* l_List_finRange(lean_object*);
lean_object* lp_mathlib_ULift_fintype___redArg(lean_object*);
lean_object* lp_mathlib_Finset_eraseNone(lean_object*);
uint8_t l_Option_instDecidableEq___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Nat_decEq___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_Finset_insertNone___lam__0(lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lp_mathlib_Fintype_truncEquivOfCardEq___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_lengthTR___redArg(lean_object*);
lean_object* l_instDecidableEqPEmpty___boxed(lean_object*, lean_object*);
lean_object* l_Nat_recCompiled___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Fintype_truncEquivFin___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_ulift(lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_trans___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Fintype_ofEquiv___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instFintypeOption___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instFintypeOption(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_fintypeOfOption___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_fintypeOfOption(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_fintypeOfOptionEquiv___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_fintypeOfOptionEquiv(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Fintype_truncRecEmptyOption___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_truncRecEmptyOption___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_truncRecEmptyOption___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Fintype_truncRecEmptyOption___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Nat_decEq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Fintype_truncRecEmptyOption___redArg___closed__0 = (const lean_object*)&lp_mathlib_Fintype_truncRecEmptyOption___redArg___closed__0_value;
static const lean_closure_object lp_mathlib_Fintype_truncRecEmptyOption___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Fintype_truncRecEmptyOption___redArg___lam__0___boxed, .m_arity = 3, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_Fintype_truncRecEmptyOption___redArg___closed__0_value)} };
static const lean_object* lp_mathlib_Fintype_truncRecEmptyOption___redArg___closed__1 = (const lean_object*)&lp_mathlib_Fintype_truncRecEmptyOption___redArg___closed__1_value;
static lean_once_cell_t lp_mathlib_Fintype_truncRecEmptyOption___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Fintype_truncRecEmptyOption___redArg___closed__2;
static lean_once_cell_t lp_mathlib_Fintype_truncRecEmptyOption___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Fintype_truncRecEmptyOption___redArg___closed__3;
static lean_once_cell_t lp_mathlib_Fintype_truncRecEmptyOption___redArg___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Fintype_truncRecEmptyOption___redArg___closed__4;
static lean_once_cell_t lp_mathlib_Fintype_truncRecEmptyOption___redArg___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Fintype_truncRecEmptyOption___redArg___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Fintype_truncRecEmptyOption___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_truncRecEmptyOption(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instFintypeOption___redArg(lean_object* v_inst_1_){
_start:
{
lean_object* v___x_2_; 
v___x_2_ = lp_mathlib_Finset_insertNone___lam__0(v_inst_1_);
return v___x_2_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instFintypeOption(lean_object* v_00_u03b1_3_, lean_object* v_inst_4_){
_start:
{
lean_object* v___x_5_; 
v___x_5_ = lp_mathlib_Finset_insertNone___lam__0(v_inst_4_);
return v___x_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_fintypeOfOption___redArg(lean_object* v_inst_6_){
_start:
{
lean_object* v___x_14__overap_7_; lean_object* v___x_8_; 
v___x_14__overap_7_ = lp_mathlib_Finset_eraseNone(lean_box(0));
v___x_8_ = lean_apply_1(v___x_14__overap_7_, v_inst_6_);
return v___x_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_fintypeOfOption(lean_object* v_00_u03b1_9_, lean_object* v_inst_10_){
_start:
{
lean_object* v___x_17__overap_11_; lean_object* v___x_12_; 
v___x_17__overap_11_ = lp_mathlib_Finset_eraseNone(lean_box(0));
v___x_12_ = lean_apply_1(v___x_17__overap_11_, v_inst_10_);
return v___x_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_fintypeOfOptionEquiv___redArg(lean_object* v_inst_13_, lean_object* v_f_14_){
_start:
{
lean_object* v___x_15_; lean_object* v___x_5__overap_16_; lean_object* v___x_17_; 
v___x_15_ = lp_mathlib_Fintype_ofEquiv___redArg(v_inst_13_, v_f_14_);
v___x_5__overap_16_ = lp_mathlib_Finset_eraseNone(lean_box(0));
v___x_17_ = lean_apply_1(v___x_5__overap_16_, v___x_15_);
return v___x_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_fintypeOfOptionEquiv(lean_object* v_00_u03b1_18_, lean_object* v_00_u03b2_19_, lean_object* v_inst_20_, lean_object* v_f_21_){
_start:
{
lean_object* v___x_22_; 
v___x_22_ = lp_mathlib_fintypeOfOptionEquiv___redArg(v_inst_20_, v_f_21_);
return v___x_22_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Fintype_truncRecEmptyOption___redArg___lam__0(lean_object* v___f_23_, lean_object* v_a_24_, lean_object* v_b_25_){
_start:
{
uint8_t v___x_26_; 
v___x_26_ = l_Option_instDecidableEq___redArg(v___f_23_, v_a_24_, v_b_25_);
return v___x_26_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_truncRecEmptyOption___redArg___lam__0___boxed(lean_object* v___f_27_, lean_object* v_a_28_, lean_object* v_b_29_){
_start:
{
uint8_t v_res_30_; lean_object* v_r_31_; 
v_res_30_ = lp_mathlib_Fintype_truncRecEmptyOption___redArg___lam__0(v___f_27_, v_a_28_, v_b_29_);
v_r_31_ = lean_box(v_res_30_);
return v_r_31_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_truncRecEmptyOption___redArg___lam__1(lean_object* v___f_32_, lean_object* v___f_33_, lean_object* v_h__option_34_, lean_object* v___f_35_, lean_object* v_of__equiv_36_, lean_object* v_n_37_, lean_object* v_ih_38_){
_start:
{
lean_object* v___x_39_; lean_object* v___x_40_; lean_object* v___x_41_; lean_object* v___x_42_; lean_object* v___x_43_; lean_object* v___x_44_; lean_object* v___x_45_; lean_object* v___x_46_; lean_object* v___x_47_; lean_object* v___x_48_; 
lean_inc(v_n_37_);
v___x_39_ = l_List_finRange(v_n_37_);
v___x_40_ = lp_mathlib_ULift_fintype___redArg(v___x_39_);
lean_inc(v___x_40_);
v___x_41_ = lp_mathlib_Finset_insertNone___lam__0(v___x_40_);
v___x_42_ = lean_unsigned_to_nat(1u);
v___x_43_ = lean_nat_add(v_n_37_, v___x_42_);
lean_dec(v_n_37_);
v___x_44_ = l_List_finRange(v___x_43_);
v___x_45_ = lp_mathlib_ULift_fintype___redArg(v___x_44_);
v___x_46_ = lp_mathlib_Fintype_truncEquivOfCardEq___redArg(v___x_41_, v___x_45_, v___f_32_, v___f_33_);
v___x_47_ = lean_apply_4(v_h__option_34_, lean_box(0), v___x_40_, v___f_35_, v_ih_38_);
v___x_48_ = lean_apply_4(v_of__equiv_36_, lean_box(0), lean_box(0), v___x_46_, v___x_47_);
return v___x_48_;
}
}
static lean_object* _init_lp_mathlib_Fintype_truncRecEmptyOption___redArg___closed__2(void){
_start:
{
lean_object* v___x_52_; lean_object* v___x_53_; 
v___x_52_ = lean_unsigned_to_nat(0u);
v___x_53_ = l_List_finRange(v___x_52_);
return v___x_53_;
}
}
static lean_object* _init_lp_mathlib_Fintype_truncRecEmptyOption___redArg___closed__3(void){
_start:
{
lean_object* v___x_54_; lean_object* v___x_55_; 
v___x_54_ = lean_obj_once(&lp_mathlib_Fintype_truncRecEmptyOption___redArg___closed__2, &lp_mathlib_Fintype_truncRecEmptyOption___redArg___closed__2_once, _init_lp_mathlib_Fintype_truncRecEmptyOption___redArg___closed__2);
v___x_55_ = lp_mathlib_ULift_fintype___redArg(v___x_54_);
return v___x_55_;
}
}
static lean_object* _init_lp_mathlib_Fintype_truncRecEmptyOption___redArg___closed__4(void){
_start:
{
lean_object* v___f_56_; lean_object* v___x_57_; lean_object* v___x_58_; lean_object* v___x_59_; lean_object* v___x_60_; 
v___f_56_ = ((lean_object*)(lp_mathlib_Fintype_truncRecEmptyOption___redArg___closed__0));
v___x_57_ = lean_alloc_closure((void*)(l_instDecidableEqPEmpty___boxed), 2, 0);
v___x_58_ = lean_obj_once(&lp_mathlib_Fintype_truncRecEmptyOption___redArg___closed__3, &lp_mathlib_Fintype_truncRecEmptyOption___redArg___closed__3_once, _init_lp_mathlib_Fintype_truncRecEmptyOption___redArg___closed__3);
v___x_59_ = lean_box(0);
v___x_60_ = lp_mathlib_Fintype_truncEquivOfCardEq___redArg(v___x_59_, v___x_58_, v___x_57_, v___f_56_);
return v___x_60_;
}
}
static lean_object* _init_lp_mathlib_Fintype_truncRecEmptyOption___redArg___closed__5(void){
_start:
{
lean_object* v___x_61_; 
v___x_61_ = lp_mathlib_Equiv_ulift(lean_box(0));
return v___x_61_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_truncRecEmptyOption___redArg(lean_object* v_of__equiv_62_, lean_object* v_h__empty_63_, lean_object* v_h__option_64_, lean_object* v_inst_65_, lean_object* v_inst_66_){
_start:
{
lean_object* v___f_67_; lean_object* v___f_68_; lean_object* v___f_69_; lean_object* v___x_70_; lean_object* v___x_71_; lean_object* v___x_72_; lean_object* v___x_73_; lean_object* v___x_74_; lean_object* v___x_75_; lean_object* v___x_76_; lean_object* v___x_77_; lean_object* v___x_78_; 
v___f_67_ = ((lean_object*)(lp_mathlib_Fintype_truncRecEmptyOption___redArg___closed__0));
v___f_68_ = ((lean_object*)(lp_mathlib_Fintype_truncRecEmptyOption___redArg___closed__1));
lean_inc_n(v_of__equiv_62_, 2);
v___f_69_ = lean_alloc_closure((void*)(lp_mathlib_Fintype_truncRecEmptyOption___redArg___lam__1), 7, 5);
lean_closure_set(v___f_69_, 0, v___f_68_);
lean_closure_set(v___f_69_, 1, v___f_67_);
lean_closure_set(v___f_69_, 2, v_h__option_64_);
lean_closure_set(v___f_69_, 3, v___f_67_);
lean_closure_set(v___f_69_, 4, v_of__equiv_62_);
v___x_70_ = l_List_lengthTR___redArg(v_inst_65_);
v___x_71_ = lean_obj_once(&lp_mathlib_Fintype_truncRecEmptyOption___redArg___closed__4, &lp_mathlib_Fintype_truncRecEmptyOption___redArg___closed__4_once, _init_lp_mathlib_Fintype_truncRecEmptyOption___redArg___closed__4);
v___x_72_ = lean_apply_4(v_of__equiv_62_, lean_box(0), lean_box(0), v___x_71_, v_h__empty_63_);
v___x_73_ = l_Nat_recCompiled___redArg(v___x_72_, v___f_69_, v___x_70_);
lean_dec(v___x_70_);
lean_dec(v___x_72_);
v___x_74_ = lp_mathlib_Fintype_truncEquivFin___redArg(v_inst_66_, v_inst_65_);
v___x_75_ = lean_obj_once(&lp_mathlib_Fintype_truncRecEmptyOption___redArg___closed__5, &lp_mathlib_Fintype_truncRecEmptyOption___redArg___closed__5_once, _init_lp_mathlib_Fintype_truncRecEmptyOption___redArg___closed__5);
v___x_76_ = lp_mathlib_Equiv_symm___redArg(v___x_74_);
v___x_77_ = lp_mathlib_Equiv_trans___redArg(v___x_75_, v___x_76_);
v___x_78_ = lean_apply_4(v_of__equiv_62_, lean_box(0), lean_box(0), v___x_77_, v___x_73_);
return v___x_78_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_truncRecEmptyOption(lean_object* v_P_79_, lean_object* v_of__equiv_80_, lean_object* v_h__empty_81_, lean_object* v_h__option_82_, lean_object* v_00_u03b1_83_, lean_object* v_inst_84_, lean_object* v_inst_85_){
_start:
{
lean_object* v___x_86_; 
v___x_86_ = lp_mathlib_Fintype_truncRecEmptyOption___redArg(v_of__equiv_80_, v_h__empty_81_, v_h__option_82_, v_inst_84_, v_inst_85_);
return v___x_86_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Fintype_EquivFin(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Option(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Fintype_Option(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fintype_EquivFin(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Option(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Fintype_Option(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Fintype_EquivFin(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Option(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Fintype_Option(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_EquivFin(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Option(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fintype_Option(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Fintype_Option(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Fintype_Option(builtin);
}
#ifdef __cplusplus
}
#endif
