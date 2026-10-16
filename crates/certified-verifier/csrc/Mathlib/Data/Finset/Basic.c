// Lean compiler output
// Module: Mathlib.Data.Finset.Basic
// Imports: public import Init public meta import Init public import Mathlib.Data.Finset.Attach public import Mathlib.Data.Finset.Disjoint public import Mathlib.Data.Finset.Erase public import Mathlib.Data.Finset.Filter public import Mathlib.Data.Finset.Range public import Mathlib.Data.Finset.SDiff public import Mathlib.Data.Multiset.Basic public import Mathlib.Logic.Equiv.Set public import Mathlib.Order.Directed public import Mathlib.Order.Interval.Set.Defs public import Mathlib.Data.Set.SymmDiff
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
uint8_t lp_mathlib_Multiset_decidableMem___aux__1___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_subtypeEquivProp(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_sumPiEquivProdPi(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_List_chooseX___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_Set_union_x27___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_trans___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_piCongrLeft___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_chooseX___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_chooseX(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_choose___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_choose(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Equiv_Finset_union___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Finset_union___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Equiv_Finset_union___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_Finset_union___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Finset_union___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Finset_union(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Finset_union___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Finset_disjUnionEquiv___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Finset_disjUnionEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Finset_disjUnionEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Equiv_piFinsetUnion___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_piFinsetUnion___redArg___closed__0;
static lean_once_cell_t lp_mathlib_Equiv_piFinsetUnion___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Equiv_piFinsetUnion___redArg___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piFinsetUnion___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piFinsetUnion(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piFinsetUnion___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_equivToSet___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_equivToSet___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Finset_equivToSet___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Finset_equivToSet___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Finset_equivToSet___closed__0 = (const lean_object*)&lp_mathlib_Finset_equivToSet___closed__0_value;
static const lean_ctor_object lp_mathlib_Finset_equivToSet___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Finset_equivToSet___closed__0_value),((lean_object*)&lp_mathlib_Finset_equivToSet___closed__0_value)}};
static const lean_object* lp_mathlib_Finset_equivToSet___closed__1 = (const lean_object*)&lp_mathlib_Finset_equivToSet___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Finset_equivToSet(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_equivToSet___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_chooseX___redArg(lean_object* v_inst_1_, lean_object* v_l_2_){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lp_mathlib_List_chooseX___redArg(v_inst_1_, v_l_2_);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_chooseX(lean_object* v_00_u03b1_4_, lean_object* v_p_5_, lean_object* v_inst_6_, lean_object* v_l_7_, lean_object* v_hp_8_){
_start:
{
lean_object* v___x_9_; 
v___x_9_ = lp_mathlib_List_chooseX___redArg(v_inst_6_, v_l_7_);
return v___x_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_choose___redArg(lean_object* v_inst_10_, lean_object* v_l_11_){
_start:
{
lean_object* v___x_12_; 
v___x_12_ = lp_mathlib_List_chooseX___redArg(v_inst_10_, v_l_11_);
return v___x_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_choose(lean_object* v_00_u03b1_13_, lean_object* v_p_14_, lean_object* v_inst_15_, lean_object* v_l_16_, lean_object* v_hp_17_){
_start:
{
lean_object* v___x_18_; 
v___x_18_ = lp_mathlib_List_chooseX___redArg(v_inst_15_, v_l_16_);
return v___x_18_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Equiv_Finset_union___redArg___lam__0(lean_object* v_inst_19_, lean_object* v_s_20_, lean_object* v_a_21_){
_start:
{
uint8_t v___x_22_; 
v___x_22_ = lp_mathlib_Multiset_decidableMem___aux__1___redArg(v_inst_19_, v_a_21_, v_s_20_);
return v___x_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Finset_union___redArg___lam__0___boxed(lean_object* v_inst_23_, lean_object* v_s_24_, lean_object* v_a_25_){
_start:
{
uint8_t v_res_26_; lean_object* v_r_27_; 
v_res_26_ = lp_mathlib_Equiv_Finset_union___redArg___lam__0(v_inst_23_, v_s_24_, v_a_25_);
v_r_27_ = lean_box(v_res_26_);
return v_r_27_;
}
}
static lean_object* _init_lp_mathlib_Equiv_Finset_union___redArg___closed__0(void){
_start:
{
lean_object* v___x_28_; 
v___x_28_ = lp_mathlib_Equiv_subtypeEquivProp(lean_box(0), lean_box(0), lean_box(0), lean_box(0));
return v___x_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Finset_union___redArg(lean_object* v_inst_29_, lean_object* v_s_30_){
_start:
{
lean_object* v___f_31_; lean_object* v___x_32_; lean_object* v___x_33_; lean_object* v___x_34_; lean_object* v___x_35_; 
v___f_31_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_Finset_union___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_31_, 0, v_inst_29_);
lean_closure_set(v___f_31_, 1, v_s_30_);
v___x_32_ = lean_obj_once(&lp_mathlib_Equiv_Finset_union___redArg___closed__0, &lp_mathlib_Equiv_Finset_union___redArg___closed__0_once, _init_lp_mathlib_Equiv_Finset_union___redArg___closed__0);
v___x_33_ = lp_mathlib_Equiv_Set_union_x27___redArg(v___f_31_);
v___x_34_ = lp_mathlib_Equiv_trans___redArg(v___x_32_, v___x_33_);
v___x_35_ = lp_mathlib_Equiv_symm___redArg(v___x_34_);
return v___x_35_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Finset_union(lean_object* v_00_u03b1_36_, lean_object* v_inst_37_, lean_object* v_s_38_, lean_object* v_t_39_, lean_object* v_h_40_){
_start:
{
lean_object* v___x_41_; 
v___x_41_ = lp_mathlib_Equiv_Finset_union___redArg(v_inst_37_, v_s_38_);
return v___x_41_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Finset_union___boxed(lean_object* v_00_u03b1_42_, lean_object* v_inst_43_, lean_object* v_s_44_, lean_object* v_t_45_, lean_object* v_h_46_){
_start:
{
lean_object* v_res_47_; 
v_res_47_ = lp_mathlib_Equiv_Finset_union(v_00_u03b1_42_, v_inst_43_, v_s_44_, v_t_45_, v_h_46_);
lean_dec(v_t_45_);
return v_res_47_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Finset_disjUnionEquiv___redArg(lean_object* v_inst_48_, lean_object* v_s_49_){
_start:
{
lean_object* v___f_50_; lean_object* v___x_51_; lean_object* v___x_52_; lean_object* v___x_53_; lean_object* v___x_54_; 
v___f_50_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_Finset_union___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_50_, 0, v_inst_48_);
lean_closure_set(v___f_50_, 1, v_s_49_);
v___x_51_ = lean_obj_once(&lp_mathlib_Equiv_Finset_union___redArg___closed__0, &lp_mathlib_Equiv_Finset_union___redArg___closed__0_once, _init_lp_mathlib_Equiv_Finset_union___redArg___closed__0);
v___x_52_ = lp_mathlib_Equiv_Set_union_x27___redArg(v___f_50_);
v___x_53_ = lp_mathlib_Equiv_trans___redArg(v___x_51_, v___x_52_);
v___x_54_ = lp_mathlib_Equiv_symm___redArg(v___x_53_);
return v___x_54_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Finset_disjUnionEquiv(lean_object* v_00_u03b1_55_, lean_object* v_inst_56_, lean_object* v_s_57_, lean_object* v_t_58_, lean_object* v_h_59_){
_start:
{
lean_object* v___x_60_; 
v___x_60_ = lp_mathlib_Equiv_Finset_disjUnionEquiv___redArg(v_inst_56_, v_s_57_);
return v___x_60_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Finset_disjUnionEquiv___boxed(lean_object* v_00_u03b1_61_, lean_object* v_inst_62_, lean_object* v_s_63_, lean_object* v_t_64_, lean_object* v_h_65_){
_start:
{
lean_object* v_res_66_; 
v_res_66_ = lp_mathlib_Equiv_Finset_disjUnionEquiv(v_00_u03b1_61_, v_inst_62_, v_s_63_, v_t_64_, v_h_65_);
lean_dec(v_t_64_);
return v_res_66_;
}
}
static lean_object* _init_lp_mathlib_Equiv_piFinsetUnion___redArg___closed__0(void){
_start:
{
lean_object* v___x_67_; 
v___x_67_ = lp_mathlib_Equiv_sumPiEquivProdPi(lean_box(0), lean_box(0), lean_box(0));
return v___x_67_;
}
}
static lean_object* _init_lp_mathlib_Equiv_piFinsetUnion___redArg___closed__1(void){
_start:
{
lean_object* v___x_68_; lean_object* v___x_69_; 
v___x_68_ = lean_obj_once(&lp_mathlib_Equiv_piFinsetUnion___redArg___closed__0, &lp_mathlib_Equiv_piFinsetUnion___redArg___closed__0_once, _init_lp_mathlib_Equiv_piFinsetUnion___redArg___closed__0);
v___x_69_ = lp_mathlib_Equiv_symm___redArg(v___x_68_);
return v___x_69_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piFinsetUnion___redArg(lean_object* v_inst_70_, lean_object* v_s_71_){
_start:
{
lean_object* v_e_72_; lean_object* v___x_73_; lean_object* v___x_74_; lean_object* v___x_75_; 
v_e_72_ = lp_mathlib_Equiv_Finset_union___redArg(v_inst_70_, v_s_71_);
v___x_73_ = lean_obj_once(&lp_mathlib_Equiv_piFinsetUnion___redArg___closed__1, &lp_mathlib_Equiv_piFinsetUnion___redArg___closed__1_once, _init_lp_mathlib_Equiv_piFinsetUnion___redArg___closed__1);
v___x_74_ = lp_mathlib_Equiv_piCongrLeft___redArg(v_e_72_);
v___x_75_ = lp_mathlib_Equiv_trans___redArg(v___x_73_, v___x_74_);
return v___x_75_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piFinsetUnion(lean_object* v_00_u03b9_76_, lean_object* v_inst_77_, lean_object* v_00_u03b1_78_, lean_object* v_s_79_, lean_object* v_t_80_, lean_object* v_h_81_){
_start:
{
lean_object* v___x_82_; 
v___x_82_ = lp_mathlib_Equiv_piFinsetUnion___redArg(v_inst_77_, v_s_79_);
return v___x_82_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piFinsetUnion___boxed(lean_object* v_00_u03b9_83_, lean_object* v_inst_84_, lean_object* v_00_u03b1_85_, lean_object* v_s_86_, lean_object* v_t_87_, lean_object* v_h_88_){
_start:
{
lean_object* v_res_89_; 
v_res_89_ = lp_mathlib_Equiv_piFinsetUnion(v_00_u03b9_83_, v_inst_84_, v_00_u03b1_85_, v_s_86_, v_t_87_, v_h_88_);
lean_dec(v_t_87_);
return v_res_89_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_equivToSet___lam__0(lean_object* v_a_90_){
_start:
{
lean_inc(v_a_90_);
return v_a_90_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_equivToSet___lam__0___boxed(lean_object* v_a_91_){
_start:
{
lean_object* v_res_92_; 
v_res_92_ = lp_mathlib_Finset_equivToSet___lam__0(v_a_91_);
lean_dec(v_a_91_);
return v_res_92_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_equivToSet(lean_object* v_00_u03b1_96_, lean_object* v_s_97_){
_start:
{
lean_object* v___x_98_; 
v___x_98_ = ((lean_object*)(lp_mathlib_Finset_equivToSet___closed__1));
return v___x_98_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_equivToSet___boxed(lean_object* v_00_u03b1_99_, lean_object* v_s_100_){
_start:
{
lean_object* v_res_101_; 
v_res_101_ = lp_mathlib_Finset_equivToSet(v_00_u03b1_99_, v_s_100_);
lean_dec(v_s_100_);
return v_res_101_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Attach(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Disjoint(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Erase(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Filter(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Range(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_SDiff(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Multiset_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Logic_Equiv_Set(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Directed(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Interval_Set_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_SymmDiff(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
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
res = runtime_initialize_mathlib_Mathlib_Data_Finset_SDiff(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Multiset_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Equiv_Set(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Directed(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Interval_Set_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_SymmDiff(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Finset_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Finset_Attach(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Disjoint(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Erase(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Filter(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Range(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_SDiff(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Multiset_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Equiv_Set(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Directed(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Interval_Set_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_SymmDiff(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Finset_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
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
res = initialize_mathlib_Mathlib_Data_Finset_SDiff(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Multiset_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Equiv_Set(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Directed(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Interval_Set_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_SymmDiff(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Finset_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Finset_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
