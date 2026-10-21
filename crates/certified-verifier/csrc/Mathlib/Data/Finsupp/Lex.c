// Lean compiler output
// Module: Mathlib.Data.Finsupp.Lex
// Imports: public import Init public meta import Init public import Mathlib.Data.Finsupp.Order public import Mathlib.Data.DFinsupp.Lex public import Mathlib.Data.Finsupp.ToDFinsupp
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
lean_object* lp_mathlib_LinearOrder_toLattice___redArg(lean_object*);
lean_object* lp_mathlib_Lattice_toSemilatticeInf___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_refl(lean_object*);
lean_object* lp_mathlib_DFinsupp_Lex_decidableLE___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Finsupp_toDFinsupp___redArg(lean_object*);
uint8_t lp_mathlib_decidableEqOfDecidableLE___redArg(lean_object*, lean_object*, lean_object*);
uint8_t lp_mathlib_DFinsupp_Lex_decidableLE___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lp_mathlib_DFinsupp_Lex_decidableLT___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
uint8_t lp_mathlib_DFinsupp_Colex_decidableLE___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lp_mathlib_DFinsupp_Colex_decidableLT___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_DFinsupp_Colex_decidableLE___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_instLTLex(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_instLTLex___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_instLTColex(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_instLTColex___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Finsupp_Lex_partialOrder___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Finsupp_Lex_partialOrder___closed__0 = (const lean_object*)&lp_mathlib_Finsupp_Lex_partialOrder___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_Lex_partialOrder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_Lex_partialOrder___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_Colex_partialOrder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_Colex_partialOrder___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_Lex_linearOrder___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_Lex_linearOrder___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_Lex_linearOrder___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_Lex_linearOrder___redArg___lam__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_Lex_linearOrder___redArg___lam__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_Lex_linearOrder___redArg___lam__3(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finsupp_Lex_linearOrder___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_Lex_linearOrder___redArg___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finsupp_Lex_linearOrder___redArg___lam__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_Lex_linearOrder___redArg___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finsupp_Lex_linearOrder___redArg___lam__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_Lex_linearOrder___redArg___lam__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_Lex_linearOrder___redArg___lam__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_Lex_linearOrder___redArg___lam__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finsupp_Lex_linearOrder___redArg___lam__9(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_Lex_linearOrder___redArg___lam__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Finsupp_Lex_linearOrder___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Finsupp_Lex_linearOrder___redArg___lam__2, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Finsupp_Lex_linearOrder___redArg___closed__0 = (const lean_object*)&lp_mathlib_Finsupp_Lex_linearOrder___redArg___closed__0_value;
static const lean_closure_object lp_mathlib_Finsupp_Lex_linearOrder___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Finsupp_Lex_linearOrder___redArg___lam__3, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Finsupp_Lex_linearOrder___redArg___closed__1 = (const lean_object*)&lp_mathlib_Finsupp_Lex_linearOrder___redArg___closed__1_value;
static lean_once_cell_t lp_mathlib_Finsupp_Lex_linearOrder___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Finsupp_Lex_linearOrder___redArg___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_Lex_linearOrder___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_Lex_linearOrder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finsupp_Colex_linearOrder___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_Colex_linearOrder___redArg___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finsupp_Colex_linearOrder___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_Colex_linearOrder___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finsupp_Colex_linearOrder___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_Colex_linearOrder___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_Colex_linearOrder___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_Colex_linearOrder___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finsupp_Colex_linearOrder___redArg___lam__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_Colex_linearOrder___redArg___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_Colex_linearOrder___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_Colex_linearOrder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_Lex_orderBot___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_Lex_orderBot___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_Lex_orderBot___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_Lex_orderBot___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_Lex_orderBot(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_Lex_orderBot___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_Colex_orderBot___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_Colex_orderBot___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_Colex_orderBot(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_Colex_orderBot___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_instLTLex(lean_object* v_00_u03b1_1_, lean_object* v_N_2_, lean_object* v_inst_3_, lean_object* v_inst_4_, lean_object* v_inst_5_){
_start:
{
lean_object* v___x_6_; 
v___x_6_ = lean_box(0);
return v___x_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_instLTLex___boxed(lean_object* v_00_u03b1_7_, lean_object* v_N_8_, lean_object* v_inst_9_, lean_object* v_inst_10_, lean_object* v_inst_11_){
_start:
{
lean_object* v_res_12_; 
v_res_12_ = lp_mathlib_Finsupp_instLTLex(v_00_u03b1_7_, v_N_8_, v_inst_9_, v_inst_10_, v_inst_11_);
lean_dec(v_inst_9_);
return v_res_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_instLTColex(lean_object* v_00_u03b1_13_, lean_object* v_N_14_, lean_object* v_inst_15_, lean_object* v_inst_16_, lean_object* v_inst_17_){
_start:
{
lean_object* v___x_18_; 
v___x_18_ = lean_box(0);
return v___x_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_instLTColex___boxed(lean_object* v_00_u03b1_19_, lean_object* v_N_20_, lean_object* v_inst_21_, lean_object* v_inst_22_, lean_object* v_inst_23_){
_start:
{
lean_object* v_res_24_; 
v_res_24_ = lp_mathlib_Finsupp_instLTColex(v_00_u03b1_19_, v_N_20_, v_inst_21_, v_inst_22_, v_inst_23_);
lean_dec(v_inst_21_);
return v_res_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_Lex_partialOrder(lean_object* v_00_u03b1_28_, lean_object* v_N_29_, lean_object* v_inst_30_, lean_object* v_inst_31_, lean_object* v_inst_32_){
_start:
{
lean_object* v___x_33_; 
v___x_33_ = ((lean_object*)(lp_mathlib_Finsupp_Lex_partialOrder___closed__0));
return v___x_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_Lex_partialOrder___boxed(lean_object* v_00_u03b1_34_, lean_object* v_N_35_, lean_object* v_inst_36_, lean_object* v_inst_37_, lean_object* v_inst_38_){
_start:
{
lean_object* v_res_39_; 
v_res_39_ = lp_mathlib_Finsupp_Lex_partialOrder(v_00_u03b1_34_, v_N_35_, v_inst_36_, v_inst_37_, v_inst_38_);
lean_dec_ref(v_inst_38_);
lean_dec_ref(v_inst_37_);
lean_dec(v_inst_36_);
return v_res_39_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_Colex_partialOrder(lean_object* v_00_u03b1_40_, lean_object* v_N_41_, lean_object* v_inst_42_, lean_object* v_inst_43_, lean_object* v_inst_44_){
_start:
{
lean_object* v___x_45_; 
v___x_45_ = ((lean_object*)(lp_mathlib_Finsupp_Lex_partialOrder___closed__0));
return v___x_45_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_Colex_partialOrder___boxed(lean_object* v_00_u03b1_46_, lean_object* v_N_47_, lean_object* v_inst_48_, lean_object* v_inst_49_, lean_object* v_inst_50_){
_start:
{
lean_object* v_res_51_; 
v_res_51_ = lp_mathlib_Finsupp_Colex_partialOrder(v_00_u03b1_46_, v_N_47_, v_inst_48_, v_inst_49_, v_inst_50_);
lean_dec_ref(v_inst_50_);
lean_dec_ref(v_inst_49_);
lean_dec(v_inst_48_);
return v_res_51_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_Lex_linearOrder___redArg___lam__0(lean_object* v_inst_52_, lean_object* v_i_53_){
_start:
{
lean_inc(v_inst_52_);
return v_inst_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_Lex_linearOrder___redArg___lam__0___boxed(lean_object* v_inst_54_, lean_object* v_i_55_){
_start:
{
lean_object* v_res_56_; 
v_res_56_ = lp_mathlib_Finsupp_Lex_linearOrder___redArg___lam__0(v_inst_54_, v_i_55_);
lean_dec(v_i_55_);
lean_dec(v_inst_54_);
return v_res_56_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_Lex_linearOrder___redArg___lam__1(lean_object* v_inst_57_, lean_object* v_i_58_){
_start:
{
lean_inc_ref(v_inst_57_);
return v_inst_57_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_Lex_linearOrder___redArg___lam__1___boxed(lean_object* v_inst_59_, lean_object* v_i_60_){
_start:
{
lean_object* v_res_61_; 
v_res_61_ = lp_mathlib_Finsupp_Lex_linearOrder___redArg___lam__1(v_inst_59_, v_i_60_);
lean_dec(v_i_60_);
lean_dec_ref(v_inst_59_);
return v_res_61_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_Lex_linearOrder___redArg___lam__2(lean_object* v_self_62_, lean_object* v___y_63_){
_start:
{
lean_object* v_toFun_64_; lean_object* v___x_65_; 
v_toFun_64_ = lean_ctor_get(v_self_62_, 0);
lean_inc(v_toFun_64_);
lean_dec_ref(v_self_62_);
v___x_65_ = lean_apply_1(v_toFun_64_, v___y_63_);
return v___x_65_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_Lex_linearOrder___redArg___lam__3(lean_object* v_self_66_, lean_object* v___y_67_){
_start:
{
lean_object* v_toFun_68_; lean_object* v___x_69_; 
v_toFun_68_ = lean_ctor_get(v_self_66_, 0);
lean_inc(v_toFun_68_);
lean_dec_ref(v_self_66_);
v___x_69_ = lean_apply_1(v_toFun_68_, v___y_67_);
return v___x_69_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finsupp_Lex_linearOrder___redArg___lam__4(lean_object* v___f_70_, lean_object* v_inst_71_, lean_object* v___f_72_, lean_object* v___f_73_, lean_object* v___x_74_, lean_object* v___f_75_, lean_object* v___x_76_, lean_object* v_x_77_, lean_object* v_y_78_){
_start:
{
lean_object* v___x_79_; lean_object* v___x_80_; lean_object* v___x_81_; lean_object* v___x_82_; lean_object* v___x_83_; lean_object* v___x_84_; lean_object* v___x_85_; uint8_t v___x_86_; 
v___x_79_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_Lex_decidableLE___boxed), 7, 5);
lean_closure_set(v___x_79_, 0, lean_box(0));
lean_closure_set(v___x_79_, 1, lean_box(0));
lean_closure_set(v___x_79_, 2, v___f_70_);
lean_closure_set(v___x_79_, 3, v_inst_71_);
lean_closure_set(v___x_79_, 4, v___f_72_);
lean_inc_ref(v___f_73_);
lean_inc_ref(v___x_74_);
v___x_80_ = lean_apply_2(v___f_73_, v___x_74_, v_x_77_);
v___x_81_ = lp_mathlib_Finsupp_toDFinsupp___redArg(v___x_80_);
lean_inc_ref(v___f_75_);
lean_inc_ref(v___x_76_);
v___x_82_ = lean_apply_2(v___f_75_, v___x_76_, v___x_81_);
v___x_83_ = lean_apply_2(v___f_73_, v___x_74_, v_y_78_);
v___x_84_ = lp_mathlib_Finsupp_toDFinsupp___redArg(v___x_83_);
v___x_85_ = lean_apply_2(v___f_75_, v___x_76_, v___x_84_);
v___x_86_ = lp_mathlib_decidableEqOfDecidableLE___redArg(v___x_79_, v___x_82_, v___x_85_);
return v___x_86_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_Lex_linearOrder___redArg___lam__4___boxed(lean_object* v___f_87_, lean_object* v_inst_88_, lean_object* v___f_89_, lean_object* v___f_90_, lean_object* v___x_91_, lean_object* v___f_92_, lean_object* v___x_93_, lean_object* v_x_94_, lean_object* v_y_95_){
_start:
{
uint8_t v_res_96_; lean_object* v_r_97_; 
v_res_96_ = lp_mathlib_Finsupp_Lex_linearOrder___redArg___lam__4(v___f_87_, v_inst_88_, v___f_89_, v___f_90_, v___x_91_, v___f_92_, v___x_93_, v_x_94_, v_y_95_);
v_r_97_ = lean_box(v_res_96_);
return v_r_97_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finsupp_Lex_linearOrder___redArg___lam__5(lean_object* v___f_98_, lean_object* v___x_99_, lean_object* v___f_100_, lean_object* v___x_101_, lean_object* v___f_102_, lean_object* v_inst_103_, lean_object* v___f_104_, lean_object* v_x_105_, lean_object* v_y_106_){
_start:
{
lean_object* v___x_107_; lean_object* v___x_108_; lean_object* v___x_109_; lean_object* v___x_110_; lean_object* v___x_111_; lean_object* v___x_112_; uint8_t v___x_113_; 
lean_inc_ref(v___f_98_);
lean_inc_ref(v___x_99_);
v___x_107_ = lean_apply_2(v___f_98_, v___x_99_, v_x_105_);
v___x_108_ = lp_mathlib_Finsupp_toDFinsupp___redArg(v___x_107_);
lean_inc_ref(v___f_100_);
lean_inc_ref(v___x_101_);
v___x_109_ = lean_apply_2(v___f_100_, v___x_101_, v___x_108_);
v___x_110_ = lean_apply_2(v___f_98_, v___x_99_, v_y_106_);
v___x_111_ = lp_mathlib_Finsupp_toDFinsupp___redArg(v___x_110_);
v___x_112_ = lean_apply_2(v___f_100_, v___x_101_, v___x_111_);
v___x_113_ = lp_mathlib_DFinsupp_Lex_decidableLE___redArg(v___f_102_, v_inst_103_, v___f_104_, v___x_109_, v___x_112_);
return v___x_113_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_Lex_linearOrder___redArg___lam__5___boxed(lean_object* v___f_114_, lean_object* v___x_115_, lean_object* v___f_116_, lean_object* v___x_117_, lean_object* v___f_118_, lean_object* v_inst_119_, lean_object* v___f_120_, lean_object* v_x_121_, lean_object* v_y_122_){
_start:
{
uint8_t v_res_123_; lean_object* v_r_124_; 
v_res_123_ = lp_mathlib_Finsupp_Lex_linearOrder___redArg___lam__5(v___f_114_, v___x_115_, v___f_116_, v___x_117_, v___f_118_, v_inst_119_, v___f_120_, v_x_121_, v_y_122_);
v_r_124_ = lean_box(v_res_123_);
return v_r_124_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finsupp_Lex_linearOrder___redArg___lam__6(lean_object* v___f_125_, lean_object* v___x_126_, lean_object* v___f_127_, lean_object* v___x_128_, lean_object* v___f_129_, lean_object* v_inst_130_, lean_object* v___f_131_, lean_object* v_x_132_, lean_object* v_y_133_){
_start:
{
lean_object* v___x_134_; lean_object* v___x_135_; lean_object* v___x_136_; lean_object* v___x_137_; lean_object* v___x_138_; lean_object* v___x_139_; uint8_t v___x_140_; 
lean_inc_ref(v___f_125_);
lean_inc_ref(v___x_126_);
v___x_134_ = lean_apply_2(v___f_125_, v___x_126_, v_x_132_);
v___x_135_ = lp_mathlib_Finsupp_toDFinsupp___redArg(v___x_134_);
lean_inc_ref(v___f_127_);
lean_inc_ref(v___x_128_);
v___x_136_ = lean_apply_2(v___f_127_, v___x_128_, v___x_135_);
v___x_137_ = lean_apply_2(v___f_125_, v___x_126_, v_y_133_);
v___x_138_ = lp_mathlib_Finsupp_toDFinsupp___redArg(v___x_137_);
v___x_139_ = lean_apply_2(v___f_127_, v___x_128_, v___x_138_);
v___x_140_ = lp_mathlib_DFinsupp_Lex_decidableLT___redArg(v___f_129_, v_inst_130_, v___f_131_, v___x_136_, v___x_139_);
return v___x_140_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_Lex_linearOrder___redArg___lam__6___boxed(lean_object* v___f_141_, lean_object* v___x_142_, lean_object* v___f_143_, lean_object* v___x_144_, lean_object* v___f_145_, lean_object* v_inst_146_, lean_object* v___f_147_, lean_object* v_x_148_, lean_object* v_y_149_){
_start:
{
uint8_t v_res_150_; lean_object* v_r_151_; 
v_res_150_ = lp_mathlib_Finsupp_Lex_linearOrder___redArg___lam__6(v___f_141_, v___x_142_, v___f_143_, v___x_144_, v___f_145_, v_inst_146_, v___f_147_, v_x_148_, v_y_149_);
v_r_151_ = lean_box(v_res_150_);
return v_r_151_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_Lex_linearOrder___redArg___lam__7(lean_object* v___f_152_, lean_object* v___x_153_, lean_object* v___f_154_, lean_object* v___x_155_, lean_object* v___f_156_, lean_object* v_inst_157_, lean_object* v___f_158_, lean_object* v_x_159_, lean_object* v_y_160_){
_start:
{
lean_object* v___x_161_; lean_object* v___x_162_; lean_object* v___x_163_; lean_object* v___x_164_; lean_object* v___x_165_; lean_object* v___x_166_; uint8_t v___x_167_; 
lean_inc_ref(v___f_152_);
lean_inc_ref(v_x_159_);
lean_inc_ref(v___x_153_);
v___x_161_ = lean_apply_2(v___f_152_, v___x_153_, v_x_159_);
v___x_162_ = lp_mathlib_Finsupp_toDFinsupp___redArg(v___x_161_);
lean_inc_ref(v___f_154_);
lean_inc_ref(v___x_155_);
v___x_163_ = lean_apply_2(v___f_154_, v___x_155_, v___x_162_);
lean_inc_ref(v_y_160_);
v___x_164_ = lean_apply_2(v___f_152_, v___x_153_, v_y_160_);
v___x_165_ = lp_mathlib_Finsupp_toDFinsupp___redArg(v___x_164_);
v___x_166_ = lean_apply_2(v___f_154_, v___x_155_, v___x_165_);
v___x_167_ = lp_mathlib_DFinsupp_Lex_decidableLE___redArg(v___f_156_, v_inst_157_, v___f_158_, v___x_163_, v___x_166_);
if (v___x_167_ == 0)
{
lean_dec_ref(v_x_159_);
return v_y_160_;
}
else
{
lean_dec_ref(v_y_160_);
return v_x_159_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_Lex_linearOrder___redArg___lam__8(lean_object* v___f_168_, lean_object* v___x_169_, lean_object* v___f_170_, lean_object* v___x_171_, lean_object* v___f_172_, lean_object* v_inst_173_, lean_object* v___f_174_, lean_object* v_x_175_, lean_object* v_y_176_){
_start:
{
lean_object* v___x_177_; lean_object* v___x_178_; lean_object* v___x_179_; lean_object* v___x_180_; lean_object* v___x_181_; lean_object* v___x_182_; uint8_t v___x_183_; 
lean_inc_ref(v___f_168_);
lean_inc_ref(v_x_175_);
lean_inc_ref(v___x_169_);
v___x_177_ = lean_apply_2(v___f_168_, v___x_169_, v_x_175_);
v___x_178_ = lp_mathlib_Finsupp_toDFinsupp___redArg(v___x_177_);
lean_inc_ref(v___f_170_);
lean_inc_ref(v___x_171_);
v___x_179_ = lean_apply_2(v___f_170_, v___x_171_, v___x_178_);
lean_inc_ref(v_y_176_);
v___x_180_ = lean_apply_2(v___f_168_, v___x_169_, v_y_176_);
v___x_181_ = lp_mathlib_Finsupp_toDFinsupp___redArg(v___x_180_);
v___x_182_ = lean_apply_2(v___f_170_, v___x_171_, v___x_181_);
v___x_183_ = lp_mathlib_DFinsupp_Lex_decidableLE___redArg(v___f_172_, v_inst_173_, v___f_174_, v___x_179_, v___x_182_);
if (v___x_183_ == 0)
{
lean_dec_ref(v_y_176_);
return v_x_175_;
}
else
{
lean_dec_ref(v_x_175_);
return v_y_176_;
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finsupp_Lex_linearOrder___redArg___lam__9(lean_object* v___f_184_, lean_object* v___x_185_, lean_object* v___f_186_, lean_object* v___x_187_, lean_object* v___f_188_, lean_object* v_inst_189_, lean_object* v___f_190_, lean_object* v_a_191_, lean_object* v_b_192_){
_start:
{
lean_object* v___x_193_; lean_object* v___x_194_; lean_object* v___x_195_; lean_object* v___x_196_; lean_object* v___x_197_; lean_object* v___x_198_; uint8_t v___x_199_; 
lean_inc_ref(v___f_184_);
lean_inc_ref(v___x_185_);
v___x_193_ = lean_apply_2(v___f_184_, v___x_185_, v_a_191_);
v___x_194_ = lp_mathlib_Finsupp_toDFinsupp___redArg(v___x_193_);
lean_inc_ref(v___f_186_);
lean_inc_ref(v___x_187_);
v___x_195_ = lean_apply_2(v___f_186_, v___x_187_, v___x_194_);
v___x_196_ = lean_apply_2(v___f_184_, v___x_185_, v_b_192_);
v___x_197_ = lp_mathlib_Finsupp_toDFinsupp___redArg(v___x_196_);
v___x_198_ = lean_apply_2(v___f_186_, v___x_187_, v___x_197_);
lean_inc_ref(v___x_198_);
lean_inc_ref(v___x_195_);
lean_inc_ref(v___f_190_);
lean_inc_ref(v_inst_189_);
lean_inc(v___f_188_);
v___x_199_ = lp_mathlib_DFinsupp_Lex_decidableLT___redArg(v___f_188_, v_inst_189_, v___f_190_, v___x_195_, v___x_198_);
if (v___x_199_ == 0)
{
lean_object* v___x_200_; uint8_t v___x_201_; 
v___x_200_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_Lex_decidableLE___boxed), 7, 5);
lean_closure_set(v___x_200_, 0, lean_box(0));
lean_closure_set(v___x_200_, 1, lean_box(0));
lean_closure_set(v___x_200_, 2, v___f_188_);
lean_closure_set(v___x_200_, 3, v_inst_189_);
lean_closure_set(v___x_200_, 4, v___f_190_);
v___x_201_ = lp_mathlib_decidableEqOfDecidableLE___redArg(v___x_200_, v___x_195_, v___x_198_);
if (v___x_201_ == 0)
{
uint8_t v___x_202_; 
v___x_202_ = 2;
return v___x_202_;
}
else
{
uint8_t v___x_203_; 
v___x_203_ = 1;
return v___x_203_;
}
}
else
{
uint8_t v___x_204_; 
lean_dec_ref(v___x_198_);
lean_dec_ref(v___x_195_);
lean_dec_ref(v___f_190_);
lean_dec_ref(v_inst_189_);
lean_dec(v___f_188_);
v___x_204_ = 0;
return v___x_204_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_Lex_linearOrder___redArg___lam__9___boxed(lean_object* v___f_205_, lean_object* v___x_206_, lean_object* v___f_207_, lean_object* v___x_208_, lean_object* v___f_209_, lean_object* v_inst_210_, lean_object* v___f_211_, lean_object* v_a_212_, lean_object* v_b_213_){
_start:
{
uint8_t v_res_214_; lean_object* v_r_215_; 
v_res_214_ = lp_mathlib_Finsupp_Lex_linearOrder___redArg___lam__9(v___f_205_, v___x_206_, v___f_207_, v___x_208_, v___f_209_, v_inst_210_, v___f_211_, v_a_212_, v_b_213_);
v_r_215_ = lean_box(v_res_214_);
return v_r_215_;
}
}
static lean_object* _init_lp_mathlib_Finsupp_Lex_linearOrder___redArg___closed__2(void){
_start:
{
lean_object* v___x_218_; 
v___x_218_ = lp_mathlib_Equiv_refl(lean_box(0));
return v___x_218_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_Lex_linearOrder___redArg(lean_object* v_inst_219_, lean_object* v_inst_220_, lean_object* v_inst_221_){
_start:
{
lean_object* v___x_222_; lean_object* v___x_223_; lean_object* v_toPartialOrder_224_; lean_object* v___f_225_; lean_object* v___f_226_; lean_object* v___f_227_; lean_object* v___f_228_; lean_object* v___x_229_; lean_object* v___x_230_; lean_object* v___f_231_; lean_object* v___f_232_; lean_object* v___f_233_; lean_object* v___f_234_; lean_object* v___f_235_; lean_object* v___f_236_; lean_object* v___x_237_; 
v___x_222_ = lp_mathlib_LinearOrder_toLattice___redArg(v_inst_221_);
v___x_223_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v___x_222_);
v_toPartialOrder_224_ = lean_ctor_get(v___x_223_, 0);
lean_inc_ref(v_toPartialOrder_224_);
lean_dec_ref(v___x_223_);
lean_inc(v_inst_219_);
v___f_225_ = lean_alloc_closure((void*)(lp_mathlib_Finsupp_Lex_linearOrder___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_225_, 0, v_inst_219_);
v___f_226_ = lean_alloc_closure((void*)(lp_mathlib_Finsupp_Lex_linearOrder___redArg___lam__1___boxed), 2, 1);
lean_closure_set(v___f_226_, 0, v_inst_221_);
v___f_227_ = ((lean_object*)(lp_mathlib_Finsupp_Lex_linearOrder___redArg___closed__0));
v___f_228_ = ((lean_object*)(lp_mathlib_Finsupp_Lex_linearOrder___redArg___closed__1));
v___x_229_ = lp_mathlib_Finsupp_Lex_partialOrder(lean_box(0), lean_box(0), v_inst_219_, v_inst_220_, v_toPartialOrder_224_);
lean_dec_ref(v_toPartialOrder_224_);
lean_dec(v_inst_219_);
v___x_230_ = lean_obj_once(&lp_mathlib_Finsupp_Lex_linearOrder___redArg___closed__2, &lp_mathlib_Finsupp_Lex_linearOrder___redArg___closed__2_once, _init_lp_mathlib_Finsupp_Lex_linearOrder___redArg___closed__2);
lean_inc_ref_n(v___f_226_, 5);
lean_inc_ref_n(v_inst_220_, 5);
lean_inc_ref_n(v___f_225_, 5);
v___f_231_ = lean_alloc_closure((void*)(lp_mathlib_Finsupp_Lex_linearOrder___redArg___lam__4___boxed), 9, 7);
lean_closure_set(v___f_231_, 0, v___f_225_);
lean_closure_set(v___f_231_, 1, v_inst_220_);
lean_closure_set(v___f_231_, 2, v___f_226_);
lean_closure_set(v___f_231_, 3, v___f_228_);
lean_closure_set(v___f_231_, 4, v___x_230_);
lean_closure_set(v___f_231_, 5, v___f_227_);
lean_closure_set(v___f_231_, 6, v___x_230_);
v___f_232_ = lean_alloc_closure((void*)(lp_mathlib_Finsupp_Lex_linearOrder___redArg___lam__5___boxed), 9, 7);
lean_closure_set(v___f_232_, 0, v___f_228_);
lean_closure_set(v___f_232_, 1, v___x_230_);
lean_closure_set(v___f_232_, 2, v___f_227_);
lean_closure_set(v___f_232_, 3, v___x_230_);
lean_closure_set(v___f_232_, 4, v___f_225_);
lean_closure_set(v___f_232_, 5, v_inst_220_);
lean_closure_set(v___f_232_, 6, v___f_226_);
v___f_233_ = lean_alloc_closure((void*)(lp_mathlib_Finsupp_Lex_linearOrder___redArg___lam__6___boxed), 9, 7);
lean_closure_set(v___f_233_, 0, v___f_228_);
lean_closure_set(v___f_233_, 1, v___x_230_);
lean_closure_set(v___f_233_, 2, v___f_227_);
lean_closure_set(v___f_233_, 3, v___x_230_);
lean_closure_set(v___f_233_, 4, v___f_225_);
lean_closure_set(v___f_233_, 5, v_inst_220_);
lean_closure_set(v___f_233_, 6, v___f_226_);
v___f_234_ = lean_alloc_closure((void*)(lp_mathlib_Finsupp_Lex_linearOrder___redArg___lam__7), 9, 7);
lean_closure_set(v___f_234_, 0, v___f_228_);
lean_closure_set(v___f_234_, 1, v___x_230_);
lean_closure_set(v___f_234_, 2, v___f_227_);
lean_closure_set(v___f_234_, 3, v___x_230_);
lean_closure_set(v___f_234_, 4, v___f_225_);
lean_closure_set(v___f_234_, 5, v_inst_220_);
lean_closure_set(v___f_234_, 6, v___f_226_);
v___f_235_ = lean_alloc_closure((void*)(lp_mathlib_Finsupp_Lex_linearOrder___redArg___lam__8), 9, 7);
lean_closure_set(v___f_235_, 0, v___f_228_);
lean_closure_set(v___f_235_, 1, v___x_230_);
lean_closure_set(v___f_235_, 2, v___f_227_);
lean_closure_set(v___f_235_, 3, v___x_230_);
lean_closure_set(v___f_235_, 4, v___f_225_);
lean_closure_set(v___f_235_, 5, v_inst_220_);
lean_closure_set(v___f_235_, 6, v___f_226_);
v___f_236_ = lean_alloc_closure((void*)(lp_mathlib_Finsupp_Lex_linearOrder___redArg___lam__9___boxed), 9, 7);
lean_closure_set(v___f_236_, 0, v___f_228_);
lean_closure_set(v___f_236_, 1, v___x_230_);
lean_closure_set(v___f_236_, 2, v___f_227_);
lean_closure_set(v___f_236_, 3, v___x_230_);
lean_closure_set(v___f_236_, 4, v___f_225_);
lean_closure_set(v___f_236_, 5, v_inst_220_);
lean_closure_set(v___f_236_, 6, v___f_226_);
v___x_237_ = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(v___x_237_, 0, v___x_229_);
lean_ctor_set(v___x_237_, 1, v___f_234_);
lean_ctor_set(v___x_237_, 2, v___f_235_);
lean_ctor_set(v___x_237_, 3, v___f_236_);
lean_ctor_set(v___x_237_, 4, v___f_232_);
lean_ctor_set(v___x_237_, 5, v___f_231_);
lean_ctor_set(v___x_237_, 6, v___f_233_);
return v___x_237_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_Lex_linearOrder(lean_object* v_00_u03b1_238_, lean_object* v_N_239_, lean_object* v_inst_240_, lean_object* v_inst_241_, lean_object* v_inst_242_){
_start:
{
lean_object* v___x_243_; 
v___x_243_ = lp_mathlib_Finsupp_Lex_linearOrder___redArg(v_inst_240_, v_inst_241_, v_inst_242_);
return v___x_243_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finsupp_Colex_linearOrder___redArg___lam__4(lean_object* v___f_244_, lean_object* v_inst_245_, lean_object* v___f_246_, lean_object* v___f_247_, lean_object* v___x_248_, lean_object* v___f_249_, lean_object* v___x_250_, lean_object* v_x_251_, lean_object* v_y_252_){
_start:
{
lean_object* v___x_253_; lean_object* v___x_254_; lean_object* v___x_255_; lean_object* v___x_256_; lean_object* v___x_257_; lean_object* v___x_258_; lean_object* v___x_259_; uint8_t v___x_260_; 
v___x_253_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_Colex_decidableLE___boxed), 7, 5);
lean_closure_set(v___x_253_, 0, lean_box(0));
lean_closure_set(v___x_253_, 1, lean_box(0));
lean_closure_set(v___x_253_, 2, v___f_244_);
lean_closure_set(v___x_253_, 3, v_inst_245_);
lean_closure_set(v___x_253_, 4, v___f_246_);
lean_inc_ref(v___f_247_);
lean_inc_ref(v___x_248_);
v___x_254_ = lean_apply_2(v___f_247_, v___x_248_, v_x_251_);
v___x_255_ = lp_mathlib_Finsupp_toDFinsupp___redArg(v___x_254_);
lean_inc_ref(v___f_249_);
lean_inc_ref(v___x_250_);
v___x_256_ = lean_apply_2(v___f_249_, v___x_250_, v___x_255_);
v___x_257_ = lean_apply_2(v___f_247_, v___x_248_, v_y_252_);
v___x_258_ = lp_mathlib_Finsupp_toDFinsupp___redArg(v___x_257_);
v___x_259_ = lean_apply_2(v___f_249_, v___x_250_, v___x_258_);
v___x_260_ = lp_mathlib_decidableEqOfDecidableLE___redArg(v___x_253_, v___x_256_, v___x_259_);
return v___x_260_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_Colex_linearOrder___redArg___lam__4___boxed(lean_object* v___f_261_, lean_object* v_inst_262_, lean_object* v___f_263_, lean_object* v___f_264_, lean_object* v___x_265_, lean_object* v___f_266_, lean_object* v___x_267_, lean_object* v_x_268_, lean_object* v_y_269_){
_start:
{
uint8_t v_res_270_; lean_object* v_r_271_; 
v_res_270_ = lp_mathlib_Finsupp_Colex_linearOrder___redArg___lam__4(v___f_261_, v_inst_262_, v___f_263_, v___f_264_, v___x_265_, v___f_266_, v___x_267_, v_x_268_, v_y_269_);
v_r_271_ = lean_box(v_res_270_);
return v_r_271_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finsupp_Colex_linearOrder___redArg___lam__0(lean_object* v___f_272_, lean_object* v___x_273_, lean_object* v___f_274_, lean_object* v___x_275_, lean_object* v___f_276_, lean_object* v_inst_277_, lean_object* v___f_278_, lean_object* v_x_279_, lean_object* v_y_280_){
_start:
{
lean_object* v___x_281_; lean_object* v___x_282_; lean_object* v___x_283_; lean_object* v___x_284_; lean_object* v___x_285_; lean_object* v___x_286_; uint8_t v___x_287_; 
lean_inc_ref(v___f_272_);
lean_inc_ref(v___x_273_);
v___x_281_ = lean_apply_2(v___f_272_, v___x_273_, v_x_279_);
v___x_282_ = lp_mathlib_Finsupp_toDFinsupp___redArg(v___x_281_);
lean_inc_ref(v___f_274_);
lean_inc_ref(v___x_275_);
v___x_283_ = lean_apply_2(v___f_274_, v___x_275_, v___x_282_);
v___x_284_ = lean_apply_2(v___f_272_, v___x_273_, v_y_280_);
v___x_285_ = lp_mathlib_Finsupp_toDFinsupp___redArg(v___x_284_);
v___x_286_ = lean_apply_2(v___f_274_, v___x_275_, v___x_285_);
v___x_287_ = lp_mathlib_DFinsupp_Colex_decidableLE___redArg(v___f_276_, v_inst_277_, v___f_278_, v___x_283_, v___x_286_);
return v___x_287_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_Colex_linearOrder___redArg___lam__0___boxed(lean_object* v___f_288_, lean_object* v___x_289_, lean_object* v___f_290_, lean_object* v___x_291_, lean_object* v___f_292_, lean_object* v_inst_293_, lean_object* v___f_294_, lean_object* v_x_295_, lean_object* v_y_296_){
_start:
{
uint8_t v_res_297_; lean_object* v_r_298_; 
v_res_297_ = lp_mathlib_Finsupp_Colex_linearOrder___redArg___lam__0(v___f_288_, v___x_289_, v___f_290_, v___x_291_, v___f_292_, v_inst_293_, v___f_294_, v_x_295_, v_y_296_);
v_r_298_ = lean_box(v_res_297_);
return v_r_298_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finsupp_Colex_linearOrder___redArg___lam__1(lean_object* v___f_299_, lean_object* v___x_300_, lean_object* v___f_301_, lean_object* v___x_302_, lean_object* v___f_303_, lean_object* v_inst_304_, lean_object* v___f_305_, lean_object* v_x_306_, lean_object* v_y_307_){
_start:
{
lean_object* v___x_308_; lean_object* v___x_309_; lean_object* v___x_310_; lean_object* v___x_311_; lean_object* v___x_312_; lean_object* v___x_313_; uint8_t v___x_314_; 
lean_inc_ref(v___f_299_);
lean_inc_ref(v___x_300_);
v___x_308_ = lean_apply_2(v___f_299_, v___x_300_, v_x_306_);
v___x_309_ = lp_mathlib_Finsupp_toDFinsupp___redArg(v___x_308_);
lean_inc_ref(v___f_301_);
lean_inc_ref(v___x_302_);
v___x_310_ = lean_apply_2(v___f_301_, v___x_302_, v___x_309_);
v___x_311_ = lean_apply_2(v___f_299_, v___x_300_, v_y_307_);
v___x_312_ = lp_mathlib_Finsupp_toDFinsupp___redArg(v___x_311_);
v___x_313_ = lean_apply_2(v___f_301_, v___x_302_, v___x_312_);
v___x_314_ = lp_mathlib_DFinsupp_Colex_decidableLT___redArg(v___f_303_, v_inst_304_, v___f_305_, v___x_310_, v___x_313_);
return v___x_314_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_Colex_linearOrder___redArg___lam__1___boxed(lean_object* v___f_315_, lean_object* v___x_316_, lean_object* v___f_317_, lean_object* v___x_318_, lean_object* v___f_319_, lean_object* v_inst_320_, lean_object* v___f_321_, lean_object* v_x_322_, lean_object* v_y_323_){
_start:
{
uint8_t v_res_324_; lean_object* v_r_325_; 
v_res_324_ = lp_mathlib_Finsupp_Colex_linearOrder___redArg___lam__1(v___f_315_, v___x_316_, v___f_317_, v___x_318_, v___f_319_, v_inst_320_, v___f_321_, v_x_322_, v_y_323_);
v_r_325_ = lean_box(v_res_324_);
return v_r_325_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_Colex_linearOrder___redArg___lam__2(lean_object* v___f_326_, lean_object* v___x_327_, lean_object* v___f_328_, lean_object* v___x_329_, lean_object* v___f_330_, lean_object* v_inst_331_, lean_object* v___f_332_, lean_object* v_x_333_, lean_object* v_y_334_){
_start:
{
lean_object* v___x_335_; lean_object* v___x_336_; lean_object* v___x_337_; lean_object* v___x_338_; lean_object* v___x_339_; lean_object* v___x_340_; uint8_t v___x_341_; 
lean_inc_ref(v___f_326_);
lean_inc_ref(v_x_333_);
lean_inc_ref(v___x_327_);
v___x_335_ = lean_apply_2(v___f_326_, v___x_327_, v_x_333_);
v___x_336_ = lp_mathlib_Finsupp_toDFinsupp___redArg(v___x_335_);
lean_inc_ref(v___f_328_);
lean_inc_ref(v___x_329_);
v___x_337_ = lean_apply_2(v___f_328_, v___x_329_, v___x_336_);
lean_inc_ref(v_y_334_);
v___x_338_ = lean_apply_2(v___f_326_, v___x_327_, v_y_334_);
v___x_339_ = lp_mathlib_Finsupp_toDFinsupp___redArg(v___x_338_);
v___x_340_ = lean_apply_2(v___f_328_, v___x_329_, v___x_339_);
v___x_341_ = lp_mathlib_DFinsupp_Colex_decidableLE___redArg(v___f_330_, v_inst_331_, v___f_332_, v___x_337_, v___x_340_);
if (v___x_341_ == 0)
{
lean_dec_ref(v_x_333_);
return v_y_334_;
}
else
{
lean_dec_ref(v_y_334_);
return v_x_333_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_Colex_linearOrder___redArg___lam__3(lean_object* v___f_342_, lean_object* v___x_343_, lean_object* v___f_344_, lean_object* v___x_345_, lean_object* v___f_346_, lean_object* v_inst_347_, lean_object* v___f_348_, lean_object* v_x_349_, lean_object* v_y_350_){
_start:
{
lean_object* v___x_351_; lean_object* v___x_352_; lean_object* v___x_353_; lean_object* v___x_354_; lean_object* v___x_355_; lean_object* v___x_356_; uint8_t v___x_357_; 
lean_inc_ref(v___f_342_);
lean_inc_ref(v_x_349_);
lean_inc_ref(v___x_343_);
v___x_351_ = lean_apply_2(v___f_342_, v___x_343_, v_x_349_);
v___x_352_ = lp_mathlib_Finsupp_toDFinsupp___redArg(v___x_351_);
lean_inc_ref(v___f_344_);
lean_inc_ref(v___x_345_);
v___x_353_ = lean_apply_2(v___f_344_, v___x_345_, v___x_352_);
lean_inc_ref(v_y_350_);
v___x_354_ = lean_apply_2(v___f_342_, v___x_343_, v_y_350_);
v___x_355_ = lp_mathlib_Finsupp_toDFinsupp___redArg(v___x_354_);
v___x_356_ = lean_apply_2(v___f_344_, v___x_345_, v___x_355_);
v___x_357_ = lp_mathlib_DFinsupp_Colex_decidableLE___redArg(v___f_346_, v_inst_347_, v___f_348_, v___x_353_, v___x_356_);
if (v___x_357_ == 0)
{
lean_dec_ref(v_y_350_);
return v_x_349_;
}
else
{
lean_dec_ref(v_x_349_);
return v_y_350_;
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finsupp_Colex_linearOrder___redArg___lam__5(lean_object* v___f_358_, lean_object* v___x_359_, lean_object* v___f_360_, lean_object* v___x_361_, lean_object* v___f_362_, lean_object* v_inst_363_, lean_object* v___f_364_, lean_object* v_a_365_, lean_object* v_b_366_){
_start:
{
lean_object* v___x_367_; lean_object* v___x_368_; lean_object* v___x_369_; lean_object* v___x_370_; lean_object* v___x_371_; lean_object* v___x_372_; uint8_t v___x_373_; 
lean_inc_ref(v___f_358_);
lean_inc_ref(v___x_359_);
v___x_367_ = lean_apply_2(v___f_358_, v___x_359_, v_a_365_);
v___x_368_ = lp_mathlib_Finsupp_toDFinsupp___redArg(v___x_367_);
lean_inc_ref(v___f_360_);
lean_inc_ref(v___x_361_);
v___x_369_ = lean_apply_2(v___f_360_, v___x_361_, v___x_368_);
v___x_370_ = lean_apply_2(v___f_358_, v___x_359_, v_b_366_);
v___x_371_ = lp_mathlib_Finsupp_toDFinsupp___redArg(v___x_370_);
v___x_372_ = lean_apply_2(v___f_360_, v___x_361_, v___x_371_);
lean_inc_ref(v___x_372_);
lean_inc_ref(v___x_369_);
lean_inc_ref(v___f_364_);
lean_inc_ref(v_inst_363_);
lean_inc(v___f_362_);
v___x_373_ = lp_mathlib_DFinsupp_Colex_decidableLT___redArg(v___f_362_, v_inst_363_, v___f_364_, v___x_369_, v___x_372_);
if (v___x_373_ == 0)
{
lean_object* v___x_374_; uint8_t v___x_375_; 
v___x_374_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_Colex_decidableLE___boxed), 7, 5);
lean_closure_set(v___x_374_, 0, lean_box(0));
lean_closure_set(v___x_374_, 1, lean_box(0));
lean_closure_set(v___x_374_, 2, v___f_362_);
lean_closure_set(v___x_374_, 3, v_inst_363_);
lean_closure_set(v___x_374_, 4, v___f_364_);
v___x_375_ = lp_mathlib_decidableEqOfDecidableLE___redArg(v___x_374_, v___x_369_, v___x_372_);
if (v___x_375_ == 0)
{
uint8_t v___x_376_; 
v___x_376_ = 2;
return v___x_376_;
}
else
{
uint8_t v___x_377_; 
v___x_377_ = 1;
return v___x_377_;
}
}
else
{
uint8_t v___x_378_; 
lean_dec_ref(v___x_372_);
lean_dec_ref(v___x_369_);
lean_dec_ref(v___f_364_);
lean_dec_ref(v_inst_363_);
lean_dec(v___f_362_);
v___x_378_ = 0;
return v___x_378_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_Colex_linearOrder___redArg___lam__5___boxed(lean_object* v___f_379_, lean_object* v___x_380_, lean_object* v___f_381_, lean_object* v___x_382_, lean_object* v___f_383_, lean_object* v_inst_384_, lean_object* v___f_385_, lean_object* v_a_386_, lean_object* v_b_387_){
_start:
{
uint8_t v_res_388_; lean_object* v_r_389_; 
v_res_388_ = lp_mathlib_Finsupp_Colex_linearOrder___redArg___lam__5(v___f_379_, v___x_380_, v___f_381_, v___x_382_, v___f_383_, v_inst_384_, v___f_385_, v_a_386_, v_b_387_);
v_r_389_ = lean_box(v_res_388_);
return v_r_389_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_Colex_linearOrder___redArg(lean_object* v_inst_390_, lean_object* v_inst_391_, lean_object* v_inst_392_){
_start:
{
lean_object* v___f_393_; lean_object* v___f_394_; lean_object* v___f_395_; lean_object* v___f_396_; lean_object* v___x_397_; lean_object* v___f_398_; lean_object* v___f_399_; lean_object* v___f_400_; lean_object* v___f_401_; lean_object* v___f_402_; lean_object* v___f_403_; lean_object* v___x_404_; lean_object* v___x_405_; 
v___f_393_ = lean_alloc_closure((void*)(lp_mathlib_Finsupp_Lex_linearOrder___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_393_, 0, v_inst_390_);
v___f_394_ = lean_alloc_closure((void*)(lp_mathlib_Finsupp_Lex_linearOrder___redArg___lam__1___boxed), 2, 1);
lean_closure_set(v___f_394_, 0, v_inst_392_);
v___f_395_ = ((lean_object*)(lp_mathlib_Finsupp_Lex_linearOrder___redArg___closed__0));
v___f_396_ = ((lean_object*)(lp_mathlib_Finsupp_Lex_linearOrder___redArg___closed__1));
v___x_397_ = lean_obj_once(&lp_mathlib_Finsupp_Lex_linearOrder___redArg___closed__2, &lp_mathlib_Finsupp_Lex_linearOrder___redArg___closed__2_once, _init_lp_mathlib_Finsupp_Lex_linearOrder___redArg___closed__2);
lean_inc_ref_n(v___f_394_, 5);
lean_inc_ref_n(v_inst_391_, 5);
lean_inc_ref_n(v___f_393_, 5);
v___f_398_ = lean_alloc_closure((void*)(lp_mathlib_Finsupp_Colex_linearOrder___redArg___lam__4___boxed), 9, 7);
lean_closure_set(v___f_398_, 0, v___f_393_);
lean_closure_set(v___f_398_, 1, v_inst_391_);
lean_closure_set(v___f_398_, 2, v___f_394_);
lean_closure_set(v___f_398_, 3, v___f_396_);
lean_closure_set(v___f_398_, 4, v___x_397_);
lean_closure_set(v___f_398_, 5, v___f_395_);
lean_closure_set(v___f_398_, 6, v___x_397_);
v___f_399_ = lean_alloc_closure((void*)(lp_mathlib_Finsupp_Colex_linearOrder___redArg___lam__0___boxed), 9, 7);
lean_closure_set(v___f_399_, 0, v___f_396_);
lean_closure_set(v___f_399_, 1, v___x_397_);
lean_closure_set(v___f_399_, 2, v___f_395_);
lean_closure_set(v___f_399_, 3, v___x_397_);
lean_closure_set(v___f_399_, 4, v___f_393_);
lean_closure_set(v___f_399_, 5, v_inst_391_);
lean_closure_set(v___f_399_, 6, v___f_394_);
v___f_400_ = lean_alloc_closure((void*)(lp_mathlib_Finsupp_Colex_linearOrder___redArg___lam__1___boxed), 9, 7);
lean_closure_set(v___f_400_, 0, v___f_396_);
lean_closure_set(v___f_400_, 1, v___x_397_);
lean_closure_set(v___f_400_, 2, v___f_395_);
lean_closure_set(v___f_400_, 3, v___x_397_);
lean_closure_set(v___f_400_, 4, v___f_393_);
lean_closure_set(v___f_400_, 5, v_inst_391_);
lean_closure_set(v___f_400_, 6, v___f_394_);
v___f_401_ = lean_alloc_closure((void*)(lp_mathlib_Finsupp_Colex_linearOrder___redArg___lam__2), 9, 7);
lean_closure_set(v___f_401_, 0, v___f_396_);
lean_closure_set(v___f_401_, 1, v___x_397_);
lean_closure_set(v___f_401_, 2, v___f_395_);
lean_closure_set(v___f_401_, 3, v___x_397_);
lean_closure_set(v___f_401_, 4, v___f_393_);
lean_closure_set(v___f_401_, 5, v_inst_391_);
lean_closure_set(v___f_401_, 6, v___f_394_);
v___f_402_ = lean_alloc_closure((void*)(lp_mathlib_Finsupp_Colex_linearOrder___redArg___lam__3), 9, 7);
lean_closure_set(v___f_402_, 0, v___f_396_);
lean_closure_set(v___f_402_, 1, v___x_397_);
lean_closure_set(v___f_402_, 2, v___f_395_);
lean_closure_set(v___f_402_, 3, v___x_397_);
lean_closure_set(v___f_402_, 4, v___f_393_);
lean_closure_set(v___f_402_, 5, v_inst_391_);
lean_closure_set(v___f_402_, 6, v___f_394_);
v___f_403_ = lean_alloc_closure((void*)(lp_mathlib_Finsupp_Colex_linearOrder___redArg___lam__5___boxed), 9, 7);
lean_closure_set(v___f_403_, 0, v___f_396_);
lean_closure_set(v___f_403_, 1, v___x_397_);
lean_closure_set(v___f_403_, 2, v___f_395_);
lean_closure_set(v___f_403_, 3, v___x_397_);
lean_closure_set(v___f_403_, 4, v___f_393_);
lean_closure_set(v___f_403_, 5, v_inst_391_);
lean_closure_set(v___f_403_, 6, v___f_394_);
v___x_404_ = ((lean_object*)(lp_mathlib_Finsupp_Lex_partialOrder___closed__0));
v___x_405_ = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(v___x_405_, 0, v___x_404_);
lean_ctor_set(v___x_405_, 1, v___f_401_);
lean_ctor_set(v___x_405_, 2, v___f_402_);
lean_ctor_set(v___x_405_, 3, v___f_403_);
lean_ctor_set(v___x_405_, 4, v___f_399_);
lean_ctor_set(v___x_405_, 5, v___f_398_);
lean_ctor_set(v___x_405_, 6, v___f_400_);
return v___x_405_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_Colex_linearOrder(lean_object* v_00_u03b1_406_, lean_object* v_N_407_, lean_object* v_inst_408_, lean_object* v_inst_409_, lean_object* v_inst_410_){
_start:
{
lean_object* v___x_411_; 
v___x_411_ = lp_mathlib_Finsupp_Colex_linearOrder___redArg(v_inst_408_, v_inst_409_, v_inst_410_);
return v___x_411_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_Lex_orderBot___redArg___lam__0(lean_object* v_toZero_412_, lean_object* v_x_413_){
_start:
{
lean_inc(v_toZero_412_);
return v_toZero_412_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_Lex_orderBot___redArg___lam__0___boxed(lean_object* v_toZero_414_, lean_object* v_x_415_){
_start:
{
lean_object* v_res_416_; 
v_res_416_ = lp_mathlib_Finsupp_Lex_orderBot___redArg___lam__0(v_toZero_414_, v_x_415_);
lean_dec(v_x_415_);
lean_dec(v_toZero_414_);
return v_res_416_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_Lex_orderBot___redArg(lean_object* v_inst_417_){
_start:
{
lean_object* v___x_418_; lean_object* v___x_419_; lean_object* v_toZero_420_; lean_object* v___x_422_; uint8_t v_isShared_423_; uint8_t v_isSharedCheck_429_; 
v___x_418_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_417_);
v___x_419_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_418_);
v_toZero_420_ = lean_ctor_get(v___x_419_, 0);
v_isSharedCheck_429_ = !lean_is_exclusive(v___x_419_);
if (v_isSharedCheck_429_ == 0)
{
lean_object* v_unused_430_; 
v_unused_430_ = lean_ctor_get(v___x_419_, 1);
lean_dec(v_unused_430_);
v___x_422_ = v___x_419_;
v_isShared_423_ = v_isSharedCheck_429_;
goto v_resetjp_421_;
}
else
{
lean_inc(v_toZero_420_);
lean_dec(v___x_419_);
v___x_422_ = lean_box(0);
v_isShared_423_ = v_isSharedCheck_429_;
goto v_resetjp_421_;
}
v_resetjp_421_:
{
lean_object* v___f_424_; lean_object* v___x_425_; lean_object* v___x_427_; 
v___f_424_ = lean_alloc_closure((void*)(lp_mathlib_Finsupp_Lex_orderBot___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_424_, 0, v_toZero_420_);
v___x_425_ = lean_box(0);
if (v_isShared_423_ == 0)
{
lean_ctor_set(v___x_422_, 1, v___f_424_);
lean_ctor_set(v___x_422_, 0, v___x_425_);
v___x_427_ = v___x_422_;
goto v_reusejp_426_;
}
else
{
lean_object* v_reuseFailAlloc_428_; 
v_reuseFailAlloc_428_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_428_, 0, v___x_425_);
lean_ctor_set(v_reuseFailAlloc_428_, 1, v___f_424_);
v___x_427_ = v_reuseFailAlloc_428_;
goto v_reusejp_426_;
}
v_reusejp_426_:
{
return v___x_427_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_Lex_orderBot___redArg___boxed(lean_object* v_inst_431_){
_start:
{
lean_object* v_res_432_; 
v_res_432_ = lp_mathlib_Finsupp_Lex_orderBot___redArg(v_inst_431_);
lean_dec_ref(v_inst_431_);
return v_res_432_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_Lex_orderBot(lean_object* v_00_u03b1_433_, lean_object* v_N_434_, lean_object* v_inst_435_, lean_object* v_inst_436_, lean_object* v_inst_437_, lean_object* v_inst_438_){
_start:
{
lean_object* v___x_439_; 
v___x_439_ = lp_mathlib_Finsupp_Lex_orderBot___redArg(v_inst_436_);
return v___x_439_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_Lex_orderBot___boxed(lean_object* v_00_u03b1_440_, lean_object* v_N_441_, lean_object* v_inst_442_, lean_object* v_inst_443_, lean_object* v_inst_444_, lean_object* v_inst_445_){
_start:
{
lean_object* v_res_446_; 
v_res_446_ = lp_mathlib_Finsupp_Lex_orderBot(v_00_u03b1_440_, v_N_441_, v_inst_442_, v_inst_443_, v_inst_444_, v_inst_445_);
lean_dec_ref(v_inst_444_);
lean_dec_ref(v_inst_443_);
lean_dec_ref(v_inst_442_);
return v_res_446_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_Colex_orderBot___redArg(lean_object* v_inst_447_){
_start:
{
lean_object* v___x_448_; lean_object* v___x_449_; lean_object* v_toZero_450_; lean_object* v___x_452_; uint8_t v_isShared_453_; uint8_t v_isSharedCheck_459_; 
v___x_448_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_447_);
v___x_449_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_448_);
v_toZero_450_ = lean_ctor_get(v___x_449_, 0);
v_isSharedCheck_459_ = !lean_is_exclusive(v___x_449_);
if (v_isSharedCheck_459_ == 0)
{
lean_object* v_unused_460_; 
v_unused_460_ = lean_ctor_get(v___x_449_, 1);
lean_dec(v_unused_460_);
v___x_452_ = v___x_449_;
v_isShared_453_ = v_isSharedCheck_459_;
goto v_resetjp_451_;
}
else
{
lean_inc(v_toZero_450_);
lean_dec(v___x_449_);
v___x_452_ = lean_box(0);
v_isShared_453_ = v_isSharedCheck_459_;
goto v_resetjp_451_;
}
v_resetjp_451_:
{
lean_object* v___f_454_; lean_object* v___x_455_; lean_object* v___x_457_; 
v___f_454_ = lean_alloc_closure((void*)(lp_mathlib_Finsupp_Lex_orderBot___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_454_, 0, v_toZero_450_);
v___x_455_ = lean_box(0);
if (v_isShared_453_ == 0)
{
lean_ctor_set(v___x_452_, 1, v___f_454_);
lean_ctor_set(v___x_452_, 0, v___x_455_);
v___x_457_ = v___x_452_;
goto v_reusejp_456_;
}
else
{
lean_object* v_reuseFailAlloc_458_; 
v_reuseFailAlloc_458_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_458_, 0, v___x_455_);
lean_ctor_set(v_reuseFailAlloc_458_, 1, v___f_454_);
v___x_457_ = v_reuseFailAlloc_458_;
goto v_reusejp_456_;
}
v_reusejp_456_:
{
return v___x_457_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_Colex_orderBot___redArg___boxed(lean_object* v_inst_461_){
_start:
{
lean_object* v_res_462_; 
v_res_462_ = lp_mathlib_Finsupp_Colex_orderBot___redArg(v_inst_461_);
lean_dec_ref(v_inst_461_);
return v_res_462_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_Colex_orderBot(lean_object* v_00_u03b1_463_, lean_object* v_N_464_, lean_object* v_inst_465_, lean_object* v_inst_466_, lean_object* v_inst_467_, lean_object* v_inst_468_){
_start:
{
lean_object* v___x_469_; 
v___x_469_ = lp_mathlib_Finsupp_Colex_orderBot___redArg(v_inst_466_);
return v___x_469_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_Colex_orderBot___boxed(lean_object* v_00_u03b1_470_, lean_object* v_N_471_, lean_object* v_inst_472_, lean_object* v_inst_473_, lean_object* v_inst_474_, lean_object* v_inst_475_){
_start:
{
lean_object* v_res_476_; 
v_res_476_ = lp_mathlib_Finsupp_Colex_orderBot(v_00_u03b1_470_, v_N_471_, v_inst_472_, v_inst_473_, v_inst_474_, v_inst_475_);
lean_dec_ref(v_inst_474_);
lean_dec_ref(v_inst_473_);
lean_dec_ref(v_inst_472_);
return v_res_476_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finsupp_Order(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_DFinsupp_Lex(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finsupp_ToDFinsupp(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Finsupp_Lex(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finsupp_Order(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_DFinsupp_Lex(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finsupp_ToDFinsupp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Finsupp_Lex(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Finsupp_Order(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_DFinsupp_Lex(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finsupp_ToDFinsupp(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Finsupp_Lex(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finsupp_Order(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_DFinsupp_Lex(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finsupp_ToDFinsupp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finsupp_Lex(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Finsupp_Lex(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Finsupp_Lex(builtin);
}
#ifdef __cplusplus
}
#endif
