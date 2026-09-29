// Lean compiler output
// Module: Batteries.Lean.Except
// Imports: public import Init public meta import Init public import Lean.Util.Trace
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
extern lean_object* l_Lean_crossEmoji;
extern lean_object* l_Lean_checkEmoji;
LEAN_EXPORT uint8_t lp_batteries_instDecidableEqExcept__batteries_decEq___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_instDecidableEqExcept__batteries_decEq___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_instDecidableEqExcept__batteries_decEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_instDecidableEqExcept__batteries_decEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_instDecidableEqExcept__batteries___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_instDecidableEqExcept__batteries___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_instDecidableEqExcept__batteries(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_instDecidableEqExcept__batteries___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Except_emoji___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Except_emoji___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Except_emoji(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Except_emoji___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Except_pmap___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Except_pmap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Lean_Except_0__ExceptT_bindCont_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Lean_Except_0__ExceptT_bindCont_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Lean_Except_0__Except_map_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Lean_Except_0__Except_map_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_instDecidableEqExcept__batteries_decEq___redArg(lean_object* v_inst_1_, lean_object* v_inst_2_, lean_object* v_x_3_, lean_object* v_x_4_){
_start:
{
if (lean_obj_tag(v_x_3_) == 0)
{
lean_dec_ref(v_inst_2_);
if (lean_obj_tag(v_x_4_) == 0)
{
lean_object* v_a_5_; lean_object* v_a_6_; lean_object* v___x_7_; uint8_t v___x_8_; 
v_a_5_ = lean_ctor_get(v_x_3_, 0);
lean_inc(v_a_5_);
lean_dec_ref_known(v_x_3_, 1);
v_a_6_ = lean_ctor_get(v_x_4_, 0);
lean_inc(v_a_6_);
lean_dec_ref_known(v_x_4_, 1);
v___x_7_ = lean_apply_2(v_inst_1_, v_a_5_, v_a_6_);
v___x_8_ = lean_unbox(v___x_7_);
return v___x_8_;
}
else
{
uint8_t v___x_9_; 
lean_dec_ref_known(v_x_4_, 1);
lean_dec_ref_known(v_x_3_, 1);
lean_dec_ref(v_inst_1_);
v___x_9_ = 0;
return v___x_9_;
}
}
else
{
lean_dec_ref(v_inst_1_);
if (lean_obj_tag(v_x_4_) == 0)
{
uint8_t v___x_10_; 
lean_dec_ref_known(v_x_4_, 1);
lean_dec_ref_known(v_x_3_, 1);
lean_dec_ref(v_inst_2_);
v___x_10_ = 0;
return v___x_10_;
}
else
{
lean_object* v_a_11_; lean_object* v_a_12_; lean_object* v___x_13_; uint8_t v___x_14_; 
v_a_11_ = lean_ctor_get(v_x_3_, 0);
lean_inc(v_a_11_);
lean_dec_ref_known(v_x_3_, 1);
v_a_12_ = lean_ctor_get(v_x_4_, 0);
lean_inc(v_a_12_);
lean_dec_ref_known(v_x_4_, 1);
v___x_13_ = lean_apply_2(v_inst_2_, v_a_11_, v_a_12_);
v___x_14_ = lean_unbox(v___x_13_);
return v___x_14_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_instDecidableEqExcept__batteries_decEq___redArg___boxed(lean_object* v_inst_15_, lean_object* v_inst_16_, lean_object* v_x_17_, lean_object* v_x_18_){
_start:
{
uint8_t v_res_19_; lean_object* v_r_20_; 
v_res_19_ = lp_batteries_instDecidableEqExcept__batteries_decEq___redArg(v_inst_15_, v_inst_16_, v_x_17_, v_x_18_);
v_r_20_ = lean_box(v_res_19_);
return v_r_20_;
}
}
LEAN_EXPORT uint8_t lp_batteries_instDecidableEqExcept__batteries_decEq(lean_object* v_00_u03b5_21_, lean_object* v_00_u03b1_22_, lean_object* v_inst_23_, lean_object* v_inst_24_, lean_object* v_x_25_, lean_object* v_x_26_){
_start:
{
uint8_t v___x_27_; 
v___x_27_ = lp_batteries_instDecidableEqExcept__batteries_decEq___redArg(v_inst_23_, v_inst_24_, v_x_25_, v_x_26_);
return v___x_27_;
}
}
LEAN_EXPORT lean_object* lp_batteries_instDecidableEqExcept__batteries_decEq___boxed(lean_object* v_00_u03b5_28_, lean_object* v_00_u03b1_29_, lean_object* v_inst_30_, lean_object* v_inst_31_, lean_object* v_x_32_, lean_object* v_x_33_){
_start:
{
uint8_t v_res_34_; lean_object* v_r_35_; 
v_res_34_ = lp_batteries_instDecidableEqExcept__batteries_decEq(v_00_u03b5_28_, v_00_u03b1_29_, v_inst_30_, v_inst_31_, v_x_32_, v_x_33_);
v_r_35_ = lean_box(v_res_34_);
return v_r_35_;
}
}
LEAN_EXPORT uint8_t lp_batteries_instDecidableEqExcept__batteries___redArg(lean_object* v_inst_36_, lean_object* v_inst_37_, lean_object* v_x_38_, lean_object* v_x_39_){
_start:
{
uint8_t v___x_40_; 
v___x_40_ = lp_batteries_instDecidableEqExcept__batteries_decEq___redArg(v_inst_36_, v_inst_37_, v_x_38_, v_x_39_);
return v___x_40_;
}
}
LEAN_EXPORT lean_object* lp_batteries_instDecidableEqExcept__batteries___redArg___boxed(lean_object* v_inst_41_, lean_object* v_inst_42_, lean_object* v_x_43_, lean_object* v_x_44_){
_start:
{
uint8_t v_res_45_; lean_object* v_r_46_; 
v_res_45_ = lp_batteries_instDecidableEqExcept__batteries___redArg(v_inst_41_, v_inst_42_, v_x_43_, v_x_44_);
v_r_46_ = lean_box(v_res_45_);
return v_r_46_;
}
}
LEAN_EXPORT uint8_t lp_batteries_instDecidableEqExcept__batteries(lean_object* v_00_u03b5_47_, lean_object* v_00_u03b1_48_, lean_object* v_inst_49_, lean_object* v_inst_50_, lean_object* v_x_51_, lean_object* v_x_52_){
_start:
{
uint8_t v___x_53_; 
v___x_53_ = lp_batteries_instDecidableEqExcept__batteries_decEq___redArg(v_inst_49_, v_inst_50_, v_x_51_, v_x_52_);
return v___x_53_;
}
}
LEAN_EXPORT lean_object* lp_batteries_instDecidableEqExcept__batteries___boxed(lean_object* v_00_u03b5_54_, lean_object* v_00_u03b1_55_, lean_object* v_inst_56_, lean_object* v_inst_57_, lean_object* v_x_58_, lean_object* v_x_59_){
_start:
{
uint8_t v_res_60_; lean_object* v_r_61_; 
v_res_60_ = lp_batteries_instDecidableEqExcept__batteries(v_00_u03b5_54_, v_00_u03b1_55_, v_inst_56_, v_inst_57_, v_x_58_, v_x_59_);
v_r_61_ = lean_box(v_res_60_);
return v_r_61_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Except_emoji___redArg(lean_object* v_x_62_){
_start:
{
if (lean_obj_tag(v_x_62_) == 0)
{
lean_object* v___x_63_; 
v___x_63_ = l_Lean_crossEmoji;
return v___x_63_;
}
else
{
lean_object* v___x_64_; 
v___x_64_ = l_Lean_checkEmoji;
return v___x_64_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Except_emoji___redArg___boxed(lean_object* v_x_65_){
_start:
{
lean_object* v_res_66_; 
v_res_66_ = lp_batteries_Except_emoji___redArg(v_x_65_);
lean_dec_ref(v_x_65_);
return v_res_66_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Except_emoji(lean_object* v_00_u03b5_67_, lean_object* v_00_u03b1_68_, lean_object* v_x_69_){
_start:
{
lean_object* v___x_70_; 
v___x_70_ = lp_batteries_Except_emoji___redArg(v_x_69_);
return v___x_70_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Except_emoji___boxed(lean_object* v_00_u03b5_71_, lean_object* v_00_u03b1_72_, lean_object* v_x_73_){
_start:
{
lean_object* v_res_74_; 
v_res_74_ = lp_batteries_Except_emoji(v_00_u03b5_71_, v_00_u03b1_72_, v_x_73_);
lean_dec_ref(v_x_73_);
return v_res_74_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Except_pmap___redArg(lean_object* v_x_75_, lean_object* v_f_76_){
_start:
{
if (lean_obj_tag(v_x_75_) == 0)
{
lean_object* v_a_77_; lean_object* v___x_79_; uint8_t v_isShared_80_; uint8_t v_isSharedCheck_84_; 
lean_dec(v_f_76_);
v_a_77_ = lean_ctor_get(v_x_75_, 0);
v_isSharedCheck_84_ = !lean_is_exclusive(v_x_75_);
if (v_isSharedCheck_84_ == 0)
{
v___x_79_ = v_x_75_;
v_isShared_80_ = v_isSharedCheck_84_;
goto v_resetjp_78_;
}
else
{
lean_inc(v_a_77_);
lean_dec(v_x_75_);
v___x_79_ = lean_box(0);
v_isShared_80_ = v_isSharedCheck_84_;
goto v_resetjp_78_;
}
v_resetjp_78_:
{
lean_object* v___x_82_; 
if (v_isShared_80_ == 0)
{
v___x_82_ = v___x_79_;
goto v_reusejp_81_;
}
else
{
lean_object* v_reuseFailAlloc_83_; 
v_reuseFailAlloc_83_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_83_, 0, v_a_77_);
v___x_82_ = v_reuseFailAlloc_83_;
goto v_reusejp_81_;
}
v_reusejp_81_:
{
return v___x_82_;
}
}
}
else
{
lean_object* v_a_85_; lean_object* v___x_87_; uint8_t v_isShared_88_; uint8_t v_isSharedCheck_93_; 
v_a_85_ = lean_ctor_get(v_x_75_, 0);
v_isSharedCheck_93_ = !lean_is_exclusive(v_x_75_);
if (v_isSharedCheck_93_ == 0)
{
v___x_87_ = v_x_75_;
v_isShared_88_ = v_isSharedCheck_93_;
goto v_resetjp_86_;
}
else
{
lean_inc(v_a_85_);
lean_dec(v_x_75_);
v___x_87_ = lean_box(0);
v_isShared_88_ = v_isSharedCheck_93_;
goto v_resetjp_86_;
}
v_resetjp_86_:
{
lean_object* v___x_89_; lean_object* v___x_91_; 
v___x_89_ = lean_apply_2(v_f_76_, v_a_85_, lean_box(0));
if (v_isShared_88_ == 0)
{
lean_ctor_set(v___x_87_, 0, v___x_89_);
v___x_91_ = v___x_87_;
goto v_reusejp_90_;
}
else
{
lean_object* v_reuseFailAlloc_92_; 
v_reuseFailAlloc_92_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_92_, 0, v___x_89_);
v___x_91_ = v_reuseFailAlloc_92_;
goto v_reusejp_90_;
}
v_reusejp_90_:
{
return v___x_91_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Except_pmap(lean_object* v_00_u03b5_94_, lean_object* v_00_u03b1_95_, lean_object* v_00_u03b2_96_, lean_object* v_x_97_, lean_object* v_f_98_){
_start:
{
lean_object* v___x_99_; 
v___x_99_ = lp_batteries_Except_pmap___redArg(v_x_97_, v_f_98_);
return v___x_99_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Lean_Except_0__ExceptT_bindCont_match__1_splitter___redArg(lean_object* v_x_100_, lean_object* v_h__1_101_, lean_object* v_h__2_102_){
_start:
{
if (lean_obj_tag(v_x_100_) == 0)
{
lean_object* v_a_103_; lean_object* v___x_104_; 
lean_dec(v_h__1_101_);
v_a_103_ = lean_ctor_get(v_x_100_, 0);
lean_inc(v_a_103_);
lean_dec_ref_known(v_x_100_, 1);
v___x_104_ = lean_apply_1(v_h__2_102_, v_a_103_);
return v___x_104_;
}
else
{
lean_object* v_a_105_; lean_object* v___x_106_; 
lean_dec(v_h__2_102_);
v_a_105_ = lean_ctor_get(v_x_100_, 0);
lean_inc(v_a_105_);
lean_dec_ref_known(v_x_100_, 1);
v___x_106_ = lean_apply_1(v_h__1_101_, v_a_105_);
return v___x_106_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Lean_Except_0__ExceptT_bindCont_match__1_splitter(lean_object* v_00_u03b5_107_, lean_object* v_00_u03b1_108_, lean_object* v_motive_109_, lean_object* v_x_110_, lean_object* v_h__1_111_, lean_object* v_h__2_112_){
_start:
{
if (lean_obj_tag(v_x_110_) == 0)
{
lean_object* v_a_113_; lean_object* v___x_114_; 
lean_dec(v_h__1_111_);
v_a_113_ = lean_ctor_get(v_x_110_, 0);
lean_inc(v_a_113_);
lean_dec_ref_known(v_x_110_, 1);
v___x_114_ = lean_apply_1(v_h__2_112_, v_a_113_);
return v___x_114_;
}
else
{
lean_object* v_a_115_; lean_object* v___x_116_; 
lean_dec(v_h__2_112_);
v_a_115_ = lean_ctor_get(v_x_110_, 0);
lean_inc(v_a_115_);
lean_dec_ref_known(v_x_110_, 1);
v___x_116_ = lean_apply_1(v_h__1_111_, v_a_115_);
return v___x_116_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Lean_Except_0__Except_map_match__1_splitter___redArg(lean_object* v_x_117_, lean_object* v_h__1_118_, lean_object* v_h__2_119_){
_start:
{
if (lean_obj_tag(v_x_117_) == 0)
{
lean_object* v_a_120_; lean_object* v___x_121_; 
lean_dec(v_h__2_119_);
v_a_120_ = lean_ctor_get(v_x_117_, 0);
lean_inc(v_a_120_);
lean_dec_ref_known(v_x_117_, 1);
v___x_121_ = lean_apply_1(v_h__1_118_, v_a_120_);
return v___x_121_;
}
else
{
lean_object* v_a_122_; lean_object* v___x_123_; 
lean_dec(v_h__1_118_);
v_a_122_ = lean_ctor_get(v_x_117_, 0);
lean_inc(v_a_122_);
lean_dec_ref_known(v_x_117_, 1);
v___x_123_ = lean_apply_1(v_h__2_119_, v_a_122_);
return v___x_123_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Lean_Except_0__Except_map_match__1_splitter(lean_object* v_00_u03b5_124_, lean_object* v_00_u03b1_125_, lean_object* v_motive_126_, lean_object* v_x_127_, lean_object* v_h__1_128_, lean_object* v_h__2_129_){
_start:
{
if (lean_obj_tag(v_x_127_) == 0)
{
lean_object* v_a_130_; lean_object* v___x_131_; 
lean_dec(v_h__2_129_);
v_a_130_ = lean_ctor_get(v_x_127_, 0);
lean_inc(v_a_130_);
lean_dec_ref_known(v_x_127_, 1);
v___x_131_ = lean_apply_1(v_h__1_128_, v_a_130_);
return v___x_131_;
}
else
{
lean_object* v_a_132_; lean_object* v___x_133_; 
lean_dec(v_h__1_128_);
v_a_132_ = lean_ctor_get(v_x_127_, 0);
lean_inc(v_a_132_);
lean_dec_ref_known(v_x_127_, 1);
v___x_133_ = lean_apply_1(v_h__2_129_, v_a_132_);
return v___x_133_;
}
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Util_Trace(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_batteries_Batteries_Lean_Except(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Util_Trace(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_batteries_Batteries_Lean_Except(uint8_t builtin) {
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
lean_object* initialize_Lean_Util_Trace(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_batteries_Batteries_Lean_Except(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Util_Trace(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Lean_Except(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_batteries_Batteries_Lean_Except(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_batteries_Batteries_Lean_Except(builtin);
}
#ifdef __cplusplus
}
#endif
