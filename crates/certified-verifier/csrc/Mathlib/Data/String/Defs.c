// Lean compiler output
// Module: Mathlib.Data.String.Defs
// Imports: public import Init public meta import Init public import Mathlib.Init
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
uint8_t lean_string_utf8_at_end(lean_object*, lean_object*);
uint32_t lean_string_utf8_get(lean_object*, lean_object*);
uint8_t lean_uint32_dec_eq(uint32_t, uint32_t);
lean_object* lean_string_utf8_next(lean_object*, lean_object*);
lean_object* lean_string_utf8_extract(lean_object*, lean_object*, lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* lean_string_push(lean_object*, uint32_t);
lean_object* l_String_intercalate(lean_object*, lean_object*);
lean_object* l_List_replicateTR___redArg(lean_object*, lean_object*);
lean_object* lean_string_mk(lean_object*);
lean_object* lean_string_utf8_byte_size(lean_object*);
lean_object* l_String_Slice_Pos_get_x3f(lean_object*, lean_object*);
lean_object* lean_uint32_to_nat(uint32_t);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_string_length(lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* lean_string_data(lean_object*);
lean_object* l_List_lengthTR___redArg(lean_object*);
lean_object* l_List_replicateTR_loop___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Char_isAscii(uint32_t);
LEAN_EXPORT lean_object* lp_mathlib_Char_isAscii___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_String_leftpad(lean_object*, uint32_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_String_leftpad___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_String_replicate(lean_object*, uint32_t);
LEAN_EXPORT lean_object* lp_mathlib_String_replicate___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_String_rightpad(lean_object*, uint32_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_String_rightpad___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_String_splitAux___at___00String_mapTokens_spec__0(uint32_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_String_splitAux___at___00String_mapTokens_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00String_mapTokens_spec__1(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_String_mapTokens___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_String_mapTokens___closed__0 = (const lean_object*)&lp_mathlib_String_mapTokens___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_String_mapTokens(uint32_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_String_mapTokens___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint32_t lp_mathlib_String_head(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_String_head___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Char_isAscii(uint32_t v_c_1_){
_start:
{
lean_object* v___x_2_; lean_object* v___x_3_; uint8_t v___x_4_; 
v___x_2_ = lean_uint32_to_nat(v_c_1_);
v___x_3_ = lean_unsigned_to_nat(128u);
v___x_4_ = lean_nat_dec_lt(v___x_2_, v___x_3_);
lean_dec(v___x_2_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Char_isAscii___boxed(lean_object* v_c_5_){
_start:
{
uint32_t v_c_boxed_6_; uint8_t v_res_7_; lean_object* v_r_8_; 
v_c_boxed_6_ = lean_unbox_uint32(v_c_5_);
lean_dec(v_c_5_);
v_res_7_ = lp_mathlib_Char_isAscii(v_c_boxed_6_);
v_r_8_ = lean_box(v_res_7_);
return v_r_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_leftpad(lean_object* v_n_9_, uint32_t v_c_10_, lean_object* v_s_11_){
_start:
{
lean_object* v___x_12_; lean_object* v___x_13_; lean_object* v___x_14_; lean_object* v___x_15_; lean_object* v___x_16_; lean_object* v___x_17_; 
v___x_12_ = lean_string_data(v_s_11_);
v___x_13_ = l_List_lengthTR___redArg(v___x_12_);
v___x_14_ = lean_nat_sub(v_n_9_, v___x_13_);
lean_dec(v___x_13_);
v___x_15_ = lean_box_uint32(v_c_10_);
v___x_16_ = l_List_replicateTR_loop___redArg(v___x_15_, v___x_14_, v___x_12_);
v___x_17_ = lean_string_mk(v___x_16_);
return v___x_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_leftpad___boxed(lean_object* v_n_18_, lean_object* v_c_19_, lean_object* v_s_20_){
_start:
{
uint32_t v_c_boxed_21_; lean_object* v_res_22_; 
v_c_boxed_21_ = lean_unbox_uint32(v_c_19_);
lean_dec(v_c_19_);
v_res_22_ = lp_mathlib_String_leftpad(v_n_18_, v_c_boxed_21_, v_s_20_);
lean_dec(v_n_18_);
return v_res_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_replicate(lean_object* v_n_23_, uint32_t v_c_24_){
_start:
{
lean_object* v___x_25_; lean_object* v___x_26_; lean_object* v___x_27_; 
v___x_25_ = lean_box_uint32(v_c_24_);
v___x_26_ = l_List_replicateTR___redArg(v_n_23_, v___x_25_);
v___x_27_ = lean_string_mk(v___x_26_);
return v___x_27_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_replicate___boxed(lean_object* v_n_28_, lean_object* v_c_29_){
_start:
{
uint32_t v_c_boxed_30_; lean_object* v_res_31_; 
v_c_boxed_30_ = lean_unbox_uint32(v_c_29_);
lean_dec(v_c_29_);
v_res_31_ = lp_mathlib_String_replicate(v_n_28_, v_c_boxed_30_);
return v_res_31_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_rightpad(lean_object* v_n_32_, uint32_t v_c_33_, lean_object* v_s_34_){
_start:
{
lean_object* v___x_35_; lean_object* v___x_36_; lean_object* v___x_37_; lean_object* v___x_38_; 
v___x_35_ = lean_string_length(v_s_34_);
v___x_36_ = lean_nat_sub(v_n_32_, v___x_35_);
v___x_37_ = lp_mathlib_String_replicate(v___x_36_, v_c_33_);
v___x_38_ = lean_string_append(v_s_34_, v___x_37_);
lean_dec_ref(v___x_37_);
return v___x_38_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_rightpad___boxed(lean_object* v_n_39_, lean_object* v_c_40_, lean_object* v_s_41_){
_start:
{
uint32_t v_c_boxed_42_; lean_object* v_res_43_; 
v_c_boxed_42_ = lean_unbox_uint32(v_c_40_);
lean_dec(v_c_40_);
v_res_43_ = lp_mathlib_String_rightpad(v_n_39_, v_c_boxed_42_, v_s_41_);
lean_dec(v_n_39_);
return v_res_43_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_splitAux___at___00String_mapTokens_spec__0(uint32_t v_c_44_, lean_object* v_s_45_, lean_object* v_b_46_, lean_object* v_i_47_, lean_object* v_r_48_){
_start:
{
uint8_t v___x_49_; 
v___x_49_ = lean_string_utf8_at_end(v_s_45_, v_i_47_);
if (v___x_49_ == 0)
{
uint32_t v___x_50_; uint8_t v___x_51_; 
v___x_50_ = lean_string_utf8_get(v_s_45_, v_i_47_);
v___x_51_ = lean_uint32_dec_eq(v___x_50_, v_c_44_);
if (v___x_51_ == 0)
{
lean_object* v___x_52_; 
v___x_52_ = lean_string_utf8_next(v_s_45_, v_i_47_);
lean_dec(v_i_47_);
v_i_47_ = v___x_52_;
goto _start;
}
else
{
lean_object* v_i_x27_54_; lean_object* v___x_55_; lean_object* v___x_56_; 
v_i_x27_54_ = lean_string_utf8_next(v_s_45_, v_i_47_);
v___x_55_ = lean_string_utf8_extract(v_s_45_, v_b_46_, v_i_47_);
lean_dec(v_i_47_);
lean_dec(v_b_46_);
v___x_56_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_56_, 0, v___x_55_);
lean_ctor_set(v___x_56_, 1, v_r_48_);
lean_inc(v_i_x27_54_);
v_b_46_ = v_i_x27_54_;
v_i_47_ = v_i_x27_54_;
v_r_48_ = v___x_56_;
goto _start;
}
}
else
{
lean_object* v___x_58_; lean_object* v_r_59_; lean_object* v___x_60_; 
v___x_58_ = lean_string_utf8_extract(v_s_45_, v_b_46_, v_i_47_);
lean_dec(v_i_47_);
lean_dec(v_b_46_);
v_r_59_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_r_59_, 0, v___x_58_);
lean_ctor_set(v_r_59_, 1, v_r_48_);
v___x_60_ = l_List_reverse___redArg(v_r_59_);
return v___x_60_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_splitAux___at___00String_mapTokens_spec__0___boxed(lean_object* v_c_61_, lean_object* v_s_62_, lean_object* v_b_63_, lean_object* v_i_64_, lean_object* v_r_65_){
_start:
{
uint32_t v_c_boxed_66_; lean_object* v_res_67_; 
v_c_boxed_66_ = lean_unbox_uint32(v_c_61_);
lean_dec(v_c_61_);
v_res_67_ = lp_mathlib_String_splitAux___at___00String_mapTokens_spec__0(v_c_boxed_66_, v_s_62_, v_b_63_, v_i_64_, v_r_65_);
lean_dec_ref(v_s_62_);
return v_res_67_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00String_mapTokens_spec__1(lean_object* v_f_68_, lean_object* v_a_69_, lean_object* v_a_70_){
_start:
{
if (lean_obj_tag(v_a_69_) == 0)
{
lean_object* v___x_71_; 
lean_dec_ref(v_f_68_);
v___x_71_ = l_List_reverse___redArg(v_a_70_);
return v___x_71_;
}
else
{
lean_object* v_head_72_; lean_object* v_tail_73_; lean_object* v___x_75_; uint8_t v_isShared_76_; uint8_t v_isSharedCheck_82_; 
v_head_72_ = lean_ctor_get(v_a_69_, 0);
v_tail_73_ = lean_ctor_get(v_a_69_, 1);
v_isSharedCheck_82_ = !lean_is_exclusive(v_a_69_);
if (v_isSharedCheck_82_ == 0)
{
v___x_75_ = v_a_69_;
v_isShared_76_ = v_isSharedCheck_82_;
goto v_resetjp_74_;
}
else
{
lean_inc(v_tail_73_);
lean_inc(v_head_72_);
lean_dec(v_a_69_);
v___x_75_ = lean_box(0);
v_isShared_76_ = v_isSharedCheck_82_;
goto v_resetjp_74_;
}
v_resetjp_74_:
{
lean_object* v___x_77_; lean_object* v___x_79_; 
lean_inc_ref(v_f_68_);
v___x_77_ = lean_apply_1(v_f_68_, v_head_72_);
if (v_isShared_76_ == 0)
{
lean_ctor_set(v___x_75_, 1, v_a_70_);
lean_ctor_set(v___x_75_, 0, v___x_77_);
v___x_79_ = v___x_75_;
goto v_reusejp_78_;
}
else
{
lean_object* v_reuseFailAlloc_81_; 
v_reuseFailAlloc_81_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_81_, 0, v___x_77_);
lean_ctor_set(v_reuseFailAlloc_81_, 1, v_a_70_);
v___x_79_ = v_reuseFailAlloc_81_;
goto v_reusejp_78_;
}
v_reusejp_78_:
{
v_a_69_ = v_tail_73_;
v_a_70_ = v___x_79_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_mapTokens(uint32_t v_c_84_, lean_object* v_f_85_, lean_object* v_a_86_){
_start:
{
lean_object* v___x_87_; lean_object* v___x_88_; lean_object* v___x_89_; lean_object* v___x_90_; lean_object* v___x_91_; lean_object* v___x_92_; lean_object* v___x_93_; 
v___x_87_ = ((lean_object*)(lp_mathlib_String_mapTokens___closed__0));
v___x_88_ = lean_string_push(v___x_87_, v_c_84_);
v___x_89_ = lean_unsigned_to_nat(0u);
v___x_90_ = lean_box(0);
v___x_91_ = lp_mathlib_String_splitAux___at___00String_mapTokens_spec__0(v_c_84_, v_a_86_, v___x_89_, v___x_89_, v___x_90_);
v___x_92_ = lp_mathlib_List_mapTR_loop___at___00String_mapTokens_spec__1(v_f_85_, v___x_91_, v___x_90_);
v___x_93_ = l_String_intercalate(v___x_88_, v___x_92_);
lean_dec_ref(v___x_88_);
return v___x_93_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_mapTokens___boxed(lean_object* v_c_94_, lean_object* v_f_95_, lean_object* v_a_96_){
_start:
{
uint32_t v_c_boxed_97_; lean_object* v_res_98_; 
v_c_boxed_97_ = lean_unbox_uint32(v_c_94_);
lean_dec(v_c_94_);
v_res_98_ = lp_mathlib_String_mapTokens(v_c_boxed_97_, v_f_95_, v_a_96_);
lean_dec_ref(v_a_96_);
return v_res_98_;
}
}
LEAN_EXPORT uint32_t lp_mathlib_String_head(lean_object* v_s_99_){
_start:
{
lean_object* v___x_100_; lean_object* v___x_101_; lean_object* v___x_102_; lean_object* v___x_103_; 
v___x_100_ = lean_unsigned_to_nat(0u);
v___x_101_ = lean_string_utf8_byte_size(v_s_99_);
v___x_102_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_102_, 0, v_s_99_);
lean_ctor_set(v___x_102_, 1, v___x_100_);
lean_ctor_set(v___x_102_, 2, v___x_101_);
v___x_103_ = l_String_Slice_Pos_get_x3f(v___x_102_, v___x_100_);
lean_dec_ref_known(v___x_102_, 3);
if (lean_obj_tag(v___x_103_) == 0)
{
uint32_t v___x_104_; 
v___x_104_ = 65;
return v___x_104_;
}
else
{
lean_object* v_val_105_; uint32_t v___x_106_; 
v_val_105_ = lean_ctor_get(v___x_103_, 0);
lean_inc(v_val_105_);
lean_dec_ref_known(v___x_103_, 1);
v___x_106_ = lean_unbox_uint32(v_val_105_);
lean_dec(v_val_105_);
return v___x_106_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_head___boxed(lean_object* v_s_107_){
_start:
{
uint32_t v_res_108_; lean_object* v_r_109_; 
v_res_108_ = lp_mathlib_String_head(v_s_107_);
v_r_109_ = lean_box_uint32(v_res_108_);
return v_r_109_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_String_Defs(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_String_Defs(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_String_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_String_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_String_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_String_Defs(builtin);
}
#ifdef __cplusplus
}
#endif
