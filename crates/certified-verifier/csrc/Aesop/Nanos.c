// Lean compiler output
// Module: Aesop.Nanos
// Imports: public import Init public meta import Init public import Lean.Data.Json.FromToJson.Basic
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
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
uint8_t lean_string_utf8_at_end(lean_object*, lean_object*);
uint32_t lean_string_utf8_get(lean_object*, lean_object*);
uint8_t lean_uint32_dec_eq(uint32_t, uint32_t);
lean_object* lean_string_utf8_next(lean_object*, lean_object*);
lean_object* lean_string_utf8_extract(lean_object*, lean_object*, lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* l_Lean_JsonNumber_fromNat(lean_object*);
lean_object* l_mkPanicMessageWithDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Nat_div___boxed(lean_object*, lean_object*);
lean_object* lean_panic_fn_borrowed(lean_object*, lean_object*);
double lean_float_of_nat(lean_object*);
double lean_float_div(double, double);
lean_object* lean_float_to_string(double);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* lean_string_utf8_byte_size(lean_object*);
lean_object* l_String_Slice_Pos_nextn(lean_object*, lean_object*, lean_object*);
lean_object* lean_string_utf8_extract_fast(lean_object*, lean_object*, lean_object*);
lean_object* l_Nat_add___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedNanos_default;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedNanos;
LEAN_EXPORT uint8_t lp_aesop_Aesop_instBEqNanos_beq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instBEqNanos_beq___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_instBEqNanos___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_instBEqNanos_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_instBEqNanos___closed__0 = (const lean_object*)&lp_aesop_Aesop_instBEqNanos___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instBEqNanos = (const lean_object*)&lp_aesop_Aesop_instBEqNanos___closed__0_value;
LEAN_EXPORT uint8_t lp_aesop_Aesop_instOrdNanos_ord(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instOrdNanos_ord___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_instOrdNanos___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_instOrdNanos_ord___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_instOrdNanos___closed__0 = (const lean_object*)&lp_aesop_Aesop_instOrdNanos___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instOrdNanos = (const lean_object*)&lp_aesop_Aesop_instOrdNanos___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Nanos_instOfNat(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Nanos_instOfNat___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Nanos_instLT;
LEAN_EXPORT uint8_t lp_aesop_Aesop_Nanos_instDecidableRelLt(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Nanos_instDecidableRelLt___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Nanos_instLE;
LEAN_EXPORT uint8_t lp_aesop_Aesop_Nanos_instDecidableRelLe(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Nanos_instDecidableRelLe___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_Nanos_instAdd___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Nat_add___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Nanos_instAdd___closed__0 = (const lean_object*)&lp_aesop_Aesop_Nanos_instAdd___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_Nanos_instAdd = (const lean_object*)&lp_aesop_Aesop_Nanos_instAdd___closed__0_value;
static const lean_closure_object lp_aesop_Aesop_Nanos_instHDivNat___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Nat_div___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Nanos_instHDivNat___closed__0 = (const lean_object*)&lp_aesop_Aesop_Nanos_instHDivNat___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_Nanos_instHDivNat = (const lean_object*)&lp_aesop_Aesop_Nanos_instHDivNat___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Nanos_instToJson___lam__0(lean_object*);
static const lean_closure_object lp_aesop_Aesop_Nanos_instToJson___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_Nanos_instToJson___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Nanos_instToJson___closed__0 = (const lean_object*)&lp_aesop_Aesop_Nanos_instToJson___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_Nanos_instToJson = (const lean_object*)&lp_aesop_Aesop_Nanos_instToJson___closed__0_value;
static const lean_string_object lp_aesop_panic___at___00Aesop_Nanos_printAsMillis_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_aesop_panic___at___00Aesop_Nanos_printAsMillis_spec__0___closed__0 = (const lean_object*)&lp_aesop_panic___at___00Aesop_Nanos_printAsMillis_spec__0___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_panic___at___00Aesop_Nanos_printAsMillis_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_String_splitAux___at___00Aesop_Nanos_printAsMillis_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_String_splitAux___at___00Aesop_Nanos_printAsMillis_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_Nanos_printAsMillis___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "Aesop.Nanos"};
static const lean_object* lp_aesop_Aesop_Nanos_printAsMillis___closed__0 = (const lean_object*)&lp_aesop_Aesop_Nanos_printAsMillis___closed__0_value;
static const lean_string_object lp_aesop_Aesop_Nanos_printAsMillis___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "Aesop.Nanos.printAsMillis"};
static const lean_object* lp_aesop_Aesop_Nanos_printAsMillis___closed__1 = (const lean_object*)&lp_aesop_Aesop_Nanos_printAsMillis___closed__1_value;
static const lean_string_object lp_aesop_Aesop_Nanos_printAsMillis___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 34, .m_capacity = 34, .m_length = 33, .m_data = "unreachable code has been reached"};
static const lean_object* lp_aesop_Aesop_Nanos_printAsMillis___closed__2 = (const lean_object*)&lp_aesop_Aesop_Nanos_printAsMillis___closed__2_value;
static lean_once_cell_t lp_aesop_Aesop_Nanos_printAsMillis___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Nanos_printAsMillis___closed__3;
static lean_once_cell_t lp_aesop_Aesop_Nanos_printAsMillis___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_aesop_Aesop_Nanos_printAsMillis___closed__4;
static const lean_string_object lp_aesop_Aesop_Nanos_printAsMillis___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "ms"};
static const lean_object* lp_aesop_Aesop_Nanos_printAsMillis___closed__5 = (const lean_object*)&lp_aesop_Aesop_Nanos_printAsMillis___closed__5_value;
static const lean_string_object lp_aesop_Aesop_Nanos_printAsMillis___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "."};
static const lean_object* lp_aesop_Aesop_Nanos_printAsMillis___closed__6 = (const lean_object*)&lp_aesop_Aesop_Nanos_printAsMillis___closed__6_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Nanos_printAsMillis(lean_object*);
static lean_object* _init_lp_aesop_Aesop_instInhabitedNanos_default(void){
_start:
{
lean_object* v___x_1_; 
v___x_1_ = lean_unsigned_to_nat(0u);
return v___x_1_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedNanos(void){
_start:
{
lean_object* v___x_2_; 
v___x_2_ = lean_unsigned_to_nat(0u);
return v___x_2_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_instBEqNanos_beq(lean_object* v_x_3_, lean_object* v_x_4_){
_start:
{
uint8_t v___x_5_; 
v___x_5_ = lean_nat_dec_eq(v_x_3_, v_x_4_);
return v___x_5_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instBEqNanos_beq___boxed(lean_object* v_x_6_, lean_object* v_x_7_){
_start:
{
uint8_t v_res_8_; lean_object* v_r_9_; 
v_res_8_ = lp_aesop_Aesop_instBEqNanos_beq(v_x_6_, v_x_7_);
lean_dec(v_x_7_);
lean_dec(v_x_6_);
v_r_9_ = lean_box(v_res_8_);
return v_r_9_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_instOrdNanos_ord(lean_object* v_x_12_, lean_object* v_x_13_){
_start:
{
uint8_t v___x_14_; 
v___x_14_ = lean_nat_dec_lt(v_x_12_, v_x_13_);
if (v___x_14_ == 0)
{
uint8_t v___x_15_; 
v___x_15_ = lean_nat_dec_eq(v_x_12_, v_x_13_);
if (v___x_15_ == 0)
{
uint8_t v___x_16_; 
v___x_16_ = 2;
return v___x_16_;
}
else
{
uint8_t v___x_17_; 
v___x_17_ = 1;
return v___x_17_;
}
}
else
{
uint8_t v___x_18_; 
v___x_18_ = 0;
return v___x_18_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instOrdNanos_ord___boxed(lean_object* v_x_19_, lean_object* v_x_20_){
_start:
{
uint8_t v_res_21_; lean_object* v_r_22_; 
v_res_21_ = lp_aesop_Aesop_instOrdNanos_ord(v_x_19_, v_x_20_);
lean_dec(v_x_20_);
lean_dec(v_x_19_);
v_r_22_ = lean_box(v_res_21_);
return v_r_22_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Nanos_instOfNat(lean_object* v_n_25_){
_start:
{
lean_inc(v_n_25_);
return v_n_25_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Nanos_instOfNat___boxed(lean_object* v_n_26_){
_start:
{
lean_object* v_res_27_; 
v_res_27_ = lp_aesop_Aesop_Nanos_instOfNat(v_n_26_);
lean_dec(v_n_26_);
return v_res_27_;
}
}
static lean_object* _init_lp_aesop_Aesop_Nanos_instLT(void){
_start:
{
lean_object* v___x_28_; 
v___x_28_ = lean_box(0);
return v___x_28_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Nanos_instDecidableRelLt(lean_object* v_x_29_, lean_object* v_x_30_){
_start:
{
uint8_t v___x_31_; 
v___x_31_ = lean_nat_dec_lt(v_x_29_, v_x_30_);
return v___x_31_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Nanos_instDecidableRelLt___boxed(lean_object* v_x_32_, lean_object* v_x_33_){
_start:
{
uint8_t v_res_34_; lean_object* v_r_35_; 
v_res_34_ = lp_aesop_Aesop_Nanos_instDecidableRelLt(v_x_32_, v_x_33_);
lean_dec(v_x_33_);
lean_dec(v_x_32_);
v_r_35_ = lean_box(v_res_34_);
return v_r_35_;
}
}
static lean_object* _init_lp_aesop_Aesop_Nanos_instLE(void){
_start:
{
lean_object* v___x_36_; 
v___x_36_ = lean_box(0);
return v___x_36_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Nanos_instDecidableRelLe(lean_object* v_x_37_, lean_object* v_x_38_){
_start:
{
uint8_t v___x_39_; 
v___x_39_ = lean_nat_dec_le(v_x_37_, v_x_38_);
return v___x_39_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Nanos_instDecidableRelLe___boxed(lean_object* v_x_40_, lean_object* v_x_41_){
_start:
{
uint8_t v_res_42_; lean_object* v_r_43_; 
v_res_42_ = lp_aesop_Aesop_Nanos_instDecidableRelLe(v_x_40_, v_x_41_);
lean_dec(v_x_41_);
lean_dec(v_x_40_);
v_r_43_ = lean_box(v_res_42_);
return v_r_43_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Nanos_instToJson___lam__0(lean_object* v_x_48_){
_start:
{
lean_object* v___x_49_; lean_object* v___x_50_; 
v___x_49_ = l_Lean_JsonNumber_fromNat(v_x_48_);
v___x_50_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v___x_50_, 0, v___x_49_);
return v___x_50_;
}
}
LEAN_EXPORT lean_object* lp_aesop_panic___at___00Aesop_Nanos_printAsMillis_spec__0(lean_object* v_msg_54_){
_start:
{
lean_object* v___x_55_; lean_object* v___x_56_; 
v___x_55_ = ((lean_object*)(lp_aesop_panic___at___00Aesop_Nanos_printAsMillis_spec__0___closed__0));
v___x_56_ = lean_panic_fn_borrowed(v___x_55_, v_msg_54_);
return v___x_56_;
}
}
LEAN_EXPORT lean_object* lp_aesop_String_splitAux___at___00Aesop_Nanos_printAsMillis_spec__1(lean_object* v_s_57_, lean_object* v_b_58_, lean_object* v_i_59_, lean_object* v_r_60_){
_start:
{
uint8_t v___x_61_; 
v___x_61_ = lean_string_utf8_at_end(v_s_57_, v_i_59_);
if (v___x_61_ == 0)
{
uint32_t v___x_62_; uint32_t v___x_63_; uint8_t v___x_64_; 
v___x_62_ = lean_string_utf8_get(v_s_57_, v_i_59_);
v___x_63_ = 46;
v___x_64_ = lean_uint32_dec_eq(v___x_62_, v___x_63_);
if (v___x_64_ == 0)
{
lean_object* v___x_65_; 
v___x_65_ = lean_string_utf8_next(v_s_57_, v_i_59_);
lean_dec(v_i_59_);
v_i_59_ = v___x_65_;
goto _start;
}
else
{
lean_object* v_i_x27_67_; lean_object* v___x_68_; lean_object* v___x_69_; 
v_i_x27_67_ = lean_string_utf8_next(v_s_57_, v_i_59_);
v___x_68_ = lean_string_utf8_extract(v_s_57_, v_b_58_, v_i_59_);
lean_dec(v_i_59_);
lean_dec(v_b_58_);
v___x_69_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_69_, 0, v___x_68_);
lean_ctor_set(v___x_69_, 1, v_r_60_);
lean_inc(v_i_x27_67_);
v_b_58_ = v_i_x27_67_;
v_i_59_ = v_i_x27_67_;
v_r_60_ = v___x_69_;
goto _start;
}
}
else
{
lean_object* v___x_71_; lean_object* v_r_72_; lean_object* v___x_73_; 
v___x_71_ = lean_string_utf8_extract(v_s_57_, v_b_58_, v_i_59_);
lean_dec(v_i_59_);
lean_dec(v_b_58_);
v_r_72_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_r_72_, 0, v___x_71_);
lean_ctor_set(v_r_72_, 1, v_r_60_);
v___x_73_ = l_List_reverse___redArg(v_r_72_);
return v___x_73_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_String_splitAux___at___00Aesop_Nanos_printAsMillis_spec__1___boxed(lean_object* v_s_74_, lean_object* v_b_75_, lean_object* v_i_76_, lean_object* v_r_77_){
_start:
{
lean_object* v_res_78_; 
v_res_78_ = lp_aesop_String_splitAux___at___00Aesop_Nanos_printAsMillis_spec__1(v_s_74_, v_b_75_, v_i_76_, v_r_77_);
lean_dec_ref(v_s_74_);
return v_res_78_;
}
}
static lean_object* _init_lp_aesop_Aesop_Nanos_printAsMillis___closed__3(void){
_start:
{
lean_object* v___x_82_; lean_object* v___x_83_; lean_object* v___x_84_; lean_object* v___x_85_; lean_object* v___x_86_; lean_object* v___x_87_; 
v___x_82_ = ((lean_object*)(lp_aesop_Aesop_Nanos_printAsMillis___closed__2));
v___x_83_ = lean_unsigned_to_nat(9u);
v___x_84_ = lean_unsigned_to_nat(51u);
v___x_85_ = ((lean_object*)(lp_aesop_Aesop_Nanos_printAsMillis___closed__1));
v___x_86_ = ((lean_object*)(lp_aesop_Aesop_Nanos_printAsMillis___closed__0));
v___x_87_ = l_mkPanicMessageWithDecl(v___x_86_, v___x_85_, v___x_84_, v___x_83_, v___x_82_);
return v___x_87_;
}
}
static double _init_lp_aesop_Aesop_Nanos_printAsMillis___closed__4(void){
_start:
{
lean_object* v___x_88_; double v___x_89_; 
v___x_88_ = lean_unsigned_to_nat(1000000u);
v___x_89_ = lean_float_of_nat(v___x_88_);
return v___x_89_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Nanos_printAsMillis(lean_object* v_n_92_){
_start:
{
double v___x_96_; double v___x_97_; double v___x_98_; lean_object* v_str_99_; lean_object* v___x_100_; lean_object* v___x_101_; lean_object* v___x_102_; 
v___x_96_ = lean_float_of_nat(v_n_92_);
v___x_97_ = lean_float_once(&lp_aesop_Aesop_Nanos_printAsMillis___closed__4, &lp_aesop_Aesop_Nanos_printAsMillis___closed__4_once, _init_lp_aesop_Aesop_Nanos_printAsMillis___closed__4);
v___x_98_ = lean_float_div(v___x_96_, v___x_97_);
v_str_99_ = lean_float_to_string(v___x_98_);
v___x_100_ = lean_unsigned_to_nat(0u);
v___x_101_ = lean_box(0);
v___x_102_ = lp_aesop_String_splitAux___at___00Aesop_Nanos_printAsMillis_spec__1(v_str_99_, v___x_100_, v___x_100_, v___x_101_);
lean_dec_ref(v_str_99_);
if (lean_obj_tag(v___x_102_) == 1)
{
lean_object* v_tail_103_; 
v_tail_103_ = lean_ctor_get(v___x_102_, 1);
lean_inc(v_tail_103_);
if (lean_obj_tag(v_tail_103_) == 0)
{
lean_object* v_head_104_; lean_object* v___x_105_; lean_object* v___x_106_; 
v_head_104_ = lean_ctor_get(v___x_102_, 0);
lean_inc(v_head_104_);
lean_dec_ref_known(v___x_102_, 2);
v___x_105_ = ((lean_object*)(lp_aesop_Aesop_Nanos_printAsMillis___closed__5));
v___x_106_ = lean_string_append(v_head_104_, v___x_105_);
return v___x_106_;
}
else
{
lean_object* v_tail_107_; 
v_tail_107_ = lean_ctor_get(v_tail_103_, 1);
if (lean_obj_tag(v_tail_107_) == 0)
{
lean_object* v_head_108_; lean_object* v_head_109_; lean_object* v___x_110_; lean_object* v___x_111_; lean_object* v___x_112_; lean_object* v___x_113_; lean_object* v___x_114_; lean_object* v___x_115_; lean_object* v___x_116_; lean_object* v___x_117_; lean_object* v___x_118_; lean_object* v___x_119_; 
v_head_108_ = lean_ctor_get(v___x_102_, 0);
lean_inc(v_head_108_);
lean_dec_ref_known(v___x_102_, 2);
v_head_109_ = lean_ctor_get(v_tail_103_, 0);
lean_inc_n(v_head_109_, 2);
lean_dec_ref_known(v_tail_103_, 2);
v___x_110_ = ((lean_object*)(lp_aesop_Aesop_Nanos_printAsMillis___closed__6));
v___x_111_ = lean_string_append(v_head_108_, v___x_110_);
v___x_112_ = lean_unsigned_to_nat(1u);
v___x_113_ = lean_string_utf8_byte_size(v_head_109_);
v___x_114_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_114_, 0, v_head_109_);
lean_ctor_set(v___x_114_, 1, v___x_100_);
lean_ctor_set(v___x_114_, 2, v___x_113_);
v___x_115_ = l_String_Slice_Pos_nextn(v___x_114_, v___x_100_, v___x_112_);
lean_dec_ref_known(v___x_114_, 3);
v___x_116_ = lean_string_utf8_extract_fast(v_head_109_, v___x_100_, v___x_115_);
lean_dec(v___x_115_);
lean_dec(v_head_109_);
v___x_117_ = lean_string_append(v___x_111_, v___x_116_);
lean_dec_ref(v___x_116_);
v___x_118_ = ((lean_object*)(lp_aesop_Aesop_Nanos_printAsMillis___closed__5));
v___x_119_ = lean_string_append(v___x_117_, v___x_118_);
return v___x_119_;
}
else
{
lean_dec_ref_known(v_tail_103_, 2);
lean_dec_ref_known(v___x_102_, 2);
goto v___jp_93_;
}
}
}
else
{
lean_dec(v___x_102_);
goto v___jp_93_;
}
v___jp_93_:
{
lean_object* v___x_94_; lean_object* v___x_95_; 
v___x_94_ = lean_obj_once(&lp_aesop_Aesop_Nanos_printAsMillis___closed__3, &lp_aesop_Aesop_Nanos_printAsMillis___closed__3_once, _init_lp_aesop_Aesop_Nanos_printAsMillis___closed__3);
v___x_95_ = lp_aesop_panic___at___00Aesop_Nanos_printAsMillis_spec__0(v___x_94_);
return v___x_95_;
}
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Data_Json_FromToJson_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Nanos(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Data_Json_FromToJson_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_aesop_Aesop_instInhabitedNanos_default = _init_lp_aesop_Aesop_instInhabitedNanos_default();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedNanos_default);
lp_aesop_Aesop_instInhabitedNanos = _init_lp_aesop_Aesop_instInhabitedNanos();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedNanos);
lp_aesop_Aesop_Nanos_instLT = _init_lp_aesop_Aesop_Nanos_instLT();
lean_mark_persistent(lp_aesop_Aesop_Nanos_instLT);
lp_aesop_Aesop_Nanos_instLE = _init_lp_aesop_Aesop_Nanos_instLE();
lean_mark_persistent(lp_aesop_Aesop_Nanos_instLE);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Nanos(uint8_t builtin) {
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
lean_object* initialize_Lean_Data_Json_FromToJson_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Nanos(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Data_Json_FromToJson_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Nanos(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Nanos(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Nanos(builtin);
}
#ifdef __cplusplus
}
#endif
