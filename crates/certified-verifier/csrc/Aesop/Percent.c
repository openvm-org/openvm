// Lean compiler output
// Module: Aesop.Percent
// Imports: public import Init public meta import Init
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
double lean_uint64_to_float(uint64_t);
lean_object* l_Float_toString___boxed(lean_object*);
lean_object* lean_panic_fn_borrowed(lean_object*, lean_object*);
lean_object* l_mkPanicMessageWithDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
double l_Float_ofScientific(lean_object*, uint8_t, lean_object*);
uint8_t lean_float_decLt(double, double);
double lean_float_sub(double, double);
double lean_float_of_nat(lean_object*);
uint8_t lean_float_decLe(double, double);
double pow(double, double);
double lean_float_div(double, double);
uint8_t lean_string_utf8_at_end(lean_object*, lean_object*);
uint32_t lean_string_utf8_get(lean_object*, lean_object*);
uint8_t lean_uint32_dec_eq(uint32_t, uint32_t);
lean_object* lean_string_utf8_next(lean_object*, lean_object*);
lean_object* lean_string_utf8_extract(lean_object*, lean_object*, lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
double lean_float_mul(double, double);
lean_object* lean_float_to_string(double);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* lean_string_utf8_byte_size(lean_object*);
lean_object* l_String_Slice_Pos_nextn(lean_object*, lean_object*, lean_object*);
lean_object* lean_string_utf8_extract_fast(lean_object*, lean_object*, lean_object*);
lean_object* l_Float_mul___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Aesop_instInhabitedPercent_default___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_aesop_Aesop_instInhabitedPercent_default___closed__0;
LEAN_EXPORT double lp_aesop_Aesop_instInhabitedPercent_default;
LEAN_EXPORT double lp_aesop_Aesop_instInhabitedPercent;
static lean_once_cell_t lp_aesop_Aesop_Percent_ofFloat___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_aesop_Aesop_Percent_ofFloat___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Percent_ofFloat(double);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Percent_ofFloat___boxed(lean_object*);
static const lean_closure_object lp_aesop_Aesop_Percent_instMul___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Float_mul___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Percent_instMul___closed__0 = (const lean_object*)&lp_aesop_Aesop_Percent_instMul___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_Percent_instMul = (const lean_object*)&lp_aesop_Aesop_Percent_instMul___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_Percent_00_u03b4___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_aesop_Aesop_Percent_00_u03b4___closed__0;
LEAN_EXPORT double lp_aesop_Aesop_Percent_00_u03b4;
LEAN_EXPORT uint8_t lp_aesop_Aesop_Percent_instBEq___lam__0(double, double);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Percent_instBEq___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_Percent_instBEq___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_Percent_instBEq___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Percent_instBEq___closed__0 = (const lean_object*)&lp_aesop_Aesop_Percent_instBEq___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_Percent_instBEq = (const lean_object*)&lp_aesop_Aesop_Percent_instBEq___closed__0_value;
LEAN_EXPORT uint8_t lp_aesop_Aesop_Percent_instOrd___lam__0(double, double);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Percent_instOrd___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_Percent_instOrd___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_Percent_instOrd___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Percent_instOrd___closed__0 = (const lean_object*)&lp_aesop_Aesop_Percent_instOrd___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_Percent_instOrd = (const lean_object*)&lp_aesop_Aesop_Percent_instOrd___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Percent_instLT;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Percent_instLE;
static const lean_closure_object lp_aesop_Aesop_Percent_instToString___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Float_toString___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Percent_instToString___closed__0 = (const lean_object*)&lp_aesop_Aesop_Percent_instToString___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_Percent_instToString = (const lean_object*)&lp_aesop_Aesop_Percent_instToString___closed__0_value;
LEAN_EXPORT double lp_aesop_Aesop_Percent_instHPowNat___lam__0(double, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Percent_instHPowNat___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_Percent_instHPowNat___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_Percent_instHPowNat___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_Percent_instHPowNat___closed__0 = (const lean_object*)&lp_aesop_Aesop_Percent_instHPowNat___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_Percent_instHPowNat = (const lean_object*)&lp_aesop_Aesop_Percent_instHPowNat___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_Percent_hundred___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_aesop_Aesop_Percent_hundred___closed__0;
LEAN_EXPORT double lp_aesop_Aesop_Percent_hundred;
static lean_once_cell_t lp_aesop_Aesop_Percent_fifty___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_aesop_Aesop_Percent_fifty___closed__0;
LEAN_EXPORT double lp_aesop_Aesop_Percent_fifty;
static const lean_string_object lp_aesop_panic___at___00Aesop_Percent_toHumanString_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_aesop_panic___at___00Aesop_Percent_toHumanString_spec__0___closed__0 = (const lean_object*)&lp_aesop_panic___at___00Aesop_Percent_toHumanString_spec__0___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_panic___at___00Aesop_Percent_toHumanString_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_String_splitAux___at___00Aesop_Percent_toHumanString_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_String_splitAux___at___00Aesop_Percent_toHumanString_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_Percent_toHumanString___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "Aesop.Percent"};
static const lean_object* lp_aesop_Aesop_Percent_toHumanString___closed__0 = (const lean_object*)&lp_aesop_Aesop_Percent_toHumanString___closed__0_value;
static const lean_string_object lp_aesop_Aesop_Percent_toHumanString___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 28, .m_capacity = 28, .m_length = 27, .m_data = "Aesop.Percent.toHumanString"};
static const lean_object* lp_aesop_Aesop_Percent_toHumanString___closed__1 = (const lean_object*)&lp_aesop_Aesop_Percent_toHumanString___closed__1_value;
static const lean_string_object lp_aesop_Aesop_Percent_toHumanString___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 34, .m_capacity = 34, .m_length = 33, .m_data = "unreachable code has been reached"};
static const lean_object* lp_aesop_Aesop_Percent_toHumanString___closed__2 = (const lean_object*)&lp_aesop_Aesop_Percent_toHumanString___closed__2_value;
static lean_once_cell_t lp_aesop_Aesop_Percent_toHumanString___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Percent_toHumanString___closed__3;
static lean_once_cell_t lp_aesop_Aesop_Percent_toHumanString___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_aesop_Aesop_Percent_toHumanString___closed__4;
static const lean_string_object lp_aesop_Aesop_Percent_toHumanString___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "%"};
static const lean_object* lp_aesop_Aesop_Percent_toHumanString___closed__5 = (const lean_object*)&lp_aesop_Aesop_Percent_toHumanString___closed__5_value;
static const lean_string_object lp_aesop_Aesop_Percent_toHumanString___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "."};
static const lean_object* lp_aesop_Aesop_Percent_toHumanString___closed__6 = (const lean_object*)&lp_aesop_Aesop_Percent_toHumanString___closed__6_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Percent_toHumanString(double);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Percent_toHumanString___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Percent_ofNat(lean_object*);
static double _init_lp_aesop_Aesop_instInhabitedPercent_default___closed__0(void){
_start:
{
uint64_t v___x_1_; double v___x_2_; 
v___x_1_ = 0ULL;
v___x_2_ = lean_uint64_to_float(v___x_1_);
return v___x_2_;
}
}
static double _init_lp_aesop_Aesop_instInhabitedPercent_default(void){
_start:
{
double v___x_3_; 
v___x_3_ = lean_float_once(&lp_aesop_Aesop_instInhabitedPercent_default___closed__0, &lp_aesop_Aesop_instInhabitedPercent_default___closed__0_once, _init_lp_aesop_Aesop_instInhabitedPercent_default___closed__0);
return v___x_3_;
}
}
static double _init_lp_aesop_Aesop_instInhabitedPercent(void){
_start:
{
double v___x_4_; 
v___x_4_ = lp_aesop_Aesop_instInhabitedPercent_default;
return v___x_4_;
}
}
static double _init_lp_aesop_Aesop_Percent_ofFloat___closed__0(void){
_start:
{
lean_object* v___x_5_; double v___x_6_; 
v___x_5_ = lean_unsigned_to_nat(0u);
v___x_6_ = lean_float_of_nat(v___x_5_);
return v___x_6_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Percent_ofFloat(double v_f_7_){
_start:
{
uint8_t v___y_9_; double v___x_13_; uint8_t v___x_14_; 
v___x_13_ = lean_float_once(&lp_aesop_Aesop_Percent_ofFloat___closed__0, &lp_aesop_Aesop_Percent_ofFloat___closed__0_once, _init_lp_aesop_Aesop_Percent_ofFloat___closed__0);
v___x_14_ = lean_float_decLe(v___x_13_, v_f_7_);
if (v___x_14_ == 0)
{
v___y_9_ = v___x_14_;
goto v___jp_8_;
}
else
{
lean_object* v___x_15_; lean_object* v___x_16_; double v___x_17_; uint8_t v___x_18_; 
v___x_15_ = lean_unsigned_to_nat(10u);
v___x_16_ = lean_unsigned_to_nat(1u);
v___x_17_ = l_Float_ofScientific(v___x_15_, v___x_14_, v___x_16_);
v___x_18_ = lean_float_decLe(v_f_7_, v___x_17_);
v___y_9_ = v___x_18_;
goto v___jp_8_;
}
v___jp_8_:
{
if (v___y_9_ == 0)
{
lean_object* v___x_10_; 
v___x_10_ = lean_box(0);
return v___x_10_;
}
else
{
lean_object* v___x_11_; lean_object* v___x_12_; 
v___x_11_ = lean_box_float(v_f_7_);
v___x_12_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_12_, 0, v___x_11_);
return v___x_12_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Percent_ofFloat___boxed(lean_object* v_f_19_){
_start:
{
double v_f_boxed_20_; lean_object* v_res_21_; 
v_f_boxed_20_ = lean_unbox_float(v_f_19_);
lean_dec_ref(v_f_19_);
v_res_21_ = lp_aesop_Aesop_Percent_ofFloat(v_f_boxed_20_);
return v_res_21_;
}
}
static double _init_lp_aesop_Aesop_Percent_00_u03b4___closed__0(void){
_start:
{
lean_object* v___x_24_; uint8_t v___x_25_; lean_object* v___x_26_; double v___x_27_; 
v___x_24_ = lean_unsigned_to_nat(5u);
v___x_25_ = 1;
v___x_26_ = lean_unsigned_to_nat(1u);
v___x_27_ = l_Float_ofScientific(v___x_26_, v___x_25_, v___x_24_);
return v___x_27_;
}
}
static double _init_lp_aesop_Aesop_Percent_00_u03b4(void){
_start:
{
double v___x_28_; 
v___x_28_ = lean_float_once(&lp_aesop_Aesop_Percent_00_u03b4___closed__0, &lp_aesop_Aesop_Percent_00_u03b4___closed__0_once, _init_lp_aesop_Aesop_Percent_00_u03b4___closed__0);
return v___x_28_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Percent_instBEq___lam__0(double v_x_29_, double v_x_30_){
_start:
{
uint8_t v___x_31_; 
v___x_31_ = lean_float_decLt(v_x_30_, v_x_29_);
if (v___x_31_ == 0)
{
double v___x_32_; double v___x_33_; uint8_t v___x_34_; 
v___x_32_ = lean_float_sub(v_x_30_, v_x_29_);
v___x_33_ = lean_float_once(&lp_aesop_Aesop_Percent_00_u03b4___closed__0, &lp_aesop_Aesop_Percent_00_u03b4___closed__0_once, _init_lp_aesop_Aesop_Percent_00_u03b4___closed__0);
v___x_34_ = lean_float_decLt(v___x_32_, v___x_33_);
return v___x_34_;
}
else
{
double v___x_35_; lean_object* v___x_36_; lean_object* v___x_37_; double v___x_38_; uint8_t v___x_39_; 
v___x_35_ = lean_float_sub(v_x_29_, v_x_30_);
v___x_36_ = lean_unsigned_to_nat(1u);
v___x_37_ = lean_unsigned_to_nat(5u);
v___x_38_ = l_Float_ofScientific(v___x_36_, v___x_31_, v___x_37_);
v___x_39_ = lean_float_decLt(v___x_35_, v___x_38_);
return v___x_39_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Percent_instBEq___lam__0___boxed(lean_object* v_x_40_, lean_object* v_x_41_){
_start:
{
double v_x_95__boxed_42_; double v_x_96__boxed_43_; uint8_t v_res_44_; lean_object* v_r_45_; 
v_x_95__boxed_42_ = lean_unbox_float(v_x_40_);
lean_dec_ref(v_x_40_);
v_x_96__boxed_43_ = lean_unbox_float(v_x_41_);
lean_dec_ref(v_x_41_);
v_res_44_ = lp_aesop_Aesop_Percent_instBEq___lam__0(v_x_95__boxed_42_, v_x_96__boxed_43_);
v_r_45_ = lean_box(v_res_44_);
return v_r_45_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_Percent_instOrd___lam__0(double v_p_48_, double v_q_49_){
_start:
{
uint8_t v___y_51_; uint8_t v___x_56_; 
v___x_56_ = lean_float_decLt(v_q_49_, v_p_48_);
if (v___x_56_ == 0)
{
double v___x_57_; double v___x_58_; uint8_t v___x_59_; 
v___x_57_ = lean_float_sub(v_q_49_, v_p_48_);
v___x_58_ = lean_float_once(&lp_aesop_Aesop_Percent_00_u03b4___closed__0, &lp_aesop_Aesop_Percent_00_u03b4___closed__0_once, _init_lp_aesop_Aesop_Percent_00_u03b4___closed__0);
v___x_59_ = lean_float_decLt(v___x_57_, v___x_58_);
v___y_51_ = v___x_59_;
goto v___jp_50_;
}
else
{
double v___x_60_; lean_object* v___x_61_; lean_object* v___x_62_; double v___x_63_; uint8_t v___x_64_; 
v___x_60_ = lean_float_sub(v_p_48_, v_q_49_);
v___x_61_ = lean_unsigned_to_nat(1u);
v___x_62_ = lean_unsigned_to_nat(5u);
v___x_63_ = l_Float_ofScientific(v___x_61_, v___x_56_, v___x_62_);
v___x_64_ = lean_float_decLt(v___x_60_, v___x_63_);
v___y_51_ = v___x_64_;
goto v___jp_50_;
}
v___jp_50_:
{
if (v___y_51_ == 0)
{
uint8_t v___x_52_; 
v___x_52_ = lean_float_decLt(v_p_48_, v_q_49_);
if (v___x_52_ == 0)
{
uint8_t v___x_53_; 
v___x_53_ = 2;
return v___x_53_;
}
else
{
uint8_t v___x_54_; 
v___x_54_ = 0;
return v___x_54_;
}
}
else
{
uint8_t v___x_55_; 
v___x_55_ = 1;
return v___x_55_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Percent_instOrd___lam__0___boxed(lean_object* v_p_65_, lean_object* v_q_66_){
_start:
{
double v_p_boxed_67_; double v_q_boxed_68_; uint8_t v_res_69_; lean_object* v_r_70_; 
v_p_boxed_67_ = lean_unbox_float(v_p_65_);
lean_dec_ref(v_p_65_);
v_q_boxed_68_ = lean_unbox_float(v_q_66_);
lean_dec_ref(v_q_66_);
v_res_69_ = lp_aesop_Aesop_Percent_instOrd___lam__0(v_p_boxed_67_, v_q_boxed_68_);
v_r_70_ = lean_box(v_res_69_);
return v_r_70_;
}
}
static lean_object* _init_lp_aesop_Aesop_Percent_instLT(void){
_start:
{
lean_object* v___x_73_; 
v___x_73_ = lean_box(0);
return v___x_73_;
}
}
static lean_object* _init_lp_aesop_Aesop_Percent_instLE(void){
_start:
{
lean_object* v___x_74_; 
v___x_74_ = lean_box(0);
return v___x_74_;
}
}
LEAN_EXPORT double lp_aesop_Aesop_Percent_instHPowNat___lam__0(double v_x_77_, lean_object* v_x_78_){
_start:
{
double v___x_79_; double v___x_80_; 
v___x_79_ = lean_float_of_nat(v_x_78_);
v___x_80_ = pow(v_x_77_, v___x_79_);
return v___x_80_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Percent_instHPowNat___lam__0___boxed(lean_object* v_x_81_, lean_object* v_x_82_){
_start:
{
double v_x_39__boxed_83_; double v_res_84_; lean_object* v_r_85_; 
v_x_39__boxed_83_ = lean_unbox_float(v_x_81_);
lean_dec_ref(v_x_81_);
v_res_84_ = lp_aesop_Aesop_Percent_instHPowNat___lam__0(v_x_39__boxed_83_, v_x_82_);
v_r_85_ = lean_box_float(v_res_84_);
return v_r_85_;
}
}
static double _init_lp_aesop_Aesop_Percent_hundred___closed__0(void){
_start:
{
lean_object* v___x_88_; double v___x_89_; 
v___x_88_ = lean_unsigned_to_nat(1u);
v___x_89_ = lean_float_of_nat(v___x_88_);
return v___x_89_;
}
}
static double _init_lp_aesop_Aesop_Percent_hundred(void){
_start:
{
double v___x_90_; 
v___x_90_ = lean_float_once(&lp_aesop_Aesop_Percent_hundred___closed__0, &lp_aesop_Aesop_Percent_hundred___closed__0_once, _init_lp_aesop_Aesop_Percent_hundred___closed__0);
return v___x_90_;
}
}
static double _init_lp_aesop_Aesop_Percent_fifty___closed__0(void){
_start:
{
lean_object* v___x_91_; uint8_t v___x_92_; lean_object* v___x_93_; double v___x_94_; 
v___x_91_ = lean_unsigned_to_nat(1u);
v___x_92_ = 1;
v___x_93_ = lean_unsigned_to_nat(5u);
v___x_94_ = l_Float_ofScientific(v___x_93_, v___x_92_, v___x_91_);
return v___x_94_;
}
}
static double _init_lp_aesop_Aesop_Percent_fifty(void){
_start:
{
double v___x_95_; 
v___x_95_ = lean_float_once(&lp_aesop_Aesop_Percent_fifty___closed__0, &lp_aesop_Aesop_Percent_fifty___closed__0_once, _init_lp_aesop_Aesop_Percent_fifty___closed__0);
return v___x_95_;
}
}
LEAN_EXPORT lean_object* lp_aesop_panic___at___00Aesop_Percent_toHumanString_spec__0(lean_object* v_msg_97_){
_start:
{
lean_object* v___x_98_; lean_object* v___x_99_; 
v___x_98_ = ((lean_object*)(lp_aesop_panic___at___00Aesop_Percent_toHumanString_spec__0___closed__0));
v___x_99_ = lean_panic_fn_borrowed(v___x_98_, v_msg_97_);
return v___x_99_;
}
}
LEAN_EXPORT lean_object* lp_aesop_String_splitAux___at___00Aesop_Percent_toHumanString_spec__1(lean_object* v_s_100_, lean_object* v_b_101_, lean_object* v_i_102_, lean_object* v_r_103_){
_start:
{
uint8_t v___x_104_; 
v___x_104_ = lean_string_utf8_at_end(v_s_100_, v_i_102_);
if (v___x_104_ == 0)
{
uint32_t v___x_105_; uint32_t v___x_106_; uint8_t v___x_107_; 
v___x_105_ = lean_string_utf8_get(v_s_100_, v_i_102_);
v___x_106_ = 46;
v___x_107_ = lean_uint32_dec_eq(v___x_105_, v___x_106_);
if (v___x_107_ == 0)
{
lean_object* v___x_108_; 
v___x_108_ = lean_string_utf8_next(v_s_100_, v_i_102_);
lean_dec(v_i_102_);
v_i_102_ = v___x_108_;
goto _start;
}
else
{
lean_object* v_i_x27_110_; lean_object* v___x_111_; lean_object* v___x_112_; 
v_i_x27_110_ = lean_string_utf8_next(v_s_100_, v_i_102_);
v___x_111_ = lean_string_utf8_extract(v_s_100_, v_b_101_, v_i_102_);
lean_dec(v_i_102_);
lean_dec(v_b_101_);
v___x_112_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_112_, 0, v___x_111_);
lean_ctor_set(v___x_112_, 1, v_r_103_);
lean_inc(v_i_x27_110_);
v_b_101_ = v_i_x27_110_;
v_i_102_ = v_i_x27_110_;
v_r_103_ = v___x_112_;
goto _start;
}
}
else
{
lean_object* v___x_114_; lean_object* v_r_115_; lean_object* v___x_116_; 
v___x_114_ = lean_string_utf8_extract(v_s_100_, v_b_101_, v_i_102_);
lean_dec(v_i_102_);
lean_dec(v_b_101_);
v_r_115_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_r_115_, 0, v___x_114_);
lean_ctor_set(v_r_115_, 1, v_r_103_);
v___x_116_ = l_List_reverse___redArg(v_r_115_);
return v___x_116_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_String_splitAux___at___00Aesop_Percent_toHumanString_spec__1___boxed(lean_object* v_s_117_, lean_object* v_b_118_, lean_object* v_i_119_, lean_object* v_r_120_){
_start:
{
lean_object* v_res_121_; 
v_res_121_ = lp_aesop_String_splitAux___at___00Aesop_Percent_toHumanString_spec__1(v_s_117_, v_b_118_, v_i_119_, v_r_120_);
lean_dec_ref(v_s_117_);
return v_res_121_;
}
}
static lean_object* _init_lp_aesop_Aesop_Percent_toHumanString___closed__3(void){
_start:
{
lean_object* v___x_125_; lean_object* v___x_126_; lean_object* v___x_127_; lean_object* v___x_128_; lean_object* v___x_129_; lean_object* v___x_130_; 
v___x_125_ = ((lean_object*)(lp_aesop_Aesop_Percent_toHumanString___closed__2));
v___x_126_ = lean_unsigned_to_nat(9u);
v___x_127_ = lean_unsigned_to_nat(72u);
v___x_128_ = ((lean_object*)(lp_aesop_Aesop_Percent_toHumanString___closed__1));
v___x_129_ = ((lean_object*)(lp_aesop_Aesop_Percent_toHumanString___closed__0));
v___x_130_ = l_mkPanicMessageWithDecl(v___x_129_, v___x_128_, v___x_127_, v___x_126_, v___x_125_);
return v___x_130_;
}
}
static double _init_lp_aesop_Aesop_Percent_toHumanString___closed__4(void){
_start:
{
lean_object* v___x_131_; double v___x_132_; 
v___x_131_ = lean_unsigned_to_nat(100u);
v___x_132_ = lean_float_of_nat(v___x_131_);
return v___x_132_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Percent_toHumanString(double v_p_135_){
_start:
{
double v___x_139_; double v___x_140_; lean_object* v_str_141_; lean_object* v___x_142_; lean_object* v___x_143_; lean_object* v___x_144_; 
v___x_139_ = lean_float_once(&lp_aesop_Aesop_Percent_toHumanString___closed__4, &lp_aesop_Aesop_Percent_toHumanString___closed__4_once, _init_lp_aesop_Aesop_Percent_toHumanString___closed__4);
v___x_140_ = lean_float_mul(v_p_135_, v___x_139_);
v_str_141_ = lean_float_to_string(v___x_140_);
v___x_142_ = lean_unsigned_to_nat(0u);
v___x_143_ = lean_box(0);
v___x_144_ = lp_aesop_String_splitAux___at___00Aesop_Percent_toHumanString_spec__1(v_str_141_, v___x_142_, v___x_142_, v___x_143_);
lean_dec_ref(v_str_141_);
if (lean_obj_tag(v___x_144_) == 1)
{
lean_object* v_tail_145_; 
v_tail_145_ = lean_ctor_get(v___x_144_, 1);
lean_inc(v_tail_145_);
if (lean_obj_tag(v_tail_145_) == 0)
{
lean_object* v_head_146_; lean_object* v___x_147_; lean_object* v___x_148_; 
v_head_146_ = lean_ctor_get(v___x_144_, 0);
lean_inc(v_head_146_);
lean_dec_ref_known(v___x_144_, 2);
v___x_147_ = ((lean_object*)(lp_aesop_Aesop_Percent_toHumanString___closed__5));
v___x_148_ = lean_string_append(v_head_146_, v___x_147_);
return v___x_148_;
}
else
{
lean_object* v_tail_149_; 
v_tail_149_ = lean_ctor_get(v_tail_145_, 1);
if (lean_obj_tag(v_tail_149_) == 0)
{
lean_object* v_head_150_; lean_object* v_head_151_; lean_object* v___x_152_; lean_object* v___x_153_; lean_object* v___x_154_; lean_object* v___x_155_; lean_object* v___x_156_; lean_object* v___x_157_; lean_object* v___x_158_; lean_object* v___x_159_; lean_object* v___x_160_; lean_object* v___x_161_; 
v_head_150_ = lean_ctor_get(v___x_144_, 0);
lean_inc(v_head_150_);
lean_dec_ref_known(v___x_144_, 2);
v_head_151_ = lean_ctor_get(v_tail_145_, 0);
lean_inc_n(v_head_151_, 2);
lean_dec_ref_known(v_tail_145_, 2);
v___x_152_ = ((lean_object*)(lp_aesop_Aesop_Percent_toHumanString___closed__6));
v___x_153_ = lean_string_append(v_head_150_, v___x_152_);
v___x_154_ = lean_unsigned_to_nat(4u);
v___x_155_ = lean_string_utf8_byte_size(v_head_151_);
v___x_156_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_156_, 0, v_head_151_);
lean_ctor_set(v___x_156_, 1, v___x_142_);
lean_ctor_set(v___x_156_, 2, v___x_155_);
v___x_157_ = l_String_Slice_Pos_nextn(v___x_156_, v___x_142_, v___x_154_);
lean_dec_ref_known(v___x_156_, 3);
v___x_158_ = lean_string_utf8_extract_fast(v_head_151_, v___x_142_, v___x_157_);
lean_dec(v___x_157_);
lean_dec(v_head_151_);
v___x_159_ = lean_string_append(v___x_153_, v___x_158_);
lean_dec_ref(v___x_158_);
v___x_160_ = ((lean_object*)(lp_aesop_Aesop_Percent_toHumanString___closed__5));
v___x_161_ = lean_string_append(v___x_159_, v___x_160_);
return v___x_161_;
}
else
{
lean_dec_ref_known(v_tail_145_, 2);
lean_dec_ref_known(v___x_144_, 2);
goto v___jp_136_;
}
}
}
else
{
lean_dec(v___x_144_);
goto v___jp_136_;
}
v___jp_136_:
{
lean_object* v___x_137_; lean_object* v___x_138_; 
v___x_137_ = lean_obj_once(&lp_aesop_Aesop_Percent_toHumanString___closed__3, &lp_aesop_Aesop_Percent_toHumanString___closed__3_once, _init_lp_aesop_Aesop_Percent_toHumanString___closed__3);
v___x_138_ = lp_aesop_panic___at___00Aesop_Percent_toHumanString_spec__0(v___x_137_);
return v___x_138_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Percent_toHumanString___boxed(lean_object* v_p_162_){
_start:
{
double v_p_boxed_163_; lean_object* v_res_164_; 
v_p_boxed_163_ = lean_unbox_float(v_p_162_);
lean_dec_ref(v_p_162_);
v_res_164_ = lp_aesop_Aesop_Percent_toHumanString(v_p_boxed_163_);
return v_res_164_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Percent_ofNat(lean_object* v_n_165_){
_start:
{
double v___x_166_; double v___x_167_; double v___x_168_; lean_object* v___x_169_; 
v___x_166_ = lean_float_of_nat(v_n_165_);
v___x_167_ = lean_float_once(&lp_aesop_Aesop_Percent_toHumanString___closed__4, &lp_aesop_Aesop_Percent_toHumanString___closed__4_once, _init_lp_aesop_Aesop_Percent_toHumanString___closed__4);
v___x_168_ = lean_float_div(v___x_166_, v___x_167_);
v___x_169_ = lp_aesop_Aesop_Percent_ofFloat(v___x_168_);
return v___x_169_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Percent(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize_runtime_module();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_aesop_Aesop_instInhabitedPercent_default = _init_lp_aesop_Aesop_instInhabitedPercent_default();
lp_aesop_Aesop_instInhabitedPercent = _init_lp_aesop_Aesop_instInhabitedPercent();
lp_aesop_Aesop_Percent_00_u03b4 = _init_lp_aesop_Aesop_Percent_00_u03b4();
lp_aesop_Aesop_Percent_instLT = _init_lp_aesop_Aesop_Percent_instLT();
lean_mark_persistent(lp_aesop_Aesop_Percent_instLT);
lp_aesop_Aesop_Percent_instLE = _init_lp_aesop_Aesop_Percent_instLE();
lean_mark_persistent(lp_aesop_Aesop_Percent_instLE);
lp_aesop_Aesop_Percent_hundred = _init_lp_aesop_Aesop_Percent_hundred();
lp_aesop_Aesop_Percent_fifty = _init_lp_aesop_Aesop_Percent_fifty();
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Percent(uint8_t builtin) {
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
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Percent(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Percent(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Percent(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Percent(builtin);
}
#ifdef __cplusplus
}
#endif
