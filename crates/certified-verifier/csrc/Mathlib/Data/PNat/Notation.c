// Lean compiler output
// Module: Mathlib.Data.PNat.Notation
// Imports: public import Init public meta import Init public import Mathlib.Data.Nat.Notation
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
lean_object* l_Lean_Name_mkStr1(lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instDecidableEqPNat___aux__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instDecidableEqPNat___aux__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instDecidableEqPNat(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instDecidableEqPNat___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_term_u2115_x2b___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 6, .m_data = "termℕ+"};
static const lean_object* lp_mathlib_term_u2115_x2b___closed__0 = (const lean_object*)&lp_mathlib_term_u2115_x2b___closed__0_value;
static const lean_ctor_object lp_mathlib_term_u2115_x2b___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term_u2115_x2b___closed__0_value),LEAN_SCALAR_PTR_LITERAL(235, 108, 13, 139, 34, 109, 100, 54)}};
static const lean_object* lp_mathlib_term_u2115_x2b___closed__1 = (const lean_object*)&lp_mathlib_term_u2115_x2b___closed__1_value;
static const lean_string_object lp_mathlib_term_u2115_x2b___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 2, .m_data = "ℕ+"};
static const lean_object* lp_mathlib_term_u2115_x2b___closed__2 = (const lean_object*)&lp_mathlib_term_u2115_x2b___closed__2_value;
static const lean_ctor_object lp_mathlib_term_u2115_x2b___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term_u2115_x2b___closed__2_value)}};
static const lean_object* lp_mathlib_term_u2115_x2b___closed__3 = (const lean_object*)&lp_mathlib_term_u2115_x2b___closed__3_value;
static const lean_ctor_object lp_mathlib_term_u2115_x2b___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_term_u2115_x2b___closed__1_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_term_u2115_x2b___closed__3_value)}};
static const lean_object* lp_mathlib_term_u2115_x2b___closed__4 = (const lean_object*)&lp_mathlib_term_u2115_x2b___closed__4_value;
LEAN_EXPORT const lean_object* lp_mathlib_term_u2115_x2b = (const lean_object*)&lp_mathlib_term_u2115_x2b___closed__4_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Data__PNat__Notation______macroRules__term_u2115_x2b__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "PNat"};
static const lean_object* lp_mathlib___aux__Mathlib__Data__PNat__Notation______macroRules__term_u2115_x2b__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__PNat__Notation______macroRules__term_u2115_x2b__1___closed__0_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Data__PNat__Notation______macroRules__term_u2115_x2b__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Data__PNat__Notation______macroRules__term_u2115_x2b__1___closed__1;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__PNat__Notation______macroRules__term_u2115_x2b__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Data__PNat__Notation______macroRules__term_u2115_x2b__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(127, 148, 96, 159, 218, 27, 102, 254)}};
static const lean_object* lp_mathlib___aux__Mathlib__Data__PNat__Notation______macroRules__term_u2115_x2b__1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__PNat__Notation______macroRules__term_u2115_x2b__1___closed__2_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__PNat__Notation______macroRules__term_u2115_x2b__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Data__PNat__Notation______macroRules__term_u2115_x2b__1___closed__2_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Data__PNat__Notation______macroRules__term_u2115_x2b__1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__PNat__Notation______macroRules__term_u2115_x2b__1___closed__3_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__PNat__Notation______macroRules__term_u2115_x2b__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Data__PNat__Notation______macroRules__term_u2115_x2b__1___closed__3_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Data__PNat__Notation______macroRules__term_u2115_x2b__1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__PNat__Notation______macroRules__term_u2115_x2b__1___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__PNat__Notation______macroRules__term_u2115_x2b__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__PNat__Notation______macroRules__term_u2115_x2b__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___aux__Mathlib__Data__PNat__Notation______unexpand__PNat__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib___aux__Mathlib__Data__PNat__Notation______unexpand__PNat__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__PNat__Notation______unexpand__PNat__1___closed__0_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__PNat__Notation______unexpand__PNat__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Data__PNat__Notation______unexpand__PNat__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib___aux__Mathlib__Data__PNat__Notation______unexpand__PNat__1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__PNat__Notation______unexpand__PNat__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__PNat__Notation______unexpand__PNat__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__PNat__Notation______unexpand__PNat__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PNat_val(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PNat_val___boxed(lean_object*);
static const lean_closure_object lp_mathlib_coePNatNat___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_PNat_val___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_coePNatNat___closed__0 = (const lean_object*)&lp_mathlib_coePNatNat___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_coePNatNat = (const lean_object*)&lp_mathlib_coePNatNat___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_instReprPNat___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instReprPNat___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_instReprPNat___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_instReprPNat___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_instReprPNat___closed__0 = (const lean_object*)&lp_mathlib_instReprPNat___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_instReprPNat = (const lean_object*)&lp_mathlib_instReprPNat___closed__0_value;
LEAN_EXPORT uint8_t lp_mathlib_instDecidableEqPNat___aux__1(lean_object* v_a_1_, lean_object* v_b_2_){
_start:
{
uint8_t v___x_3_; 
v___x_3_ = lean_nat_dec_eq(v_a_1_, v_b_2_);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDecidableEqPNat___aux__1___boxed(lean_object* v_a_4_, lean_object* v_b_5_){
_start:
{
uint8_t v_res_6_; lean_object* v_r_7_; 
v_res_6_ = lp_mathlib_instDecidableEqPNat___aux__1(v_a_4_, v_b_5_);
lean_dec(v_b_5_);
lean_dec(v_a_4_);
v_r_7_ = lean_box(v_res_6_);
return v_r_7_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instDecidableEqPNat(lean_object* v_a_8_, lean_object* v_b_9_){
_start:
{
uint8_t v___x_10_; 
v___x_10_ = lean_nat_dec_eq(v_a_8_, v_b_9_);
return v___x_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDecidableEqPNat___boxed(lean_object* v_a_11_, lean_object* v_b_12_){
_start:
{
uint8_t v_res_13_; lean_object* v_r_14_; 
v_res_13_ = lp_mathlib_instDecidableEqPNat(v_a_11_, v_b_12_);
lean_dec(v_b_12_);
lean_dec(v_a_11_);
v_r_14_ = lean_box(v_res_13_);
return v_r_14_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Data__PNat__Notation______macroRules__term_u2115_x2b__1___closed__1(void){
_start:
{
lean_object* v___x_27_; lean_object* v___x_28_; 
v___x_27_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__PNat__Notation______macroRules__term_u2115_x2b__1___closed__0));
v___x_28_ = l_String_toRawSubstring_x27(v___x_27_);
return v___x_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__PNat__Notation______macroRules__term_u2115_x2b__1(lean_object* v_x_37_, lean_object* v_a_38_, lean_object* v_a_39_){
_start:
{
lean_object* v___x_40_; uint8_t v___x_41_; 
v___x_40_ = ((lean_object*)(lp_mathlib_term_u2115_x2b___closed__1));
v___x_41_ = l_Lean_Syntax_isOfKind(v_x_37_, v___x_40_);
if (v___x_41_ == 0)
{
lean_object* v___x_42_; lean_object* v___x_43_; 
v___x_42_ = lean_box(1);
v___x_43_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_43_, 0, v___x_42_);
lean_ctor_set(v___x_43_, 1, v_a_39_);
return v___x_43_;
}
else
{
lean_object* v_quotContext_44_; lean_object* v_currMacroScope_45_; lean_object* v_ref_46_; uint8_t v___x_47_; lean_object* v___x_48_; lean_object* v___x_49_; lean_object* v___x_50_; lean_object* v___x_51_; lean_object* v___x_52_; lean_object* v___x_53_; lean_object* v___x_54_; 
v_quotContext_44_ = lean_ctor_get(v_a_38_, 1);
v_currMacroScope_45_ = lean_ctor_get(v_a_38_, 2);
v_ref_46_ = lean_ctor_get(v_a_38_, 5);
v___x_47_ = 0;
v___x_48_ = l_Lean_SourceInfo_fromRef(v_ref_46_, v___x_47_);
v___x_49_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Data__PNat__Notation______macroRules__term_u2115_x2b__1___closed__1, &lp_mathlib___aux__Mathlib__Data__PNat__Notation______macroRules__term_u2115_x2b__1___closed__1_once, _init_lp_mathlib___aux__Mathlib__Data__PNat__Notation______macroRules__term_u2115_x2b__1___closed__1);
v___x_50_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__PNat__Notation______macroRules__term_u2115_x2b__1___closed__2));
lean_inc(v_currMacroScope_45_);
lean_inc(v_quotContext_44_);
v___x_51_ = l_Lean_addMacroScope(v_quotContext_44_, v___x_50_, v_currMacroScope_45_);
v___x_52_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__PNat__Notation______macroRules__term_u2115_x2b__1___closed__4));
v___x_53_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_53_, 0, v___x_48_);
lean_ctor_set(v___x_53_, 1, v___x_49_);
lean_ctor_set(v___x_53_, 2, v___x_51_);
lean_ctor_set(v___x_53_, 3, v___x_52_);
v___x_54_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_54_, 0, v___x_53_);
lean_ctor_set(v___x_54_, 1, v_a_39_);
return v___x_54_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__PNat__Notation______macroRules__term_u2115_x2b__1___boxed(lean_object* v_x_55_, lean_object* v_a_56_, lean_object* v_a_57_){
_start:
{
lean_object* v_res_58_; 
v_res_58_ = lp_mathlib___aux__Mathlib__Data__PNat__Notation______macroRules__term_u2115_x2b__1(v_x_55_, v_a_56_, v_a_57_);
lean_dec_ref(v_a_56_);
return v_res_58_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__PNat__Notation______unexpand__PNat__1(lean_object* v_x_62_, lean_object* v_a_63_, lean_object* v_a_64_){
_start:
{
lean_object* v___x_65_; uint8_t v___x_66_; 
v___x_65_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__PNat__Notation______unexpand__PNat__1___closed__1));
lean_inc(v_x_62_);
v___x_66_ = l_Lean_Syntax_isOfKind(v_x_62_, v___x_65_);
if (v___x_66_ == 0)
{
lean_object* v___x_67_; lean_object* v___x_68_; 
lean_dec(v_x_62_);
v___x_67_ = lean_box(0);
v___x_68_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_68_, 0, v___x_67_);
lean_ctor_set(v___x_68_, 1, v_a_64_);
return v___x_68_;
}
else
{
lean_object* v_ref_69_; uint8_t v___x_70_; lean_object* v___x_71_; lean_object* v___x_72_; lean_object* v___x_73_; lean_object* v___x_74_; lean_object* v___x_75_; lean_object* v___x_76_; 
v_ref_69_ = l_Lean_replaceRef(v_x_62_, v_a_63_);
lean_dec(v_x_62_);
v___x_70_ = 0;
v___x_71_ = l_Lean_SourceInfo_fromRef(v_ref_69_, v___x_70_);
lean_dec(v_ref_69_);
v___x_72_ = ((lean_object*)(lp_mathlib_term_u2115_x2b___closed__1));
v___x_73_ = ((lean_object*)(lp_mathlib_term_u2115_x2b___closed__2));
lean_inc(v___x_71_);
v___x_74_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_74_, 0, v___x_71_);
lean_ctor_set(v___x_74_, 1, v___x_73_);
v___x_75_ = l_Lean_Syntax_node1(v___x_71_, v___x_72_, v___x_74_);
v___x_76_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_76_, 0, v___x_75_);
lean_ctor_set(v___x_76_, 1, v_a_64_);
return v___x_76_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__PNat__Notation______unexpand__PNat__1___boxed(lean_object* v_x_77_, lean_object* v_a_78_, lean_object* v_a_79_){
_start:
{
lean_object* v_res_80_; 
v_res_80_ = lp_mathlib___aux__Mathlib__Data__PNat__Notation______unexpand__PNat__1(v_x_77_, v_a_78_, v_a_79_);
lean_dec(v_a_78_);
return v_res_80_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PNat_val(lean_object* v_self_81_){
_start:
{
lean_inc(v_self_81_);
return v_self_81_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PNat_val___boxed(lean_object* v_self_82_){
_start:
{
lean_object* v_res_83_; 
v_res_83_ = lp_mathlib_PNat_val(v_self_82_);
lean_dec(v_self_82_);
return v_res_83_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instReprPNat___lam__0(lean_object* v_n_86_, lean_object* v_n_x27_87_){
_start:
{
lean_object* v___x_88_; lean_object* v___x_89_; 
v___x_88_ = l_Nat_reprFast(v_n_86_);
v___x_89_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_89_, 0, v___x_88_);
return v___x_89_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instReprPNat___lam__0___boxed(lean_object* v_n_90_, lean_object* v_n_x27_91_){
_start:
{
lean_object* v_res_92_; 
v_res_92_ = lp_mathlib_instReprPNat___lam__0(v_n_90_, v_n_x27_91_);
lean_dec(v_n_x27_91_);
return v_res_92_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Notation(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_PNat_Notation(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Notation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_PNat_Notation(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Nat_Notation(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_PNat_Notation(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Notation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_PNat_Notation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_PNat_Notation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_PNat_Notation(builtin);
}
#ifdef __cplusplus
}
#endif
