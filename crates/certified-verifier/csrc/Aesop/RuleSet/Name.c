// Lean compiler output
// Module: Aesop.RuleSet.Name
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
lean_object* l_Lean_Name_mkStr1(lean_object*);
uint8_t lean_name_eq(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
size_t lean_usize_add(size_t, size_t);
static const lean_string_object lp_aesop_Aesop_defaultRuleSetName___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "default"};
static const lean_object* lp_aesop_Aesop_defaultRuleSetName___closed__0 = (const lean_object*)&lp_aesop_Aesop_defaultRuleSetName___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_defaultRuleSetName___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_defaultRuleSetName___closed__0_value),LEAN_SCALAR_PTR_LITERAL(29, 214, 131, 210, 10, 90, 37, 134)}};
static const lean_object* lp_aesop_Aesop_defaultRuleSetName___closed__1 = (const lean_object*)&lp_aesop_Aesop_defaultRuleSetName___closed__1_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_defaultRuleSetName = (const lean_object*)&lp_aesop_Aesop_defaultRuleSetName___closed__1_value;
static const lean_string_object lp_aesop_Aesop_builtinRuleSetName___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "builtin"};
static const lean_object* lp_aesop_Aesop_builtinRuleSetName___closed__0 = (const lean_object*)&lp_aesop_Aesop_builtinRuleSetName___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_builtinRuleSetName___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_builtinRuleSetName___closed__0_value),LEAN_SCALAR_PTR_LITERAL(78, 115, 70, 234, 108, 55, 8, 53)}};
static const lean_object* lp_aesop_Aesop_builtinRuleSetName___closed__1 = (const lean_object*)&lp_aesop_Aesop_builtinRuleSetName___closed__1_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_builtinRuleSetName = (const lean_object*)&lp_aesop_Aesop_builtinRuleSetName___closed__1_value;
static const lean_string_object lp_aesop_Aesop_localRuleSetName___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "local"};
static const lean_object* lp_aesop_Aesop_localRuleSetName___closed__0 = (const lean_object*)&lp_aesop_Aesop_localRuleSetName___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_localRuleSetName___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_localRuleSetName___closed__0_value),LEAN_SCALAR_PTR_LITERAL(120, 104, 189, 185, 38, 81, 44, 71)}};
static const lean_object* lp_aesop_Aesop_localRuleSetName___closed__1 = (const lean_object*)&lp_aesop_Aesop_localRuleSetName___closed__1_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_localRuleSetName = (const lean_object*)&lp_aesop_Aesop_localRuleSetName___closed__1_value;
static const lean_array_object lp_aesop_Aesop_builtinRuleSetNames___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 246}, .m_size = 2, .m_capacity = 2, .m_data = {((lean_object*)&lp_aesop_Aesop_defaultRuleSetName___closed__1_value),((lean_object*)&lp_aesop_Aesop_builtinRuleSetName___closed__1_value)}};
static const lean_object* lp_aesop_Aesop_builtinRuleSetNames___closed__0 = (const lean_object*)&lp_aesop_Aesop_builtinRuleSetNames___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_builtinRuleSetNames = (const lean_object*)&lp_aesop_Aesop_builtinRuleSetNames___closed__0_value;
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Aesop_RuleSetName_isReserved_spec__0_spec__0(lean_object*, lean_object*, size_t, size_t);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Aesop_RuleSetName_isReserved_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Array_contains___at___00Aesop_RuleSetName_isReserved_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_contains___at___00Aesop_RuleSetName_isReserved_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_RuleSetName_isReserved(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleSetName_isReserved___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Aesop_RuleSetName_isReserved_spec__0_spec__0(lean_object* v_a_20_, lean_object* v_as_21_, size_t v_i_22_, size_t v_stop_23_){
_start:
{
uint8_t v___x_24_; 
v___x_24_ = lean_usize_dec_eq(v_i_22_, v_stop_23_);
if (v___x_24_ == 0)
{
lean_object* v___x_25_; uint8_t v___x_26_; 
v___x_25_ = lean_array_uget_borrowed(v_as_21_, v_i_22_);
v___x_26_ = lean_name_eq(v_a_20_, v___x_25_);
if (v___x_26_ == 0)
{
size_t v___x_27_; size_t v___x_28_; 
v___x_27_ = ((size_t)1ULL);
v___x_28_ = lean_usize_add(v_i_22_, v___x_27_);
v_i_22_ = v___x_28_;
goto _start;
}
else
{
return v___x_26_;
}
}
else
{
uint8_t v___x_30_; 
v___x_30_ = 0;
return v___x_30_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Aesop_RuleSetName_isReserved_spec__0_spec__0___boxed(lean_object* v_a_31_, lean_object* v_as_32_, lean_object* v_i_33_, lean_object* v_stop_34_){
_start:
{
size_t v_i_boxed_35_; size_t v_stop_boxed_36_; uint8_t v_res_37_; lean_object* v_r_38_; 
v_i_boxed_35_ = lean_unbox_usize(v_i_33_);
lean_dec(v_i_33_);
v_stop_boxed_36_ = lean_unbox_usize(v_stop_34_);
lean_dec(v_stop_34_);
v_res_37_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Aesop_RuleSetName_isReserved_spec__0_spec__0(v_a_31_, v_as_32_, v_i_boxed_35_, v_stop_boxed_36_);
lean_dec_ref(v_as_32_);
lean_dec(v_a_31_);
v_r_38_ = lean_box(v_res_37_);
return v_r_38_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Array_contains___at___00Aesop_RuleSetName_isReserved_spec__0(lean_object* v_as_39_, lean_object* v_a_40_){
_start:
{
lean_object* v___x_41_; lean_object* v___x_42_; uint8_t v___x_43_; 
v___x_41_ = lean_unsigned_to_nat(0u);
v___x_42_ = lean_array_get_size(v_as_39_);
v___x_43_ = lean_nat_dec_lt(v___x_41_, v___x_42_);
if (v___x_43_ == 0)
{
return v___x_43_;
}
else
{
if (v___x_43_ == 0)
{
return v___x_43_;
}
else
{
size_t v___x_44_; size_t v___x_45_; uint8_t v___x_46_; 
v___x_44_ = ((size_t)0ULL);
v___x_45_ = lean_usize_of_nat(v___x_42_);
v___x_46_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00Aesop_RuleSetName_isReserved_spec__0_spec__0(v_a_40_, v_as_39_, v___x_44_, v___x_45_);
return v___x_46_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_contains___at___00Aesop_RuleSetName_isReserved_spec__0___boxed(lean_object* v_as_47_, lean_object* v_a_48_){
_start:
{
uint8_t v_res_49_; lean_object* v_r_50_; 
v_res_49_ = lp_aesop_Array_contains___at___00Aesop_RuleSetName_isReserved_spec__0(v_as_47_, v_a_48_);
lean_dec(v_a_48_);
lean_dec_ref(v_as_47_);
v_r_50_ = lean_box(v_res_49_);
return v_r_50_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_RuleSetName_isReserved(lean_object* v_n_51_){
_start:
{
lean_object* v___x_52_; uint8_t v___x_53_; 
v___x_52_ = ((lean_object*)(lp_aesop_Aesop_localRuleSetName));
v___x_53_ = lean_name_eq(v_n_51_, v___x_52_);
if (v___x_53_ == 0)
{
lean_object* v___x_54_; uint8_t v___x_55_; 
v___x_54_ = ((lean_object*)(lp_aesop_Aesop_builtinRuleSetNames));
v___x_55_ = lp_aesop_Array_contains___at___00Aesop_RuleSetName_isReserved_spec__0(v___x_54_, v_n_51_);
return v___x_55_;
}
else
{
return v___x_53_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleSetName_isReserved___boxed(lean_object* v_n_56_){
_start:
{
uint8_t v_res_57_; lean_object* v_r_58_; 
v_res_57_ = lp_aesop_Aesop_RuleSetName_isReserved(v_n_56_);
lean_dec(v_n_56_);
v_r_58_ = lean_box(v_res_57_);
return v_r_58_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_RuleSet_Name(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize_runtime_module();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_RuleSet_Name(uint8_t builtin) {
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
LEAN_EXPORT lean_object* initialize_aesop_Aesop_RuleSet_Name(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_RuleSet_Name(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_RuleSet_Name(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_RuleSet_Name(builtin);
}
#ifdef __cplusplus
}
#endif
