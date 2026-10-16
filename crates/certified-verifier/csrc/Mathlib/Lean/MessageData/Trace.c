// Lean compiler output
// Module: Mathlib.Lean.MessageData.Trace
// Imports: public import Init public meta import Init import Mathlib.Init public import Lean.Message
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
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint64_t lean_string_hash(lean_object*);
uint64_t lean_uint64_shift_right(uint64_t, uint64_t);
uint64_t lean_uint64_xor(uint64_t, uint64_t);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_of_nat(lean_object*);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* lean_nat_div(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_string_utf8_byte_size(lean_object*);
uint8_t lean_string_memcmp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_string_utf8_next_fast(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
uint32_t lean_string_utf8_get_fast(lean_object*, lean_object*);
uint8_t lean_uint32_dec_eq(uint32_t, uint32_t);
lean_object* lean_string_utf8_extract_fast(lean_object*, lean_object*, lean_object*);
lean_object* l_String_splitOnAux(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_Slice_trimAscii(lean_object*);
lean_object* l_String_Slice_toString(lean_object*);
size_t lean_usize_add(size_t, size_t);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* l_Lean_MessageData_toString(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
static const lean_string_object lp_mathlib_Lean_MessageData_traceResultOf___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 2, .m_data = "💥️"};
static const lean_object* lp_mathlib_Lean_MessageData_traceResultOf___closed__0 = (const lean_object*)&lp_mathlib_Lean_MessageData_traceResultOf___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_MessageData_traceResultOf___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_MessageData_traceResultOf___closed__1;
static const lean_ctor_object lp_mathlib_Lean_MessageData_traceResultOf___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(2) << 1) | 1))}};
static const lean_object* lp_mathlib_Lean_MessageData_traceResultOf___closed__2 = (const lean_object*)&lp_mathlib_Lean_MessageData_traceResultOf___closed__2_value;
static const lean_string_object lp_mathlib_Lean_MessageData_traceResultOf___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 2, .m_data = "❌️"};
static const lean_object* lp_mathlib_Lean_MessageData_traceResultOf___closed__3 = (const lean_object*)&lp_mathlib_Lean_MessageData_traceResultOf___closed__3_value;
static lean_once_cell_t lp_mathlib_Lean_MessageData_traceResultOf___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_MessageData_traceResultOf___closed__4;
static const lean_ctor_object lp_mathlib_Lean_MessageData_traceResultOf___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_mathlib_Lean_MessageData_traceResultOf___closed__5 = (const lean_object*)&lp_mathlib_Lean_MessageData_traceResultOf___closed__5_value;
static const lean_string_object lp_mathlib_Lean_MessageData_traceResultOf___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 2, .m_data = "✅️"};
static const lean_object* lp_mathlib_Lean_MessageData_traceResultOf___closed__6 = (const lean_object*)&lp_mathlib_Lean_MessageData_traceResultOf___closed__6_value;
static lean_once_cell_t lp_mathlib_Lean_MessageData_traceResultOf___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_MessageData_traceResultOf___closed__7;
static const lean_ctor_object lp_mathlib_Lean_MessageData_traceResultOf___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Lean_MessageData_traceResultOf___closed__8 = (const lean_object*)&lp_mathlib_Lean_MessageData_traceResultOf___closed__8_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_MessageData_traceResultOf(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MessageData_traceResultOf___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_dropPrefix___at___00Lean_MessageData_stripTraceResultPrefix_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_dropPrefix___at___00Lean_MessageData_stripTraceResultPrefix_spec__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MessageData_stripTraceResultPrefix(lean_object*);
static const lean_string_object lp_mathlib_Lean_MessageData_extractInstName___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "apply "};
static const lean_object* lp_mathlib_Lean_MessageData_extractInstName___closed__0 = (const lean_object*)&lp_mathlib_Lean_MessageData_extractInstName___closed__0_value;
static const lean_string_object lp_mathlib_Lean_MessageData_extractInstName___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " to "};
static const lean_object* lp_mathlib_Lean_MessageData_extractInstName___closed__1 = (const lean_object*)&lp_mathlib_Lean_MessageData_extractInstName___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_MessageData_extractInstName(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MessageData_extractInstName___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Lean_MessageData_dedupByString_spec__0_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Lean_MessageData_dedupByString_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Lean_MessageData_dedupByString_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Lean_MessageData_dedupByString_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Lean_MessageData_dedupByString_spec__1_spec__2_spec__3_spec__5___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Lean_MessageData_dedupByString_spec__1_spec__2_spec__3___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Lean_MessageData_dedupByString_spec__1_spec__2___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Lean_MessageData_dedupByString_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_MessageData_dedupByString_spec__2(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_MessageData_dedupByString_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_MessageData_dedupByString___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_MessageData_dedupByString___closed__0;
static lean_once_cell_t lp_mathlib_Lean_MessageData_dedupByString___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_MessageData_dedupByString___closed__1;
static const lean_array_object lp_mathlib_Lean_MessageData_dedupByString___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Lean_MessageData_dedupByString___closed__2 = (const lean_object*)&lp_mathlib_Lean_MessageData_dedupByString___closed__2_value;
static lean_once_cell_t lp_mathlib_Lean_MessageData_dedupByString___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_MessageData_dedupByString___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Lean_MessageData_dedupByString(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MessageData_dedupByString___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Lean_MessageData_dedupByString_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Lean_MessageData_dedupByString_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Lean_MessageData_dedupByString_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Lean_MessageData_dedupByString_spec__0_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Lean_MessageData_dedupByString_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Lean_MessageData_dedupByString_spec__1_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Lean_MessageData_dedupByString_spec__1_spec__2_spec__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Lean_MessageData_dedupByString_spec__1_spec__2_spec__3_spec__5(lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Lean_MessageData_traceResultOf___closed__1(void){
_start:
{
lean_object* v___x_2_; lean_object* v___x_3_; 
v___x_2_ = ((lean_object*)(lp_mathlib_Lean_MessageData_traceResultOf___closed__0));
v___x_3_ = lean_string_utf8_byte_size(v___x_2_);
return v___x_3_;
}
}
static lean_object* _init_lp_mathlib_Lean_MessageData_traceResultOf___closed__4(void){
_start:
{
lean_object* v___x_8_; lean_object* v___x_9_; 
v___x_8_ = ((lean_object*)(lp_mathlib_Lean_MessageData_traceResultOf___closed__3));
v___x_9_ = lean_string_utf8_byte_size(v___x_8_);
return v___x_9_;
}
}
static lean_object* _init_lp_mathlib_Lean_MessageData_traceResultOf___closed__7(void){
_start:
{
lean_object* v___x_14_; lean_object* v___x_15_; 
v___x_14_ = ((lean_object*)(lp_mathlib_Lean_MessageData_traceResultOf___closed__6));
v___x_15_ = lean_string_utf8_byte_size(v___x_14_);
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MessageData_traceResultOf(lean_object* v_headerStr_19_){
_start:
{
lean_object* v___x_38_; lean_object* v___x_39_; lean_object* v___x_40_; uint8_t v___x_41_; 
v___x_38_ = ((lean_object*)(lp_mathlib_Lean_MessageData_traceResultOf___closed__6));
v___x_39_ = lean_string_utf8_byte_size(v_headerStr_19_);
v___x_40_ = lean_obj_once(&lp_mathlib_Lean_MessageData_traceResultOf___closed__7, &lp_mathlib_Lean_MessageData_traceResultOf___closed__7_once, _init_lp_mathlib_Lean_MessageData_traceResultOf___closed__7);
v___x_41_ = lean_nat_dec_le(v___x_40_, v___x_39_);
if (v___x_41_ == 0)
{
goto v___jp_30_;
}
else
{
lean_object* v___x_42_; uint8_t v___x_43_; 
v___x_42_ = lean_unsigned_to_nat(0u);
v___x_43_ = lean_string_memcmp(v_headerStr_19_, v___x_38_, v___x_42_, v___x_42_, v___x_40_);
if (v___x_43_ == 0)
{
goto v___jp_30_;
}
else
{
lean_object* v___x_44_; 
v___x_44_ = ((lean_object*)(lp_mathlib_Lean_MessageData_traceResultOf___closed__8));
return v___x_44_;
}
}
v___jp_20_:
{
lean_object* v___x_21_; lean_object* v___x_22_; lean_object* v___x_23_; uint8_t v___x_24_; 
v___x_21_ = ((lean_object*)(lp_mathlib_Lean_MessageData_traceResultOf___closed__0));
v___x_22_ = lean_string_utf8_byte_size(v_headerStr_19_);
v___x_23_ = lean_obj_once(&lp_mathlib_Lean_MessageData_traceResultOf___closed__1, &lp_mathlib_Lean_MessageData_traceResultOf___closed__1_once, _init_lp_mathlib_Lean_MessageData_traceResultOf___closed__1);
v___x_24_ = lean_nat_dec_le(v___x_23_, v___x_22_);
if (v___x_24_ == 0)
{
lean_object* v___x_25_; 
v___x_25_ = lean_box(0);
return v___x_25_;
}
else
{
lean_object* v___x_26_; uint8_t v___x_27_; 
v___x_26_ = lean_unsigned_to_nat(0u);
v___x_27_ = lean_string_memcmp(v_headerStr_19_, v___x_21_, v___x_26_, v___x_26_, v___x_23_);
if (v___x_27_ == 0)
{
lean_object* v___x_28_; 
v___x_28_ = lean_box(0);
return v___x_28_;
}
else
{
lean_object* v___x_29_; 
v___x_29_ = ((lean_object*)(lp_mathlib_Lean_MessageData_traceResultOf___closed__2));
return v___x_29_;
}
}
}
v___jp_30_:
{
lean_object* v___x_31_; lean_object* v___x_32_; lean_object* v___x_33_; uint8_t v___x_34_; 
v___x_31_ = ((lean_object*)(lp_mathlib_Lean_MessageData_traceResultOf___closed__3));
v___x_32_ = lean_string_utf8_byte_size(v_headerStr_19_);
v___x_33_ = lean_obj_once(&lp_mathlib_Lean_MessageData_traceResultOf___closed__4, &lp_mathlib_Lean_MessageData_traceResultOf___closed__4_once, _init_lp_mathlib_Lean_MessageData_traceResultOf___closed__4);
v___x_34_ = lean_nat_dec_le(v___x_33_, v___x_32_);
if (v___x_34_ == 0)
{
goto v___jp_20_;
}
else
{
lean_object* v___x_35_; uint8_t v___x_36_; 
v___x_35_ = lean_unsigned_to_nat(0u);
v___x_36_ = lean_string_memcmp(v_headerStr_19_, v___x_31_, v___x_35_, v___x_35_, v___x_33_);
if (v___x_36_ == 0)
{
goto v___jp_20_;
}
else
{
lean_object* v___x_37_; 
v___x_37_ = ((lean_object*)(lp_mathlib_Lean_MessageData_traceResultOf___closed__5));
return v___x_37_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MessageData_traceResultOf___boxed(lean_object* v_headerStr_45_){
_start:
{
lean_object* v_res_46_; 
v_res_46_ = lp_mathlib_Lean_MessageData_traceResultOf(v_headerStr_45_);
lean_dec_ref(v_headerStr_45_);
return v_res_46_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_dropPrefix___at___00Lean_MessageData_stripTraceResultPrefix_spec__0(lean_object* v_s_47_){
_start:
{
lean_object* v_str_48_; lean_object* v_startInclusive_49_; lean_object* v_endExclusive_50_; uint8_t v___y_52_; lean_object* v___x_66_; lean_object* v___x_67_; uint8_t v___x_68_; 
v_str_48_ = lean_ctor_get(v_s_47_, 0);
v_startInclusive_49_ = lean_ctor_get(v_s_47_, 1);
v_endExclusive_50_ = lean_ctor_get(v_s_47_, 2);
v___x_66_ = lean_unsigned_to_nat(0u);
v___x_67_ = lean_nat_sub(v_endExclusive_50_, v_startInclusive_49_);
v___x_68_ = lean_nat_dec_eq(v___x_66_, v___x_67_);
lean_dec(v___x_67_);
if (v___x_68_ == 0)
{
uint32_t v___x_69_; uint8_t v___y_71_; uint32_t v___x_76_; uint8_t v___x_77_; 
v___x_69_ = lean_string_utf8_get_fast(v_str_48_, v_startInclusive_49_);
v___x_76_ = 32;
v___x_77_ = lean_uint32_dec_eq(v___x_69_, v___x_76_);
if (v___x_77_ == 0)
{
uint32_t v___x_78_; uint8_t v___x_79_; 
v___x_78_ = 9;
v___x_79_ = lean_uint32_dec_eq(v___x_69_, v___x_78_);
v___y_71_ = v___x_79_;
goto v___jp_70_;
}
else
{
v___y_71_ = v___x_77_;
goto v___jp_70_;
}
v___jp_70_:
{
if (v___y_71_ == 0)
{
uint32_t v___x_72_; uint8_t v___x_73_; 
v___x_72_ = 13;
v___x_73_ = lean_uint32_dec_eq(v___x_69_, v___x_72_);
if (v___x_73_ == 0)
{
uint32_t v___x_74_; uint8_t v___x_75_; 
v___x_74_ = 10;
v___x_75_ = lean_uint32_dec_eq(v___x_69_, v___x_74_);
v___y_52_ = v___x_75_;
goto v___jp_51_;
}
else
{
v___y_52_ = v___x_73_;
goto v___jp_51_;
}
}
else
{
return v_s_47_;
}
}
}
else
{
return v_s_47_;
}
v___jp_51_:
{
if (v___y_52_ == 0)
{
lean_object* v___x_54_; uint8_t v_isShared_55_; uint8_t v_isSharedCheck_62_; 
lean_inc(v_endExclusive_50_);
lean_inc(v_startInclusive_49_);
lean_inc_ref(v_str_48_);
v_isSharedCheck_62_ = !lean_is_exclusive(v_s_47_);
if (v_isSharedCheck_62_ == 0)
{
lean_object* v_unused_63_; lean_object* v_unused_64_; lean_object* v_unused_65_; 
v_unused_63_ = lean_ctor_get(v_s_47_, 2);
lean_dec(v_unused_63_);
v_unused_64_ = lean_ctor_get(v_s_47_, 1);
lean_dec(v_unused_64_);
v_unused_65_ = lean_ctor_get(v_s_47_, 0);
lean_dec(v_unused_65_);
v___x_54_ = v_s_47_;
v_isShared_55_ = v_isSharedCheck_62_;
goto v_resetjp_53_;
}
else
{
lean_dec(v_s_47_);
v___x_54_ = lean_box(0);
v_isShared_55_ = v_isSharedCheck_62_;
goto v_resetjp_53_;
}
v_resetjp_53_:
{
lean_object* v___x_56_; lean_object* v___x_57_; lean_object* v___x_58_; lean_object* v___x_60_; 
v___x_56_ = lean_string_utf8_next_fast(v_str_48_, v_startInclusive_49_);
v___x_57_ = lean_nat_sub(v___x_56_, v_startInclusive_49_);
v___x_58_ = lean_nat_add(v_startInclusive_49_, v___x_57_);
lean_dec(v___x_57_);
lean_dec(v_startInclusive_49_);
if (v_isShared_55_ == 0)
{
lean_ctor_set(v___x_54_, 1, v___x_58_);
v___x_60_ = v___x_54_;
goto v_reusejp_59_;
}
else
{
lean_object* v_reuseFailAlloc_61_; 
v_reuseFailAlloc_61_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_61_, 0, v_str_48_);
lean_ctor_set(v_reuseFailAlloc_61_, 1, v___x_58_);
lean_ctor_set(v_reuseFailAlloc_61_, 2, v_endExclusive_50_);
v___x_60_ = v_reuseFailAlloc_61_;
goto v_reusejp_59_;
}
v_reusejp_59_:
{
return v___x_60_;
}
}
}
else
{
return v_s_47_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_String_Slice_dropPrefix___at___00Lean_MessageData_stripTraceResultPrefix_spec__1(lean_object* v_s_80_){
_start:
{
lean_object* v_str_81_; lean_object* v_startInclusive_82_; lean_object* v_endExclusive_83_; lean_object* v___x_84_; lean_object* v___x_85_; uint8_t v___x_86_; 
v_str_81_ = lean_ctor_get(v_s_80_, 0);
v_startInclusive_82_ = lean_ctor_get(v_s_80_, 1);
v_endExclusive_83_ = lean_ctor_get(v_s_80_, 2);
v___x_84_ = lean_unsigned_to_nat(0u);
v___x_85_ = lean_nat_sub(v_endExclusive_83_, v_startInclusive_82_);
v___x_86_ = lean_nat_dec_eq(v___x_84_, v___x_85_);
lean_dec(v___x_85_);
if (v___x_86_ == 0)
{
uint32_t v___x_87_; uint32_t v___x_88_; uint8_t v___x_89_; 
v___x_87_ = 32;
v___x_88_ = lean_string_utf8_get_fast(v_str_81_, v_startInclusive_82_);
v___x_89_ = lean_uint32_dec_eq(v___x_88_, v___x_87_);
if (v___x_89_ == 0)
{
return v_s_80_;
}
else
{
lean_object* v___x_91_; uint8_t v_isShared_92_; uint8_t v_isSharedCheck_99_; 
lean_inc(v_endExclusive_83_);
lean_inc(v_startInclusive_82_);
lean_inc_ref(v_str_81_);
v_isSharedCheck_99_ = !lean_is_exclusive(v_s_80_);
if (v_isSharedCheck_99_ == 0)
{
lean_object* v_unused_100_; lean_object* v_unused_101_; lean_object* v_unused_102_; 
v_unused_100_ = lean_ctor_get(v_s_80_, 2);
lean_dec(v_unused_100_);
v_unused_101_ = lean_ctor_get(v_s_80_, 1);
lean_dec(v_unused_101_);
v_unused_102_ = lean_ctor_get(v_s_80_, 0);
lean_dec(v_unused_102_);
v___x_91_ = v_s_80_;
v_isShared_92_ = v_isSharedCheck_99_;
goto v_resetjp_90_;
}
else
{
lean_dec(v_s_80_);
v___x_91_ = lean_box(0);
v_isShared_92_ = v_isSharedCheck_99_;
goto v_resetjp_90_;
}
v_resetjp_90_:
{
lean_object* v___x_93_; lean_object* v___x_94_; lean_object* v___x_95_; lean_object* v___x_97_; 
v___x_93_ = lean_string_utf8_next_fast(v_str_81_, v_startInclusive_82_);
v___x_94_ = lean_nat_sub(v___x_93_, v_startInclusive_82_);
v___x_95_ = lean_nat_add(v_startInclusive_82_, v___x_94_);
lean_dec(v___x_94_);
lean_dec(v_startInclusive_82_);
if (v_isShared_92_ == 0)
{
lean_ctor_set(v___x_91_, 1, v___x_95_);
v___x_97_ = v___x_91_;
goto v_reusejp_96_;
}
else
{
lean_object* v_reuseFailAlloc_98_; 
v_reuseFailAlloc_98_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_98_, 0, v_str_81_);
lean_ctor_set(v_reuseFailAlloc_98_, 1, v___x_95_);
lean_ctor_set(v_reuseFailAlloc_98_, 2, v_endExclusive_83_);
v___x_97_ = v_reuseFailAlloc_98_;
goto v_reusejp_96_;
}
v_reusejp_96_:
{
return v___x_97_;
}
}
}
}
else
{
return v_s_80_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MessageData_stripTraceResultPrefix(lean_object* v_s_103_){
_start:
{
lean_object* v___x_104_; 
v___x_104_ = lp_mathlib_Lean_MessageData_traceResultOf(v_s_103_);
if (lean_obj_tag(v___x_104_) == 0)
{
return v_s_103_;
}
else
{
lean_object* v___x_105_; lean_object* v___x_106_; lean_object* v___x_107_; lean_object* v___x_108_; lean_object* v___x_109_; lean_object* v_str_110_; lean_object* v_startInclusive_111_; lean_object* v_endExclusive_112_; lean_object* v___x_113_; 
lean_dec_ref_known(v___x_104_, 1);
v___x_105_ = lean_unsigned_to_nat(0u);
v___x_106_ = lean_string_utf8_byte_size(v_s_103_);
v___x_107_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_107_, 0, v_s_103_);
lean_ctor_set(v___x_107_, 1, v___x_105_);
lean_ctor_set(v___x_107_, 2, v___x_106_);
v___x_108_ = lp_mathlib_String_Slice_dropPrefix___at___00Lean_MessageData_stripTraceResultPrefix_spec__0(v___x_107_);
v___x_109_ = lp_mathlib_String_Slice_dropPrefix___at___00Lean_MessageData_stripTraceResultPrefix_spec__1(v___x_108_);
v_str_110_ = lean_ctor_get(v___x_109_, 0);
lean_inc_ref(v_str_110_);
v_startInclusive_111_ = lean_ctor_get(v___x_109_, 1);
lean_inc(v_startInclusive_111_);
v_endExclusive_112_ = lean_ctor_get(v___x_109_, 2);
lean_inc(v_endExclusive_112_);
lean_dec_ref(v___x_109_);
v___x_113_ = lean_string_utf8_extract_fast(v_str_110_, v_startInclusive_111_, v_endExclusive_112_);
lean_dec(v_endExclusive_112_);
lean_dec(v_startInclusive_111_);
lean_dec_ref(v_str_110_);
return v___x_113_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MessageData_extractInstName(lean_object* v_s_116_){
_start:
{
lean_object* v___x_117_; lean_object* v___x_118_; lean_object* v___x_119_; lean_object* v___x_120_; 
v___x_117_ = ((lean_object*)(lp_mathlib_Lean_MessageData_extractInstName___closed__0));
v___x_118_ = lean_unsigned_to_nat(0u);
v___x_119_ = lean_box(0);
v___x_120_ = l_String_splitOnAux(v_s_116_, v___x_117_, v___x_118_, v___x_118_, v___x_118_, v___x_119_);
if (lean_obj_tag(v___x_120_) == 1)
{
lean_object* v_tail_121_; 
v_tail_121_ = lean_ctor_get(v___x_120_, 1);
lean_inc(v_tail_121_);
lean_dec_ref_known(v___x_120_, 2);
if (lean_obj_tag(v_tail_121_) == 1)
{
lean_object* v_tail_122_; 
v_tail_122_ = lean_ctor_get(v_tail_121_, 1);
lean_inc(v_tail_122_);
if (lean_obj_tag(v_tail_122_) == 0)
{
lean_object* v_head_123_; lean_object* v___x_124_; lean_object* v___x_125_; 
v_head_123_ = lean_ctor_get(v_tail_121_, 0);
lean_inc(v_head_123_);
lean_dec_ref_known(v_tail_121_, 2);
v___x_124_ = ((lean_object*)(lp_mathlib_Lean_MessageData_extractInstName___closed__1));
v___x_125_ = l_String_splitOnAux(v_head_123_, v___x_124_, v___x_118_, v___x_118_, v___x_118_, v_tail_122_);
lean_dec(v_head_123_);
if (lean_obj_tag(v___x_125_) == 1)
{
lean_object* v_head_126_; lean_object* v___x_127_; lean_object* v___x_128_; lean_object* v___x_129_; lean_object* v___x_130_; 
v_head_126_ = lean_ctor_get(v___x_125_, 0);
lean_inc(v_head_126_);
lean_dec_ref_known(v___x_125_, 2);
v___x_127_ = lean_string_utf8_byte_size(v_head_126_);
v___x_128_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_128_, 0, v_head_126_);
lean_ctor_set(v___x_128_, 1, v___x_118_);
lean_ctor_set(v___x_128_, 2, v___x_127_);
v___x_129_ = l_String_Slice_trimAscii(v___x_128_);
v___x_130_ = l_String_Slice_toString(v___x_129_);
lean_dec_ref(v___x_129_);
return v___x_130_;
}
else
{
lean_dec(v___x_125_);
lean_inc_ref(v_s_116_);
return v_s_116_;
}
}
else
{
lean_dec_ref_known(v_tail_121_, 2);
lean_dec(v_tail_122_);
lean_inc_ref(v_s_116_);
return v_s_116_;
}
}
else
{
lean_dec(v_tail_121_);
lean_inc_ref(v_s_116_);
return v_s_116_;
}
}
else
{
lean_dec(v___x_120_);
lean_inc_ref(v_s_116_);
return v_s_116_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MessageData_extractInstName___boxed(lean_object* v_s_131_){
_start:
{
lean_object* v_res_132_; 
v_res_132_ = lp_mathlib_Lean_MessageData_extractInstName(v_s_131_);
lean_dec_ref(v_s_131_);
return v_res_132_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Lean_MessageData_dedupByString_spec__0_spec__0___redArg(lean_object* v_a_133_, lean_object* v_x_134_){
_start:
{
if (lean_obj_tag(v_x_134_) == 0)
{
uint8_t v___x_135_; 
v___x_135_ = 0;
return v___x_135_;
}
else
{
lean_object* v_key_136_; lean_object* v_tail_137_; uint8_t v___x_138_; 
v_key_136_ = lean_ctor_get(v_x_134_, 0);
v_tail_137_ = lean_ctor_get(v_x_134_, 2);
v___x_138_ = lean_string_dec_eq(v_key_136_, v_a_133_);
if (v___x_138_ == 0)
{
v_x_134_ = v_tail_137_;
goto _start;
}
else
{
return v___x_138_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Lean_MessageData_dedupByString_spec__0_spec__0___redArg___boxed(lean_object* v_a_140_, lean_object* v_x_141_){
_start:
{
uint8_t v_res_142_; lean_object* v_r_143_; 
v_res_142_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Lean_MessageData_dedupByString_spec__0_spec__0___redArg(v_a_140_, v_x_141_);
lean_dec(v_x_141_);
lean_dec_ref(v_a_140_);
v_r_143_ = lean_box(v_res_142_);
return v_r_143_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Lean_MessageData_dedupByString_spec__0___redArg(lean_object* v_m_144_, lean_object* v_a_145_){
_start:
{
lean_object* v_buckets_146_; lean_object* v___x_147_; uint64_t v___x_148_; uint64_t v___x_149_; uint64_t v___x_150_; uint64_t v_fold_151_; uint64_t v___x_152_; uint64_t v___x_153_; uint64_t v___x_154_; size_t v___x_155_; size_t v___x_156_; size_t v___x_157_; size_t v___x_158_; size_t v___x_159_; lean_object* v___x_160_; uint8_t v___x_161_; 
v_buckets_146_ = lean_ctor_get(v_m_144_, 1);
v___x_147_ = lean_array_get_size(v_buckets_146_);
v___x_148_ = lean_string_hash(v_a_145_);
v___x_149_ = 32ULL;
v___x_150_ = lean_uint64_shift_right(v___x_148_, v___x_149_);
v_fold_151_ = lean_uint64_xor(v___x_148_, v___x_150_);
v___x_152_ = 16ULL;
v___x_153_ = lean_uint64_shift_right(v_fold_151_, v___x_152_);
v___x_154_ = lean_uint64_xor(v_fold_151_, v___x_153_);
v___x_155_ = lean_uint64_to_usize(v___x_154_);
v___x_156_ = lean_usize_of_nat(v___x_147_);
v___x_157_ = ((size_t)1ULL);
v___x_158_ = lean_usize_sub(v___x_156_, v___x_157_);
v___x_159_ = lean_usize_land(v___x_155_, v___x_158_);
v___x_160_ = lean_array_uget_borrowed(v_buckets_146_, v___x_159_);
v___x_161_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Lean_MessageData_dedupByString_spec__0_spec__0___redArg(v_a_145_, v___x_160_);
return v___x_161_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Lean_MessageData_dedupByString_spec__0___redArg___boxed(lean_object* v_m_162_, lean_object* v_a_163_){
_start:
{
uint8_t v_res_164_; lean_object* v_r_165_; 
v_res_164_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Lean_MessageData_dedupByString_spec__0___redArg(v_m_162_, v_a_163_);
lean_dec_ref(v_a_163_);
lean_dec_ref(v_m_162_);
v_r_165_ = lean_box(v_res_164_);
return v_r_165_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Lean_MessageData_dedupByString_spec__1_spec__2_spec__3_spec__5___redArg(lean_object* v_x_166_, lean_object* v_x_167_){
_start:
{
if (lean_obj_tag(v_x_167_) == 0)
{
return v_x_166_;
}
else
{
lean_object* v_key_168_; lean_object* v_value_169_; lean_object* v_tail_170_; lean_object* v___x_172_; uint8_t v_isShared_173_; uint8_t v_isSharedCheck_193_; 
v_key_168_ = lean_ctor_get(v_x_167_, 0);
v_value_169_ = lean_ctor_get(v_x_167_, 1);
v_tail_170_ = lean_ctor_get(v_x_167_, 2);
v_isSharedCheck_193_ = !lean_is_exclusive(v_x_167_);
if (v_isSharedCheck_193_ == 0)
{
v___x_172_ = v_x_167_;
v_isShared_173_ = v_isSharedCheck_193_;
goto v_resetjp_171_;
}
else
{
lean_inc(v_tail_170_);
lean_inc(v_value_169_);
lean_inc(v_key_168_);
lean_dec(v_x_167_);
v___x_172_ = lean_box(0);
v_isShared_173_ = v_isSharedCheck_193_;
goto v_resetjp_171_;
}
v_resetjp_171_:
{
lean_object* v___x_174_; uint64_t v___x_175_; uint64_t v___x_176_; uint64_t v___x_177_; uint64_t v_fold_178_; uint64_t v___x_179_; uint64_t v___x_180_; uint64_t v___x_181_; size_t v___x_182_; size_t v___x_183_; size_t v___x_184_; size_t v___x_185_; size_t v___x_186_; lean_object* v___x_187_; lean_object* v___x_189_; 
v___x_174_ = lean_array_get_size(v_x_166_);
v___x_175_ = lean_string_hash(v_key_168_);
v___x_176_ = 32ULL;
v___x_177_ = lean_uint64_shift_right(v___x_175_, v___x_176_);
v_fold_178_ = lean_uint64_xor(v___x_175_, v___x_177_);
v___x_179_ = 16ULL;
v___x_180_ = lean_uint64_shift_right(v_fold_178_, v___x_179_);
v___x_181_ = lean_uint64_xor(v_fold_178_, v___x_180_);
v___x_182_ = lean_uint64_to_usize(v___x_181_);
v___x_183_ = lean_usize_of_nat(v___x_174_);
v___x_184_ = ((size_t)1ULL);
v___x_185_ = lean_usize_sub(v___x_183_, v___x_184_);
v___x_186_ = lean_usize_land(v___x_182_, v___x_185_);
v___x_187_ = lean_array_uget_borrowed(v_x_166_, v___x_186_);
lean_inc(v___x_187_);
if (v_isShared_173_ == 0)
{
lean_ctor_set(v___x_172_, 2, v___x_187_);
v___x_189_ = v___x_172_;
goto v_reusejp_188_;
}
else
{
lean_object* v_reuseFailAlloc_192_; 
v_reuseFailAlloc_192_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_192_, 0, v_key_168_);
lean_ctor_set(v_reuseFailAlloc_192_, 1, v_value_169_);
lean_ctor_set(v_reuseFailAlloc_192_, 2, v___x_187_);
v___x_189_ = v_reuseFailAlloc_192_;
goto v_reusejp_188_;
}
v_reusejp_188_:
{
lean_object* v___x_190_; 
v___x_190_ = lean_array_uset(v_x_166_, v___x_186_, v___x_189_);
v_x_166_ = v___x_190_;
v_x_167_ = v_tail_170_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Lean_MessageData_dedupByString_spec__1_spec__2_spec__3___redArg(lean_object* v_i_194_, lean_object* v_source_195_, lean_object* v_target_196_){
_start:
{
lean_object* v___x_197_; uint8_t v___x_198_; 
v___x_197_ = lean_array_get_size(v_source_195_);
v___x_198_ = lean_nat_dec_lt(v_i_194_, v___x_197_);
if (v___x_198_ == 0)
{
lean_dec_ref(v_source_195_);
lean_dec(v_i_194_);
return v_target_196_;
}
else
{
lean_object* v_es_199_; lean_object* v___x_200_; lean_object* v_source_201_; lean_object* v_target_202_; lean_object* v___x_203_; lean_object* v___x_204_; 
v_es_199_ = lean_array_fget(v_source_195_, v_i_194_);
v___x_200_ = lean_box(0);
v_source_201_ = lean_array_fset(v_source_195_, v_i_194_, v___x_200_);
v_target_202_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Lean_MessageData_dedupByString_spec__1_spec__2_spec__3_spec__5___redArg(v_target_196_, v_es_199_);
v___x_203_ = lean_unsigned_to_nat(1u);
v___x_204_ = lean_nat_add(v_i_194_, v___x_203_);
lean_dec(v_i_194_);
v_i_194_ = v___x_204_;
v_source_195_ = v_source_201_;
v_target_196_ = v_target_202_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Lean_MessageData_dedupByString_spec__1_spec__2___redArg(lean_object* v_data_206_){
_start:
{
lean_object* v___x_207_; lean_object* v___x_208_; lean_object* v_nbuckets_209_; lean_object* v___x_210_; lean_object* v___x_211_; lean_object* v___x_212_; lean_object* v___x_213_; 
v___x_207_ = lean_array_get_size(v_data_206_);
v___x_208_ = lean_unsigned_to_nat(2u);
v_nbuckets_209_ = lean_nat_mul(v___x_207_, v___x_208_);
v___x_210_ = lean_unsigned_to_nat(0u);
v___x_211_ = lean_box(0);
v___x_212_ = lean_mk_array(v_nbuckets_209_, v___x_211_);
v___x_213_ = lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Lean_MessageData_dedupByString_spec__1_spec__2_spec__3___redArg(v___x_210_, v_data_206_, v___x_212_);
return v___x_213_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Lean_MessageData_dedupByString_spec__1___redArg(lean_object* v_m_214_, lean_object* v_a_215_, lean_object* v_b_216_){
_start:
{
lean_object* v_size_217_; lean_object* v_buckets_218_; lean_object* v___x_219_; uint64_t v___x_220_; uint64_t v___x_221_; uint64_t v___x_222_; uint64_t v_fold_223_; uint64_t v___x_224_; uint64_t v___x_225_; uint64_t v___x_226_; size_t v___x_227_; size_t v___x_228_; size_t v___x_229_; size_t v___x_230_; size_t v___x_231_; lean_object* v_bkt_232_; uint8_t v___x_233_; 
v_size_217_ = lean_ctor_get(v_m_214_, 0);
v_buckets_218_ = lean_ctor_get(v_m_214_, 1);
v___x_219_ = lean_array_get_size(v_buckets_218_);
v___x_220_ = lean_string_hash(v_a_215_);
v___x_221_ = 32ULL;
v___x_222_ = lean_uint64_shift_right(v___x_220_, v___x_221_);
v_fold_223_ = lean_uint64_xor(v___x_220_, v___x_222_);
v___x_224_ = 16ULL;
v___x_225_ = lean_uint64_shift_right(v_fold_223_, v___x_224_);
v___x_226_ = lean_uint64_xor(v_fold_223_, v___x_225_);
v___x_227_ = lean_uint64_to_usize(v___x_226_);
v___x_228_ = lean_usize_of_nat(v___x_219_);
v___x_229_ = ((size_t)1ULL);
v___x_230_ = lean_usize_sub(v___x_228_, v___x_229_);
v___x_231_ = lean_usize_land(v___x_227_, v___x_230_);
v_bkt_232_ = lean_array_uget_borrowed(v_buckets_218_, v___x_231_);
v___x_233_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Lean_MessageData_dedupByString_spec__0_spec__0___redArg(v_a_215_, v_bkt_232_);
if (v___x_233_ == 0)
{
lean_object* v___x_235_; uint8_t v_isShared_236_; uint8_t v_isSharedCheck_254_; 
lean_inc_ref(v_buckets_218_);
lean_inc(v_size_217_);
v_isSharedCheck_254_ = !lean_is_exclusive(v_m_214_);
if (v_isSharedCheck_254_ == 0)
{
lean_object* v_unused_255_; lean_object* v_unused_256_; 
v_unused_255_ = lean_ctor_get(v_m_214_, 1);
lean_dec(v_unused_255_);
v_unused_256_ = lean_ctor_get(v_m_214_, 0);
lean_dec(v_unused_256_);
v___x_235_ = v_m_214_;
v_isShared_236_ = v_isSharedCheck_254_;
goto v_resetjp_234_;
}
else
{
lean_dec(v_m_214_);
v___x_235_ = lean_box(0);
v_isShared_236_ = v_isSharedCheck_254_;
goto v_resetjp_234_;
}
v_resetjp_234_:
{
lean_object* v___x_237_; lean_object* v_size_x27_238_; lean_object* v___x_239_; lean_object* v_buckets_x27_240_; lean_object* v___x_241_; lean_object* v___x_242_; lean_object* v___x_243_; lean_object* v___x_244_; lean_object* v___x_245_; uint8_t v___x_246_; 
v___x_237_ = lean_unsigned_to_nat(1u);
v_size_x27_238_ = lean_nat_add(v_size_217_, v___x_237_);
lean_dec(v_size_217_);
lean_inc(v_bkt_232_);
v___x_239_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_239_, 0, v_a_215_);
lean_ctor_set(v___x_239_, 1, v_b_216_);
lean_ctor_set(v___x_239_, 2, v_bkt_232_);
v_buckets_x27_240_ = lean_array_uset(v_buckets_218_, v___x_231_, v___x_239_);
v___x_241_ = lean_unsigned_to_nat(4u);
v___x_242_ = lean_nat_mul(v_size_x27_238_, v___x_241_);
v___x_243_ = lean_unsigned_to_nat(3u);
v___x_244_ = lean_nat_div(v___x_242_, v___x_243_);
lean_dec(v___x_242_);
v___x_245_ = lean_array_get_size(v_buckets_x27_240_);
v___x_246_ = lean_nat_dec_le(v___x_244_, v___x_245_);
lean_dec(v___x_244_);
if (v___x_246_ == 0)
{
lean_object* v_val_247_; lean_object* v___x_249_; 
v_val_247_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Lean_MessageData_dedupByString_spec__1_spec__2___redArg(v_buckets_x27_240_);
if (v_isShared_236_ == 0)
{
lean_ctor_set(v___x_235_, 1, v_val_247_);
lean_ctor_set(v___x_235_, 0, v_size_x27_238_);
v___x_249_ = v___x_235_;
goto v_reusejp_248_;
}
else
{
lean_object* v_reuseFailAlloc_250_; 
v_reuseFailAlloc_250_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_250_, 0, v_size_x27_238_);
lean_ctor_set(v_reuseFailAlloc_250_, 1, v_val_247_);
v___x_249_ = v_reuseFailAlloc_250_;
goto v_reusejp_248_;
}
v_reusejp_248_:
{
return v___x_249_;
}
}
else
{
lean_object* v___x_252_; 
if (v_isShared_236_ == 0)
{
lean_ctor_set(v___x_235_, 1, v_buckets_x27_240_);
lean_ctor_set(v___x_235_, 0, v_size_x27_238_);
v___x_252_ = v___x_235_;
goto v_reusejp_251_;
}
else
{
lean_object* v_reuseFailAlloc_253_; 
v_reuseFailAlloc_253_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_253_, 0, v_size_x27_238_);
lean_ctor_set(v_reuseFailAlloc_253_, 1, v_buckets_x27_240_);
v___x_252_ = v_reuseFailAlloc_253_;
goto v_reusejp_251_;
}
v_reusejp_251_:
{
return v___x_252_;
}
}
}
}
else
{
lean_dec(v_b_216_);
lean_dec_ref(v_a_215_);
return v_m_214_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_MessageData_dedupByString_spec__2(lean_object* v_as_257_, size_t v_sz_258_, size_t v_i_259_, lean_object* v_b_260_){
_start:
{
lean_object* v_a_263_; uint8_t v___x_267_; 
v___x_267_ = lean_usize_dec_lt(v_i_259_, v_sz_258_);
if (v___x_267_ == 0)
{
return v_b_260_;
}
else
{
lean_object* v_a_268_; lean_object* v___x_269_; lean_object* v_fst_270_; lean_object* v_snd_271_; lean_object* v___x_273_; uint8_t v_isShared_274_; uint8_t v_isSharedCheck_285_; 
v_a_268_ = lean_array_uget_borrowed(v_as_257_, v_i_259_);
lean_inc(v_a_268_);
v___x_269_ = l_Lean_MessageData_toString(v_a_268_);
v_fst_270_ = lean_ctor_get(v_b_260_, 0);
v_snd_271_ = lean_ctor_get(v_b_260_, 1);
v_isSharedCheck_285_ = !lean_is_exclusive(v_b_260_);
if (v_isSharedCheck_285_ == 0)
{
v___x_273_ = v_b_260_;
v_isShared_274_ = v_isSharedCheck_285_;
goto v_resetjp_272_;
}
else
{
lean_inc(v_snd_271_);
lean_inc(v_fst_270_);
lean_dec(v_b_260_);
v___x_273_ = lean_box(0);
v_isShared_274_ = v_isSharedCheck_285_;
goto v_resetjp_272_;
}
v_resetjp_272_:
{
uint8_t v___x_275_; 
v___x_275_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Lean_MessageData_dedupByString_spec__0___redArg(v_fst_270_, v___x_269_);
if (v___x_275_ == 0)
{
lean_object* v___x_276_; lean_object* v___x_277_; lean_object* v___x_278_; lean_object* v___x_280_; 
v___x_276_ = lean_box(0);
v___x_277_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Lean_MessageData_dedupByString_spec__1___redArg(v_fst_270_, v___x_269_, v___x_276_);
lean_inc(v_a_268_);
v___x_278_ = lean_array_push(v_snd_271_, v_a_268_);
if (v_isShared_274_ == 0)
{
lean_ctor_set(v___x_273_, 1, v___x_278_);
lean_ctor_set(v___x_273_, 0, v___x_277_);
v___x_280_ = v___x_273_;
goto v_reusejp_279_;
}
else
{
lean_object* v_reuseFailAlloc_281_; 
v_reuseFailAlloc_281_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_281_, 0, v___x_277_);
lean_ctor_set(v_reuseFailAlloc_281_, 1, v___x_278_);
v___x_280_ = v_reuseFailAlloc_281_;
goto v_reusejp_279_;
}
v_reusejp_279_:
{
v_a_263_ = v___x_280_;
goto v___jp_262_;
}
}
else
{
lean_object* v___x_283_; 
lean_dec_ref(v___x_269_);
if (v_isShared_274_ == 0)
{
v___x_283_ = v___x_273_;
goto v_reusejp_282_;
}
else
{
lean_object* v_reuseFailAlloc_284_; 
v_reuseFailAlloc_284_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_284_, 0, v_fst_270_);
lean_ctor_set(v_reuseFailAlloc_284_, 1, v_snd_271_);
v___x_283_ = v_reuseFailAlloc_284_;
goto v_reusejp_282_;
}
v_reusejp_282_:
{
v_a_263_ = v___x_283_;
goto v___jp_262_;
}
}
}
}
v___jp_262_:
{
size_t v___x_264_; size_t v___x_265_; 
v___x_264_ = ((size_t)1ULL);
v___x_265_ = lean_usize_add(v_i_259_, v___x_264_);
v_i_259_ = v___x_265_;
v_b_260_ = v_a_263_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_MessageData_dedupByString_spec__2___boxed(lean_object* v_as_286_, lean_object* v_sz_287_, lean_object* v_i_288_, lean_object* v_b_289_, lean_object* v___y_290_){
_start:
{
size_t v_sz_boxed_291_; size_t v_i_boxed_292_; lean_object* v_res_293_; 
v_sz_boxed_291_ = lean_unbox_usize(v_sz_287_);
lean_dec(v_sz_287_);
v_i_boxed_292_ = lean_unbox_usize(v_i_288_);
lean_dec(v_i_288_);
v_res_293_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_MessageData_dedupByString_spec__2(v_as_286_, v_sz_boxed_291_, v_i_boxed_292_, v_b_289_);
lean_dec_ref(v_as_286_);
return v_res_293_;
}
}
static lean_object* _init_lp_mathlib_Lean_MessageData_dedupByString___closed__0(void){
_start:
{
lean_object* v___x_294_; lean_object* v___x_295_; lean_object* v___x_296_; 
v___x_294_ = lean_box(0);
v___x_295_ = lean_unsigned_to_nat(16u);
v___x_296_ = lean_mk_array(v___x_295_, v___x_294_);
return v___x_296_;
}
}
static lean_object* _init_lp_mathlib_Lean_MessageData_dedupByString___closed__1(void){
_start:
{
lean_object* v___x_297_; lean_object* v___x_298_; lean_object* v_seen_299_; 
v___x_297_ = lean_obj_once(&lp_mathlib_Lean_MessageData_dedupByString___closed__0, &lp_mathlib_Lean_MessageData_dedupByString___closed__0_once, _init_lp_mathlib_Lean_MessageData_dedupByString___closed__0);
v___x_298_ = lean_unsigned_to_nat(0u);
v_seen_299_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_seen_299_, 0, v___x_298_);
lean_ctor_set(v_seen_299_, 1, v___x_297_);
return v_seen_299_;
}
}
static lean_object* _init_lp_mathlib_Lean_MessageData_dedupByString___closed__3(void){
_start:
{
lean_object* v_unique_302_; lean_object* v_seen_303_; lean_object* v___x_304_; 
v_unique_302_ = ((lean_object*)(lp_mathlib_Lean_MessageData_dedupByString___closed__2));
v_seen_303_ = lean_obj_once(&lp_mathlib_Lean_MessageData_dedupByString___closed__1, &lp_mathlib_Lean_MessageData_dedupByString___closed__1_once, _init_lp_mathlib_Lean_MessageData_dedupByString___closed__1);
v___x_304_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_304_, 0, v_seen_303_);
lean_ctor_set(v___x_304_, 1, v_unique_302_);
return v___x_304_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MessageData_dedupByString(lean_object* v_msgs_305_){
_start:
{
lean_object* v___x_307_; size_t v_sz_308_; size_t v___x_309_; lean_object* v___x_310_; lean_object* v_snd_311_; 
v___x_307_ = lean_obj_once(&lp_mathlib_Lean_MessageData_dedupByString___closed__3, &lp_mathlib_Lean_MessageData_dedupByString___closed__3_once, _init_lp_mathlib_Lean_MessageData_dedupByString___closed__3);
v_sz_308_ = lean_array_size(v_msgs_305_);
v___x_309_ = ((size_t)0ULL);
v___x_310_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_MessageData_dedupByString_spec__2(v_msgs_305_, v_sz_308_, v___x_309_, v___x_307_);
v_snd_311_ = lean_ctor_get(v___x_310_, 1);
lean_inc(v_snd_311_);
lean_dec_ref(v___x_310_);
return v_snd_311_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MessageData_dedupByString___boxed(lean_object* v_msgs_312_, lean_object* v_a_313_){
_start:
{
lean_object* v_res_314_; 
v_res_314_ = lp_mathlib_Lean_MessageData_dedupByString(v_msgs_312_);
lean_dec_ref(v_msgs_312_);
return v_res_314_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Lean_MessageData_dedupByString_spec__0(lean_object* v_00_u03b2_315_, lean_object* v_m_316_, lean_object* v_a_317_){
_start:
{
uint8_t v___x_318_; 
v___x_318_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Lean_MessageData_dedupByString_spec__0___redArg(v_m_316_, v_a_317_);
return v___x_318_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Lean_MessageData_dedupByString_spec__0___boxed(lean_object* v_00_u03b2_319_, lean_object* v_m_320_, lean_object* v_a_321_){
_start:
{
uint8_t v_res_322_; lean_object* v_r_323_; 
v_res_322_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_contains___at___00Lean_MessageData_dedupByString_spec__0(v_00_u03b2_319_, v_m_320_, v_a_321_);
lean_dec_ref(v_a_321_);
lean_dec_ref(v_m_320_);
v_r_323_ = lean_box(v_res_322_);
return v_r_323_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Lean_MessageData_dedupByString_spec__1(lean_object* v_00_u03b2_324_, lean_object* v_m_325_, lean_object* v_a_326_, lean_object* v_b_327_){
_start:
{
lean_object* v___x_328_; 
v___x_328_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Lean_MessageData_dedupByString_spec__1___redArg(v_m_325_, v_a_326_, v_b_327_);
return v___x_328_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Lean_MessageData_dedupByString_spec__0_spec__0(lean_object* v_00_u03b2_329_, lean_object* v_a_330_, lean_object* v_x_331_){
_start:
{
uint8_t v___x_332_; 
v___x_332_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Lean_MessageData_dedupByString_spec__0_spec__0___redArg(v_a_330_, v_x_331_);
return v___x_332_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Lean_MessageData_dedupByString_spec__0_spec__0___boxed(lean_object* v_00_u03b2_333_, lean_object* v_a_334_, lean_object* v_x_335_){
_start:
{
uint8_t v_res_336_; lean_object* v_r_337_; 
v_res_336_ = lp_mathlib_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_contains___at___00Lean_MessageData_dedupByString_spec__0_spec__0(v_00_u03b2_333_, v_a_334_, v_x_335_);
lean_dec(v_x_335_);
lean_dec_ref(v_a_334_);
v_r_337_ = lean_box(v_res_336_);
return v_r_337_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Lean_MessageData_dedupByString_spec__1_spec__2(lean_object* v_00_u03b2_338_, lean_object* v_data_339_){
_start:
{
lean_object* v___x_340_; 
v___x_340_ = lp_mathlib_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Lean_MessageData_dedupByString_spec__1_spec__2___redArg(v_data_339_);
return v___x_340_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Lean_MessageData_dedupByString_spec__1_spec__2_spec__3(lean_object* v_00_u03b2_341_, lean_object* v_i_342_, lean_object* v_source_343_, lean_object* v_target_344_){
_start:
{
lean_object* v___x_345_; 
v___x_345_ = lp_mathlib___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Lean_MessageData_dedupByString_spec__1_spec__2_spec__3___redArg(v_i_342_, v_source_343_, v_target_344_);
return v___x_345_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Lean_MessageData_dedupByString_spec__1_spec__2_spec__3_spec__5(lean_object* v_00_u03b2_346_, lean_object* v_x_347_, lean_object* v_x_348_){
_start:
{
lean_object* v___x_349_; 
v___x_349_ = lp_mathlib_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insertIfNew___at___00Lean_MessageData_dedupByString_spec__1_spec__2_spec__3_spec__5___redArg(v_x_347_, v_x_348_);
return v___x_349_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Message(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Lean_MessageData_Trace(uint8_t builtin) {
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
res = runtime_initialize_Lean_Message(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Lean_MessageData_Trace(uint8_t builtin) {
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
lean_object* initialize_Lean_Message(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Lean_MessageData_Trace(uint8_t builtin) {
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
res = initialize_Lean_Message(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Lean_MessageData_Trace(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Lean_MessageData_Trace(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Lean_MessageData_Trace(builtin);
}
#ifdef __cplusplus
}
#endif
