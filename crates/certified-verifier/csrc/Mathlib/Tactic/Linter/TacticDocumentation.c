// Lean compiler output
// Module: Mathlib.Tactic.Linter.TacticDocumentation
// Imports: public import Init public meta import Init public meta import Lean.Elab.Tactic.Doc public meta import Lean.Parser.Tactic.Doc public import Mathlib.Tactic.Linter.Header public import Batteries.Tactic.Lint.Basic public import Lean.Elab.Tactic.Doc
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
lean_object* l_Lean_stringToMessageData(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
uint8_t lean_name_eq(lean_object*, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
uint8_t l_Lean_Parser_Tactic_Doc_isTactic(lean_object*, lean_object*);
lean_object* l_Lean_Parser_Tactic_Doc_alternativeOfTactic(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_Doc_allTacticDocs(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
lean_object* l_Lean_MessageData_ofList(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_push(lean_object*, lean_object*);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
lean_object* lean_string_utf8_byte_size(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_Std_DTreeMap_Internal_Impl_insert___at___00Lean_NameMap_insert_spec__0___redArg(lean_object*, lean_object*, lean_object*);
uint8_t l___private_Lean_Data_Name_0__Lean_Name_quickCmpImpl(lean_object*, lean_object*);
lean_object* l_Lean_Name_toString(lean_object*, uint8_t);
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_TacticDocumentation_0__isNonemptyDoc_spec__0(lean_object*, size_t, size_t);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_TacticDocumentation_0__isNonemptyDoc_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Tactic_Linter_TacticDocumentation_0__isNonemptyDoc(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_TacticDocumentation_0__isNonemptyDoc___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00tacticDocs_spec__1(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00tacticDocs_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00tacticDocs_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00tacticDocs_spec__0___redArg___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_tacticDocs___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "tactic `"};
static const lean_object* lp_mathlib_tacticDocs___lam__0___closed__0 = (const lean_object*)&lp_mathlib_tacticDocs___lam__0___closed__0_value;
static lean_once_cell_t lp_mathlib_tacticDocs___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_tacticDocs___lam__0___closed__1;
static const lean_string_object lp_mathlib_tacticDocs___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 31, .m_capacity = 31, .m_length = 30, .m_data = "` missing documentation string"};
static const lean_object* lp_mathlib_tacticDocs___lam__0___closed__2 = (const lean_object*)&lp_mathlib_tacticDocs___lam__0___closed__2_value;
static lean_once_cell_t lp_mathlib_tacticDocs___lam__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_tacticDocs___lam__0___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_tacticDocs___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_tacticDocs___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_tacticDocs___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_tacticDocs___lam__0___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_tacticDocs___closed__0 = (const lean_object*)&lp_mathlib_tacticDocs___closed__0_value;
static const lean_string_object lp_mathlib_tacticDocs___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 38, .m_capacity = 38, .m_length = 37, .m_data = "No tactics are missing documentation."};
static const lean_object* lp_mathlib_tacticDocs___closed__1 = (const lean_object*)&lp_mathlib_tacticDocs___closed__1_value;
static const lean_ctor_object lp_mathlib_tacticDocs___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_tacticDocs___closed__1_value)}};
static const lean_object* lp_mathlib_tacticDocs___closed__2 = (const lean_object*)&lp_mathlib_tacticDocs___closed__2_value;
static lean_once_cell_t lp_mathlib_tacticDocs___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_tacticDocs___closed__3;
static const lean_string_object lp_mathlib_tacticDocs___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 43, .m_capacity = 43, .m_length = 42, .m_data = "TACTICS ARE MISSING DOCUMENTATION STRINGS:"};
static const lean_object* lp_mathlib_tacticDocs___closed__4 = (const lean_object*)&lp_mathlib_tacticDocs___closed__4_value;
static const lean_ctor_object lp_mathlib_tacticDocs___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_tacticDocs___closed__4_value)}};
static const lean_object* lp_mathlib_tacticDocs___closed__5 = (const lean_object*)&lp_mathlib_tacticDocs___closed__5_value;
static lean_once_cell_t lp_mathlib_tacticDocs___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_tacticDocs___closed__6;
static lean_once_cell_t lp_mathlib_tacticDocs___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_tacticDocs___closed__7;
LEAN_EXPORT lean_object* lp_mathlib_tacticDocs;
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00tacticDocs_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00tacticDocs_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00tacticAlt_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00tacticAlt_spec__3(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00tacticAlt_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00tacticAlt_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00tacticAlt_spec__0___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00tacticAlt_spec__0___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00tacticAlt_spec__0(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00tacticAlt_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00tacticAlt_spec__1(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00tacticAlt_spec__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_tacticAlt___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = "` has multiple declarations "};
static const lean_object* lp_mathlib_tacticAlt___lam__0___closed__0 = (const lean_object*)&lp_mathlib_tacticAlt___lam__0___closed__0_value;
static lean_once_cell_t lp_mathlib_tacticAlt___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_tacticAlt___lam__0___closed__1;
static const lean_string_object lp_mathlib_tacticAlt___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 145, .m_capacity = 145, .m_length = 144, .m_data = " one of which should be marked as `@[tactic_alt]` of the other(s).\nHint: you can use the `tactic_extension` command to extend the documentation."};
static const lean_object* lp_mathlib_tacticAlt___lam__0___closed__2 = (const lean_object*)&lp_mathlib_tacticAlt___lam__0___closed__2_value;
static lean_once_cell_t lp_mathlib_tacticAlt___lam__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_tacticAlt___lam__0___closed__3;
static const lean_array_object lp_mathlib_tacticAlt___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_tacticAlt___lam__0___closed__4 = (const lean_object*)&lp_mathlib_tacticAlt___lam__0___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib_tacticAlt___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_tacticAlt___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_tacticAlt___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_tacticAlt___lam__0___boxed, .m_arity = 6, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_tacticAlt___closed__0 = (const lean_object*)&lp_mathlib_tacticAlt___closed__0_value;
static const lean_string_object lp_mathlib_tacticAlt___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 46, .m_capacity = 46, .m_length = 45, .m_data = "No tactics sharing the same user-facing name."};
static const lean_object* lp_mathlib_tacticAlt___closed__1 = (const lean_object*)&lp_mathlib_tacticAlt___closed__1_value;
static const lean_ctor_object lp_mathlib_tacticAlt___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_tacticAlt___closed__1_value)}};
static const lean_object* lp_mathlib_tacticAlt___closed__2 = (const lean_object*)&lp_mathlib_tacticAlt___closed__2_value;
static lean_once_cell_t lp_mathlib_tacticAlt___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_tacticAlt___closed__3;
static const lean_string_object lp_mathlib_tacticAlt___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 48, .m_capacity = 48, .m_length = 47, .m_data = "TACTICS ARE MISSING `@[tactic_alt]` ATTRIBUTES:"};
static const lean_object* lp_mathlib_tacticAlt___closed__4 = (const lean_object*)&lp_mathlib_tacticAlt___closed__4_value;
static const lean_ctor_object lp_mathlib_tacticAlt___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_tacticAlt___closed__4_value)}};
static const lean_object* lp_mathlib_tacticAlt___closed__5 = (const lean_object*)&lp_mathlib_tacticAlt___closed__5_value;
static lean_once_cell_t lp_mathlib_tacticAlt___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_tacticAlt___closed__6;
static lean_once_cell_t lp_mathlib_tacticAlt___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_tacticAlt___closed__7;
LEAN_EXPORT lean_object* lp_mathlib_tacticAlt;
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_TacticDocumentation_0__isNonemptyDoc_spec__0(lean_object* v_as_1_, size_t v_i_2_, size_t v_stop_3_){
_start:
{
uint8_t v___x_4_; 
v___x_4_ = lean_usize_dec_eq(v_i_2_, v_stop_3_);
if (v___x_4_ == 0)
{
uint8_t v___x_5_; lean_object* v___x_6_; lean_object* v___x_7_; lean_object* v___x_8_; uint8_t v___x_9_; 
v___x_5_ = 1;
v___x_6_ = lean_array_uget_borrowed(v_as_1_, v_i_2_);
v___x_7_ = lean_string_utf8_byte_size(v___x_6_);
v___x_8_ = lean_unsigned_to_nat(0u);
v___x_9_ = lean_nat_dec_eq(v___x_7_, v___x_8_);
if (v___x_9_ == 0)
{
return v___x_5_;
}
else
{
if (v___x_4_ == 0)
{
size_t v___x_10_; size_t v___x_11_; 
v___x_10_ = ((size_t)1ULL);
v___x_11_ = lean_usize_add(v_i_2_, v___x_10_);
v_i_2_ = v___x_11_;
goto _start;
}
else
{
return v___x_5_;
}
}
}
else
{
uint8_t v___x_13_; 
v___x_13_ = 0;
return v___x_13_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_TacticDocumentation_0__isNonemptyDoc_spec__0___boxed(lean_object* v_as_14_, lean_object* v_i_15_, lean_object* v_stop_16_){
_start:
{
size_t v_i_boxed_17_; size_t v_stop_boxed_18_; uint8_t v_res_19_; lean_object* v_r_20_; 
v_i_boxed_17_ = lean_unbox_usize(v_i_15_);
lean_dec(v_i_15_);
v_stop_boxed_18_ = lean_unbox_usize(v_stop_16_);
lean_dec(v_stop_16_);
v_res_19_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_TacticDocumentation_0__isNonemptyDoc_spec__0(v_as_14_, v_i_boxed_17_, v_stop_boxed_18_);
lean_dec_ref(v_as_14_);
v_r_20_ = lean_box(v_res_19_);
return v_r_20_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Tactic_Linter_TacticDocumentation_0__isNonemptyDoc(lean_object* v_doc_21_){
_start:
{
lean_object* v_docString_22_; 
v_docString_22_ = lean_ctor_get(v_doc_21_, 3);
if (lean_obj_tag(v_docString_22_) == 0)
{
lean_object* v_extensionDocs_23_; lean_object* v___x_24_; lean_object* v___x_25_; uint8_t v___x_26_; 
v_extensionDocs_23_ = lean_ctor_get(v_doc_21_, 4);
v___x_24_ = lean_unsigned_to_nat(0u);
v___x_25_ = lean_array_get_size(v_extensionDocs_23_);
v___x_26_ = lean_nat_dec_lt(v___x_24_, v___x_25_);
if (v___x_26_ == 0)
{
return v___x_26_;
}
else
{
if (v___x_26_ == 0)
{
return v___x_26_;
}
else
{
size_t v___x_27_; size_t v___x_28_; uint8_t v___x_29_; 
v___x_27_ = ((size_t)0ULL);
v___x_28_ = lean_usize_of_nat(v___x_25_);
v___x_29_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00__private_Mathlib_Tactic_Linter_TacticDocumentation_0__isNonemptyDoc_spec__0(v_extensionDocs_23_, v___x_27_, v___x_28_);
return v___x_29_;
}
}
}
else
{
uint8_t v___x_30_; 
v___x_30_ = 1;
return v___x_30_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Linter_TacticDocumentation_0__isNonemptyDoc___boxed(lean_object* v_doc_31_){
_start:
{
uint8_t v_res_32_; lean_object* v_r_33_; 
v_res_32_ = lp_mathlib___private_Mathlib_Tactic_Linter_TacticDocumentation_0__isNonemptyDoc(v_doc_31_);
lean_dec_ref(v_doc_31_);
v_r_33_ = lean_box(v_res_32_);
return v_r_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00tacticDocs_spec__1(lean_object* v_as_34_, size_t v_i_35_, size_t v_stop_36_, lean_object* v_b_37_){
_start:
{
uint8_t v___x_38_; 
v___x_38_ = lean_usize_dec_eq(v_i_35_, v_stop_36_);
if (v___x_38_ == 0)
{
lean_object* v___x_39_; lean_object* v_internalName_40_; lean_object* v___x_41_; size_t v___x_42_; size_t v___x_43_; 
v___x_39_ = lean_array_uget_borrowed(v_as_34_, v_i_35_);
v_internalName_40_ = lean_ctor_get(v___x_39_, 0);
lean_inc(v___x_39_);
lean_inc(v_internalName_40_);
v___x_41_ = l_Std_DTreeMap_Internal_Impl_insert___at___00Lean_NameMap_insert_spec__0___redArg(v_internalName_40_, v___x_39_, v_b_37_);
v___x_42_ = ((size_t)1ULL);
v___x_43_ = lean_usize_add(v_i_35_, v___x_42_);
v_i_35_ = v___x_43_;
v_b_37_ = v___x_41_;
goto _start;
}
else
{
return v_b_37_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00tacticDocs_spec__1___boxed(lean_object* v_as_45_, lean_object* v_i_46_, lean_object* v_stop_47_, lean_object* v_b_48_){
_start:
{
size_t v_i_boxed_49_; size_t v_stop_boxed_50_; lean_object* v_res_51_; 
v_i_boxed_49_ = lean_unbox_usize(v_i_46_);
lean_dec(v_i_46_);
v_stop_boxed_50_ = lean_unbox_usize(v_stop_47_);
lean_dec(v_stop_47_);
v_res_51_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00tacticDocs_spec__1(v_as_45_, v_i_boxed_49_, v_stop_boxed_50_, v_b_48_);
lean_dec_ref(v_as_45_);
return v_res_51_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00tacticDocs_spec__0___redArg(lean_object* v_t_52_, lean_object* v_k_53_){
_start:
{
if (lean_obj_tag(v_t_52_) == 0)
{
lean_object* v_k_54_; lean_object* v_v_55_; lean_object* v_l_56_; lean_object* v_r_57_; uint8_t v___x_58_; 
v_k_54_ = lean_ctor_get(v_t_52_, 1);
v_v_55_ = lean_ctor_get(v_t_52_, 2);
v_l_56_ = lean_ctor_get(v_t_52_, 3);
v_r_57_ = lean_ctor_get(v_t_52_, 4);
v___x_58_ = l___private_Lean_Data_Name_0__Lean_Name_quickCmpImpl(v_k_53_, v_k_54_);
switch(v___x_58_)
{
case 0:
{
v_t_52_ = v_l_56_;
goto _start;
}
case 1:
{
lean_object* v___x_60_; 
lean_inc(v_v_55_);
v___x_60_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_60_, 0, v_v_55_);
return v___x_60_;
}
default: 
{
v_t_52_ = v_r_57_;
goto _start;
}
}
}
else
{
lean_object* v___x_62_; 
v___x_62_ = lean_box(0);
return v___x_62_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00tacticDocs_spec__0___redArg___boxed(lean_object* v_t_63_, lean_object* v_k_64_){
_start:
{
lean_object* v_res_65_; 
v_res_65_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00tacticDocs_spec__0___redArg(v_t_63_, v_k_64_);
lean_dec(v_k_64_);
lean_dec(v_t_63_);
return v_res_65_;
}
}
static lean_object* _init_lp_mathlib_tacticDocs___lam__0___closed__1(void){
_start:
{
lean_object* v___x_67_; lean_object* v___x_68_; 
v___x_67_ = ((lean_object*)(lp_mathlib_tacticDocs___lam__0___closed__0));
v___x_68_ = l_Lean_stringToMessageData(v___x_67_);
return v___x_68_;
}
}
static lean_object* _init_lp_mathlib_tacticDocs___lam__0___closed__3(void){
_start:
{
lean_object* v___x_70_; lean_object* v___x_71_; 
v___x_70_ = ((lean_object*)(lp_mathlib_tacticDocs___lam__0___closed__2));
v___x_71_ = l_Lean_stringToMessageData(v___x_70_);
return v___x_71_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_tacticDocs___lam__0(lean_object* v_tac_72_, lean_object* v___y_73_, lean_object* v___y_74_, lean_object* v___y_75_, lean_object* v___y_76_){
_start:
{
lean_object* v___x_78_; lean_object* v_env_82_; uint8_t v___x_83_; 
v___x_78_ = lean_st_ref_get(v___y_76_);
v_env_82_ = lean_ctor_get(v___x_78_, 0);
lean_inc_ref_n(v_env_82_, 2);
lean_dec(v___x_78_);
v___x_83_ = l_Lean_Parser_Tactic_Doc_isTactic(v_env_82_, v_tac_72_);
if (v___x_83_ == 0)
{
lean_dec_ref(v_env_82_);
lean_dec(v_tac_72_);
goto v___jp_79_;
}
else
{
lean_object* v___x_84_; 
lean_inc(v_tac_72_);
v___x_84_ = l_Lean_Parser_Tactic_Doc_alternativeOfTactic(v_env_82_, v_tac_72_);
if (lean_obj_tag(v___x_84_) == 0)
{
lean_object* v___x_85_; 
v___x_85_ = l_Lean_Elab_Tactic_Doc_allTacticDocs(v___x_83_, v___y_73_, v___y_74_, v___y_75_, v___y_76_);
if (lean_obj_tag(v___x_85_) == 0)
{
lean_object* v_a_86_; lean_object* v___x_88_; uint8_t v_isShared_89_; uint8_t v_isSharedCheck_131_; 
v_a_86_ = lean_ctor_get(v___x_85_, 0);
v_isSharedCheck_131_ = !lean_is_exclusive(v___x_85_);
if (v_isSharedCheck_131_ == 0)
{
v___x_88_ = v___x_85_;
v_isShared_89_ = v_isSharedCheck_131_;
goto v_resetjp_87_;
}
else
{
lean_inc(v_a_86_);
lean_dec(v___x_85_);
v___x_88_ = lean_box(0);
v_isShared_89_ = v_isSharedCheck_131_;
goto v_resetjp_87_;
}
v_resetjp_87_:
{
lean_object* v___y_91_; lean_object* v___y_102_; lean_object* v___y_103_; lean_object* v___y_115_; lean_object* v___x_120_; lean_object* v___x_121_; lean_object* v___x_122_; uint8_t v___x_123_; 
v___x_120_ = lean_box(1);
v___x_121_ = lean_unsigned_to_nat(0u);
v___x_122_ = lean_array_get_size(v_a_86_);
v___x_123_ = lean_nat_dec_lt(v___x_121_, v___x_122_);
if (v___x_123_ == 0)
{
lean_dec(v_a_86_);
v___y_115_ = v___x_120_;
goto v___jp_114_;
}
else
{
uint8_t v___x_124_; 
v___x_124_ = lean_nat_dec_le(v___x_122_, v___x_122_);
if (v___x_124_ == 0)
{
if (v___x_123_ == 0)
{
lean_dec(v_a_86_);
v___y_115_ = v___x_120_;
goto v___jp_114_;
}
else
{
size_t v___x_125_; size_t v___x_126_; lean_object* v___x_127_; 
v___x_125_ = ((size_t)0ULL);
v___x_126_ = lean_usize_of_nat(v___x_122_);
v___x_127_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00tacticDocs_spec__1(v_a_86_, v___x_125_, v___x_126_, v___x_120_);
lean_dec(v_a_86_);
v___y_115_ = v___x_127_;
goto v___jp_114_;
}
}
else
{
size_t v___x_128_; size_t v___x_129_; lean_object* v___x_130_; 
v___x_128_ = ((size_t)0ULL);
v___x_129_ = lean_usize_of_nat(v___x_122_);
v___x_130_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00tacticDocs_spec__1(v_a_86_, v___x_128_, v___x_129_, v___x_120_);
lean_dec(v_a_86_);
v___y_115_ = v___x_130_;
goto v___jp_114_;
}
}
v___jp_90_:
{
lean_object* v___x_92_; lean_object* v___x_93_; lean_object* v___x_94_; lean_object* v___x_95_; lean_object* v___x_96_; lean_object* v___x_97_; lean_object* v___x_99_; 
v___x_92_ = lean_obj_once(&lp_mathlib_tacticDocs___lam__0___closed__1, &lp_mathlib_tacticDocs___lam__0___closed__1_once, _init_lp_mathlib_tacticDocs___lam__0___closed__1);
v___x_93_ = l_Lean_stringToMessageData(v___y_91_);
v___x_94_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_94_, 0, v___x_92_);
lean_ctor_set(v___x_94_, 1, v___x_93_);
v___x_95_ = lean_obj_once(&lp_mathlib_tacticDocs___lam__0___closed__3, &lp_mathlib_tacticDocs___lam__0___closed__3_once, _init_lp_mathlib_tacticDocs___lam__0___closed__3);
v___x_96_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_96_, 0, v___x_94_);
lean_ctor_set(v___x_96_, 1, v___x_95_);
v___x_97_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_97_, 0, v___x_96_);
if (v_isShared_89_ == 0)
{
lean_ctor_set(v___x_88_, 0, v___x_97_);
v___x_99_ = v___x_88_;
goto v_reusejp_98_;
}
else
{
lean_object* v_reuseFailAlloc_100_; 
v_reuseFailAlloc_100_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_100_, 0, v___x_97_);
v___x_99_ = v_reuseFailAlloc_100_;
goto v_reusejp_98_;
}
v_reusejp_98_:
{
return v___x_99_;
}
}
v___jp_101_:
{
if (lean_obj_tag(v___y_102_) == 1)
{
lean_object* v_val_104_; lean_object* v___x_106_; uint8_t v_isShared_107_; uint8_t v_isSharedCheck_113_; 
v_val_104_ = lean_ctor_get(v___y_102_, 0);
v_isSharedCheck_113_ = !lean_is_exclusive(v___y_102_);
if (v_isSharedCheck_113_ == 0)
{
v___x_106_ = v___y_102_;
v_isShared_107_ = v_isSharedCheck_113_;
goto v_resetjp_105_;
}
else
{
lean_inc(v_val_104_);
lean_dec(v___y_102_);
v___x_106_ = lean_box(0);
v_isShared_107_ = v_isSharedCheck_113_;
goto v_resetjp_105_;
}
v_resetjp_105_:
{
uint8_t v___x_108_; 
v___x_108_ = lp_mathlib___private_Mathlib_Tactic_Linter_TacticDocumentation_0__isNonemptyDoc(v_val_104_);
lean_dec(v_val_104_);
if (v___x_108_ == 0)
{
lean_del_object(v___x_106_);
v___y_91_ = v___y_103_;
goto v___jp_90_;
}
else
{
lean_object* v___x_109_; lean_object* v___x_111_; 
lean_dec_ref(v___y_103_);
lean_del_object(v___x_88_);
v___x_109_ = lean_box(0);
if (v_isShared_107_ == 0)
{
lean_ctor_set_tag(v___x_106_, 0);
lean_ctor_set(v___x_106_, 0, v___x_109_);
v___x_111_ = v___x_106_;
goto v_reusejp_110_;
}
else
{
lean_object* v_reuseFailAlloc_112_; 
v_reuseFailAlloc_112_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_112_, 0, v___x_109_);
v___x_111_ = v_reuseFailAlloc_112_;
goto v_reusejp_110_;
}
v_reusejp_110_:
{
return v___x_111_;
}
}
}
}
else
{
lean_dec(v___y_102_);
v___y_91_ = v___y_103_;
goto v___jp_90_;
}
}
v___jp_114_:
{
lean_object* v___x_116_; 
v___x_116_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00tacticDocs_spec__0___redArg(v___y_115_, v_tac_72_);
lean_dec(v___y_115_);
if (lean_obj_tag(v___x_116_) == 0)
{
lean_object* v___x_117_; 
v___x_117_ = l_Lean_Name_toString(v_tac_72_, v___x_83_);
v___y_102_ = v___x_116_;
v___y_103_ = v___x_117_;
goto v___jp_101_;
}
else
{
lean_object* v_val_118_; lean_object* v_userName_119_; 
lean_dec(v_tac_72_);
v_val_118_ = lean_ctor_get(v___x_116_, 0);
lean_inc(v_val_118_);
v_userName_119_ = lean_ctor_get(v_val_118_, 1);
lean_inc_ref(v_userName_119_);
lean_dec(v_val_118_);
v___y_102_ = v___x_116_;
v___y_103_ = v_userName_119_;
goto v___jp_101_;
}
}
}
}
else
{
lean_object* v_a_132_; lean_object* v___x_134_; uint8_t v_isShared_135_; uint8_t v_isSharedCheck_139_; 
lean_dec(v_tac_72_);
v_a_132_ = lean_ctor_get(v___x_85_, 0);
v_isSharedCheck_139_ = !lean_is_exclusive(v___x_85_);
if (v_isSharedCheck_139_ == 0)
{
v___x_134_ = v___x_85_;
v_isShared_135_ = v_isSharedCheck_139_;
goto v_resetjp_133_;
}
else
{
lean_inc(v_a_132_);
lean_dec(v___x_85_);
v___x_134_ = lean_box(0);
v_isShared_135_ = v_isSharedCheck_139_;
goto v_resetjp_133_;
}
v_resetjp_133_:
{
lean_object* v___x_137_; 
if (v_isShared_135_ == 0)
{
v___x_137_ = v___x_134_;
goto v_reusejp_136_;
}
else
{
lean_object* v_reuseFailAlloc_138_; 
v_reuseFailAlloc_138_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_138_, 0, v_a_132_);
v___x_137_ = v_reuseFailAlloc_138_;
goto v_reusejp_136_;
}
v_reusejp_136_:
{
return v___x_137_;
}
}
}
}
else
{
lean_dec_ref_known(v___x_84_, 1);
lean_dec(v_tac_72_);
goto v___jp_79_;
}
}
v___jp_79_:
{
lean_object* v___x_80_; lean_object* v___x_81_; 
v___x_80_ = lean_box(0);
v___x_81_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_81_, 0, v___x_80_);
return v___x_81_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_tacticDocs___lam__0___boxed(lean_object* v_tac_140_, lean_object* v___y_141_, lean_object* v___y_142_, lean_object* v___y_143_, lean_object* v___y_144_, lean_object* v___y_145_){
_start:
{
lean_object* v_res_146_; 
v_res_146_ = lp_mathlib_tacticDocs___lam__0(v_tac_140_, v___y_141_, v___y_142_, v___y_143_, v___y_144_);
lean_dec(v___y_144_);
lean_dec_ref(v___y_143_);
lean_dec(v___y_142_);
lean_dec_ref(v___y_141_);
return v_res_146_;
}
}
static lean_object* _init_lp_mathlib_tacticDocs___closed__3(void){
_start:
{
lean_object* v___x_151_; lean_object* v___x_152_; 
v___x_151_ = ((lean_object*)(lp_mathlib_tacticDocs___closed__2));
v___x_152_ = l_Lean_MessageData_ofFormat(v___x_151_);
return v___x_152_;
}
}
static lean_object* _init_lp_mathlib_tacticDocs___closed__6(void){
_start:
{
lean_object* v___x_156_; lean_object* v___x_157_; 
v___x_156_ = ((lean_object*)(lp_mathlib_tacticDocs___closed__5));
v___x_157_ = l_Lean_MessageData_ofFormat(v___x_156_);
return v___x_157_;
}
}
static lean_object* _init_lp_mathlib_tacticDocs___closed__7(void){
_start:
{
uint8_t v___x_158_; lean_object* v___x_159_; lean_object* v___x_160_; lean_object* v___f_161_; lean_object* v___x_162_; 
v___x_158_ = 1;
v___x_159_ = lean_obj_once(&lp_mathlib_tacticDocs___closed__6, &lp_mathlib_tacticDocs___closed__6_once, _init_lp_mathlib_tacticDocs___closed__6);
v___x_160_ = lean_obj_once(&lp_mathlib_tacticDocs___closed__3, &lp_mathlib_tacticDocs___closed__3_once, _init_lp_mathlib_tacticDocs___closed__3);
v___f_161_ = ((lean_object*)(lp_mathlib_tacticDocs___closed__0));
v___x_162_ = lean_alloc_ctor(0, 3, 1);
lean_ctor_set(v___x_162_, 0, v___f_161_);
lean_ctor_set(v___x_162_, 1, v___x_160_);
lean_ctor_set(v___x_162_, 2, v___x_159_);
lean_ctor_set_uint8(v___x_162_, sizeof(void*)*3, v___x_158_);
return v___x_162_;
}
}
static lean_object* _init_lp_mathlib_tacticDocs(void){
_start:
{
lean_object* v___x_163_; 
v___x_163_ = lean_obj_once(&lp_mathlib_tacticDocs___closed__7, &lp_mathlib_tacticDocs___closed__7_once, _init_lp_mathlib_tacticDocs___closed__7);
return v___x_163_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00tacticDocs_spec__0(lean_object* v_00_u03b4_164_, lean_object* v_t_165_, lean_object* v_k_166_){
_start:
{
lean_object* v___x_167_; 
v___x_167_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00tacticDocs_spec__0___redArg(v_t_165_, v_k_166_);
return v___x_167_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00tacticDocs_spec__0___boxed(lean_object* v_00_u03b4_168_, lean_object* v_t_169_, lean_object* v_k_170_){
_start:
{
lean_object* v_res_171_; 
v_res_171_ = lp_mathlib_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00tacticDocs_spec__0(v_00_u03b4_168_, v_t_169_, v_k_170_);
lean_dec(v_k_170_);
lean_dec(v_t_169_);
return v_res_171_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00tacticAlt_spec__2(lean_object* v_a_172_, lean_object* v_a_173_){
_start:
{
if (lean_obj_tag(v_a_172_) == 0)
{
lean_object* v___x_174_; 
v___x_174_ = l_List_reverse___redArg(v_a_173_);
return v___x_174_;
}
else
{
lean_object* v_head_175_; lean_object* v_tail_176_; lean_object* v___x_178_; uint8_t v_isShared_179_; uint8_t v_isSharedCheck_185_; 
v_head_175_ = lean_ctor_get(v_a_172_, 0);
v_tail_176_ = lean_ctor_get(v_a_172_, 1);
v_isSharedCheck_185_ = !lean_is_exclusive(v_a_172_);
if (v_isSharedCheck_185_ == 0)
{
v___x_178_ = v_a_172_;
v_isShared_179_ = v_isSharedCheck_185_;
goto v_resetjp_177_;
}
else
{
lean_inc(v_tail_176_);
lean_inc(v_head_175_);
lean_dec(v_a_172_);
v___x_178_ = lean_box(0);
v_isShared_179_ = v_isSharedCheck_185_;
goto v_resetjp_177_;
}
v_resetjp_177_:
{
lean_object* v___x_180_; lean_object* v___x_182_; 
v___x_180_ = l_Lean_MessageData_ofName(v_head_175_);
if (v_isShared_179_ == 0)
{
lean_ctor_set(v___x_178_, 1, v_a_173_);
lean_ctor_set(v___x_178_, 0, v___x_180_);
v___x_182_ = v___x_178_;
goto v_reusejp_181_;
}
else
{
lean_object* v_reuseFailAlloc_184_; 
v_reuseFailAlloc_184_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_184_, 0, v___x_180_);
lean_ctor_set(v_reuseFailAlloc_184_, 1, v_a_173_);
v___x_182_ = v_reuseFailAlloc_184_;
goto v_reusejp_181_;
}
v_reusejp_181_:
{
v_a_172_ = v_tail_176_;
v_a_173_ = v___x_182_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00tacticAlt_spec__3(lean_object* v___x_186_, lean_object* v___x_187_, lean_object* v_as_188_, size_t v_i_189_, size_t v_stop_190_, lean_object* v_b_191_){
_start:
{
lean_object* v___y_193_; uint8_t v___x_197_; 
v___x_197_ = lean_usize_dec_eq(v_i_189_, v_stop_190_);
if (v___x_197_ == 0)
{
lean_object* v___x_198_; uint8_t v___y_200_; lean_object* v_internalName_202_; lean_object* v_userName_203_; uint8_t v___x_204_; 
v___x_198_ = lean_array_uget_borrowed(v_as_188_, v_i_189_);
v_internalName_202_ = lean_ctor_get(v___x_198_, 0);
v_userName_203_ = lean_ctor_get(v___x_198_, 1);
v___x_204_ = lean_string_dec_eq(v_userName_203_, v___x_186_);
if (v___x_204_ == 0)
{
v___y_200_ = v___x_204_;
goto v___jp_199_;
}
else
{
lean_object* v___x_205_; 
lean_inc(v_internalName_202_);
lean_inc_ref(v___x_187_);
v___x_205_ = l_Lean_Parser_Tactic_Doc_alternativeOfTactic(v___x_187_, v_internalName_202_);
if (lean_obj_tag(v___x_205_) == 0)
{
v___y_200_ = v___x_204_;
goto v___jp_199_;
}
else
{
lean_dec_ref_known(v___x_205_, 1);
v___y_193_ = v_b_191_;
goto v___jp_192_;
}
}
v___jp_199_:
{
if (v___y_200_ == 0)
{
v___y_193_ = v_b_191_;
goto v___jp_192_;
}
else
{
lean_object* v___x_201_; 
lean_inc(v___x_198_);
v___x_201_ = lean_array_push(v_b_191_, v___x_198_);
v___y_193_ = v___x_201_;
goto v___jp_192_;
}
}
}
else
{
lean_dec_ref(v___x_187_);
return v_b_191_;
}
v___jp_192_:
{
size_t v___x_194_; size_t v___x_195_; 
v___x_194_ = ((size_t)1ULL);
v___x_195_ = lean_usize_add(v_i_189_, v___x_194_);
v_i_189_ = v___x_195_;
v_b_191_ = v___y_193_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00tacticAlt_spec__3___boxed(lean_object* v___x_206_, lean_object* v___x_207_, lean_object* v_as_208_, lean_object* v_i_209_, lean_object* v_stop_210_, lean_object* v_b_211_){
_start:
{
size_t v_i_boxed_212_; size_t v_stop_boxed_213_; lean_object* v_res_214_; 
v_i_boxed_212_ = lean_unbox_usize(v_i_209_);
lean_dec(v_i_209_);
v_stop_boxed_213_ = lean_unbox_usize(v_stop_210_);
lean_dec(v_stop_210_);
v_res_214_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00tacticAlt_spec__3(v___x_206_, v___x_207_, v_as_208_, v_i_boxed_212_, v_stop_boxed_213_, v_b_211_);
lean_dec_ref(v_as_208_);
lean_dec_ref(v___x_206_);
return v_res_214_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00tacticAlt_spec__0(lean_object* v_tac_218_, lean_object* v_as_219_, size_t v_sz_220_, size_t v_i_221_, lean_object* v_b_222_){
_start:
{
uint8_t v___x_223_; 
v___x_223_ = lean_usize_dec_lt(v_i_221_, v_sz_220_);
if (v___x_223_ == 0)
{
lean_inc_ref(v_b_222_);
return v_b_222_;
}
else
{
lean_object* v_a_224_; lean_object* v_internalName_225_; lean_object* v___x_226_; uint8_t v___x_227_; 
v_a_224_ = lean_array_uget_borrowed(v_as_219_, v_i_221_);
v_internalName_225_ = lean_ctor_get(v_a_224_, 0);
v___x_226_ = lean_box(0);
v___x_227_ = lean_name_eq(v_internalName_225_, v_tac_218_);
if (v___x_227_ == 0)
{
lean_object* v___x_228_; size_t v___x_229_; size_t v___x_230_; 
v___x_228_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00tacticAlt_spec__0___closed__0));
v___x_229_ = ((size_t)1ULL);
v___x_230_ = lean_usize_add(v_i_221_, v___x_229_);
v_i_221_ = v___x_230_;
v_b_222_ = v___x_228_;
goto _start;
}
else
{
lean_object* v___x_232_; lean_object* v___x_233_; lean_object* v___x_234_; 
lean_inc(v_a_224_);
v___x_232_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_232_, 0, v_a_224_);
v___x_233_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_233_, 0, v___x_232_);
v___x_234_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_234_, 0, v___x_233_);
lean_ctor_set(v___x_234_, 1, v___x_226_);
return v___x_234_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00tacticAlt_spec__0___boxed(lean_object* v_tac_235_, lean_object* v_as_236_, lean_object* v_sz_237_, lean_object* v_i_238_, lean_object* v_b_239_){
_start:
{
size_t v_sz_boxed_240_; size_t v_i_boxed_241_; lean_object* v_res_242_; 
v_sz_boxed_240_ = lean_unbox_usize(v_sz_237_);
lean_dec(v_sz_237_);
v_i_boxed_241_ = lean_unbox_usize(v_i_238_);
lean_dec(v_i_238_);
v_res_242_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00tacticAlt_spec__0(v_tac_235_, v_as_236_, v_sz_boxed_240_, v_i_boxed_241_, v_b_239_);
lean_dec_ref(v_b_239_);
lean_dec_ref(v_as_236_);
lean_dec(v_tac_235_);
return v_res_242_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00tacticAlt_spec__1(size_t v_sz_243_, size_t v_i_244_, lean_object* v_bs_245_){
_start:
{
uint8_t v___x_246_; 
v___x_246_ = lean_usize_dec_lt(v_i_244_, v_sz_243_);
if (v___x_246_ == 0)
{
return v_bs_245_;
}
else
{
lean_object* v_v_247_; lean_object* v_internalName_248_; lean_object* v___x_249_; lean_object* v_bs_x27_250_; size_t v___x_251_; size_t v___x_252_; lean_object* v___x_253_; 
v_v_247_ = lean_array_uget_borrowed(v_bs_245_, v_i_244_);
v_internalName_248_ = lean_ctor_get(v_v_247_, 0);
lean_inc(v_internalName_248_);
v___x_249_ = lean_unsigned_to_nat(0u);
v_bs_x27_250_ = lean_array_uset(v_bs_245_, v_i_244_, v___x_249_);
v___x_251_ = ((size_t)1ULL);
v___x_252_ = lean_usize_add(v_i_244_, v___x_251_);
v___x_253_ = lean_array_uset(v_bs_x27_250_, v_i_244_, v_internalName_248_);
v_i_244_ = v___x_252_;
v_bs_245_ = v___x_253_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00tacticAlt_spec__1___boxed(lean_object* v_sz_255_, lean_object* v_i_256_, lean_object* v_bs_257_){
_start:
{
size_t v_sz_boxed_258_; size_t v_i_boxed_259_; lean_object* v_res_260_; 
v_sz_boxed_258_ = lean_unbox_usize(v_sz_255_);
lean_dec(v_sz_255_);
v_i_boxed_259_ = lean_unbox_usize(v_i_256_);
lean_dec(v_i_256_);
v_res_260_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00tacticAlt_spec__1(v_sz_boxed_258_, v_i_boxed_259_, v_bs_257_);
return v_res_260_;
}
}
static lean_object* _init_lp_mathlib_tacticAlt___lam__0___closed__1(void){
_start:
{
lean_object* v___x_262_; lean_object* v___x_263_; 
v___x_262_ = ((lean_object*)(lp_mathlib_tacticAlt___lam__0___closed__0));
v___x_263_ = l_Lean_stringToMessageData(v___x_262_);
return v___x_263_;
}
}
static lean_object* _init_lp_mathlib_tacticAlt___lam__0___closed__3(void){
_start:
{
lean_object* v___x_265_; lean_object* v___x_266_; 
v___x_265_ = ((lean_object*)(lp_mathlib_tacticAlt___lam__0___closed__2));
v___x_266_ = l_Lean_stringToMessageData(v___x_265_);
return v___x_266_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_tacticAlt___lam__0(lean_object* v_tac_269_, lean_object* v___y_270_, lean_object* v___y_271_, lean_object* v___y_272_, lean_object* v___y_273_){
_start:
{
lean_object* v___x_275_; lean_object* v_env_279_; uint8_t v___x_280_; 
v___x_275_ = lean_st_ref_get(v___y_273_);
v_env_279_ = lean_ctor_get(v___x_275_, 0);
lean_inc_ref_n(v_env_279_, 2);
lean_dec(v___x_275_);
v___x_280_ = l_Lean_Parser_Tactic_Doc_isTactic(v_env_279_, v_tac_269_);
if (v___x_280_ == 0)
{
lean_dec_ref(v_env_279_);
lean_dec(v_tac_269_);
goto v___jp_276_;
}
else
{
lean_object* v___x_281_; 
lean_inc(v_tac_269_);
lean_inc_ref(v_env_279_);
v___x_281_ = l_Lean_Parser_Tactic_Doc_alternativeOfTactic(v_env_279_, v_tac_269_);
if (lean_obj_tag(v___x_281_) == 0)
{
lean_object* v___x_282_; 
v___x_282_ = l_Lean_Elab_Tactic_Doc_allTacticDocs(v___x_280_, v___y_270_, v___y_271_, v___y_272_, v___y_273_);
if (lean_obj_tag(v___x_282_) == 0)
{
lean_object* v_a_283_; lean_object* v___x_285_; uint8_t v_isShared_286_; uint8_t v_isSharedCheck_353_; 
v_a_283_ = lean_ctor_get(v___x_282_, 0);
v_isSharedCheck_353_ = !lean_is_exclusive(v___x_282_);
if (v_isSharedCheck_353_ == 0)
{
v___x_285_ = v___x_282_;
v_isShared_286_ = v_isSharedCheck_353_;
goto v_resetjp_284_;
}
else
{
lean_inc(v_a_283_);
lean_dec(v___x_282_);
v___x_285_ = lean_box(0);
v_isShared_286_ = v_isSharedCheck_353_;
goto v_resetjp_284_;
}
v_resetjp_284_:
{
lean_object* v___x_292_; lean_object* v___x_293_; size_t v_sz_294_; size_t v___x_295_; lean_object* v___x_296_; lean_object* v_fst_297_; lean_object* v___x_299_; uint8_t v_isShared_300_; uint8_t v_isSharedCheck_351_; 
v___x_292_ = lean_box(0);
v___x_293_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00tacticAlt_spec__0___closed__0));
v_sz_294_ = lean_array_size(v_a_283_);
v___x_295_ = ((size_t)0ULL);
v___x_296_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00tacticAlt_spec__0(v_tac_269_, v_a_283_, v_sz_294_, v___x_295_, v___x_293_);
lean_dec(v_tac_269_);
v_fst_297_ = lean_ctor_get(v___x_296_, 0);
v_isSharedCheck_351_ = !lean_is_exclusive(v___x_296_);
if (v_isSharedCheck_351_ == 0)
{
lean_object* v_unused_352_; 
v_unused_352_ = lean_ctor_get(v___x_296_, 1);
lean_dec(v_unused_352_);
v___x_299_ = v___x_296_;
v_isShared_300_ = v_isSharedCheck_351_;
goto v_resetjp_298_;
}
else
{
lean_inc(v_fst_297_);
lean_dec(v___x_296_);
v___x_299_ = lean_box(0);
v_isShared_300_ = v_isSharedCheck_351_;
goto v_resetjp_298_;
}
v___jp_287_:
{
lean_object* v___x_288_; lean_object* v___x_290_; 
v___x_288_ = lean_box(0);
if (v_isShared_286_ == 0)
{
lean_ctor_set(v___x_285_, 0, v___x_288_);
v___x_290_ = v___x_285_;
goto v_reusejp_289_;
}
else
{
lean_object* v_reuseFailAlloc_291_; 
v_reuseFailAlloc_291_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_291_, 0, v___x_288_);
v___x_290_ = v_reuseFailAlloc_291_;
goto v_reusejp_289_;
}
v_reusejp_289_:
{
return v___x_290_;
}
}
v_resetjp_298_:
{
if (lean_obj_tag(v_fst_297_) == 0)
{
lean_del_object(v___x_299_);
lean_dec(v_a_283_);
lean_dec_ref(v_env_279_);
goto v___jp_287_;
}
else
{
lean_object* v_val_301_; lean_object* v___x_303_; uint8_t v_isShared_304_; uint8_t v_isSharedCheck_350_; 
v_val_301_ = lean_ctor_get(v_fst_297_, 0);
v_isSharedCheck_350_ = !lean_is_exclusive(v_fst_297_);
if (v_isSharedCheck_350_ == 0)
{
v___x_303_ = v_fst_297_;
v_isShared_304_ = v_isSharedCheck_350_;
goto v_resetjp_302_;
}
else
{
lean_inc(v_val_301_);
lean_dec(v_fst_297_);
v___x_303_ = lean_box(0);
v_isShared_304_ = v_isSharedCheck_350_;
goto v_resetjp_302_;
}
v_resetjp_302_:
{
if (lean_obj_tag(v_val_301_) == 1)
{
lean_object* v_val_305_; lean_object* v___x_307_; uint8_t v_isShared_308_; uint8_t v_isSharedCheck_349_; 
lean_del_object(v___x_285_);
v_val_305_ = lean_ctor_get(v_val_301_, 0);
v_isSharedCheck_349_ = !lean_is_exclusive(v_val_301_);
if (v_isSharedCheck_349_ == 0)
{
v___x_307_ = v_val_301_;
v_isShared_308_ = v_isSharedCheck_349_;
goto v_resetjp_306_;
}
else
{
lean_inc(v_val_305_);
lean_dec(v_val_301_);
v___x_307_ = lean_box(0);
v_isShared_308_ = v_isSharedCheck_349_;
goto v_resetjp_306_;
}
v_resetjp_306_:
{
lean_object* v_userName_309_; lean_object* v___y_311_; lean_object* v___x_340_; lean_object* v___x_341_; lean_object* v___x_342_; uint8_t v___x_343_; 
v_userName_309_ = lean_ctor_get(v_val_305_, 1);
lean_inc_ref(v_userName_309_);
lean_dec(v_val_305_);
v___x_340_ = lean_unsigned_to_nat(0u);
v___x_341_ = lean_array_get_size(v_a_283_);
v___x_342_ = ((lean_object*)(lp_mathlib_tacticAlt___lam__0___closed__4));
v___x_343_ = lean_nat_dec_lt(v___x_340_, v___x_341_);
if (v___x_343_ == 0)
{
lean_dec(v_a_283_);
lean_dec_ref(v_env_279_);
v___y_311_ = v___x_342_;
goto v___jp_310_;
}
else
{
uint8_t v___x_344_; 
v___x_344_ = lean_nat_dec_le(v___x_341_, v___x_341_);
if (v___x_344_ == 0)
{
if (v___x_343_ == 0)
{
lean_dec(v_a_283_);
lean_dec_ref(v_env_279_);
v___y_311_ = v___x_342_;
goto v___jp_310_;
}
else
{
size_t v___x_345_; lean_object* v___x_346_; 
v___x_345_ = lean_usize_of_nat(v___x_341_);
v___x_346_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00tacticAlt_spec__3(v_userName_309_, v_env_279_, v_a_283_, v___x_295_, v___x_345_, v___x_342_);
lean_dec(v_a_283_);
v___y_311_ = v___x_346_;
goto v___jp_310_;
}
}
else
{
size_t v___x_347_; lean_object* v___x_348_; 
v___x_347_ = lean_usize_of_nat(v___x_341_);
v___x_348_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00tacticAlt_spec__3(v_userName_309_, v_env_279_, v_a_283_, v___x_295_, v___x_347_, v___x_342_);
lean_dec(v_a_283_);
v___y_311_ = v___x_348_;
goto v___jp_310_;
}
}
v___jp_310_:
{
lean_object* v___x_312_; lean_object* v___x_313_; uint8_t v___x_314_; 
v___x_312_ = lean_array_get_size(v___y_311_);
v___x_313_ = lean_unsigned_to_nat(1u);
v___x_314_ = lean_nat_dec_le(v___x_312_, v___x_313_);
if (v___x_314_ == 0)
{
size_t v_sz_315_; lean_object* v___x_316_; lean_object* v___x_317_; lean_object* v___x_318_; lean_object* v___x_320_; 
v_sz_315_ = lean_array_size(v___y_311_);
v___x_316_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00tacticAlt_spec__1(v_sz_315_, v___x_295_, v___y_311_);
v___x_317_ = lean_obj_once(&lp_mathlib_tacticDocs___lam__0___closed__1, &lp_mathlib_tacticDocs___lam__0___closed__1_once, _init_lp_mathlib_tacticDocs___lam__0___closed__1);
v___x_318_ = l_Lean_stringToMessageData(v_userName_309_);
if (v_isShared_300_ == 0)
{
lean_ctor_set_tag(v___x_299_, 7);
lean_ctor_set(v___x_299_, 1, v___x_318_);
lean_ctor_set(v___x_299_, 0, v___x_317_);
v___x_320_ = v___x_299_;
goto v_reusejp_319_;
}
else
{
lean_object* v_reuseFailAlloc_336_; 
v_reuseFailAlloc_336_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_336_, 0, v___x_317_);
lean_ctor_set(v_reuseFailAlloc_336_, 1, v___x_318_);
v___x_320_ = v_reuseFailAlloc_336_;
goto v_reusejp_319_;
}
v_reusejp_319_:
{
lean_object* v___x_321_; lean_object* v___x_322_; lean_object* v___x_323_; lean_object* v___x_324_; lean_object* v___x_325_; lean_object* v___x_326_; lean_object* v___x_327_; lean_object* v___x_328_; lean_object* v___x_329_; lean_object* v___x_331_; 
v___x_321_ = lean_obj_once(&lp_mathlib_tacticAlt___lam__0___closed__1, &lp_mathlib_tacticAlt___lam__0___closed__1_once, _init_lp_mathlib_tacticAlt___lam__0___closed__1);
v___x_322_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_322_, 0, v___x_320_);
lean_ctor_set(v___x_322_, 1, v___x_321_);
v___x_323_ = lean_array_to_list(v___x_316_);
v___x_324_ = lean_box(0);
v___x_325_ = lp_mathlib_List_mapTR_loop___at___00tacticAlt_spec__2(v___x_323_, v___x_324_);
v___x_326_ = l_Lean_MessageData_ofList(v___x_325_);
v___x_327_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_327_, 0, v___x_322_);
lean_ctor_set(v___x_327_, 1, v___x_326_);
v___x_328_ = lean_obj_once(&lp_mathlib_tacticAlt___lam__0___closed__3, &lp_mathlib_tacticAlt___lam__0___closed__3_once, _init_lp_mathlib_tacticAlt___lam__0___closed__3);
v___x_329_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_329_, 0, v___x_327_);
lean_ctor_set(v___x_329_, 1, v___x_328_);
if (v_isShared_308_ == 0)
{
lean_ctor_set(v___x_307_, 0, v___x_329_);
v___x_331_ = v___x_307_;
goto v_reusejp_330_;
}
else
{
lean_object* v_reuseFailAlloc_335_; 
v_reuseFailAlloc_335_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_335_, 0, v___x_329_);
v___x_331_ = v_reuseFailAlloc_335_;
goto v_reusejp_330_;
}
v_reusejp_330_:
{
lean_object* v___x_333_; 
if (v_isShared_304_ == 0)
{
lean_ctor_set_tag(v___x_303_, 0);
lean_ctor_set(v___x_303_, 0, v___x_331_);
v___x_333_ = v___x_303_;
goto v_reusejp_332_;
}
else
{
lean_object* v_reuseFailAlloc_334_; 
v_reuseFailAlloc_334_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_334_, 0, v___x_331_);
v___x_333_ = v_reuseFailAlloc_334_;
goto v_reusejp_332_;
}
v_reusejp_332_:
{
return v___x_333_;
}
}
}
}
else
{
lean_object* v___x_338_; 
lean_dec_ref(v___y_311_);
lean_dec_ref(v_userName_309_);
lean_del_object(v___x_307_);
lean_del_object(v___x_299_);
if (v_isShared_304_ == 0)
{
lean_ctor_set_tag(v___x_303_, 0);
lean_ctor_set(v___x_303_, 0, v___x_292_);
v___x_338_ = v___x_303_;
goto v_reusejp_337_;
}
else
{
lean_object* v_reuseFailAlloc_339_; 
v_reuseFailAlloc_339_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_339_, 0, v___x_292_);
v___x_338_ = v_reuseFailAlloc_339_;
goto v_reusejp_337_;
}
v_reusejp_337_:
{
return v___x_338_;
}
}
}
}
}
else
{
lean_del_object(v___x_303_);
lean_dec(v_val_301_);
lean_del_object(v___x_299_);
lean_dec(v_a_283_);
lean_dec_ref(v_env_279_);
goto v___jp_287_;
}
}
}
}
}
}
else
{
lean_object* v_a_354_; lean_object* v___x_356_; uint8_t v_isShared_357_; uint8_t v_isSharedCheck_361_; 
lean_dec_ref(v_env_279_);
lean_dec(v_tac_269_);
v_a_354_ = lean_ctor_get(v___x_282_, 0);
v_isSharedCheck_361_ = !lean_is_exclusive(v___x_282_);
if (v_isSharedCheck_361_ == 0)
{
v___x_356_ = v___x_282_;
v_isShared_357_ = v_isSharedCheck_361_;
goto v_resetjp_355_;
}
else
{
lean_inc(v_a_354_);
lean_dec(v___x_282_);
v___x_356_ = lean_box(0);
v_isShared_357_ = v_isSharedCheck_361_;
goto v_resetjp_355_;
}
v_resetjp_355_:
{
lean_object* v___x_359_; 
if (v_isShared_357_ == 0)
{
v___x_359_ = v___x_356_;
goto v_reusejp_358_;
}
else
{
lean_object* v_reuseFailAlloc_360_; 
v_reuseFailAlloc_360_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_360_, 0, v_a_354_);
v___x_359_ = v_reuseFailAlloc_360_;
goto v_reusejp_358_;
}
v_reusejp_358_:
{
return v___x_359_;
}
}
}
}
else
{
lean_dec_ref_known(v___x_281_, 1);
lean_dec_ref(v_env_279_);
lean_dec(v_tac_269_);
goto v___jp_276_;
}
}
v___jp_276_:
{
lean_object* v___x_277_; lean_object* v___x_278_; 
v___x_277_ = lean_box(0);
v___x_278_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_278_, 0, v___x_277_);
return v___x_278_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_tacticAlt___lam__0___boxed(lean_object* v_tac_362_, lean_object* v___y_363_, lean_object* v___y_364_, lean_object* v___y_365_, lean_object* v___y_366_, lean_object* v___y_367_){
_start:
{
lean_object* v_res_368_; 
v_res_368_ = lp_mathlib_tacticAlt___lam__0(v_tac_362_, v___y_363_, v___y_364_, v___y_365_, v___y_366_);
lean_dec(v___y_366_);
lean_dec_ref(v___y_365_);
lean_dec(v___y_364_);
lean_dec_ref(v___y_363_);
return v_res_368_;
}
}
static lean_object* _init_lp_mathlib_tacticAlt___closed__3(void){
_start:
{
lean_object* v___x_373_; lean_object* v___x_374_; 
v___x_373_ = ((lean_object*)(lp_mathlib_tacticAlt___closed__2));
v___x_374_ = l_Lean_MessageData_ofFormat(v___x_373_);
return v___x_374_;
}
}
static lean_object* _init_lp_mathlib_tacticAlt___closed__6(void){
_start:
{
lean_object* v___x_378_; lean_object* v___x_379_; 
v___x_378_ = ((lean_object*)(lp_mathlib_tacticAlt___closed__5));
v___x_379_ = l_Lean_MessageData_ofFormat(v___x_378_);
return v___x_379_;
}
}
static lean_object* _init_lp_mathlib_tacticAlt___closed__7(void){
_start:
{
uint8_t v___x_380_; lean_object* v___x_381_; lean_object* v___x_382_; lean_object* v___f_383_; lean_object* v___x_384_; 
v___x_380_ = 1;
v___x_381_ = lean_obj_once(&lp_mathlib_tacticAlt___closed__6, &lp_mathlib_tacticAlt___closed__6_once, _init_lp_mathlib_tacticAlt___closed__6);
v___x_382_ = lean_obj_once(&lp_mathlib_tacticAlt___closed__3, &lp_mathlib_tacticAlt___closed__3_once, _init_lp_mathlib_tacticAlt___closed__3);
v___f_383_ = ((lean_object*)(lp_mathlib_tacticAlt___closed__0));
v___x_384_ = lean_alloc_ctor(0, 3, 1);
lean_ctor_set(v___x_384_, 0, v___f_383_);
lean_ctor_set(v___x_384_, 1, v___x_382_);
lean_ctor_set(v___x_384_, 2, v___x_381_);
lean_ctor_set_uint8(v___x_384_, sizeof(void*)*3, v___x_380_);
return v___x_384_;
}
}
static lean_object* _init_lp_mathlib_tacticAlt(void){
_start:
{
lean_object* v___x_385_; 
v___x_385_ = lean_obj_once(&lp_mathlib_tacticAlt___closed__7, &lp_mathlib_tacticAlt___closed__7_once, _init_lp_mathlib_tacticAlt___closed__7);
return v___x_385_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linter_Header(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Tactic_Lint_Basic(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Tactic_Doc(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linter_TacticDocumentation(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linter_Header(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Tactic_Lint_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Tactic_Doc(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Tactic_Doc(uint8_t builtin);
lean_object* runtime_initialize_Lean_Parser_Tactic_Doc(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Linter_TacticDocumentation(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Tactic_Doc(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Parser_Tactic_Doc(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_tacticDocs = _init_lp_mathlib_tacticDocs();
lean_mark_persistent(lp_mathlib_tacticDocs);
lp_mathlib_tacticAlt = _init_lp_mathlib_tacticAlt();
lean_mark_persistent(lp_mathlib_tacticAlt);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Elab_Tactic_Doc(uint8_t builtin);
lean_object* initialize_Lean_Parser_Tactic_Doc(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Linter_Header(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Tactic_Lint_Basic(uint8_t builtin);
lean_object* initialize_Lean_Elab_Tactic_Doc(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Linter_TacticDocumentation(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Tactic_Doc(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Parser_Tactic_Doc(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Linter_Header(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Tactic_Lint_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Tactic_Doc(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linter_TacticDocumentation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Linter_TacticDocumentation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Linter_TacticDocumentation(builtin);
}
#ifdef __cplusplus
}
#endif
