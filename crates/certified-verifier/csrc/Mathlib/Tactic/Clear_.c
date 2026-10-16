// Lean compiler output
// Module: Mathlib.Tactic.Clear_
// Imports: public import Init public meta import Init public import Mathlib.Init public meta import Lean.Meta.Tactic.Clear public meta import Lean.Elab.Tactic.Basic
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
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
size_t lean_array_size(lean_object*);
size_t lean_usize_add(size_t, size_t);
uint8_t lean_uint32_dec_eq(uint32_t, uint32_t);
lean_object* l_Lean_LocalDecl_type(lean_object*);
lean_object* l_Lean_Meta_isClass_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_LocalDecl_fvarId(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_LocalDecl_userName(lean_object*);
lean_object* lean_string_utf8_byte_size(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_String_Slice_Pos_get_x3f(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_Elab_Tactic_getMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_MVarId_tryClearMany(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_replaceMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_withMainContext___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_clear___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Tactic_clear___00__closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_clear___00__closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_clear___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_clear___00__closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_clear___00__closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_clear___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "clear_"};
static const lean_object* lp_mathlib_Mathlib_Tactic_clear___00__closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_clear___00__closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_clear___00__closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_clear___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_clear___00__closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_clear___00__closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_clear___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_clear___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_clear___00__closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_clear___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(198, 46, 94, 123, 234, 182, 207, 148)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_clear___00__closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_clear___00__closed__3_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_clear___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 6}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_clear___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_clear___00__closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_clear___00__closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_clear___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_clear___00__closed__3_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_clear___00__closed__4_value)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_clear___00__closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_clear___00__closed__5_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_clear__ = (const lean_object*)&lp_mathlib_Mathlib_Tactic_clear___00__closed__5_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1_spec__1___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1_spec__1___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1_spec__1___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1_spec__1___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1_spec__0_spec__0_spec__3_spec__4(uint8_t, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1_spec__0_spec__0_spec__3_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1_spec__0_spec__0_spec__3(uint8_t, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1_spec__0_spec__0_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1_spec__0_spec__0(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1_spec__0_spec__0_spec__2(lean_object*, uint8_t, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1_spec__0_spec__0_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1_spec__0_spec__1_spec__5(uint8_t, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1_spec__0_spec__1_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1_spec__0_spec__1(uint8_t, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1_spec__0(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1___lam__0___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1___lam__0(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1_spec__1___redArg___closed__0(void){
_start:
{
lean_object* v___x_16_; lean_object* v___x_17_; lean_object* v___x_18_; 
v___x_16_ = lean_box(0);
v___x_17_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_18_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_18_, 0, v___x_17_);
lean_ctor_set(v___x_18_, 1, v___x_16_);
return v___x_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1_spec__1___redArg(){
_start:
{
lean_object* v___x_20_; lean_object* v___x_21_; 
v___x_20_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1_spec__1___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1_spec__1___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1_spec__1___redArg___closed__0);
v___x_21_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_21_, 0, v___x_20_);
return v___x_21_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1_spec__1___redArg___boxed(lean_object* v___y_22_){
_start:
{
lean_object* v_res_23_; 
v_res_23_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1_spec__1___redArg();
return v_res_23_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1_spec__1(lean_object* v_00_u03b1_24_, lean_object* v___y_25_, lean_object* v___y_26_, lean_object* v___y_27_, lean_object* v___y_28_, lean_object* v___y_29_, lean_object* v___y_30_, lean_object* v___y_31_, lean_object* v___y_32_){
_start:
{
lean_object* v___x_34_; 
v___x_34_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1_spec__1___redArg();
return v___x_34_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1_spec__1___boxed(lean_object* v_00_u03b1_35_, lean_object* v___y_36_, lean_object* v___y_37_, lean_object* v___y_38_, lean_object* v___y_39_, lean_object* v___y_40_, lean_object* v___y_41_, lean_object* v___y_42_, lean_object* v___y_43_, lean_object* v___y_44_){
_start:
{
lean_object* v_res_45_; 
v_res_45_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1_spec__1(v_00_u03b1_35_, v___y_36_, v___y_37_, v___y_38_, v___y_39_, v___y_40_, v___y_41_, v___y_42_, v___y_43_);
lean_dec(v___y_43_);
lean_dec_ref(v___y_42_);
lean_dec(v___y_41_);
lean_dec_ref(v___y_40_);
lean_dec(v___y_39_);
lean_dec_ref(v___y_38_);
lean_dec(v___y_37_);
lean_dec_ref(v___y_36_);
return v_res_45_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1_spec__0_spec__0_spec__3_spec__4(uint8_t v___x_46_, lean_object* v_as_47_, size_t v_sz_48_, size_t v_i_49_, lean_object* v_b_50_, lean_object* v___y_51_, lean_object* v___y_52_, lean_object* v___y_53_, lean_object* v___y_54_){
_start:
{
uint8_t v___x_56_; 
v___x_56_ = lean_usize_dec_lt(v_i_49_, v_sz_48_);
if (v___x_56_ == 0)
{
lean_object* v___x_57_; 
v___x_57_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_57_, 0, v_b_50_);
return v___x_57_;
}
else
{
lean_object* v_snd_58_; lean_object* v___x_60_; uint8_t v_isShared_61_; uint8_t v_isSharedCheck_100_; 
v_snd_58_ = lean_ctor_get(v_b_50_, 1);
v_isSharedCheck_100_ = !lean_is_exclusive(v_b_50_);
if (v_isSharedCheck_100_ == 0)
{
lean_object* v_unused_101_; 
v_unused_101_ = lean_ctor_get(v_b_50_, 0);
lean_dec(v_unused_101_);
v___x_60_ = v_b_50_;
v_isShared_61_ = v_isSharedCheck_100_;
goto v_resetjp_59_;
}
else
{
lean_inc(v_snd_58_);
lean_dec(v_b_50_);
v___x_60_ = lean_box(0);
v_isShared_61_ = v_isSharedCheck_100_;
goto v_resetjp_59_;
}
v_resetjp_59_:
{
lean_object* v___x_62_; lean_object* v_a_64_; lean_object* v_a_71_; 
v___x_62_ = lean_box(0);
v_a_71_ = lean_array_uget_borrowed(v_as_47_, v_i_49_);
if (lean_obj_tag(v_a_71_) == 0)
{
v_a_64_ = v_snd_58_;
goto v___jp_63_;
}
else
{
lean_object* v_val_72_; uint32_t v___y_74_; lean_object* v___x_90_; 
v_val_72_ = lean_ctor_get(v_a_71_, 0);
v___x_90_ = l_Lean_LocalDecl_userName(v_val_72_);
if (lean_obj_tag(v___x_90_) == 1)
{
lean_object* v_str_91_; lean_object* v___x_92_; lean_object* v___x_93_; uint8_t v___x_94_; 
v_str_91_ = lean_ctor_get(v___x_90_, 1);
lean_inc_ref(v_str_91_);
lean_dec_ref_known(v___x_90_, 2);
v___x_92_ = lean_string_utf8_byte_size(v_str_91_);
v___x_93_ = lean_unsigned_to_nat(0u);
v___x_94_ = lean_nat_dec_eq(v___x_92_, v___x_93_);
if (v___x_94_ == 0)
{
if (v___x_46_ == 0)
{
lean_dec_ref(v_str_91_);
v_a_64_ = v_snd_58_;
goto v___jp_63_;
}
else
{
lean_object* v___x_95_; lean_object* v___x_96_; 
v___x_95_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_95_, 0, v_str_91_);
lean_ctor_set(v___x_95_, 1, v___x_93_);
lean_ctor_set(v___x_95_, 2, v___x_92_);
v___x_96_ = l_String_Slice_Pos_get_x3f(v___x_95_, v___x_93_);
lean_dec_ref_known(v___x_95_, 3);
if (lean_obj_tag(v___x_96_) == 0)
{
uint32_t v___x_97_; 
v___x_97_ = 65;
v___y_74_ = v___x_97_;
goto v___jp_73_;
}
else
{
lean_object* v_val_98_; uint32_t v___x_99_; 
v_val_98_ = lean_ctor_get(v___x_96_, 0);
lean_inc(v_val_98_);
lean_dec_ref_known(v___x_96_, 1);
v___x_99_ = lean_unbox_uint32(v_val_98_);
lean_dec(v_val_98_);
v___y_74_ = v___x_99_;
goto v___jp_73_;
}
}
}
else
{
lean_dec_ref(v_str_91_);
v_a_64_ = v_snd_58_;
goto v___jp_63_;
}
}
else
{
lean_dec(v___x_90_);
v_a_64_ = v_snd_58_;
goto v___jp_63_;
}
v___jp_73_:
{
uint32_t v___x_75_; uint8_t v___x_76_; 
v___x_75_ = 95;
v___x_76_ = lean_uint32_dec_eq(v___y_74_, v___x_75_);
if (v___x_76_ == 0)
{
v_a_64_ = v_snd_58_;
goto v___jp_63_;
}
else
{
lean_object* v___x_77_; lean_object* v___x_78_; 
v___x_77_ = l_Lean_LocalDecl_type(v_val_72_);
v___x_78_ = l_Lean_Meta_isClass_x3f(v___x_77_, v___y_51_, v___y_52_, v___y_53_, v___y_54_);
if (lean_obj_tag(v___x_78_) == 0)
{
lean_object* v_a_79_; 
v_a_79_ = lean_ctor_get(v___x_78_, 0);
lean_inc(v_a_79_);
lean_dec_ref_known(v___x_78_, 1);
if (lean_obj_tag(v_a_79_) == 0)
{
lean_object* v___x_80_; lean_object* v___x_81_; 
v___x_80_ = l_Lean_LocalDecl_fvarId(v_val_72_);
v___x_81_ = lean_array_push(v_snd_58_, v___x_80_);
v_a_64_ = v___x_81_;
goto v___jp_63_;
}
else
{
lean_dec(v_a_79_);
v_a_64_ = v_snd_58_;
goto v___jp_63_;
}
}
else
{
lean_object* v_a_82_; lean_object* v___x_84_; uint8_t v_isShared_85_; uint8_t v_isSharedCheck_89_; 
lean_del_object(v___x_60_);
lean_dec(v_snd_58_);
v_a_82_ = lean_ctor_get(v___x_78_, 0);
v_isSharedCheck_89_ = !lean_is_exclusive(v___x_78_);
if (v_isSharedCheck_89_ == 0)
{
v___x_84_ = v___x_78_;
v_isShared_85_ = v_isSharedCheck_89_;
goto v_resetjp_83_;
}
else
{
lean_inc(v_a_82_);
lean_dec(v___x_78_);
v___x_84_ = lean_box(0);
v_isShared_85_ = v_isSharedCheck_89_;
goto v_resetjp_83_;
}
v_resetjp_83_:
{
lean_object* v___x_87_; 
if (v_isShared_85_ == 0)
{
v___x_87_ = v___x_84_;
goto v_reusejp_86_;
}
else
{
lean_object* v_reuseFailAlloc_88_; 
v_reuseFailAlloc_88_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_88_, 0, v_a_82_);
v___x_87_ = v_reuseFailAlloc_88_;
goto v_reusejp_86_;
}
v_reusejp_86_:
{
return v___x_87_;
}
}
}
}
}
}
v___jp_63_:
{
lean_object* v___x_66_; 
if (v_isShared_61_ == 0)
{
lean_ctor_set(v___x_60_, 1, v_a_64_);
lean_ctor_set(v___x_60_, 0, v___x_62_);
v___x_66_ = v___x_60_;
goto v_reusejp_65_;
}
else
{
lean_object* v_reuseFailAlloc_70_; 
v_reuseFailAlloc_70_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_70_, 0, v___x_62_);
lean_ctor_set(v_reuseFailAlloc_70_, 1, v_a_64_);
v___x_66_ = v_reuseFailAlloc_70_;
goto v_reusejp_65_;
}
v_reusejp_65_:
{
size_t v___x_67_; size_t v___x_68_; 
v___x_67_ = ((size_t)1ULL);
v___x_68_ = lean_usize_add(v_i_49_, v___x_67_);
v_i_49_ = v___x_68_;
v_b_50_ = v___x_66_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1_spec__0_spec__0_spec__3_spec__4___boxed(lean_object* v___x_102_, lean_object* v_as_103_, lean_object* v_sz_104_, lean_object* v_i_105_, lean_object* v_b_106_, lean_object* v___y_107_, lean_object* v___y_108_, lean_object* v___y_109_, lean_object* v___y_110_, lean_object* v___y_111_){
_start:
{
uint8_t v___x_5682__boxed_112_; size_t v_sz_boxed_113_; size_t v_i_boxed_114_; lean_object* v_res_115_; 
v___x_5682__boxed_112_ = lean_unbox(v___x_102_);
v_sz_boxed_113_ = lean_unbox_usize(v_sz_104_);
lean_dec(v_sz_104_);
v_i_boxed_114_ = lean_unbox_usize(v_i_105_);
lean_dec(v_i_105_);
v_res_115_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1_spec__0_spec__0_spec__3_spec__4(v___x_5682__boxed_112_, v_as_103_, v_sz_boxed_113_, v_i_boxed_114_, v_b_106_, v___y_107_, v___y_108_, v___y_109_, v___y_110_);
lean_dec(v___y_110_);
lean_dec_ref(v___y_109_);
lean_dec(v___y_108_);
lean_dec_ref(v___y_107_);
lean_dec_ref(v_as_103_);
return v_res_115_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1_spec__0_spec__0_spec__3(uint8_t v___x_116_, lean_object* v_as_117_, size_t v_sz_118_, size_t v_i_119_, lean_object* v_b_120_, lean_object* v___y_121_, lean_object* v___y_122_, lean_object* v___y_123_, lean_object* v___y_124_){
_start:
{
uint8_t v___x_126_; 
v___x_126_ = lean_usize_dec_lt(v_i_119_, v_sz_118_);
if (v___x_126_ == 0)
{
lean_object* v___x_127_; 
v___x_127_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_127_, 0, v_b_120_);
return v___x_127_;
}
else
{
lean_object* v_snd_128_; lean_object* v___x_130_; uint8_t v_isShared_131_; uint8_t v_isSharedCheck_170_; 
v_snd_128_ = lean_ctor_get(v_b_120_, 1);
v_isSharedCheck_170_ = !lean_is_exclusive(v_b_120_);
if (v_isSharedCheck_170_ == 0)
{
lean_object* v_unused_171_; 
v_unused_171_ = lean_ctor_get(v_b_120_, 0);
lean_dec(v_unused_171_);
v___x_130_ = v_b_120_;
v_isShared_131_ = v_isSharedCheck_170_;
goto v_resetjp_129_;
}
else
{
lean_inc(v_snd_128_);
lean_dec(v_b_120_);
v___x_130_ = lean_box(0);
v_isShared_131_ = v_isSharedCheck_170_;
goto v_resetjp_129_;
}
v_resetjp_129_:
{
lean_object* v___x_132_; lean_object* v_a_134_; lean_object* v_a_141_; 
v___x_132_ = lean_box(0);
v_a_141_ = lean_array_uget_borrowed(v_as_117_, v_i_119_);
if (lean_obj_tag(v_a_141_) == 0)
{
v_a_134_ = v_snd_128_;
goto v___jp_133_;
}
else
{
lean_object* v_val_142_; uint32_t v___y_144_; lean_object* v___x_160_; 
v_val_142_ = lean_ctor_get(v_a_141_, 0);
v___x_160_ = l_Lean_LocalDecl_userName(v_val_142_);
if (lean_obj_tag(v___x_160_) == 1)
{
lean_object* v_str_161_; lean_object* v___x_162_; lean_object* v___x_163_; uint8_t v___x_164_; 
v_str_161_ = lean_ctor_get(v___x_160_, 1);
lean_inc_ref(v_str_161_);
lean_dec_ref_known(v___x_160_, 2);
v___x_162_ = lean_string_utf8_byte_size(v_str_161_);
v___x_163_ = lean_unsigned_to_nat(0u);
v___x_164_ = lean_nat_dec_eq(v___x_162_, v___x_163_);
if (v___x_164_ == 0)
{
if (v___x_116_ == 0)
{
lean_dec_ref(v_str_161_);
v_a_134_ = v_snd_128_;
goto v___jp_133_;
}
else
{
lean_object* v___x_165_; lean_object* v___x_166_; 
v___x_165_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_165_, 0, v_str_161_);
lean_ctor_set(v___x_165_, 1, v___x_163_);
lean_ctor_set(v___x_165_, 2, v___x_162_);
v___x_166_ = l_String_Slice_Pos_get_x3f(v___x_165_, v___x_163_);
lean_dec_ref_known(v___x_165_, 3);
if (lean_obj_tag(v___x_166_) == 0)
{
uint32_t v___x_167_; 
v___x_167_ = 65;
v___y_144_ = v___x_167_;
goto v___jp_143_;
}
else
{
lean_object* v_val_168_; uint32_t v___x_169_; 
v_val_168_ = lean_ctor_get(v___x_166_, 0);
lean_inc(v_val_168_);
lean_dec_ref_known(v___x_166_, 1);
v___x_169_ = lean_unbox_uint32(v_val_168_);
lean_dec(v_val_168_);
v___y_144_ = v___x_169_;
goto v___jp_143_;
}
}
}
else
{
lean_dec_ref(v_str_161_);
v_a_134_ = v_snd_128_;
goto v___jp_133_;
}
}
else
{
lean_dec(v___x_160_);
v_a_134_ = v_snd_128_;
goto v___jp_133_;
}
v___jp_143_:
{
uint32_t v___x_145_; uint8_t v___x_146_; 
v___x_145_ = 95;
v___x_146_ = lean_uint32_dec_eq(v___y_144_, v___x_145_);
if (v___x_146_ == 0)
{
v_a_134_ = v_snd_128_;
goto v___jp_133_;
}
else
{
lean_object* v___x_147_; lean_object* v___x_148_; 
v___x_147_ = l_Lean_LocalDecl_type(v_val_142_);
v___x_148_ = l_Lean_Meta_isClass_x3f(v___x_147_, v___y_121_, v___y_122_, v___y_123_, v___y_124_);
if (lean_obj_tag(v___x_148_) == 0)
{
lean_object* v_a_149_; 
v_a_149_ = lean_ctor_get(v___x_148_, 0);
lean_inc(v_a_149_);
lean_dec_ref_known(v___x_148_, 1);
if (lean_obj_tag(v_a_149_) == 0)
{
lean_object* v___x_150_; lean_object* v___x_151_; 
v___x_150_ = l_Lean_LocalDecl_fvarId(v_val_142_);
v___x_151_ = lean_array_push(v_snd_128_, v___x_150_);
v_a_134_ = v___x_151_;
goto v___jp_133_;
}
else
{
lean_dec(v_a_149_);
v_a_134_ = v_snd_128_;
goto v___jp_133_;
}
}
else
{
lean_object* v_a_152_; lean_object* v___x_154_; uint8_t v_isShared_155_; uint8_t v_isSharedCheck_159_; 
lean_del_object(v___x_130_);
lean_dec(v_snd_128_);
v_a_152_ = lean_ctor_get(v___x_148_, 0);
v_isSharedCheck_159_ = !lean_is_exclusive(v___x_148_);
if (v_isSharedCheck_159_ == 0)
{
v___x_154_ = v___x_148_;
v_isShared_155_ = v_isSharedCheck_159_;
goto v_resetjp_153_;
}
else
{
lean_inc(v_a_152_);
lean_dec(v___x_148_);
v___x_154_ = lean_box(0);
v_isShared_155_ = v_isSharedCheck_159_;
goto v_resetjp_153_;
}
v_resetjp_153_:
{
lean_object* v___x_157_; 
if (v_isShared_155_ == 0)
{
v___x_157_ = v___x_154_;
goto v_reusejp_156_;
}
else
{
lean_object* v_reuseFailAlloc_158_; 
v_reuseFailAlloc_158_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_158_, 0, v_a_152_);
v___x_157_ = v_reuseFailAlloc_158_;
goto v_reusejp_156_;
}
v_reusejp_156_:
{
return v___x_157_;
}
}
}
}
}
}
v___jp_133_:
{
lean_object* v___x_136_; 
if (v_isShared_131_ == 0)
{
lean_ctor_set(v___x_130_, 1, v_a_134_);
lean_ctor_set(v___x_130_, 0, v___x_132_);
v___x_136_ = v___x_130_;
goto v_reusejp_135_;
}
else
{
lean_object* v_reuseFailAlloc_140_; 
v_reuseFailAlloc_140_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_140_, 0, v___x_132_);
lean_ctor_set(v_reuseFailAlloc_140_, 1, v_a_134_);
v___x_136_ = v_reuseFailAlloc_140_;
goto v_reusejp_135_;
}
v_reusejp_135_:
{
size_t v___x_137_; size_t v___x_138_; lean_object* v___x_139_; 
v___x_137_ = ((size_t)1ULL);
v___x_138_ = lean_usize_add(v_i_119_, v___x_137_);
v___x_139_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1_spec__0_spec__0_spec__3_spec__4(v___x_116_, v_as_117_, v_sz_118_, v___x_138_, v___x_136_, v___y_121_, v___y_122_, v___y_123_, v___y_124_);
return v___x_139_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1_spec__0_spec__0_spec__3___boxed(lean_object* v___x_172_, lean_object* v_as_173_, lean_object* v_sz_174_, lean_object* v_i_175_, lean_object* v_b_176_, lean_object* v___y_177_, lean_object* v___y_178_, lean_object* v___y_179_, lean_object* v___y_180_, lean_object* v___y_181_){
_start:
{
uint8_t v___x_5789__boxed_182_; size_t v_sz_boxed_183_; size_t v_i_boxed_184_; lean_object* v_res_185_; 
v___x_5789__boxed_182_ = lean_unbox(v___x_172_);
v_sz_boxed_183_ = lean_unbox_usize(v_sz_174_);
lean_dec(v_sz_174_);
v_i_boxed_184_ = lean_unbox_usize(v_i_175_);
lean_dec(v_i_175_);
v_res_185_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1_spec__0_spec__0_spec__3(v___x_5789__boxed_182_, v_as_173_, v_sz_boxed_183_, v_i_boxed_184_, v_b_176_, v___y_177_, v___y_178_, v___y_179_, v___y_180_);
lean_dec(v___y_180_);
lean_dec_ref(v___y_179_);
lean_dec(v___y_178_);
lean_dec_ref(v___y_177_);
lean_dec_ref(v_as_173_);
return v_res_185_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1_spec__0_spec__0(lean_object* v_init_186_, uint8_t v___x_187_, lean_object* v_n_188_, lean_object* v_b_189_, lean_object* v___y_190_, lean_object* v___y_191_, lean_object* v___y_192_, lean_object* v___y_193_){
_start:
{
if (lean_obj_tag(v_n_188_) == 0)
{
lean_object* v_cs_195_; lean_object* v___x_196_; lean_object* v___x_197_; size_t v_sz_198_; size_t v___x_199_; lean_object* v___x_200_; 
v_cs_195_ = lean_ctor_get(v_n_188_, 0);
v___x_196_ = lean_box(0);
v___x_197_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_197_, 0, v___x_196_);
lean_ctor_set(v___x_197_, 1, v_b_189_);
v_sz_198_ = lean_array_size(v_cs_195_);
v___x_199_ = ((size_t)0ULL);
v___x_200_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1_spec__0_spec__0_spec__2(v_init_186_, v___x_187_, v_cs_195_, v_sz_198_, v___x_199_, v___x_197_, v___y_190_, v___y_191_, v___y_192_, v___y_193_);
if (lean_obj_tag(v___x_200_) == 0)
{
lean_object* v_a_201_; lean_object* v___x_203_; uint8_t v_isShared_204_; uint8_t v_isSharedCheck_215_; 
v_a_201_ = lean_ctor_get(v___x_200_, 0);
v_isSharedCheck_215_ = !lean_is_exclusive(v___x_200_);
if (v_isSharedCheck_215_ == 0)
{
v___x_203_ = v___x_200_;
v_isShared_204_ = v_isSharedCheck_215_;
goto v_resetjp_202_;
}
else
{
lean_inc(v_a_201_);
lean_dec(v___x_200_);
v___x_203_ = lean_box(0);
v_isShared_204_ = v_isSharedCheck_215_;
goto v_resetjp_202_;
}
v_resetjp_202_:
{
lean_object* v_fst_205_; 
v_fst_205_ = lean_ctor_get(v_a_201_, 0);
if (lean_obj_tag(v_fst_205_) == 0)
{
lean_object* v_snd_206_; lean_object* v___x_207_; lean_object* v___x_209_; 
v_snd_206_ = lean_ctor_get(v_a_201_, 1);
lean_inc(v_snd_206_);
lean_dec(v_a_201_);
v___x_207_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_207_, 0, v_snd_206_);
if (v_isShared_204_ == 0)
{
lean_ctor_set(v___x_203_, 0, v___x_207_);
v___x_209_ = v___x_203_;
goto v_reusejp_208_;
}
else
{
lean_object* v_reuseFailAlloc_210_; 
v_reuseFailAlloc_210_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_210_, 0, v___x_207_);
v___x_209_ = v_reuseFailAlloc_210_;
goto v_reusejp_208_;
}
v_reusejp_208_:
{
return v___x_209_;
}
}
else
{
lean_object* v_val_211_; lean_object* v___x_213_; 
lean_inc_ref(v_fst_205_);
lean_dec(v_a_201_);
v_val_211_ = lean_ctor_get(v_fst_205_, 0);
lean_inc(v_val_211_);
lean_dec_ref_known(v_fst_205_, 1);
if (v_isShared_204_ == 0)
{
lean_ctor_set(v___x_203_, 0, v_val_211_);
v___x_213_ = v___x_203_;
goto v_reusejp_212_;
}
else
{
lean_object* v_reuseFailAlloc_214_; 
v_reuseFailAlloc_214_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_214_, 0, v_val_211_);
v___x_213_ = v_reuseFailAlloc_214_;
goto v_reusejp_212_;
}
v_reusejp_212_:
{
return v___x_213_;
}
}
}
}
else
{
lean_object* v_a_216_; lean_object* v___x_218_; uint8_t v_isShared_219_; uint8_t v_isSharedCheck_223_; 
v_a_216_ = lean_ctor_get(v___x_200_, 0);
v_isSharedCheck_223_ = !lean_is_exclusive(v___x_200_);
if (v_isSharedCheck_223_ == 0)
{
v___x_218_ = v___x_200_;
v_isShared_219_ = v_isSharedCheck_223_;
goto v_resetjp_217_;
}
else
{
lean_inc(v_a_216_);
lean_dec(v___x_200_);
v___x_218_ = lean_box(0);
v_isShared_219_ = v_isSharedCheck_223_;
goto v_resetjp_217_;
}
v_resetjp_217_:
{
lean_object* v___x_221_; 
if (v_isShared_219_ == 0)
{
v___x_221_ = v___x_218_;
goto v_reusejp_220_;
}
else
{
lean_object* v_reuseFailAlloc_222_; 
v_reuseFailAlloc_222_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_222_, 0, v_a_216_);
v___x_221_ = v_reuseFailAlloc_222_;
goto v_reusejp_220_;
}
v_reusejp_220_:
{
return v___x_221_;
}
}
}
}
else
{
lean_object* v_vs_224_; lean_object* v___x_225_; lean_object* v___x_226_; size_t v_sz_227_; size_t v___x_228_; lean_object* v___x_229_; 
v_vs_224_ = lean_ctor_get(v_n_188_, 0);
v___x_225_ = lean_box(0);
v___x_226_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_226_, 0, v___x_225_);
lean_ctor_set(v___x_226_, 1, v_b_189_);
v_sz_227_ = lean_array_size(v_vs_224_);
v___x_228_ = ((size_t)0ULL);
v___x_229_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1_spec__0_spec__0_spec__3(v___x_187_, v_vs_224_, v_sz_227_, v___x_228_, v___x_226_, v___y_190_, v___y_191_, v___y_192_, v___y_193_);
if (lean_obj_tag(v___x_229_) == 0)
{
lean_object* v_a_230_; lean_object* v___x_232_; uint8_t v_isShared_233_; uint8_t v_isSharedCheck_244_; 
v_a_230_ = lean_ctor_get(v___x_229_, 0);
v_isSharedCheck_244_ = !lean_is_exclusive(v___x_229_);
if (v_isSharedCheck_244_ == 0)
{
v___x_232_ = v___x_229_;
v_isShared_233_ = v_isSharedCheck_244_;
goto v_resetjp_231_;
}
else
{
lean_inc(v_a_230_);
lean_dec(v___x_229_);
v___x_232_ = lean_box(0);
v_isShared_233_ = v_isSharedCheck_244_;
goto v_resetjp_231_;
}
v_resetjp_231_:
{
lean_object* v_fst_234_; 
v_fst_234_ = lean_ctor_get(v_a_230_, 0);
if (lean_obj_tag(v_fst_234_) == 0)
{
lean_object* v_snd_235_; lean_object* v___x_236_; lean_object* v___x_238_; 
v_snd_235_ = lean_ctor_get(v_a_230_, 1);
lean_inc(v_snd_235_);
lean_dec(v_a_230_);
v___x_236_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_236_, 0, v_snd_235_);
if (v_isShared_233_ == 0)
{
lean_ctor_set(v___x_232_, 0, v___x_236_);
v___x_238_ = v___x_232_;
goto v_reusejp_237_;
}
else
{
lean_object* v_reuseFailAlloc_239_; 
v_reuseFailAlloc_239_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_239_, 0, v___x_236_);
v___x_238_ = v_reuseFailAlloc_239_;
goto v_reusejp_237_;
}
v_reusejp_237_:
{
return v___x_238_;
}
}
else
{
lean_object* v_val_240_; lean_object* v___x_242_; 
lean_inc_ref(v_fst_234_);
lean_dec(v_a_230_);
v_val_240_ = lean_ctor_get(v_fst_234_, 0);
lean_inc(v_val_240_);
lean_dec_ref_known(v_fst_234_, 1);
if (v_isShared_233_ == 0)
{
lean_ctor_set(v___x_232_, 0, v_val_240_);
v___x_242_ = v___x_232_;
goto v_reusejp_241_;
}
else
{
lean_object* v_reuseFailAlloc_243_; 
v_reuseFailAlloc_243_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_243_, 0, v_val_240_);
v___x_242_ = v_reuseFailAlloc_243_;
goto v_reusejp_241_;
}
v_reusejp_241_:
{
return v___x_242_;
}
}
}
}
else
{
lean_object* v_a_245_; lean_object* v___x_247_; uint8_t v_isShared_248_; uint8_t v_isSharedCheck_252_; 
v_a_245_ = lean_ctor_get(v___x_229_, 0);
v_isSharedCheck_252_ = !lean_is_exclusive(v___x_229_);
if (v_isSharedCheck_252_ == 0)
{
v___x_247_ = v___x_229_;
v_isShared_248_ = v_isSharedCheck_252_;
goto v_resetjp_246_;
}
else
{
lean_inc(v_a_245_);
lean_dec(v___x_229_);
v___x_247_ = lean_box(0);
v_isShared_248_ = v_isSharedCheck_252_;
goto v_resetjp_246_;
}
v_resetjp_246_:
{
lean_object* v___x_250_; 
if (v_isShared_248_ == 0)
{
v___x_250_ = v___x_247_;
goto v_reusejp_249_;
}
else
{
lean_object* v_reuseFailAlloc_251_; 
v_reuseFailAlloc_251_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_251_, 0, v_a_245_);
v___x_250_ = v_reuseFailAlloc_251_;
goto v_reusejp_249_;
}
v_reusejp_249_:
{
return v___x_250_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1_spec__0_spec__0_spec__2(lean_object* v_init_253_, uint8_t v___x_254_, lean_object* v_as_255_, size_t v_sz_256_, size_t v_i_257_, lean_object* v_b_258_, lean_object* v___y_259_, lean_object* v___y_260_, lean_object* v___y_261_, lean_object* v___y_262_){
_start:
{
uint8_t v___x_264_; 
v___x_264_ = lean_usize_dec_lt(v_i_257_, v_sz_256_);
if (v___x_264_ == 0)
{
lean_object* v___x_265_; 
v___x_265_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_265_, 0, v_b_258_);
return v___x_265_;
}
else
{
lean_object* v_snd_266_; lean_object* v___x_268_; uint8_t v_isShared_269_; uint8_t v_isSharedCheck_300_; 
v_snd_266_ = lean_ctor_get(v_b_258_, 1);
v_isSharedCheck_300_ = !lean_is_exclusive(v_b_258_);
if (v_isSharedCheck_300_ == 0)
{
lean_object* v_unused_301_; 
v_unused_301_ = lean_ctor_get(v_b_258_, 0);
lean_dec(v_unused_301_);
v___x_268_ = v_b_258_;
v_isShared_269_ = v_isSharedCheck_300_;
goto v_resetjp_267_;
}
else
{
lean_inc(v_snd_266_);
lean_dec(v_b_258_);
v___x_268_ = lean_box(0);
v_isShared_269_ = v_isSharedCheck_300_;
goto v_resetjp_267_;
}
v_resetjp_267_:
{
lean_object* v_a_270_; lean_object* v___x_271_; 
v_a_270_ = lean_array_uget_borrowed(v_as_255_, v_i_257_);
lean_inc(v_snd_266_);
v___x_271_ = lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1_spec__0_spec__0(v_init_253_, v___x_254_, v_a_270_, v_snd_266_, v___y_259_, v___y_260_, v___y_261_, v___y_262_);
if (lean_obj_tag(v___x_271_) == 0)
{
lean_object* v_a_272_; lean_object* v___x_274_; uint8_t v_isShared_275_; uint8_t v_isSharedCheck_291_; 
v_a_272_ = lean_ctor_get(v___x_271_, 0);
v_isSharedCheck_291_ = !lean_is_exclusive(v___x_271_);
if (v_isSharedCheck_291_ == 0)
{
v___x_274_ = v___x_271_;
v_isShared_275_ = v_isSharedCheck_291_;
goto v_resetjp_273_;
}
else
{
lean_inc(v_a_272_);
lean_dec(v___x_271_);
v___x_274_ = lean_box(0);
v_isShared_275_ = v_isSharedCheck_291_;
goto v_resetjp_273_;
}
v_resetjp_273_:
{
if (lean_obj_tag(v_a_272_) == 0)
{
lean_object* v___x_276_; lean_object* v___x_278_; 
v___x_276_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_276_, 0, v_a_272_);
if (v_isShared_269_ == 0)
{
lean_ctor_set(v___x_268_, 0, v___x_276_);
v___x_278_ = v___x_268_;
goto v_reusejp_277_;
}
else
{
lean_object* v_reuseFailAlloc_282_; 
v_reuseFailAlloc_282_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_282_, 0, v___x_276_);
lean_ctor_set(v_reuseFailAlloc_282_, 1, v_snd_266_);
v___x_278_ = v_reuseFailAlloc_282_;
goto v_reusejp_277_;
}
v_reusejp_277_:
{
lean_object* v___x_280_; 
if (v_isShared_275_ == 0)
{
lean_ctor_set(v___x_274_, 0, v___x_278_);
v___x_280_ = v___x_274_;
goto v_reusejp_279_;
}
else
{
lean_object* v_reuseFailAlloc_281_; 
v_reuseFailAlloc_281_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_281_, 0, v___x_278_);
v___x_280_ = v_reuseFailAlloc_281_;
goto v_reusejp_279_;
}
v_reusejp_279_:
{
return v___x_280_;
}
}
}
else
{
lean_object* v_a_283_; lean_object* v___x_284_; lean_object* v___x_286_; 
lean_del_object(v___x_274_);
lean_dec(v_snd_266_);
v_a_283_ = lean_ctor_get(v_a_272_, 0);
lean_inc(v_a_283_);
lean_dec_ref_known(v_a_272_, 1);
v___x_284_ = lean_box(0);
if (v_isShared_269_ == 0)
{
lean_ctor_set(v___x_268_, 1, v_a_283_);
lean_ctor_set(v___x_268_, 0, v___x_284_);
v___x_286_ = v___x_268_;
goto v_reusejp_285_;
}
else
{
lean_object* v_reuseFailAlloc_290_; 
v_reuseFailAlloc_290_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_290_, 0, v___x_284_);
lean_ctor_set(v_reuseFailAlloc_290_, 1, v_a_283_);
v___x_286_ = v_reuseFailAlloc_290_;
goto v_reusejp_285_;
}
v_reusejp_285_:
{
size_t v___x_287_; size_t v___x_288_; 
v___x_287_ = ((size_t)1ULL);
v___x_288_ = lean_usize_add(v_i_257_, v___x_287_);
v_i_257_ = v___x_288_;
v_b_258_ = v___x_286_;
goto _start;
}
}
}
}
else
{
lean_object* v_a_292_; lean_object* v___x_294_; uint8_t v_isShared_295_; uint8_t v_isSharedCheck_299_; 
lean_del_object(v___x_268_);
lean_dec(v_snd_266_);
v_a_292_ = lean_ctor_get(v___x_271_, 0);
v_isSharedCheck_299_ = !lean_is_exclusive(v___x_271_);
if (v_isSharedCheck_299_ == 0)
{
v___x_294_ = v___x_271_;
v_isShared_295_ = v_isSharedCheck_299_;
goto v_resetjp_293_;
}
else
{
lean_inc(v_a_292_);
lean_dec(v___x_271_);
v___x_294_ = lean_box(0);
v_isShared_295_ = v_isSharedCheck_299_;
goto v_resetjp_293_;
}
v_resetjp_293_:
{
lean_object* v___x_297_; 
if (v_isShared_295_ == 0)
{
v___x_297_ = v___x_294_;
goto v_reusejp_296_;
}
else
{
lean_object* v_reuseFailAlloc_298_; 
v_reuseFailAlloc_298_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_298_, 0, v_a_292_);
v___x_297_ = v_reuseFailAlloc_298_;
goto v_reusejp_296_;
}
v_reusejp_296_:
{
return v___x_297_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1_spec__0_spec__0_spec__2___boxed(lean_object* v_init_302_, lean_object* v___x_303_, lean_object* v_as_304_, lean_object* v_sz_305_, lean_object* v_i_306_, lean_object* v_b_307_, lean_object* v___y_308_, lean_object* v___y_309_, lean_object* v___y_310_, lean_object* v___y_311_, lean_object* v___y_312_){
_start:
{
uint8_t v___x_5896__boxed_313_; size_t v_sz_boxed_314_; size_t v_i_boxed_315_; lean_object* v_res_316_; 
v___x_5896__boxed_313_ = lean_unbox(v___x_303_);
v_sz_boxed_314_ = lean_unbox_usize(v_sz_305_);
lean_dec(v_sz_305_);
v_i_boxed_315_ = lean_unbox_usize(v_i_306_);
lean_dec(v_i_306_);
v_res_316_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1_spec__0_spec__0_spec__2(v_init_302_, v___x_5896__boxed_313_, v_as_304_, v_sz_boxed_314_, v_i_boxed_315_, v_b_307_, v___y_308_, v___y_309_, v___y_310_, v___y_311_);
lean_dec(v___y_311_);
lean_dec_ref(v___y_310_);
lean_dec(v___y_309_);
lean_dec_ref(v___y_308_);
lean_dec_ref(v_as_304_);
lean_dec_ref(v_init_302_);
return v_res_316_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1_spec__0_spec__0___boxed(lean_object* v_init_317_, lean_object* v___x_318_, lean_object* v_n_319_, lean_object* v_b_320_, lean_object* v___y_321_, lean_object* v___y_322_, lean_object* v___y_323_, lean_object* v___y_324_, lean_object* v___y_325_){
_start:
{
uint8_t v___x_5918__boxed_326_; lean_object* v_res_327_; 
v___x_5918__boxed_326_ = lean_unbox(v___x_318_);
v_res_327_ = lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1_spec__0_spec__0(v_init_317_, v___x_5918__boxed_326_, v_n_319_, v_b_320_, v___y_321_, v___y_322_, v___y_323_, v___y_324_);
lean_dec(v___y_324_);
lean_dec_ref(v___y_323_);
lean_dec(v___y_322_);
lean_dec_ref(v___y_321_);
lean_dec_ref(v_n_319_);
lean_dec_ref(v_init_317_);
return v_res_327_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1_spec__0_spec__1_spec__5(uint8_t v___x_328_, lean_object* v_as_329_, size_t v_sz_330_, size_t v_i_331_, lean_object* v_b_332_, lean_object* v___y_333_, lean_object* v___y_334_, lean_object* v___y_335_, lean_object* v___y_336_){
_start:
{
uint8_t v___x_338_; 
v___x_338_ = lean_usize_dec_lt(v_i_331_, v_sz_330_);
if (v___x_338_ == 0)
{
lean_object* v___x_339_; 
v___x_339_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_339_, 0, v_b_332_);
return v___x_339_;
}
else
{
lean_object* v_snd_340_; lean_object* v___x_342_; uint8_t v_isShared_343_; uint8_t v_isSharedCheck_382_; 
v_snd_340_ = lean_ctor_get(v_b_332_, 1);
v_isSharedCheck_382_ = !lean_is_exclusive(v_b_332_);
if (v_isSharedCheck_382_ == 0)
{
lean_object* v_unused_383_; 
v_unused_383_ = lean_ctor_get(v_b_332_, 0);
lean_dec(v_unused_383_);
v___x_342_ = v_b_332_;
v_isShared_343_ = v_isSharedCheck_382_;
goto v_resetjp_341_;
}
else
{
lean_inc(v_snd_340_);
lean_dec(v_b_332_);
v___x_342_ = lean_box(0);
v_isShared_343_ = v_isSharedCheck_382_;
goto v_resetjp_341_;
}
v_resetjp_341_:
{
lean_object* v___x_344_; lean_object* v_a_346_; lean_object* v_a_353_; 
v___x_344_ = lean_box(0);
v_a_353_ = lean_array_uget_borrowed(v_as_329_, v_i_331_);
if (lean_obj_tag(v_a_353_) == 0)
{
v_a_346_ = v_snd_340_;
goto v___jp_345_;
}
else
{
lean_object* v_val_354_; uint32_t v___y_356_; lean_object* v___x_372_; 
v_val_354_ = lean_ctor_get(v_a_353_, 0);
v___x_372_ = l_Lean_LocalDecl_userName(v_val_354_);
if (lean_obj_tag(v___x_372_) == 1)
{
lean_object* v_str_373_; lean_object* v___x_374_; lean_object* v___x_375_; uint8_t v___x_376_; 
v_str_373_ = lean_ctor_get(v___x_372_, 1);
lean_inc_ref(v_str_373_);
lean_dec_ref_known(v___x_372_, 2);
v___x_374_ = lean_string_utf8_byte_size(v_str_373_);
v___x_375_ = lean_unsigned_to_nat(0u);
v___x_376_ = lean_nat_dec_eq(v___x_374_, v___x_375_);
if (v___x_376_ == 0)
{
if (v___x_328_ == 0)
{
lean_dec_ref(v_str_373_);
v_a_346_ = v_snd_340_;
goto v___jp_345_;
}
else
{
lean_object* v___x_377_; lean_object* v___x_378_; 
v___x_377_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_377_, 0, v_str_373_);
lean_ctor_set(v___x_377_, 1, v___x_375_);
lean_ctor_set(v___x_377_, 2, v___x_374_);
v___x_378_ = l_String_Slice_Pos_get_x3f(v___x_377_, v___x_375_);
lean_dec_ref_known(v___x_377_, 3);
if (lean_obj_tag(v___x_378_) == 0)
{
uint32_t v___x_379_; 
v___x_379_ = 65;
v___y_356_ = v___x_379_;
goto v___jp_355_;
}
else
{
lean_object* v_val_380_; uint32_t v___x_381_; 
v_val_380_ = lean_ctor_get(v___x_378_, 0);
lean_inc(v_val_380_);
lean_dec_ref_known(v___x_378_, 1);
v___x_381_ = lean_unbox_uint32(v_val_380_);
lean_dec(v_val_380_);
v___y_356_ = v___x_381_;
goto v___jp_355_;
}
}
}
else
{
lean_dec_ref(v_str_373_);
v_a_346_ = v_snd_340_;
goto v___jp_345_;
}
}
else
{
lean_dec(v___x_372_);
v_a_346_ = v_snd_340_;
goto v___jp_345_;
}
v___jp_355_:
{
uint32_t v___x_357_; uint8_t v___x_358_; 
v___x_357_ = 95;
v___x_358_ = lean_uint32_dec_eq(v___y_356_, v___x_357_);
if (v___x_358_ == 0)
{
v_a_346_ = v_snd_340_;
goto v___jp_345_;
}
else
{
lean_object* v___x_359_; lean_object* v___x_360_; 
v___x_359_ = l_Lean_LocalDecl_type(v_val_354_);
v___x_360_ = l_Lean_Meta_isClass_x3f(v___x_359_, v___y_333_, v___y_334_, v___y_335_, v___y_336_);
if (lean_obj_tag(v___x_360_) == 0)
{
lean_object* v_a_361_; 
v_a_361_ = lean_ctor_get(v___x_360_, 0);
lean_inc(v_a_361_);
lean_dec_ref_known(v___x_360_, 1);
if (lean_obj_tag(v_a_361_) == 0)
{
lean_object* v___x_362_; lean_object* v___x_363_; 
v___x_362_ = l_Lean_LocalDecl_fvarId(v_val_354_);
v___x_363_ = lean_array_push(v_snd_340_, v___x_362_);
v_a_346_ = v___x_363_;
goto v___jp_345_;
}
else
{
lean_dec(v_a_361_);
v_a_346_ = v_snd_340_;
goto v___jp_345_;
}
}
else
{
lean_object* v_a_364_; lean_object* v___x_366_; uint8_t v_isShared_367_; uint8_t v_isSharedCheck_371_; 
lean_del_object(v___x_342_);
lean_dec(v_snd_340_);
v_a_364_ = lean_ctor_get(v___x_360_, 0);
v_isSharedCheck_371_ = !lean_is_exclusive(v___x_360_);
if (v_isSharedCheck_371_ == 0)
{
v___x_366_ = v___x_360_;
v_isShared_367_ = v_isSharedCheck_371_;
goto v_resetjp_365_;
}
else
{
lean_inc(v_a_364_);
lean_dec(v___x_360_);
v___x_366_ = lean_box(0);
v_isShared_367_ = v_isSharedCheck_371_;
goto v_resetjp_365_;
}
v_resetjp_365_:
{
lean_object* v___x_369_; 
if (v_isShared_367_ == 0)
{
v___x_369_ = v___x_366_;
goto v_reusejp_368_;
}
else
{
lean_object* v_reuseFailAlloc_370_; 
v_reuseFailAlloc_370_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_370_, 0, v_a_364_);
v___x_369_ = v_reuseFailAlloc_370_;
goto v_reusejp_368_;
}
v_reusejp_368_:
{
return v___x_369_;
}
}
}
}
}
}
v___jp_345_:
{
lean_object* v___x_348_; 
if (v_isShared_343_ == 0)
{
lean_ctor_set(v___x_342_, 1, v_a_346_);
lean_ctor_set(v___x_342_, 0, v___x_344_);
v___x_348_ = v___x_342_;
goto v_reusejp_347_;
}
else
{
lean_object* v_reuseFailAlloc_352_; 
v_reuseFailAlloc_352_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_352_, 0, v___x_344_);
lean_ctor_set(v_reuseFailAlloc_352_, 1, v_a_346_);
v___x_348_ = v_reuseFailAlloc_352_;
goto v_reusejp_347_;
}
v_reusejp_347_:
{
size_t v___x_349_; size_t v___x_350_; 
v___x_349_ = ((size_t)1ULL);
v___x_350_ = lean_usize_add(v_i_331_, v___x_349_);
v_i_331_ = v___x_350_;
v_b_332_ = v___x_348_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1_spec__0_spec__1_spec__5___boxed(lean_object* v___x_384_, lean_object* v_as_385_, lean_object* v_sz_386_, lean_object* v_i_387_, lean_object* v_b_388_, lean_object* v___y_389_, lean_object* v___y_390_, lean_object* v___y_391_, lean_object* v___y_392_, lean_object* v___y_393_){
_start:
{
uint8_t v___x_6109__boxed_394_; size_t v_sz_boxed_395_; size_t v_i_boxed_396_; lean_object* v_res_397_; 
v___x_6109__boxed_394_ = lean_unbox(v___x_384_);
v_sz_boxed_395_ = lean_unbox_usize(v_sz_386_);
lean_dec(v_sz_386_);
v_i_boxed_396_ = lean_unbox_usize(v_i_387_);
lean_dec(v_i_387_);
v_res_397_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1_spec__0_spec__1_spec__5(v___x_6109__boxed_394_, v_as_385_, v_sz_boxed_395_, v_i_boxed_396_, v_b_388_, v___y_389_, v___y_390_, v___y_391_, v___y_392_);
lean_dec(v___y_392_);
lean_dec_ref(v___y_391_);
lean_dec(v___y_390_);
lean_dec_ref(v___y_389_);
lean_dec_ref(v_as_385_);
return v_res_397_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1_spec__0_spec__1(uint8_t v___x_398_, lean_object* v_as_399_, size_t v_sz_400_, size_t v_i_401_, lean_object* v_b_402_, lean_object* v___y_403_, lean_object* v___y_404_, lean_object* v___y_405_, lean_object* v___y_406_){
_start:
{
uint8_t v___x_408_; 
v___x_408_ = lean_usize_dec_lt(v_i_401_, v_sz_400_);
if (v___x_408_ == 0)
{
lean_object* v___x_409_; 
v___x_409_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_409_, 0, v_b_402_);
return v___x_409_;
}
else
{
lean_object* v_snd_410_; lean_object* v___x_412_; uint8_t v_isShared_413_; uint8_t v_isSharedCheck_452_; 
v_snd_410_ = lean_ctor_get(v_b_402_, 1);
v_isSharedCheck_452_ = !lean_is_exclusive(v_b_402_);
if (v_isSharedCheck_452_ == 0)
{
lean_object* v_unused_453_; 
v_unused_453_ = lean_ctor_get(v_b_402_, 0);
lean_dec(v_unused_453_);
v___x_412_ = v_b_402_;
v_isShared_413_ = v_isSharedCheck_452_;
goto v_resetjp_411_;
}
else
{
lean_inc(v_snd_410_);
lean_dec(v_b_402_);
v___x_412_ = lean_box(0);
v_isShared_413_ = v_isSharedCheck_452_;
goto v_resetjp_411_;
}
v_resetjp_411_:
{
lean_object* v___x_414_; lean_object* v_a_416_; lean_object* v_a_423_; 
v___x_414_ = lean_box(0);
v_a_423_ = lean_array_uget_borrowed(v_as_399_, v_i_401_);
if (lean_obj_tag(v_a_423_) == 0)
{
v_a_416_ = v_snd_410_;
goto v___jp_415_;
}
else
{
lean_object* v_val_424_; uint32_t v___y_426_; lean_object* v___x_442_; 
v_val_424_ = lean_ctor_get(v_a_423_, 0);
v___x_442_ = l_Lean_LocalDecl_userName(v_val_424_);
if (lean_obj_tag(v___x_442_) == 1)
{
lean_object* v_str_443_; lean_object* v___x_444_; lean_object* v___x_445_; uint8_t v___x_446_; 
v_str_443_ = lean_ctor_get(v___x_442_, 1);
lean_inc_ref(v_str_443_);
lean_dec_ref_known(v___x_442_, 2);
v___x_444_ = lean_string_utf8_byte_size(v_str_443_);
v___x_445_ = lean_unsigned_to_nat(0u);
v___x_446_ = lean_nat_dec_eq(v___x_444_, v___x_445_);
if (v___x_446_ == 0)
{
if (v___x_398_ == 0)
{
lean_dec_ref(v_str_443_);
v_a_416_ = v_snd_410_;
goto v___jp_415_;
}
else
{
lean_object* v___x_447_; lean_object* v___x_448_; 
v___x_447_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_447_, 0, v_str_443_);
lean_ctor_set(v___x_447_, 1, v___x_445_);
lean_ctor_set(v___x_447_, 2, v___x_444_);
v___x_448_ = l_String_Slice_Pos_get_x3f(v___x_447_, v___x_445_);
lean_dec_ref_known(v___x_447_, 3);
if (lean_obj_tag(v___x_448_) == 0)
{
uint32_t v___x_449_; 
v___x_449_ = 65;
v___y_426_ = v___x_449_;
goto v___jp_425_;
}
else
{
lean_object* v_val_450_; uint32_t v___x_451_; 
v_val_450_ = lean_ctor_get(v___x_448_, 0);
lean_inc(v_val_450_);
lean_dec_ref_known(v___x_448_, 1);
v___x_451_ = lean_unbox_uint32(v_val_450_);
lean_dec(v_val_450_);
v___y_426_ = v___x_451_;
goto v___jp_425_;
}
}
}
else
{
lean_dec_ref(v_str_443_);
v_a_416_ = v_snd_410_;
goto v___jp_415_;
}
}
else
{
lean_dec(v___x_442_);
v_a_416_ = v_snd_410_;
goto v___jp_415_;
}
v___jp_425_:
{
uint32_t v___x_427_; uint8_t v___x_428_; 
v___x_427_ = 95;
v___x_428_ = lean_uint32_dec_eq(v___y_426_, v___x_427_);
if (v___x_428_ == 0)
{
v_a_416_ = v_snd_410_;
goto v___jp_415_;
}
else
{
lean_object* v___x_429_; lean_object* v___x_430_; 
v___x_429_ = l_Lean_LocalDecl_type(v_val_424_);
v___x_430_ = l_Lean_Meta_isClass_x3f(v___x_429_, v___y_403_, v___y_404_, v___y_405_, v___y_406_);
if (lean_obj_tag(v___x_430_) == 0)
{
lean_object* v_a_431_; 
v_a_431_ = lean_ctor_get(v___x_430_, 0);
lean_inc(v_a_431_);
lean_dec_ref_known(v___x_430_, 1);
if (lean_obj_tag(v_a_431_) == 0)
{
lean_object* v___x_432_; lean_object* v___x_433_; 
v___x_432_ = l_Lean_LocalDecl_fvarId(v_val_424_);
v___x_433_ = lean_array_push(v_snd_410_, v___x_432_);
v_a_416_ = v___x_433_;
goto v___jp_415_;
}
else
{
lean_dec(v_a_431_);
v_a_416_ = v_snd_410_;
goto v___jp_415_;
}
}
else
{
lean_object* v_a_434_; lean_object* v___x_436_; uint8_t v_isShared_437_; uint8_t v_isSharedCheck_441_; 
lean_del_object(v___x_412_);
lean_dec(v_snd_410_);
v_a_434_ = lean_ctor_get(v___x_430_, 0);
v_isSharedCheck_441_ = !lean_is_exclusive(v___x_430_);
if (v_isSharedCheck_441_ == 0)
{
v___x_436_ = v___x_430_;
v_isShared_437_ = v_isSharedCheck_441_;
goto v_resetjp_435_;
}
else
{
lean_inc(v_a_434_);
lean_dec(v___x_430_);
v___x_436_ = lean_box(0);
v_isShared_437_ = v_isSharedCheck_441_;
goto v_resetjp_435_;
}
v_resetjp_435_:
{
lean_object* v___x_439_; 
if (v_isShared_437_ == 0)
{
v___x_439_ = v___x_436_;
goto v_reusejp_438_;
}
else
{
lean_object* v_reuseFailAlloc_440_; 
v_reuseFailAlloc_440_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_440_, 0, v_a_434_);
v___x_439_ = v_reuseFailAlloc_440_;
goto v_reusejp_438_;
}
v_reusejp_438_:
{
return v___x_439_;
}
}
}
}
}
}
v___jp_415_:
{
lean_object* v___x_418_; 
if (v_isShared_413_ == 0)
{
lean_ctor_set(v___x_412_, 1, v_a_416_);
lean_ctor_set(v___x_412_, 0, v___x_414_);
v___x_418_ = v___x_412_;
goto v_reusejp_417_;
}
else
{
lean_object* v_reuseFailAlloc_422_; 
v_reuseFailAlloc_422_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_422_, 0, v___x_414_);
lean_ctor_set(v_reuseFailAlloc_422_, 1, v_a_416_);
v___x_418_ = v_reuseFailAlloc_422_;
goto v_reusejp_417_;
}
v_reusejp_417_:
{
size_t v___x_419_; size_t v___x_420_; lean_object* v___x_421_; 
v___x_419_ = ((size_t)1ULL);
v___x_420_ = lean_usize_add(v_i_401_, v___x_419_);
v___x_421_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1_spec__0_spec__1_spec__5(v___x_398_, v_as_399_, v_sz_400_, v___x_420_, v___x_418_, v___y_403_, v___y_404_, v___y_405_, v___y_406_);
return v___x_421_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1_spec__0_spec__1___boxed(lean_object* v___x_454_, lean_object* v_as_455_, lean_object* v_sz_456_, lean_object* v_i_457_, lean_object* v_b_458_, lean_object* v___y_459_, lean_object* v___y_460_, lean_object* v___y_461_, lean_object* v___y_462_, lean_object* v___y_463_){
_start:
{
uint8_t v___x_6216__boxed_464_; size_t v_sz_boxed_465_; size_t v_i_boxed_466_; lean_object* v_res_467_; 
v___x_6216__boxed_464_ = lean_unbox(v___x_454_);
v_sz_boxed_465_ = lean_unbox_usize(v_sz_456_);
lean_dec(v_sz_456_);
v_i_boxed_466_ = lean_unbox_usize(v_i_457_);
lean_dec(v_i_457_);
v_res_467_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1_spec__0_spec__1(v___x_6216__boxed_464_, v_as_455_, v_sz_boxed_465_, v_i_boxed_466_, v_b_458_, v___y_459_, v___y_460_, v___y_461_, v___y_462_);
lean_dec(v___y_462_);
lean_dec_ref(v___y_461_);
lean_dec(v___y_460_);
lean_dec_ref(v___y_459_);
lean_dec_ref(v_as_455_);
return v_res_467_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1_spec__0(uint8_t v___x_468_, lean_object* v_t_469_, lean_object* v_init_470_, lean_object* v___y_471_, lean_object* v___y_472_, lean_object* v___y_473_, lean_object* v___y_474_){
_start:
{
lean_object* v_root_476_; lean_object* v_tail_477_; lean_object* v___x_478_; 
v_root_476_ = lean_ctor_get(v_t_469_, 0);
v_tail_477_ = lean_ctor_get(v_t_469_, 1);
lean_inc_ref(v_init_470_);
v___x_478_ = lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1_spec__0_spec__0(v_init_470_, v___x_468_, v_root_476_, v_init_470_, v___y_471_, v___y_472_, v___y_473_, v___y_474_);
lean_dec_ref(v_init_470_);
if (lean_obj_tag(v___x_478_) == 0)
{
lean_object* v_a_479_; lean_object* v___x_481_; uint8_t v_isShared_482_; uint8_t v_isSharedCheck_515_; 
v_a_479_ = lean_ctor_get(v___x_478_, 0);
v_isSharedCheck_515_ = !lean_is_exclusive(v___x_478_);
if (v_isSharedCheck_515_ == 0)
{
v___x_481_ = v___x_478_;
v_isShared_482_ = v_isSharedCheck_515_;
goto v_resetjp_480_;
}
else
{
lean_inc(v_a_479_);
lean_dec(v___x_478_);
v___x_481_ = lean_box(0);
v_isShared_482_ = v_isSharedCheck_515_;
goto v_resetjp_480_;
}
v_resetjp_480_:
{
if (lean_obj_tag(v_a_479_) == 0)
{
lean_object* v_a_483_; lean_object* v___x_485_; 
v_a_483_ = lean_ctor_get(v_a_479_, 0);
lean_inc(v_a_483_);
lean_dec_ref_known(v_a_479_, 1);
if (v_isShared_482_ == 0)
{
lean_ctor_set(v___x_481_, 0, v_a_483_);
v___x_485_ = v___x_481_;
goto v_reusejp_484_;
}
else
{
lean_object* v_reuseFailAlloc_486_; 
v_reuseFailAlloc_486_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_486_, 0, v_a_483_);
v___x_485_ = v_reuseFailAlloc_486_;
goto v_reusejp_484_;
}
v_reusejp_484_:
{
return v___x_485_;
}
}
else
{
lean_object* v_a_487_; lean_object* v___x_488_; lean_object* v___x_489_; size_t v_sz_490_; size_t v___x_491_; lean_object* v___x_492_; 
lean_del_object(v___x_481_);
v_a_487_ = lean_ctor_get(v_a_479_, 0);
lean_inc(v_a_487_);
lean_dec_ref_known(v_a_479_, 1);
v___x_488_ = lean_box(0);
v___x_489_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_489_, 0, v___x_488_);
lean_ctor_set(v___x_489_, 1, v_a_487_);
v_sz_490_ = lean_array_size(v_tail_477_);
v___x_491_ = ((size_t)0ULL);
v___x_492_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1_spec__0_spec__1(v___x_468_, v_tail_477_, v_sz_490_, v___x_491_, v___x_489_, v___y_471_, v___y_472_, v___y_473_, v___y_474_);
if (lean_obj_tag(v___x_492_) == 0)
{
lean_object* v_a_493_; lean_object* v___x_495_; uint8_t v_isShared_496_; uint8_t v_isSharedCheck_506_; 
v_a_493_ = lean_ctor_get(v___x_492_, 0);
v_isSharedCheck_506_ = !lean_is_exclusive(v___x_492_);
if (v_isSharedCheck_506_ == 0)
{
v___x_495_ = v___x_492_;
v_isShared_496_ = v_isSharedCheck_506_;
goto v_resetjp_494_;
}
else
{
lean_inc(v_a_493_);
lean_dec(v___x_492_);
v___x_495_ = lean_box(0);
v_isShared_496_ = v_isSharedCheck_506_;
goto v_resetjp_494_;
}
v_resetjp_494_:
{
lean_object* v_fst_497_; 
v_fst_497_ = lean_ctor_get(v_a_493_, 0);
if (lean_obj_tag(v_fst_497_) == 0)
{
lean_object* v_snd_498_; lean_object* v___x_500_; 
v_snd_498_ = lean_ctor_get(v_a_493_, 1);
lean_inc(v_snd_498_);
lean_dec(v_a_493_);
if (v_isShared_496_ == 0)
{
lean_ctor_set(v___x_495_, 0, v_snd_498_);
v___x_500_ = v___x_495_;
goto v_reusejp_499_;
}
else
{
lean_object* v_reuseFailAlloc_501_; 
v_reuseFailAlloc_501_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_501_, 0, v_snd_498_);
v___x_500_ = v_reuseFailAlloc_501_;
goto v_reusejp_499_;
}
v_reusejp_499_:
{
return v___x_500_;
}
}
else
{
lean_object* v_val_502_; lean_object* v___x_504_; 
lean_inc_ref(v_fst_497_);
lean_dec(v_a_493_);
v_val_502_ = lean_ctor_get(v_fst_497_, 0);
lean_inc(v_val_502_);
lean_dec_ref_known(v_fst_497_, 1);
if (v_isShared_496_ == 0)
{
lean_ctor_set(v___x_495_, 0, v_val_502_);
v___x_504_ = v___x_495_;
goto v_reusejp_503_;
}
else
{
lean_object* v_reuseFailAlloc_505_; 
v_reuseFailAlloc_505_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_505_, 0, v_val_502_);
v___x_504_ = v_reuseFailAlloc_505_;
goto v_reusejp_503_;
}
v_reusejp_503_:
{
return v___x_504_;
}
}
}
}
else
{
lean_object* v_a_507_; lean_object* v___x_509_; uint8_t v_isShared_510_; uint8_t v_isSharedCheck_514_; 
v_a_507_ = lean_ctor_get(v___x_492_, 0);
v_isSharedCheck_514_ = !lean_is_exclusive(v___x_492_);
if (v_isSharedCheck_514_ == 0)
{
v___x_509_ = v___x_492_;
v_isShared_510_ = v_isSharedCheck_514_;
goto v_resetjp_508_;
}
else
{
lean_inc(v_a_507_);
lean_dec(v___x_492_);
v___x_509_ = lean_box(0);
v_isShared_510_ = v_isSharedCheck_514_;
goto v_resetjp_508_;
}
v_resetjp_508_:
{
lean_object* v___x_512_; 
if (v_isShared_510_ == 0)
{
v___x_512_ = v___x_509_;
goto v_reusejp_511_;
}
else
{
lean_object* v_reuseFailAlloc_513_; 
v_reuseFailAlloc_513_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_513_, 0, v_a_507_);
v___x_512_ = v_reuseFailAlloc_513_;
goto v_reusejp_511_;
}
v_reusejp_511_:
{
return v___x_512_;
}
}
}
}
}
}
else
{
lean_object* v_a_516_; lean_object* v___x_518_; uint8_t v_isShared_519_; uint8_t v_isSharedCheck_523_; 
v_a_516_ = lean_ctor_get(v___x_478_, 0);
v_isSharedCheck_523_ = !lean_is_exclusive(v___x_478_);
if (v_isSharedCheck_523_ == 0)
{
v___x_518_ = v___x_478_;
v_isShared_519_ = v_isSharedCheck_523_;
goto v_resetjp_517_;
}
else
{
lean_inc(v_a_516_);
lean_dec(v___x_478_);
v___x_518_ = lean_box(0);
v_isShared_519_ = v_isSharedCheck_523_;
goto v_resetjp_517_;
}
v_resetjp_517_:
{
lean_object* v___x_521_; 
if (v_isShared_519_ == 0)
{
v___x_521_ = v___x_518_;
goto v_reusejp_520_;
}
else
{
lean_object* v_reuseFailAlloc_522_; 
v_reuseFailAlloc_522_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_522_, 0, v_a_516_);
v___x_521_ = v_reuseFailAlloc_522_;
goto v_reusejp_520_;
}
v_reusejp_520_:
{
return v___x_521_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1_spec__0___boxed(lean_object* v___x_524_, lean_object* v_t_525_, lean_object* v_init_526_, lean_object* v___y_527_, lean_object* v___y_528_, lean_object* v___y_529_, lean_object* v___y_530_, lean_object* v___y_531_){
_start:
{
uint8_t v___x_6323__boxed_532_; lean_object* v_res_533_; 
v___x_6323__boxed_532_ = lean_unbox(v___x_524_);
v_res_533_ = lp_mathlib_Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1_spec__0(v___x_6323__boxed_532_, v_t_525_, v_init_526_, v___y_527_, v___y_528_, v___y_529_, v___y_530_);
lean_dec(v___y_530_);
lean_dec_ref(v___y_529_);
lean_dec(v___y_528_);
lean_dec_ref(v___y_527_);
lean_dec_ref(v_t_525_);
return v_res_533_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1___lam__0(uint8_t v___x_536_, lean_object* v___y_537_, lean_object* v___y_538_, lean_object* v___y_539_, lean_object* v___y_540_, lean_object* v___y_541_, lean_object* v___y_542_, lean_object* v___y_543_, lean_object* v___y_544_){
_start:
{
lean_object* v___x_546_; 
v___x_546_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_538_, v___y_541_, v___y_542_, v___y_543_, v___y_544_);
if (lean_obj_tag(v___x_546_) == 0)
{
lean_object* v_lctx_547_; lean_object* v_a_548_; lean_object* v_decls_549_; lean_object* v_toClear_550_; lean_object* v___x_551_; 
v_lctx_547_ = lean_ctor_get(v___y_541_, 2);
v_a_548_ = lean_ctor_get(v___x_546_, 0);
lean_inc(v_a_548_);
lean_dec_ref_known(v___x_546_, 1);
v_decls_549_ = lean_ctor_get(v_lctx_547_, 1);
v_toClear_550_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1___lam__0___closed__0));
v___x_551_ = lp_mathlib_Lean_PersistentArray_forIn___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1_spec__0(v___x_536_, v_decls_549_, v_toClear_550_, v___y_541_, v___y_542_, v___y_543_, v___y_544_);
if (lean_obj_tag(v___x_551_) == 0)
{
lean_object* v_a_552_; lean_object* v___x_553_; 
v_a_552_ = lean_ctor_get(v___x_551_, 0);
lean_inc(v_a_552_);
lean_dec_ref_known(v___x_551_, 1);
v___x_553_ = l_Lean_MVarId_tryClearMany(v_a_548_, v_a_552_, v___y_541_, v___y_542_, v___y_543_, v___y_544_);
lean_dec(v_a_552_);
if (lean_obj_tag(v___x_553_) == 0)
{
lean_object* v_a_554_; lean_object* v___x_555_; lean_object* v___x_556_; lean_object* v___x_557_; 
v_a_554_ = lean_ctor_get(v___x_553_, 0);
lean_inc(v_a_554_);
lean_dec_ref_known(v___x_553_, 1);
v___x_555_ = lean_box(0);
v___x_556_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_556_, 0, v_a_554_);
lean_ctor_set(v___x_556_, 1, v___x_555_);
v___x_557_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v___x_556_, v___y_538_, v___y_541_, v___y_542_, v___y_543_, v___y_544_);
return v___x_557_;
}
else
{
lean_object* v_a_558_; lean_object* v___x_560_; uint8_t v_isShared_561_; uint8_t v_isSharedCheck_565_; 
v_a_558_ = lean_ctor_get(v___x_553_, 0);
v_isSharedCheck_565_ = !lean_is_exclusive(v___x_553_);
if (v_isSharedCheck_565_ == 0)
{
v___x_560_ = v___x_553_;
v_isShared_561_ = v_isSharedCheck_565_;
goto v_resetjp_559_;
}
else
{
lean_inc(v_a_558_);
lean_dec(v___x_553_);
v___x_560_ = lean_box(0);
v_isShared_561_ = v_isSharedCheck_565_;
goto v_resetjp_559_;
}
v_resetjp_559_:
{
lean_object* v___x_563_; 
if (v_isShared_561_ == 0)
{
v___x_563_ = v___x_560_;
goto v_reusejp_562_;
}
else
{
lean_object* v_reuseFailAlloc_564_; 
v_reuseFailAlloc_564_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_564_, 0, v_a_558_);
v___x_563_ = v_reuseFailAlloc_564_;
goto v_reusejp_562_;
}
v_reusejp_562_:
{
return v___x_563_;
}
}
}
}
else
{
lean_object* v_a_566_; lean_object* v___x_568_; uint8_t v_isShared_569_; uint8_t v_isSharedCheck_573_; 
lean_dec(v_a_548_);
v_a_566_ = lean_ctor_get(v___x_551_, 0);
v_isSharedCheck_573_ = !lean_is_exclusive(v___x_551_);
if (v_isSharedCheck_573_ == 0)
{
v___x_568_ = v___x_551_;
v_isShared_569_ = v_isSharedCheck_573_;
goto v_resetjp_567_;
}
else
{
lean_inc(v_a_566_);
lean_dec(v___x_551_);
v___x_568_ = lean_box(0);
v_isShared_569_ = v_isSharedCheck_573_;
goto v_resetjp_567_;
}
v_resetjp_567_:
{
lean_object* v___x_571_; 
if (v_isShared_569_ == 0)
{
v___x_571_ = v___x_568_;
goto v_reusejp_570_;
}
else
{
lean_object* v_reuseFailAlloc_572_; 
v_reuseFailAlloc_572_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_572_, 0, v_a_566_);
v___x_571_ = v_reuseFailAlloc_572_;
goto v_reusejp_570_;
}
v_reusejp_570_:
{
return v___x_571_;
}
}
}
}
else
{
lean_object* v_a_574_; lean_object* v___x_576_; uint8_t v_isShared_577_; uint8_t v_isSharedCheck_581_; 
v_a_574_ = lean_ctor_get(v___x_546_, 0);
v_isSharedCheck_581_ = !lean_is_exclusive(v___x_546_);
if (v_isSharedCheck_581_ == 0)
{
v___x_576_ = v___x_546_;
v_isShared_577_ = v_isSharedCheck_581_;
goto v_resetjp_575_;
}
else
{
lean_inc(v_a_574_);
lean_dec(v___x_546_);
v___x_576_ = lean_box(0);
v_isShared_577_ = v_isSharedCheck_581_;
goto v_resetjp_575_;
}
v_resetjp_575_:
{
lean_object* v___x_579_; 
if (v_isShared_577_ == 0)
{
v___x_579_ = v___x_576_;
goto v_reusejp_578_;
}
else
{
lean_object* v_reuseFailAlloc_580_; 
v_reuseFailAlloc_580_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_580_, 0, v_a_574_);
v___x_579_ = v_reuseFailAlloc_580_;
goto v_reusejp_578_;
}
v_reusejp_578_:
{
return v___x_579_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1___lam__0___boxed(lean_object* v___x_582_, lean_object* v___y_583_, lean_object* v___y_584_, lean_object* v___y_585_, lean_object* v___y_586_, lean_object* v___y_587_, lean_object* v___y_588_, lean_object* v___y_589_, lean_object* v___y_590_, lean_object* v___y_591_){
_start:
{
uint8_t v___x_6429__boxed_592_; lean_object* v_res_593_; 
v___x_6429__boxed_592_ = lean_unbox(v___x_582_);
v_res_593_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1___lam__0(v___x_6429__boxed_592_, v___y_583_, v___y_584_, v___y_585_, v___y_586_, v___y_587_, v___y_588_, v___y_589_, v___y_590_);
lean_dec(v___y_590_);
lean_dec_ref(v___y_589_);
lean_dec(v___y_588_);
lean_dec_ref(v___y_587_);
lean_dec(v___y_586_);
lean_dec_ref(v___y_585_);
lean_dec(v___y_584_);
lean_dec_ref(v___y_583_);
return v_res_593_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1(lean_object* v_x_594_, lean_object* v_a_595_, lean_object* v_a_596_, lean_object* v_a_597_, lean_object* v_a_598_, lean_object* v_a_599_, lean_object* v_a_600_, lean_object* v_a_601_, lean_object* v_a_602_){
_start:
{
lean_object* v___x_604_; uint8_t v___x_605_; 
v___x_604_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_clear___00__closed__3));
v___x_605_ = l_Lean_Syntax_isOfKind(v_x_594_, v___x_604_);
if (v___x_605_ == 0)
{
lean_object* v___x_606_; 
v___x_606_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1_spec__1___redArg();
return v___x_606_;
}
else
{
lean_object* v___x_607_; lean_object* v___f_608_; lean_object* v___x_609_; 
v___x_607_ = lean_box(v___x_605_);
v___f_608_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1___lam__0___boxed), 10, 1);
lean_closure_set(v___f_608_, 0, v___x_607_);
v___x_609_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_608_, v_a_595_, v_a_596_, v_a_597_, v_a_598_, v_a_599_, v_a_600_, v_a_601_, v_a_602_);
return v___x_609_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1___boxed(lean_object* v_x_610_, lean_object* v_a_611_, lean_object* v_a_612_, lean_object* v_a_613_, lean_object* v_a_614_, lean_object* v_a_615_, lean_object* v_a_616_, lean_object* v_a_617_, lean_object* v_a_618_, lean_object* v_a_619_){
_start:
{
lean_object* v_res_620_; 
v_res_620_ = lp_mathlib_Mathlib_Tactic___aux__Mathlib__Tactic__Clear________elabRules__Mathlib__Tactic__clear____1(v_x_610_, v_a_611_, v_a_612_, v_a_613_, v_a_614_, v_a_615_, v_a_616_, v_a_617_, v_a_618_);
lean_dec(v_a_618_);
lean_dec_ref(v_a_617_);
lean_dec(v_a_616_);
lean_dec_ref(v_a_615_);
lean_dec(v_a_614_);
lean_dec_ref(v_a_613_);
lean_dec(v_a_612_);
lean_dec_ref(v_a_611_);
return v_res_620_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Clear__(uint8_t builtin) {
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
lean_object* runtime_initialize_Lean_Meta_Tactic_Clear(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Tactic_Basic(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Clear__(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Tactic_Clear(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Tactic_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* initialize_Lean_Meta_Tactic_Clear(uint8_t builtin);
lean_object* initialize_Lean_Elab_Tactic_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Clear__(uint8_t builtin) {
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
res = initialize_Lean_Meta_Tactic_Clear(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Tactic_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Clear__(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Clear__(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Clear__(builtin);
}
#ifdef __cplusplus
}
#endif
