// Lean compiler output
// Module: Qq.MatchImpl
// Imports: public import Init public meta import Init public import Qq.MetaM
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
lean_object* l_Lean_Expr_const___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_Expr_app___override(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Expr_abstractM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkLambda(lean_object*, uint8_t, lean_object*, lean_object*);
lean_object* l_Lean_Expr_letE___override(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_mkConst(lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* l_Lean_Expr_fvar___override(lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Syntax_stripPos(lean_object*);
LEAN_EXPORT lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Syntax_stripPos_spec__0(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Syntax_stripPos_spec__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_Qq_Qq_Impl_PatVarDecl_fvarTy___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_Qq_Qq_Impl_PatVarDecl_fvarTy___closed__0 = (const lean_object*)&lp_Qq_Qq_Impl_PatVarDecl_fvarTy___closed__0_value;
static const lean_string_object lp_Qq_Qq_Impl_PatVarDecl_fvarTy___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Level"};
static const lean_object* lp_Qq_Qq_Impl_PatVarDecl_fvarTy___closed__1 = (const lean_object*)&lp_Qq_Qq_Impl_PatVarDecl_fvarTy___closed__1_value;
static const lean_ctor_object lp_Qq_Qq_Impl_PatVarDecl_fvarTy___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Qq_Impl_PatVarDecl_fvarTy___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_Qq_Qq_Impl_PatVarDecl_fvarTy___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl_PatVarDecl_fvarTy___closed__2_value_aux_0),((lean_object*)&lp_Qq_Qq_Impl_PatVarDecl_fvarTy___closed__1_value),LEAN_SCALAR_PTR_LITERAL(165, 140, 25, 163, 179, 32, 245, 95)}};
static const lean_object* lp_Qq_Qq_Impl_PatVarDecl_fvarTy___closed__2 = (const lean_object*)&lp_Qq_Qq_Impl_PatVarDecl_fvarTy___closed__2_value;
static lean_once_cell_t lp_Qq_Qq_Impl_PatVarDecl_fvarTy___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq_Impl_PatVarDecl_fvarTy___closed__3;
static const lean_string_object lp_Qq_Qq_Impl_PatVarDecl_fvarTy___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Expr"};
static const lean_object* lp_Qq_Qq_Impl_PatVarDecl_fvarTy___closed__4 = (const lean_object*)&lp_Qq_Qq_Impl_PatVarDecl_fvarTy___closed__4_value;
static const lean_ctor_object lp_Qq_Qq_Impl_PatVarDecl_fvarTy___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Qq_Impl_PatVarDecl_fvarTy___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_Qq_Qq_Impl_PatVarDecl_fvarTy___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl_PatVarDecl_fvarTy___closed__5_value_aux_0),((lean_object*)&lp_Qq_Qq_Impl_PatVarDecl_fvarTy___closed__4_value),LEAN_SCALAR_PTR_LITERAL(84, 208, 74, 211, 93, 83, 88, 82)}};
static const lean_object* lp_Qq_Qq_Impl_PatVarDecl_fvarTy___closed__5 = (const lean_object*)&lp_Qq_Qq_Impl_PatVarDecl_fvarTy___closed__5_value;
static lean_once_cell_t lp_Qq_Qq_Impl_PatVarDecl_fvarTy___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq_Impl_PatVarDecl_fvarTy___closed__6;
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_PatVarDecl_fvarTy(lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_PatVarDecl_fvarTy___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_PatVarDecl_fvar(lean_object*);
static const lean_string_object lp_Qq_Qq_Impl_mkIsDefEqType___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Bool"};
static const lean_object* lp_Qq_Qq_Impl_mkIsDefEqType___closed__0 = (const lean_object*)&lp_Qq_Qq_Impl_mkIsDefEqType___closed__0_value;
static const lean_ctor_object lp_Qq_Qq_Impl_mkIsDefEqType___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Qq_Impl_mkIsDefEqType___closed__0_value),LEAN_SCALAR_PTR_LITERAL(250, 44, 198, 216, 184, 195, 199, 178)}};
static const lean_object* lp_Qq_Qq_Impl_mkIsDefEqType___closed__1 = (const lean_object*)&lp_Qq_Qq_Impl_mkIsDefEqType___closed__1_value;
static lean_once_cell_t lp_Qq_Qq_Impl_mkIsDefEqType___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq_Impl_mkIsDefEqType___closed__2;
static const lean_string_object lp_Qq_Qq_Impl_mkIsDefEqType___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Prod"};
static const lean_object* lp_Qq_Qq_Impl_mkIsDefEqType___closed__3 = (const lean_object*)&lp_Qq_Qq_Impl_mkIsDefEqType___closed__3_value;
static const lean_ctor_object lp_Qq_Qq_Impl_mkIsDefEqType___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Qq_Impl_mkIsDefEqType___closed__3_value),LEAN_SCALAR_PTR_LITERAL(121, 119, 164, 206, 221, 118, 48, 212)}};
static const lean_object* lp_Qq_Qq_Impl_mkIsDefEqType___closed__4 = (const lean_object*)&lp_Qq_Qq_Impl_mkIsDefEqType___closed__4_value;
static const lean_ctor_object lp_Qq_Qq_Impl_mkIsDefEqType___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_Qq_Qq_Impl_mkIsDefEqType___closed__5 = (const lean_object*)&lp_Qq_Qq_Impl_mkIsDefEqType___closed__5_value;
static const lean_ctor_object lp_Qq_Qq_Impl_mkIsDefEqType___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Qq_Impl_mkIsDefEqType___closed__5_value)}};
static const lean_object* lp_Qq_Qq_Impl_mkIsDefEqType___closed__6 = (const lean_object*)&lp_Qq_Qq_Impl_mkIsDefEqType___closed__6_value;
static lean_once_cell_t lp_Qq_Qq_Impl_mkIsDefEqType___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq_Impl_mkIsDefEqType___closed__7;
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_mkIsDefEqType(lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_mkIsDefEqType___boxed(lean_object*);
static const lean_string_object lp_Qq_Qq_Impl_mkIsDefEqResult___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "false"};
static const lean_object* lp_Qq_Qq_Impl_mkIsDefEqResult___closed__0 = (const lean_object*)&lp_Qq_Qq_Impl_mkIsDefEqResult___closed__0_value;
static const lean_ctor_object lp_Qq_Qq_Impl_mkIsDefEqResult___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Qq_Impl_mkIsDefEqType___closed__0_value),LEAN_SCALAR_PTR_LITERAL(250, 44, 198, 216, 184, 195, 199, 178)}};
static const lean_ctor_object lp_Qq_Qq_Impl_mkIsDefEqResult___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl_mkIsDefEqResult___closed__1_value_aux_0),((lean_object*)&lp_Qq_Qq_Impl_mkIsDefEqResult___closed__0_value),LEAN_SCALAR_PTR_LITERAL(117, 151, 161, 190, 111, 237, 188, 218)}};
static const lean_object* lp_Qq_Qq_Impl_mkIsDefEqResult___closed__1 = (const lean_object*)&lp_Qq_Qq_Impl_mkIsDefEqResult___closed__1_value;
static lean_once_cell_t lp_Qq_Qq_Impl_mkIsDefEqResult___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq_Impl_mkIsDefEqResult___closed__2;
static const lean_string_object lp_Qq_Qq_Impl_mkIsDefEqResult___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "true"};
static const lean_object* lp_Qq_Qq_Impl_mkIsDefEqResult___closed__3 = (const lean_object*)&lp_Qq_Qq_Impl_mkIsDefEqResult___closed__3_value;
static const lean_ctor_object lp_Qq_Qq_Impl_mkIsDefEqResult___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Qq_Impl_mkIsDefEqType___closed__0_value),LEAN_SCALAR_PTR_LITERAL(250, 44, 198, 216, 184, 195, 199, 178)}};
static const lean_ctor_object lp_Qq_Qq_Impl_mkIsDefEqResult___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl_mkIsDefEqResult___closed__4_value_aux_0),((lean_object*)&lp_Qq_Qq_Impl_mkIsDefEqResult___closed__3_value),LEAN_SCALAR_PTR_LITERAL(22, 245, 194, 28, 184, 9, 113, 128)}};
static const lean_object* lp_Qq_Qq_Impl_mkIsDefEqResult___closed__4 = (const lean_object*)&lp_Qq_Qq_Impl_mkIsDefEqResult___closed__4_value;
static lean_once_cell_t lp_Qq_Qq_Impl_mkIsDefEqResult___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq_Impl_mkIsDefEqResult___closed__5;
static const lean_string_object lp_Qq_Qq_Impl_mkIsDefEqResult___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "mk"};
static const lean_object* lp_Qq_Qq_Impl_mkIsDefEqResult___closed__6 = (const lean_object*)&lp_Qq_Qq_Impl_mkIsDefEqResult___closed__6_value;
static const lean_ctor_object lp_Qq_Qq_Impl_mkIsDefEqResult___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Qq_Impl_mkIsDefEqType___closed__3_value),LEAN_SCALAR_PTR_LITERAL(121, 119, 164, 206, 221, 118, 48, 212)}};
static const lean_ctor_object lp_Qq_Qq_Impl_mkIsDefEqResult___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl_mkIsDefEqResult___closed__7_value_aux_0),((lean_object*)&lp_Qq_Qq_Impl_mkIsDefEqResult___closed__6_value),LEAN_SCALAR_PTR_LITERAL(117, 121, 37, 123, 104, 28, 189, 89)}};
static const lean_object* lp_Qq_Qq_Impl_mkIsDefEqResult___closed__7 = (const lean_object*)&lp_Qq_Qq_Impl_mkIsDefEqResult___closed__7_value;
static lean_once_cell_t lp_Qq_Qq_Impl_mkIsDefEqResult___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq_Impl_mkIsDefEqResult___closed__8;
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_mkIsDefEqResult(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_mkIsDefEqResult___boxed(lean_object*, lean_object*);
static const lean_string_object lp_Qq_Qq_Impl_mkIsDefEqResultVal___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "snd"};
static const lean_object* lp_Qq_Qq_Impl_mkIsDefEqResultVal___closed__0 = (const lean_object*)&lp_Qq_Qq_Impl_mkIsDefEqResultVal___closed__0_value;
static const lean_ctor_object lp_Qq_Qq_Impl_mkIsDefEqResultVal___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_Qq_Qq_Impl_mkIsDefEqType___closed__3_value),LEAN_SCALAR_PTR_LITERAL(121, 119, 164, 206, 221, 118, 48, 212)}};
static const lean_ctor_object lp_Qq_Qq_Impl_mkIsDefEqResultVal___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_Qq_Qq_Impl_mkIsDefEqResultVal___closed__1_value_aux_0),((lean_object*)&lp_Qq_Qq_Impl_mkIsDefEqResultVal___closed__0_value),LEAN_SCALAR_PTR_LITERAL(35, 40, 163, 84, 60, 49, 151, 224)}};
static const lean_object* lp_Qq_Qq_Impl_mkIsDefEqResultVal___closed__1 = (const lean_object*)&lp_Qq_Qq_Impl_mkIsDefEqResultVal___closed__1_value;
static lean_once_cell_t lp_Qq_Qq_Impl_mkIsDefEqResultVal___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_Qq_Qq_Impl_mkIsDefEqResultVal___closed__2;
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_mkIsDefEqResultVal(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_mkIsDefEqResultVal___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_mkLambda_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_mkLambda_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_mkLet_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_mkLet_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_mkLambdaQ___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_mkLambdaQ___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_mkLambdaQ(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_mkLambdaQ___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_Syntax_stripPos(lean_object* v_x_1_){
_start:
{
switch(lean_obj_tag(v_x_1_))
{
case 0:
{
return v_x_1_;
}
case 1:
{
lean_object* v_kind_2_; lean_object* v_args_3_; lean_object* v___x_5_; uint8_t v_isShared_6_; uint8_t v_isSharedCheck_14_; 
v_kind_2_ = lean_ctor_get(v_x_1_, 1);
v_args_3_ = lean_ctor_get(v_x_1_, 2);
v_isSharedCheck_14_ = !lean_is_exclusive(v_x_1_);
if (v_isSharedCheck_14_ == 0)
{
lean_object* v_unused_15_; 
v_unused_15_ = lean_ctor_get(v_x_1_, 0);
lean_dec(v_unused_15_);
v___x_5_ = v_x_1_;
v_isShared_6_ = v_isSharedCheck_14_;
goto v_resetjp_4_;
}
else
{
lean_inc(v_args_3_);
lean_inc(v_kind_2_);
lean_dec(v_x_1_);
v___x_5_ = lean_box(0);
v_isShared_6_ = v_isSharedCheck_14_;
goto v_resetjp_4_;
}
v_resetjp_4_:
{
lean_object* v___x_7_; size_t v_sz_8_; size_t v___x_9_; lean_object* v___x_10_; lean_object* v___x_12_; 
v___x_7_ = lean_box(2);
v_sz_8_ = lean_array_size(v_args_3_);
v___x_9_ = ((size_t)0ULL);
v___x_10_ = lp_Qq___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Syntax_stripPos_spec__0(v_sz_8_, v___x_9_, v_args_3_);
if (v_isShared_6_ == 0)
{
lean_ctor_set(v___x_5_, 2, v___x_10_);
lean_ctor_set(v___x_5_, 0, v___x_7_);
v___x_12_ = v___x_5_;
goto v_reusejp_11_;
}
else
{
lean_object* v_reuseFailAlloc_13_; 
v_reuseFailAlloc_13_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_13_, 0, v___x_7_);
lean_ctor_set(v_reuseFailAlloc_13_, 1, v_kind_2_);
lean_ctor_set(v_reuseFailAlloc_13_, 2, v___x_10_);
v___x_12_ = v_reuseFailAlloc_13_;
goto v_reusejp_11_;
}
v_reusejp_11_:
{
return v___x_12_;
}
}
}
case 2:
{
lean_object* v_val_16_; lean_object* v___x_18_; uint8_t v_isShared_19_; uint8_t v_isSharedCheck_24_; 
v_val_16_ = lean_ctor_get(v_x_1_, 1);
v_isSharedCheck_24_ = !lean_is_exclusive(v_x_1_);
if (v_isSharedCheck_24_ == 0)
{
lean_object* v_unused_25_; 
v_unused_25_ = lean_ctor_get(v_x_1_, 0);
lean_dec(v_unused_25_);
v___x_18_ = v_x_1_;
v_isShared_19_ = v_isSharedCheck_24_;
goto v_resetjp_17_;
}
else
{
lean_inc(v_val_16_);
lean_dec(v_x_1_);
v___x_18_ = lean_box(0);
v_isShared_19_ = v_isSharedCheck_24_;
goto v_resetjp_17_;
}
v_resetjp_17_:
{
lean_object* v___x_20_; lean_object* v___x_22_; 
v___x_20_ = lean_box(2);
if (v_isShared_19_ == 0)
{
lean_ctor_set(v___x_18_, 0, v___x_20_);
v___x_22_ = v___x_18_;
goto v_reusejp_21_;
}
else
{
lean_object* v_reuseFailAlloc_23_; 
v_reuseFailAlloc_23_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v_reuseFailAlloc_23_, 0, v___x_20_);
lean_ctor_set(v_reuseFailAlloc_23_, 1, v_val_16_);
v___x_22_ = v_reuseFailAlloc_23_;
goto v_reusejp_21_;
}
v_reusejp_21_:
{
return v___x_22_;
}
}
}
default: 
{
lean_object* v_rawVal_26_; lean_object* v_val_27_; lean_object* v_preresolved_28_; lean_object* v___x_30_; uint8_t v_isShared_31_; uint8_t v_isSharedCheck_36_; 
v_rawVal_26_ = lean_ctor_get(v_x_1_, 1);
v_val_27_ = lean_ctor_get(v_x_1_, 2);
v_preresolved_28_ = lean_ctor_get(v_x_1_, 3);
v_isSharedCheck_36_ = !lean_is_exclusive(v_x_1_);
if (v_isSharedCheck_36_ == 0)
{
lean_object* v_unused_37_; 
v_unused_37_ = lean_ctor_get(v_x_1_, 0);
lean_dec(v_unused_37_);
v___x_30_ = v_x_1_;
v_isShared_31_ = v_isSharedCheck_36_;
goto v_resetjp_29_;
}
else
{
lean_inc(v_preresolved_28_);
lean_inc(v_val_27_);
lean_inc(v_rawVal_26_);
lean_dec(v_x_1_);
v___x_30_ = lean_box(0);
v_isShared_31_ = v_isSharedCheck_36_;
goto v_resetjp_29_;
}
v_resetjp_29_:
{
lean_object* v___x_32_; lean_object* v___x_34_; 
v___x_32_ = lean_box(2);
if (v_isShared_31_ == 0)
{
lean_ctor_set(v___x_30_, 0, v___x_32_);
v___x_34_ = v___x_30_;
goto v_reusejp_33_;
}
else
{
lean_object* v_reuseFailAlloc_35_; 
v_reuseFailAlloc_35_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v_reuseFailAlloc_35_, 0, v___x_32_);
lean_ctor_set(v_reuseFailAlloc_35_, 1, v_rawVal_26_);
lean_ctor_set(v_reuseFailAlloc_35_, 2, v_val_27_);
lean_ctor_set(v_reuseFailAlloc_35_, 3, v_preresolved_28_);
v___x_34_ = v_reuseFailAlloc_35_;
goto v_reusejp_33_;
}
v_reusejp_33_:
{
return v___x_34_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Syntax_stripPos_spec__0(size_t v_sz_38_, size_t v_i_39_, lean_object* v_bs_40_){
_start:
{
uint8_t v___x_41_; 
v___x_41_ = lean_usize_dec_lt(v_i_39_, v_sz_38_);
if (v___x_41_ == 0)
{
return v_bs_40_;
}
else
{
lean_object* v_v_42_; lean_object* v___x_43_; lean_object* v_bs_x27_44_; lean_object* v___x_45_; size_t v___x_46_; size_t v___x_47_; lean_object* v___x_48_; 
v_v_42_ = lean_array_uget(v_bs_40_, v_i_39_);
v___x_43_ = lean_unsigned_to_nat(0u);
v_bs_x27_44_ = lean_array_uset(v_bs_40_, v_i_39_, v___x_43_);
v___x_45_ = lp_Qq_Lean_Syntax_stripPos(v_v_42_);
v___x_46_ = ((size_t)1ULL);
v___x_47_ = lean_usize_add(v_i_39_, v___x_46_);
v___x_48_ = lean_array_uset(v_bs_x27_44_, v_i_39_, v___x_45_);
v_i_39_ = v___x_47_;
v_bs_40_ = v___x_48_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Syntax_stripPos_spec__0___boxed(lean_object* v_sz_50_, lean_object* v_i_51_, lean_object* v_bs_52_){
_start:
{
size_t v_sz_boxed_53_; size_t v_i_boxed_54_; lean_object* v_res_55_; 
v_sz_boxed_53_ = lean_unbox_usize(v_sz_50_);
lean_dec(v_sz_50_);
v_i_boxed_54_ = lean_unbox_usize(v_i_51_);
lean_dec(v_i_51_);
v_res_55_ = lp_Qq___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Syntax_stripPos_spec__0(v_sz_boxed_53_, v_i_boxed_54_, v_bs_52_);
return v_res_55_;
}
}
static lean_object* _init_lp_Qq_Qq_Impl_PatVarDecl_fvarTy___closed__3(void){
_start:
{
lean_object* v___x_61_; lean_object* v___x_62_; lean_object* v___x_63_; 
v___x_61_ = lean_box(0);
v___x_62_ = ((lean_object*)(lp_Qq_Qq_Impl_PatVarDecl_fvarTy___closed__2));
v___x_63_ = l_Lean_Expr_const___override(v___x_62_, v___x_61_);
return v___x_63_;
}
}
static lean_object* _init_lp_Qq_Qq_Impl_PatVarDecl_fvarTy___closed__6(void){
_start:
{
lean_object* v___x_68_; lean_object* v___x_69_; lean_object* v___x_70_; 
v___x_68_ = lean_box(0);
v___x_69_ = ((lean_object*)(lp_Qq_Qq_Impl_PatVarDecl_fvarTy___closed__5));
v___x_70_ = l_Lean_Expr_const___override(v___x_69_, v___x_68_);
return v___x_70_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_PatVarDecl_fvarTy(lean_object* v_x_71_){
_start:
{
lean_object* v_ty_72_; 
v_ty_72_ = lean_ctor_get(v_x_71_, 0);
if (lean_obj_tag(v_ty_72_) == 0)
{
lean_object* v___x_73_; 
v___x_73_ = lean_obj_once(&lp_Qq_Qq_Impl_PatVarDecl_fvarTy___closed__3, &lp_Qq_Qq_Impl_PatVarDecl_fvarTy___closed__3_once, _init_lp_Qq_Qq_Impl_PatVarDecl_fvarTy___closed__3);
return v___x_73_;
}
else
{
lean_object* v___x_74_; 
v___x_74_ = lean_obj_once(&lp_Qq_Qq_Impl_PatVarDecl_fvarTy___closed__6, &lp_Qq_Qq_Impl_PatVarDecl_fvarTy___closed__6_once, _init_lp_Qq_Qq_Impl_PatVarDecl_fvarTy___closed__6);
return v___x_74_;
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_PatVarDecl_fvarTy___boxed(lean_object* v_x_75_){
_start:
{
lean_object* v_res_76_; 
v_res_76_ = lp_Qq_Qq_Impl_PatVarDecl_fvarTy(v_x_75_);
lean_dec_ref(v_x_75_);
return v_res_76_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_PatVarDecl_fvar(lean_object* v_decl_77_){
_start:
{
lean_object* v_fvarId_78_; lean_object* v___x_79_; 
v_fvarId_78_ = lean_ctor_get(v_decl_77_, 1);
lean_inc(v_fvarId_78_);
lean_dec_ref(v_decl_77_);
v___x_79_ = l_Lean_Expr_fvar___override(v_fvarId_78_);
return v___x_79_;
}
}
static lean_object* _init_lp_Qq_Qq_Impl_mkIsDefEqType___closed__2(void){
_start:
{
lean_object* v___x_83_; lean_object* v___x_84_; lean_object* v___x_85_; 
v___x_83_ = lean_box(0);
v___x_84_ = ((lean_object*)(lp_Qq_Qq_Impl_mkIsDefEqType___closed__1));
v___x_85_ = l_Lean_Expr_const___override(v___x_84_, v___x_83_);
return v___x_85_;
}
}
static lean_object* _init_lp_Qq_Qq_Impl_mkIsDefEqType___closed__7(void){
_start:
{
lean_object* v___x_95_; lean_object* v___x_96_; lean_object* v___x_97_; 
v___x_95_ = ((lean_object*)(lp_Qq_Qq_Impl_mkIsDefEqType___closed__6));
v___x_96_ = ((lean_object*)(lp_Qq_Qq_Impl_mkIsDefEqType___closed__4));
v___x_97_ = l_Lean_Expr_const___override(v___x_96_, v___x_95_);
return v___x_97_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_mkIsDefEqType(lean_object* v_x_98_){
_start:
{
if (lean_obj_tag(v_x_98_) == 0)
{
lean_object* v___x_99_; 
v___x_99_ = lean_obj_once(&lp_Qq_Qq_Impl_mkIsDefEqType___closed__2, &lp_Qq_Qq_Impl_mkIsDefEqType___closed__2_once, _init_lp_Qq_Qq_Impl_mkIsDefEqType___closed__2);
return v___x_99_;
}
else
{
lean_object* v_head_100_; lean_object* v_tail_101_; lean_object* v_a_102_; lean_object* v_a_103_; lean_object* v___x_104_; lean_object* v___x_105_; lean_object* v___x_106_; 
v_head_100_ = lean_ctor_get(v_x_98_, 0);
v_tail_101_ = lean_ctor_get(v_x_98_, 1);
v_a_102_ = lp_Qq_Qq_Impl_mkIsDefEqType(v_tail_101_);
v_a_103_ = lp_Qq_Qq_Impl_PatVarDecl_fvarTy(v_head_100_);
v___x_104_ = lean_obj_once(&lp_Qq_Qq_Impl_mkIsDefEqType___closed__7, &lp_Qq_Qq_Impl_mkIsDefEqType___closed__7_once, _init_lp_Qq_Qq_Impl_mkIsDefEqType___closed__7);
v___x_105_ = l_Lean_Expr_app___override(v___x_104_, v_a_103_);
v___x_106_ = l_Lean_Expr_app___override(v___x_105_, v_a_102_);
return v___x_106_;
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_mkIsDefEqType___boxed(lean_object* v_x_107_){
_start:
{
lean_object* v_res_108_; 
v_res_108_ = lp_Qq_Qq_Impl_mkIsDefEqType(v_x_107_);
lean_dec(v_x_107_);
return v_res_108_;
}
}
static lean_object* _init_lp_Qq_Qq_Impl_mkIsDefEqResult___closed__2(void){
_start:
{
lean_object* v___x_113_; lean_object* v___x_114_; lean_object* v___x_115_; 
v___x_113_ = lean_box(0);
v___x_114_ = ((lean_object*)(lp_Qq_Qq_Impl_mkIsDefEqResult___closed__1));
v___x_115_ = l_Lean_mkConst(v___x_114_, v___x_113_);
return v___x_115_;
}
}
static lean_object* _init_lp_Qq_Qq_Impl_mkIsDefEqResult___closed__5(void){
_start:
{
lean_object* v___x_120_; lean_object* v___x_121_; lean_object* v___x_122_; 
v___x_120_ = lean_box(0);
v___x_121_ = ((lean_object*)(lp_Qq_Qq_Impl_mkIsDefEqResult___closed__4));
v___x_122_ = l_Lean_mkConst(v___x_121_, v___x_120_);
return v___x_122_;
}
}
static lean_object* _init_lp_Qq_Qq_Impl_mkIsDefEqResult___closed__8(void){
_start:
{
lean_object* v___x_127_; lean_object* v___x_128_; lean_object* v___x_129_; 
v___x_127_ = ((lean_object*)(lp_Qq_Qq_Impl_mkIsDefEqType___closed__6));
v___x_128_ = ((lean_object*)(lp_Qq_Qq_Impl_mkIsDefEqResult___closed__7));
v___x_129_ = l_Lean_Expr_const___override(v___x_128_, v___x_127_);
return v___x_129_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_mkIsDefEqResult(uint8_t v_val_130_, lean_object* v_x_131_){
_start:
{
if (lean_obj_tag(v_x_131_) == 0)
{
if (v_val_130_ == 0)
{
lean_object* v___x_132_; 
v___x_132_ = lean_obj_once(&lp_Qq_Qq_Impl_mkIsDefEqResult___closed__2, &lp_Qq_Qq_Impl_mkIsDefEqResult___closed__2_once, _init_lp_Qq_Qq_Impl_mkIsDefEqResult___closed__2);
return v___x_132_;
}
else
{
lean_object* v___x_133_; 
v___x_133_ = lean_obj_once(&lp_Qq_Qq_Impl_mkIsDefEqResult___closed__5, &lp_Qq_Qq_Impl_mkIsDefEqResult___closed__5_once, _init_lp_Qq_Qq_Impl_mkIsDefEqResult___closed__5);
return v___x_133_;
}
}
else
{
lean_object* v_head_134_; lean_object* v_tail_135_; lean_object* v_ty_136_; lean_object* v_a_137_; lean_object* v_a_138_; lean_object* v___x_139_; lean_object* v___y_141_; 
v_head_134_ = lean_ctor_get(v_x_131_, 0);
lean_inc(v_head_134_);
v_tail_135_ = lean_ctor_get(v_x_131_, 1);
lean_inc_n(v_tail_135_, 2);
lean_dec_ref_known(v_x_131_, 2);
v_ty_136_ = lean_ctor_get(v_head_134_, 0);
lean_inc(v_ty_136_);
v_a_137_ = lp_Qq_Qq_Impl_mkIsDefEqResult(v_val_130_, v_tail_135_);
v_a_138_ = lp_Qq_Qq_Impl_PatVarDecl_fvar(v_head_134_);
v___x_139_ = lean_obj_once(&lp_Qq_Qq_Impl_mkIsDefEqResult___closed__8, &lp_Qq_Qq_Impl_mkIsDefEqResult___closed__8_once, _init_lp_Qq_Qq_Impl_mkIsDefEqResult___closed__8);
if (lean_obj_tag(v_ty_136_) == 0)
{
lean_object* v___x_147_; 
v___x_147_ = lean_obj_once(&lp_Qq_Qq_Impl_PatVarDecl_fvarTy___closed__3, &lp_Qq_Qq_Impl_PatVarDecl_fvarTy___closed__3_once, _init_lp_Qq_Qq_Impl_PatVarDecl_fvarTy___closed__3);
v___y_141_ = v___x_147_;
goto v___jp_140_;
}
else
{
lean_object* v___x_148_; 
lean_dec_ref_known(v_ty_136_, 1);
v___x_148_ = lean_obj_once(&lp_Qq_Qq_Impl_PatVarDecl_fvarTy___closed__6, &lp_Qq_Qq_Impl_PatVarDecl_fvarTy___closed__6_once, _init_lp_Qq_Qq_Impl_PatVarDecl_fvarTy___closed__6);
v___y_141_ = v___x_148_;
goto v___jp_140_;
}
v___jp_140_:
{
lean_object* v___x_142_; lean_object* v___x_143_; lean_object* v___x_144_; lean_object* v___x_145_; lean_object* v___x_146_; 
lean_inc_ref(v___y_141_);
v___x_142_ = l_Lean_Expr_app___override(v___x_139_, v___y_141_);
v___x_143_ = lp_Qq_Qq_Impl_mkIsDefEqType(v_tail_135_);
lean_dec(v_tail_135_);
v___x_144_ = l_Lean_Expr_app___override(v___x_142_, v___x_143_);
v___x_145_ = l_Lean_Expr_app___override(v___x_144_, v_a_138_);
v___x_146_ = l_Lean_Expr_app___override(v___x_145_, v_a_137_);
return v___x_146_;
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_mkIsDefEqResult___boxed(lean_object* v_val_149_, lean_object* v_x_150_){
_start:
{
uint8_t v_val_boxed_151_; lean_object* v_res_152_; 
v_val_boxed_151_ = lean_unbox(v_val_149_);
v_res_152_ = lp_Qq_Qq_Impl_mkIsDefEqResult(v_val_boxed_151_, v_x_150_);
return v_res_152_;
}
}
static lean_object* _init_lp_Qq_Qq_Impl_mkIsDefEqResultVal___closed__2(void){
_start:
{
lean_object* v___x_157_; lean_object* v___x_158_; lean_object* v___x_159_; 
v___x_157_ = ((lean_object*)(lp_Qq_Qq_Impl_mkIsDefEqType___closed__6));
v___x_158_ = ((lean_object*)(lp_Qq_Qq_Impl_mkIsDefEqResultVal___closed__1));
v___x_159_ = l_Lean_Expr_const___override(v___x_158_, v___x_157_);
return v___x_159_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_mkIsDefEqResultVal(lean_object* v_x_160_, lean_object* v_x_161_){
_start:
{
if (lean_obj_tag(v_x_160_) == 0)
{
return v_x_161_;
}
else
{
lean_object* v_head_162_; lean_object* v_tail_163_; lean_object* v_ty_164_; lean_object* v___x_165_; lean_object* v___x_166_; lean_object* v___y_168_; 
v_head_162_ = lean_ctor_get(v_x_160_, 0);
v_tail_163_ = lean_ctor_get(v_x_160_, 1);
v_ty_164_ = lean_ctor_get(v_head_162_, 0);
v___x_165_ = lp_Qq_Qq_Impl_mkIsDefEqType(v_tail_163_);
v___x_166_ = lean_obj_once(&lp_Qq_Qq_Impl_mkIsDefEqResultVal___closed__2, &lp_Qq_Qq_Impl_mkIsDefEqResultVal___closed__2_once, _init_lp_Qq_Qq_Impl_mkIsDefEqResultVal___closed__2);
if (lean_obj_tag(v_ty_164_) == 0)
{
lean_object* v___x_173_; 
v___x_173_ = lean_obj_once(&lp_Qq_Qq_Impl_PatVarDecl_fvarTy___closed__3, &lp_Qq_Qq_Impl_PatVarDecl_fvarTy___closed__3_once, _init_lp_Qq_Qq_Impl_PatVarDecl_fvarTy___closed__3);
v___y_168_ = v___x_173_;
goto v___jp_167_;
}
else
{
lean_object* v___x_174_; 
v___x_174_ = lean_obj_once(&lp_Qq_Qq_Impl_PatVarDecl_fvarTy___closed__6, &lp_Qq_Qq_Impl_PatVarDecl_fvarTy___closed__6_once, _init_lp_Qq_Qq_Impl_PatVarDecl_fvarTy___closed__6);
v___y_168_ = v___x_174_;
goto v___jp_167_;
}
v___jp_167_:
{
lean_object* v___x_169_; lean_object* v___x_170_; lean_object* v___x_171_; 
lean_inc_ref(v___y_168_);
v___x_169_ = l_Lean_Expr_app___override(v___x_166_, v___y_168_);
v___x_170_ = l_Lean_Expr_app___override(v___x_169_, v___x_165_);
v___x_171_ = l_Lean_Expr_app___override(v___x_170_, v_x_161_);
v_x_160_ = v_tail_163_;
v_x_161_ = v___x_171_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_mkIsDefEqResultVal___boxed(lean_object* v_x_175_, lean_object* v_x_176_){
_start:
{
lean_object* v_res_177_; 
v_res_177_ = lp_Qq_Qq_Impl_mkIsDefEqResultVal(v_x_175_, v_x_176_);
lean_dec(v_x_175_);
return v_res_177_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_mkLambda_x27(lean_object* v_n_178_, lean_object* v_fvar_179_, lean_object* v_ty_180_, lean_object* v_body_181_, lean_object* v_a_182_, lean_object* v_a_183_, lean_object* v_a_184_, lean_object* v_a_185_){
_start:
{
lean_object* v___x_187_; lean_object* v___x_188_; lean_object* v___x_189_; lean_object* v___x_190_; 
v___x_187_ = lean_unsigned_to_nat(1u);
v___x_188_ = lean_mk_empty_array_with_capacity(v___x_187_);
v___x_189_ = lean_array_push(v___x_188_, v_fvar_179_);
v___x_190_ = l_Lean_Expr_abstractM(v_body_181_, v___x_189_, v_a_182_, v_a_183_, v_a_184_, v_a_185_);
lean_dec_ref(v___x_189_);
if (lean_obj_tag(v___x_190_) == 0)
{
lean_object* v_a_191_; lean_object* v___x_193_; uint8_t v_isShared_194_; uint8_t v_isSharedCheck_200_; 
v_a_191_ = lean_ctor_get(v___x_190_, 0);
v_isSharedCheck_200_ = !lean_is_exclusive(v___x_190_);
if (v_isSharedCheck_200_ == 0)
{
v___x_193_ = v___x_190_;
v_isShared_194_ = v_isSharedCheck_200_;
goto v_resetjp_192_;
}
else
{
lean_inc(v_a_191_);
lean_dec(v___x_190_);
v___x_193_ = lean_box(0);
v_isShared_194_ = v_isSharedCheck_200_;
goto v_resetjp_192_;
}
v_resetjp_192_:
{
uint8_t v___x_195_; lean_object* v___x_196_; lean_object* v___x_198_; 
v___x_195_ = 0;
v___x_196_ = l_Lean_mkLambda(v_n_178_, v___x_195_, v_ty_180_, v_a_191_);
if (v_isShared_194_ == 0)
{
lean_ctor_set(v___x_193_, 0, v___x_196_);
v___x_198_ = v___x_193_;
goto v_reusejp_197_;
}
else
{
lean_object* v_reuseFailAlloc_199_; 
v_reuseFailAlloc_199_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_199_, 0, v___x_196_);
v___x_198_ = v_reuseFailAlloc_199_;
goto v_reusejp_197_;
}
v_reusejp_197_:
{
return v___x_198_;
}
}
}
else
{
lean_dec_ref(v_ty_180_);
lean_dec(v_n_178_);
return v___x_190_;
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_mkLambda_x27___boxed(lean_object* v_n_201_, lean_object* v_fvar_202_, lean_object* v_ty_203_, lean_object* v_body_204_, lean_object* v_a_205_, lean_object* v_a_206_, lean_object* v_a_207_, lean_object* v_a_208_, lean_object* v_a_209_){
_start:
{
lean_object* v_res_210_; 
v_res_210_ = lp_Qq_Qq_Impl_mkLambda_x27(v_n_201_, v_fvar_202_, v_ty_203_, v_body_204_, v_a_205_, v_a_206_, v_a_207_, v_a_208_);
lean_dec(v_a_208_);
lean_dec_ref(v_a_207_);
lean_dec(v_a_206_);
lean_dec_ref(v_a_205_);
return v_res_210_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_mkLet_x27(lean_object* v_n_211_, lean_object* v_fvar_212_, lean_object* v_ty_213_, lean_object* v_val_214_, lean_object* v_body_215_, lean_object* v_a_216_, lean_object* v_a_217_, lean_object* v_a_218_, lean_object* v_a_219_){
_start:
{
lean_object* v___x_221_; lean_object* v___x_222_; lean_object* v___x_223_; lean_object* v___x_224_; 
v___x_221_ = lean_unsigned_to_nat(1u);
v___x_222_ = lean_mk_empty_array_with_capacity(v___x_221_);
v___x_223_ = lean_array_push(v___x_222_, v_fvar_212_);
v___x_224_ = l_Lean_Expr_abstractM(v_body_215_, v___x_223_, v_a_216_, v_a_217_, v_a_218_, v_a_219_);
lean_dec_ref(v___x_223_);
if (lean_obj_tag(v___x_224_) == 0)
{
lean_object* v_a_225_; lean_object* v___x_227_; uint8_t v_isShared_228_; uint8_t v_isSharedCheck_234_; 
v_a_225_ = lean_ctor_get(v___x_224_, 0);
v_isSharedCheck_234_ = !lean_is_exclusive(v___x_224_);
if (v_isSharedCheck_234_ == 0)
{
v___x_227_ = v___x_224_;
v_isShared_228_ = v_isSharedCheck_234_;
goto v_resetjp_226_;
}
else
{
lean_inc(v_a_225_);
lean_dec(v___x_224_);
v___x_227_ = lean_box(0);
v_isShared_228_ = v_isSharedCheck_234_;
goto v_resetjp_226_;
}
v_resetjp_226_:
{
uint8_t v___x_229_; lean_object* v___x_230_; lean_object* v___x_232_; 
v___x_229_ = 0;
v___x_230_ = l_Lean_Expr_letE___override(v_n_211_, v_ty_213_, v_val_214_, v_a_225_, v___x_229_);
if (v_isShared_228_ == 0)
{
lean_ctor_set(v___x_227_, 0, v___x_230_);
v___x_232_ = v___x_227_;
goto v_reusejp_231_;
}
else
{
lean_object* v_reuseFailAlloc_233_; 
v_reuseFailAlloc_233_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_233_, 0, v___x_230_);
v___x_232_ = v_reuseFailAlloc_233_;
goto v_reusejp_231_;
}
v_reusejp_231_:
{
return v___x_232_;
}
}
}
else
{
lean_dec_ref(v_val_214_);
lean_dec_ref(v_ty_213_);
lean_dec(v_n_211_);
return v___x_224_;
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_mkLet_x27___boxed(lean_object* v_n_235_, lean_object* v_fvar_236_, lean_object* v_ty_237_, lean_object* v_val_238_, lean_object* v_body_239_, lean_object* v_a_240_, lean_object* v_a_241_, lean_object* v_a_242_, lean_object* v_a_243_, lean_object* v_a_244_){
_start:
{
lean_object* v_res_245_; 
v_res_245_ = lp_Qq_Qq_Impl_mkLet_x27(v_n_235_, v_fvar_236_, v_ty_237_, v_val_238_, v_body_239_, v_a_240_, v_a_241_, v_a_242_, v_a_243_);
lean_dec(v_a_243_);
lean_dec_ref(v_a_242_);
lean_dec(v_a_241_);
lean_dec_ref(v_a_240_);
return v_res_245_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_mkLambdaQ___redArg(lean_object* v_00_u03b1_246_, lean_object* v_n_247_, lean_object* v_fvar_248_, lean_object* v_body_249_, lean_object* v_a_250_, lean_object* v_a_251_, lean_object* v_a_252_, lean_object* v_a_253_){
_start:
{
lean_object* v___x_255_; lean_object* v___x_256_; lean_object* v___x_257_; lean_object* v___x_258_; 
v___x_255_ = lean_unsigned_to_nat(1u);
v___x_256_ = lean_mk_empty_array_with_capacity(v___x_255_);
v___x_257_ = lean_array_push(v___x_256_, v_fvar_248_);
v___x_258_ = l_Lean_Expr_abstractM(v_body_249_, v___x_257_, v_a_250_, v_a_251_, v_a_252_, v_a_253_);
lean_dec_ref(v___x_257_);
if (lean_obj_tag(v___x_258_) == 0)
{
lean_object* v_a_259_; lean_object* v___x_261_; uint8_t v_isShared_262_; uint8_t v_isSharedCheck_268_; 
v_a_259_ = lean_ctor_get(v___x_258_, 0);
v_isSharedCheck_268_ = !lean_is_exclusive(v___x_258_);
if (v_isSharedCheck_268_ == 0)
{
v___x_261_ = v___x_258_;
v_isShared_262_ = v_isSharedCheck_268_;
goto v_resetjp_260_;
}
else
{
lean_inc(v_a_259_);
lean_dec(v___x_258_);
v___x_261_ = lean_box(0);
v_isShared_262_ = v_isSharedCheck_268_;
goto v_resetjp_260_;
}
v_resetjp_260_:
{
uint8_t v___x_263_; lean_object* v___x_264_; lean_object* v___x_266_; 
v___x_263_ = 0;
v___x_264_ = l_Lean_mkLambda(v_n_247_, v___x_263_, v_00_u03b1_246_, v_a_259_);
if (v_isShared_262_ == 0)
{
lean_ctor_set(v___x_261_, 0, v___x_264_);
v___x_266_ = v___x_261_;
goto v_reusejp_265_;
}
else
{
lean_object* v_reuseFailAlloc_267_; 
v_reuseFailAlloc_267_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_267_, 0, v___x_264_);
v___x_266_ = v_reuseFailAlloc_267_;
goto v_reusejp_265_;
}
v_reusejp_265_:
{
return v___x_266_;
}
}
}
else
{
lean_dec(v_n_247_);
lean_dec_ref(v_00_u03b1_246_);
return v___x_258_;
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_mkLambdaQ___redArg___boxed(lean_object* v_00_u03b1_269_, lean_object* v_n_270_, lean_object* v_fvar_271_, lean_object* v_body_272_, lean_object* v_a_273_, lean_object* v_a_274_, lean_object* v_a_275_, lean_object* v_a_276_, lean_object* v_a_277_){
_start:
{
lean_object* v_res_278_; 
v_res_278_ = lp_Qq_Qq_Impl_mkLambdaQ___redArg(v_00_u03b1_269_, v_n_270_, v_fvar_271_, v_body_272_, v_a_273_, v_a_274_, v_a_275_, v_a_276_);
lean_dec(v_a_276_);
lean_dec_ref(v_a_275_);
lean_dec(v_a_274_);
lean_dec_ref(v_a_273_);
return v_res_278_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_mkLambdaQ(lean_object* v_00_u03b1_279_, lean_object* v_00_u03b2_280_, lean_object* v_n_281_, lean_object* v_fvar_282_, lean_object* v_body_283_, lean_object* v_a_284_, lean_object* v_a_285_, lean_object* v_a_286_, lean_object* v_a_287_){
_start:
{
lean_object* v___x_289_; 
v___x_289_ = lp_Qq_Qq_Impl_mkLambdaQ___redArg(v_00_u03b1_279_, v_n_281_, v_fvar_282_, v_body_283_, v_a_284_, v_a_285_, v_a_286_, v_a_287_);
return v___x_289_;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_Impl_mkLambdaQ___boxed(lean_object* v_00_u03b1_290_, lean_object* v_00_u03b2_291_, lean_object* v_n_292_, lean_object* v_fvar_293_, lean_object* v_body_294_, lean_object* v_a_295_, lean_object* v_a_296_, lean_object* v_a_297_, lean_object* v_a_298_, lean_object* v_a_299_){
_start:
{
lean_object* v_res_300_; 
v_res_300_ = lp_Qq_Qq_Impl_mkLambdaQ(v_00_u03b1_290_, v_00_u03b2_291_, v_n_292_, v_fvar_293_, v_body_294_, v_a_295_, v_a_296_, v_a_297_, v_a_298_);
lean_dec(v_a_298_);
lean_dec_ref(v_a_297_);
lean_dec(v_a_296_);
lean_dec_ref(v_a_295_);
lean_dec_ref(v_00_u03b2_291_);
return v_res_300_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Qq_Qq_MetaM(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_Qq_Qq_MatchImpl(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Qq_Qq_MetaM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_Qq_Qq_MatchImpl(uint8_t builtin) {
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
lean_object* initialize_Qq_Qq_MetaM(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_Qq_Qq_MatchImpl(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Qq_Qq_MetaM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Qq_Qq_MatchImpl(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_Qq_Qq_MatchImpl(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_Qq_Qq_MatchImpl(builtin);
}
#ifdef __cplusplus
}
#endif
