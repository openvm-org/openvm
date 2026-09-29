// Lean compiler output
// Module: Mathlib.Data.ENat.Monoid
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Order.Ring.Nat public import Mathlib.Algebra.Order.Ring.WithTop public import Mathlib.Data.ENat.Basic import Mathlib.Algebra.Group.Nat.Units import Mathlib.Data.Nat.Cast.Order.Basic
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
extern lean_object* lp_mathlib_instOneENat;
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
extern lean_object* lp_mathlib_instAddENat;
extern lean_object* lp_mathlib_instZeroENat;
lean_object* lp_mathlib_WithTop_some(lean_object*, lean_object*);
lean_object* lean_nat_pow(lean_object*, lean_object*);
extern lean_object* lp_mathlib_instLinearOrderENat;
lean_object* lp_mathlib_ENat_map(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_ENat_toNat(lean_object*);
lean_object* lp_mathlib_Units_instInhabited___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instAddMonoidWithOneENat___aux__4(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instAddMonoidWithOneENat___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_instAddMonoidWithOneENat___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_instAddMonoidWithOneENat___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_instAddMonoidWithOneENat___closed__0 = (const lean_object*)&lp_mathlib_instAddMonoidWithOneENat___closed__0_value;
static const lean_closure_object lp_mathlib_instAddMonoidWithOneENat___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_WithTop_some, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_instAddMonoidWithOneENat___closed__1 = (const lean_object*)&lp_mathlib_instAddMonoidWithOneENat___closed__1_value;
static lean_once_cell_t lp_mathlib_instAddMonoidWithOneENat___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_instAddMonoidWithOneENat___closed__2;
static lean_once_cell_t lp_mathlib_instAddMonoidWithOneENat___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_instAddMonoidWithOneENat___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_instAddMonoidWithOneENat;
static const lean_ctor_object lp_mathlib_instCommSemiringENat___aux__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_instCommSemiringENat___aux__2___closed__0 = (const lean_object*)&lp_mathlib_instCommSemiringENat___aux__2___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_instCommSemiringENat___aux__2(lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_instCommSemiringENat___aux__7___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_mathlib_instCommSemiringENat___aux__7___closed__0 = (const lean_object*)&lp_mathlib_instCommSemiringENat___aux__7___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_instCommSemiringENat___aux__7(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCommSemiringENat___aux__7___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCommSemiringENat___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCommSemiringENat___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCommSemiringENat___lam__1___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_instCommSemiringENat___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_instCommSemiringENat___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_instCommSemiringENat___closed__0 = (const lean_object*)&lp_mathlib_instCommSemiringENat___closed__0_value;
static const lean_closure_object lp_mathlib_instCommSemiringENat___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_instCommSemiringENat___lam__1___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_instCommSemiringENat___closed__1 = (const lean_object*)&lp_mathlib_instCommSemiringENat___closed__1_value;
static lean_once_cell_t lp_mathlib_instCommSemiringENat___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_instCommSemiringENat___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_instCommSemiringENat;
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderedAddCommMonoidWithTopENat;
static const lean_closure_object lp_mathlib_ENat_toNatHom___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_ENat_toNat, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_ENat_toNatHom___closed__0 = (const lean_object*)&lp_mathlib_ENat_toNatHom___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_ENat_toNatHom = (const lean_object*)&lp_mathlib_ENat_toNatHom___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_ENat_instUniqueUnits;
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_ENatMap___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_ENatMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_ENatMap(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_ENatMap___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_ENatMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_ENatMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_ENatMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_ENatMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_ENatMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RingHom_ENatMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instAddMonoidWithOneENat___aux__4(lean_object* v_n_1_, lean_object* v_a_2_){
_start:
{
if (lean_obj_tag(v_a_2_) == 0)
{
lean_object* v_zero_3_; uint8_t v_isZero_4_; 
v_zero_3_ = lean_unsigned_to_nat(0u);
v_isZero_4_ = lean_nat_dec_eq(v_n_1_, v_zero_3_);
if (v_isZero_4_ == 1)
{
lean_object* v___x_5_; 
v___x_5_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_5_, 0, v_n_1_);
return v___x_5_;
}
else
{
lean_object* v___x_6_; 
lean_dec(v_n_1_);
v___x_6_ = lean_box(0);
return v___x_6_;
}
}
else
{
lean_object* v_val_7_; lean_object* v___x_9_; uint8_t v_isShared_10_; uint8_t v_isSharedCheck_15_; 
v_val_7_ = lean_ctor_get(v_a_2_, 0);
v_isSharedCheck_15_ = !lean_is_exclusive(v_a_2_);
if (v_isSharedCheck_15_ == 0)
{
v___x_9_ = v_a_2_;
v_isShared_10_ = v_isSharedCheck_15_;
goto v_resetjp_8_;
}
else
{
lean_inc(v_val_7_);
lean_dec(v_a_2_);
v___x_9_ = lean_box(0);
v_isShared_10_ = v_isSharedCheck_15_;
goto v_resetjp_8_;
}
v_resetjp_8_:
{
lean_object* v___x_11_; lean_object* v___x_13_; 
v___x_11_ = lean_nat_mul(v_n_1_, v_val_7_);
lean_dec(v_val_7_);
lean_dec(v_n_1_);
if (v_isShared_10_ == 0)
{
lean_ctor_set(v___x_9_, 0, v___x_11_);
v___x_13_ = v___x_9_;
goto v_reusejp_12_;
}
else
{
lean_object* v_reuseFailAlloc_14_; 
v_reuseFailAlloc_14_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_14_, 0, v___x_11_);
v___x_13_ = v_reuseFailAlloc_14_;
goto v_reusejp_12_;
}
v_reusejp_12_:
{
return v___x_13_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_instAddMonoidWithOneENat___lam__0(lean_object* v___y_16_, lean_object* v___y_17_){
_start:
{
if (lean_obj_tag(v___y_17_) == 0)
{
lean_object* v_zero_18_; uint8_t v_isZero_19_; 
v_zero_18_ = lean_unsigned_to_nat(0u);
v_isZero_19_ = lean_nat_dec_eq(v___y_16_, v_zero_18_);
if (v_isZero_19_ == 1)
{
lean_object* v___x_20_; 
v___x_20_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_20_, 0, v___y_16_);
return v___x_20_;
}
else
{
lean_object* v___x_21_; 
lean_dec(v___y_16_);
v___x_21_ = lean_box(0);
return v___x_21_;
}
}
else
{
lean_object* v_val_22_; lean_object* v___x_24_; uint8_t v_isShared_25_; uint8_t v_isSharedCheck_30_; 
v_val_22_ = lean_ctor_get(v___y_17_, 0);
v_isSharedCheck_30_ = !lean_is_exclusive(v___y_17_);
if (v_isSharedCheck_30_ == 0)
{
v___x_24_ = v___y_17_;
v_isShared_25_ = v_isSharedCheck_30_;
goto v_resetjp_23_;
}
else
{
lean_inc(v_val_22_);
lean_dec(v___y_17_);
v___x_24_ = lean_box(0);
v_isShared_25_ = v_isSharedCheck_30_;
goto v_resetjp_23_;
}
v_resetjp_23_:
{
lean_object* v___x_26_; lean_object* v___x_28_; 
v___x_26_ = lean_nat_mul(v___y_16_, v_val_22_);
lean_dec(v_val_22_);
lean_dec(v___y_16_);
if (v_isShared_25_ == 0)
{
lean_ctor_set(v___x_24_, 0, v___x_26_);
v___x_28_ = v___x_24_;
goto v_reusejp_27_;
}
else
{
lean_object* v_reuseFailAlloc_29_; 
v_reuseFailAlloc_29_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_29_, 0, v___x_26_);
v___x_28_ = v_reuseFailAlloc_29_;
goto v_reusejp_27_;
}
v_reusejp_27_:
{
return v___x_28_;
}
}
}
}
}
static lean_object* _init_lp_mathlib_instAddMonoidWithOneENat___closed__2(void){
_start:
{
lean_object* v___f_33_; lean_object* v___x_34_; lean_object* v___x_35_; lean_object* v___x_36_; 
v___f_33_ = ((lean_object*)(lp_mathlib_instAddMonoidWithOneENat___closed__0));
v___x_34_ = lp_mathlib_instAddENat;
v___x_35_ = lp_mathlib_instZeroENat;
v___x_36_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_36_, 0, v___x_35_);
lean_ctor_set(v___x_36_, 1, v___x_34_);
lean_ctor_set(v___x_36_, 2, v___f_33_);
return v___x_36_;
}
}
static lean_object* _init_lp_mathlib_instAddMonoidWithOneENat___closed__3(void){
_start:
{
lean_object* v___x_37_; lean_object* v___x_38_; lean_object* v___x_39_; lean_object* v___x_40_; 
v___x_37_ = lp_mathlib_instOneENat;
v___x_38_ = lean_obj_once(&lp_mathlib_instAddMonoidWithOneENat___closed__2, &lp_mathlib_instAddMonoidWithOneENat___closed__2_once, _init_lp_mathlib_instAddMonoidWithOneENat___closed__2);
v___x_39_ = ((lean_object*)(lp_mathlib_instAddMonoidWithOneENat___closed__1));
v___x_40_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_40_, 0, v___x_39_);
lean_ctor_set(v___x_40_, 1, v___x_38_);
lean_ctor_set(v___x_40_, 2, v___x_37_);
return v___x_40_;
}
}
static lean_object* _init_lp_mathlib_instAddMonoidWithOneENat(void){
_start:
{
lean_object* v___x_41_; 
v___x_41_ = lean_obj_once(&lp_mathlib_instAddMonoidWithOneENat___closed__3, &lp_mathlib_instAddMonoidWithOneENat___closed__3_once, _init_lp_mathlib_instAddMonoidWithOneENat___closed__3);
return v___x_41_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCommSemiringENat___aux__2(lean_object* v_x_44_, lean_object* v_x_45_){
_start:
{
lean_object* v___x_46_; lean_object* v_a_48_; 
v___x_46_ = lean_box(0);
if (lean_obj_tag(v_x_44_) == 0)
{
if (lean_obj_tag(v_x_45_) == 0)
{
return v___x_46_;
}
else
{
lean_object* v_val_52_; 
v_val_52_ = lean_ctor_get(v_x_45_, 0);
lean_inc(v_val_52_);
lean_dec_ref_known(v_x_45_, 1);
v_a_48_ = v_val_52_;
goto v___jp_47_;
}
}
else
{
if (lean_obj_tag(v_x_45_) == 0)
{
lean_object* v_val_53_; 
v_val_53_ = lean_ctor_get(v_x_44_, 0);
lean_inc(v_val_53_);
lean_dec_ref_known(v_x_44_, 1);
v_a_48_ = v_val_53_;
goto v___jp_47_;
}
else
{
lean_object* v_val_54_; lean_object* v_val_55_; lean_object* v___x_57_; uint8_t v_isShared_58_; uint8_t v_isSharedCheck_63_; 
v_val_54_ = lean_ctor_get(v_x_44_, 0);
lean_inc(v_val_54_);
lean_dec_ref_known(v_x_44_, 1);
v_val_55_ = lean_ctor_get(v_x_45_, 0);
v_isSharedCheck_63_ = !lean_is_exclusive(v_x_45_);
if (v_isSharedCheck_63_ == 0)
{
v___x_57_ = v_x_45_;
v_isShared_58_ = v_isSharedCheck_63_;
goto v_resetjp_56_;
}
else
{
lean_inc(v_val_55_);
lean_dec(v_x_45_);
v___x_57_ = lean_box(0);
v_isShared_58_ = v_isSharedCheck_63_;
goto v_resetjp_56_;
}
v_resetjp_56_:
{
lean_object* v___x_59_; lean_object* v___x_61_; 
v___x_59_ = lean_nat_mul(v_val_54_, v_val_55_);
lean_dec(v_val_55_);
lean_dec(v_val_54_);
if (v_isShared_58_ == 0)
{
lean_ctor_set(v___x_57_, 0, v___x_59_);
v___x_61_ = v___x_57_;
goto v_reusejp_60_;
}
else
{
lean_object* v_reuseFailAlloc_62_; 
v_reuseFailAlloc_62_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_62_, 0, v___x_59_);
v___x_61_ = v_reuseFailAlloc_62_;
goto v_reusejp_60_;
}
v_reusejp_60_:
{
return v___x_61_;
}
}
}
}
v___jp_47_:
{
lean_object* v___x_49_; uint8_t v___x_50_; 
v___x_49_ = lean_unsigned_to_nat(0u);
v___x_50_ = lean_nat_dec_eq(v_a_48_, v___x_49_);
lean_dec(v_a_48_);
if (v___x_50_ == 0)
{
return v___x_46_;
}
else
{
lean_object* v___x_51_; 
v___x_51_ = ((lean_object*)(lp_mathlib_instCommSemiringENat___aux__2___closed__0));
return v___x_51_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCommSemiringENat___aux__7(lean_object* v_n_66_, lean_object* v_a_67_){
_start:
{
if (lean_obj_tag(v_a_67_) == 0)
{
lean_object* v_zero_68_; uint8_t v_isZero_69_; 
v_zero_68_ = lean_unsigned_to_nat(0u);
v_isZero_69_ = lean_nat_dec_eq(v_n_66_, v_zero_68_);
if (v_isZero_69_ == 1)
{
lean_object* v___x_70_; 
v___x_70_ = ((lean_object*)(lp_mathlib_instCommSemiringENat___aux__7___closed__0));
return v___x_70_;
}
else
{
lean_object* v___x_71_; 
v___x_71_ = lean_box(0);
return v___x_71_;
}
}
else
{
lean_object* v_val_72_; lean_object* v___x_74_; uint8_t v_isShared_75_; uint8_t v_isSharedCheck_80_; 
v_val_72_ = lean_ctor_get(v_a_67_, 0);
v_isSharedCheck_80_ = !lean_is_exclusive(v_a_67_);
if (v_isSharedCheck_80_ == 0)
{
v___x_74_ = v_a_67_;
v_isShared_75_ = v_isSharedCheck_80_;
goto v_resetjp_73_;
}
else
{
lean_inc(v_val_72_);
lean_dec(v_a_67_);
v___x_74_ = lean_box(0);
v_isShared_75_ = v_isSharedCheck_80_;
goto v_resetjp_73_;
}
v_resetjp_73_:
{
lean_object* v___x_76_; lean_object* v___x_78_; 
v___x_76_ = lean_nat_pow(v_val_72_, v_n_66_);
lean_dec(v_val_72_);
if (v_isShared_75_ == 0)
{
lean_ctor_set(v___x_74_, 0, v___x_76_);
v___x_78_ = v___x_74_;
goto v_reusejp_77_;
}
else
{
lean_object* v_reuseFailAlloc_79_; 
v_reuseFailAlloc_79_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_79_, 0, v___x_76_);
v___x_78_ = v_reuseFailAlloc_79_;
goto v_reusejp_77_;
}
v_reusejp_77_:
{
return v___x_78_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCommSemiringENat___aux__7___boxed(lean_object* v_n_81_, lean_object* v_a_82_){
_start:
{
lean_object* v_res_83_; 
v_res_83_ = lp_mathlib_instCommSemiringENat___aux__7(v_n_81_, v_a_82_);
lean_dec(v_n_81_);
return v_res_83_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCommSemiringENat___lam__0(lean_object* v___y_84_, lean_object* v___y_85_){
_start:
{
lean_object* v___x_86_; lean_object* v_a_88_; 
v___x_86_ = lean_box(0);
if (lean_obj_tag(v___y_84_) == 0)
{
if (lean_obj_tag(v___y_85_) == 0)
{
return v___x_86_;
}
else
{
lean_object* v_val_92_; 
v_val_92_ = lean_ctor_get(v___y_85_, 0);
lean_inc(v_val_92_);
lean_dec_ref_known(v___y_85_, 1);
v_a_88_ = v_val_92_;
goto v___jp_87_;
}
}
else
{
if (lean_obj_tag(v___y_85_) == 0)
{
lean_object* v_val_93_; 
v_val_93_ = lean_ctor_get(v___y_84_, 0);
lean_inc(v_val_93_);
lean_dec_ref_known(v___y_84_, 1);
v_a_88_ = v_val_93_;
goto v___jp_87_;
}
else
{
lean_object* v_val_94_; lean_object* v_val_95_; lean_object* v___x_97_; uint8_t v_isShared_98_; uint8_t v_isSharedCheck_103_; 
v_val_94_ = lean_ctor_get(v___y_84_, 0);
lean_inc(v_val_94_);
lean_dec_ref_known(v___y_84_, 1);
v_val_95_ = lean_ctor_get(v___y_85_, 0);
v_isSharedCheck_103_ = !lean_is_exclusive(v___y_85_);
if (v_isSharedCheck_103_ == 0)
{
v___x_97_ = v___y_85_;
v_isShared_98_ = v_isSharedCheck_103_;
goto v_resetjp_96_;
}
else
{
lean_inc(v_val_95_);
lean_dec(v___y_85_);
v___x_97_ = lean_box(0);
v_isShared_98_ = v_isSharedCheck_103_;
goto v_resetjp_96_;
}
v_resetjp_96_:
{
lean_object* v___x_99_; lean_object* v___x_101_; 
v___x_99_ = lean_nat_mul(v_val_94_, v_val_95_);
lean_dec(v_val_95_);
lean_dec(v_val_94_);
if (v_isShared_98_ == 0)
{
lean_ctor_set(v___x_97_, 0, v___x_99_);
v___x_101_ = v___x_97_;
goto v_reusejp_100_;
}
else
{
lean_object* v_reuseFailAlloc_102_; 
v_reuseFailAlloc_102_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_102_, 0, v___x_99_);
v___x_101_ = v_reuseFailAlloc_102_;
goto v_reusejp_100_;
}
v_reusejp_100_:
{
return v___x_101_;
}
}
}
}
v___jp_87_:
{
lean_object* v___x_89_; uint8_t v___x_90_; 
v___x_89_ = lean_unsigned_to_nat(0u);
v___x_90_ = lean_nat_dec_eq(v_a_88_, v___x_89_);
lean_dec(v_a_88_);
if (v___x_90_ == 0)
{
return v___x_86_;
}
else
{
lean_object* v___x_91_; 
v___x_91_ = ((lean_object*)(lp_mathlib_instCommSemiringENat___aux__2___closed__0));
return v___x_91_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCommSemiringENat___lam__1(lean_object* v___y_104_, lean_object* v___y_105_){
_start:
{
if (lean_obj_tag(v___y_105_) == 0)
{
lean_object* v_zero_106_; uint8_t v_isZero_107_; 
v_zero_106_ = lean_unsigned_to_nat(0u);
v_isZero_107_ = lean_nat_dec_eq(v___y_104_, v_zero_106_);
if (v_isZero_107_ == 1)
{
lean_object* v___x_108_; 
v___x_108_ = ((lean_object*)(lp_mathlib_instCommSemiringENat___aux__7___closed__0));
return v___x_108_;
}
else
{
lean_object* v___x_109_; 
v___x_109_ = lean_box(0);
return v___x_109_;
}
}
else
{
lean_object* v_val_110_; lean_object* v___x_112_; uint8_t v_isShared_113_; uint8_t v_isSharedCheck_118_; 
v_val_110_ = lean_ctor_get(v___y_105_, 0);
v_isSharedCheck_118_ = !lean_is_exclusive(v___y_105_);
if (v_isSharedCheck_118_ == 0)
{
v___x_112_ = v___y_105_;
v_isShared_113_ = v_isSharedCheck_118_;
goto v_resetjp_111_;
}
else
{
lean_inc(v_val_110_);
lean_dec(v___y_105_);
v___x_112_ = lean_box(0);
v_isShared_113_ = v_isSharedCheck_118_;
goto v_resetjp_111_;
}
v_resetjp_111_:
{
lean_object* v___x_114_; lean_object* v___x_116_; 
v___x_114_ = lean_nat_pow(v_val_110_, v___y_104_);
lean_dec(v_val_110_);
if (v_isShared_113_ == 0)
{
lean_ctor_set(v___x_112_, 0, v___x_114_);
v___x_116_ = v___x_112_;
goto v_reusejp_115_;
}
else
{
lean_object* v_reuseFailAlloc_117_; 
v_reuseFailAlloc_117_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_117_, 0, v___x_114_);
v___x_116_ = v_reuseFailAlloc_117_;
goto v_reusejp_115_;
}
v_reusejp_115_:
{
return v___x_116_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCommSemiringENat___lam__1___boxed(lean_object* v___y_119_, lean_object* v___y_120_){
_start:
{
lean_object* v_res_121_; 
v_res_121_ = lp_mathlib_instCommSemiringENat___lam__1(v___y_119_, v___y_120_);
lean_dec(v___y_119_);
return v_res_121_;
}
}
static lean_object* _init_lp_mathlib_instCommSemiringENat___closed__2(void){
_start:
{
lean_object* v___f_124_; lean_object* v___f_125_; lean_object* v___x_126_; lean_object* v___x_127_; 
v___f_124_ = ((lean_object*)(lp_mathlib_instCommSemiringENat___closed__1));
v___f_125_ = ((lean_object*)(lp_mathlib_instCommSemiringENat___closed__0));
v___x_126_ = lp_mathlib_instOneENat;
v___x_127_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_127_, 0, v___x_126_);
lean_ctor_set(v___x_127_, 1, v___f_125_);
lean_ctor_set(v___x_127_, 2, v___f_124_);
return v___x_127_;
}
}
static lean_object* _init_lp_mathlib_instCommSemiringENat(void){
_start:
{
lean_object* v___x_128_; lean_object* v_toAddMonoid_129_; lean_object* v___x_130_; lean_object* v___x_131_; lean_object* v___x_132_; 
v___x_128_ = lp_mathlib_instAddMonoidWithOneENat;
v_toAddMonoid_129_ = lean_ctor_get(v___x_128_, 1);
v___x_130_ = ((lean_object*)(lp_mathlib_instAddMonoidWithOneENat___closed__1));
v___x_131_ = lean_obj_once(&lp_mathlib_instCommSemiringENat___closed__2, &lp_mathlib_instCommSemiringENat___closed__2_once, _init_lp_mathlib_instCommSemiringENat___closed__2);
lean_inc_ref(v_toAddMonoid_129_);
v___x_132_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_132_, 0, v_toAddMonoid_129_);
lean_ctor_set(v___x_132_, 1, v___x_131_);
lean_ctor_set(v___x_132_, 2, v___x_130_);
return v___x_132_;
}
}
static lean_object* _init_lp_mathlib_instLinearOrderedAddCommMonoidWithTopENat(void){
_start:
{
lean_object* v___x_133_; lean_object* v_toAddCommMonoid_134_; lean_object* v___x_135_; lean_object* v___x_136_; lean_object* v___x_137_; 
v___x_133_ = lp_mathlib_instCommSemiringENat;
v_toAddCommMonoid_134_ = lean_ctor_get(v___x_133_, 0);
v___x_135_ = lp_mathlib_instLinearOrderENat;
v___x_136_ = lean_box(0);
lean_inc_ref(v_toAddCommMonoid_134_);
v___x_137_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_137_, 0, v_toAddCommMonoid_134_);
lean_ctor_set(v___x_137_, 1, v___x_135_);
lean_ctor_set(v___x_137_, 2, v___x_136_);
return v___x_137_;
}
}
static lean_object* _init_lp_mathlib_ENat_instUniqueUnits(void){
_start:
{
lean_object* v___x_140_; lean_object* v_toMonoid_141_; lean_object* v___x_142_; 
v___x_140_ = lp_mathlib_instCommSemiringENat;
v_toMonoid_141_ = lean_ctor_get(v___x_140_, 1);
v___x_142_ = lp_mathlib_Units_instInhabited___redArg(v_toMonoid_141_);
return v___x_142_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_ENatMap___redArg___lam__0(lean_object* v_f_143_, lean_object* v___y_144_){
_start:
{
lean_object* v___x_145_; 
v___x_145_ = lean_apply_1(v_f_143_, v___y_144_);
return v___x_145_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_ENatMap___redArg(lean_object* v_f_146_){
_start:
{
lean_object* v___f_147_; lean_object* v___x_148_; 
v___f_147_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoidHom_ENatMap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_147_, 0, v_f_146_);
v___x_148_ = lean_alloc_closure((void*)(lp_mathlib_ENat_map), 3, 2);
lean_closure_set(v___x_148_, 0, lean_box(0));
lean_closure_set(v___x_148_, 1, v___f_147_);
return v___x_148_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_ENatMap(lean_object* v_N_149_, lean_object* v_inst_150_, lean_object* v_f_151_){
_start:
{
lean_object* v___x_152_; 
v___x_152_ = lp_mathlib_AddMonoidHom_ENatMap___redArg(v_f_151_);
return v___x_152_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_ENatMap___boxed(lean_object* v_N_153_, lean_object* v_inst_154_, lean_object* v_f_155_){
_start:
{
lean_object* v_res_156_; 
v_res_156_ = lp_mathlib_AddMonoidHom_ENatMap(v_N_153_, v_inst_154_, v_f_155_);
lean_dec_ref(v_inst_154_);
return v_res_156_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_ENatMap___redArg(lean_object* v_f_157_){
_start:
{
lean_object* v___f_158_; lean_object* v___x_159_; 
v___f_158_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoidHom_ENatMap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_158_, 0, v_f_157_);
v___x_159_ = lean_alloc_closure((void*)(lp_mathlib_ENat_map), 3, 2);
lean_closure_set(v___x_159_, 0, lean_box(0));
lean_closure_set(v___x_159_, 1, v___f_158_);
return v___x_159_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_ENatMap(lean_object* v_S_160_, lean_object* v_inst_161_, lean_object* v_inst_162_, lean_object* v_inst_163_, lean_object* v_f_164_, lean_object* v_hf_165_){
_start:
{
lean_object* v___x_166_; 
v___x_166_ = lp_mathlib_MonoidWithZeroHom_ENatMap___redArg(v_f_164_);
return v___x_166_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_ENatMap___boxed(lean_object* v_S_167_, lean_object* v_inst_168_, lean_object* v_inst_169_, lean_object* v_inst_170_, lean_object* v_f_171_, lean_object* v_hf_172_){
_start:
{
lean_object* v_res_173_; 
v_res_173_ = lp_mathlib_MonoidWithZeroHom_ENatMap(v_S_167_, v_inst_168_, v_inst_169_, v_inst_170_, v_f_171_, v_hf_172_);
lean_dec_ref(v_inst_169_);
lean_dec_ref(v_inst_168_);
return v_res_173_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_ENatMap___redArg(lean_object* v_f_174_){
_start:
{
lean_object* v___x_175_; 
v___x_175_ = lp_mathlib_MonoidWithZeroHom_ENatMap___redArg(v_f_174_);
return v___x_175_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_ENatMap(lean_object* v_S_176_, lean_object* v_inst_177_, lean_object* v_inst_178_, lean_object* v_inst_179_, lean_object* v_inst_180_, lean_object* v_inst_181_, lean_object* v_f_182_, lean_object* v_hf_183_){
_start:
{
lean_object* v___x_184_; 
v___x_184_ = lp_mathlib_MonoidWithZeroHom_ENatMap___redArg(v_f_182_);
return v___x_184_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RingHom_ENatMap___boxed(lean_object* v_S_185_, lean_object* v_inst_186_, lean_object* v_inst_187_, lean_object* v_inst_188_, lean_object* v_inst_189_, lean_object* v_inst_190_, lean_object* v_f_191_, lean_object* v_hf_192_){
_start:
{
lean_object* v_res_193_; 
v_res_193_ = lp_mathlib_RingHom_ENatMap(v_S_185_, v_inst_186_, v_inst_187_, v_inst_188_, v_inst_189_, v_inst_190_, v_f_191_, v_hf_192_);
lean_dec_ref(v_inst_189_);
lean_dec_ref(v_inst_187_);
lean_dec_ref(v_inst_186_);
return v_res_193_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Ring_Nat(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Ring_WithTop(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_ENat_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Nat_Units(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Cast_Order_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_ENat_Monoid(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Ring_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Ring_WithTop(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_ENat_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Nat_Units(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Cast_Order_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_instAddMonoidWithOneENat = _init_lp_mathlib_instAddMonoidWithOneENat();
lean_mark_persistent(lp_mathlib_instAddMonoidWithOneENat);
lp_mathlib_instCommSemiringENat = _init_lp_mathlib_instCommSemiringENat();
lean_mark_persistent(lp_mathlib_instCommSemiringENat);
lp_mathlib_instLinearOrderedAddCommMonoidWithTopENat = _init_lp_mathlib_instLinearOrderedAddCommMonoidWithTopENat();
lean_mark_persistent(lp_mathlib_instLinearOrderedAddCommMonoidWithTopENat);
lp_mathlib_ENat_instUniqueUnits = _init_lp_mathlib_ENat_instUniqueUnits();
lean_mark_persistent(lp_mathlib_ENat_instUniqueUnits);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_ENat_Monoid(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Ring_Nat(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Ring_WithTop(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_ENat_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Nat_Units(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Cast_Order_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_ENat_Monoid(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Ring_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Ring_WithTop(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_ENat_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Nat_Units(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Cast_Order_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_ENat_Monoid(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_ENat_Monoid(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_ENat_Monoid(builtin);
}
#ifdef __cplusplus
}
#endif
