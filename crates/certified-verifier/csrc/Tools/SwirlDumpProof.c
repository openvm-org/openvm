// Lean compiler output
// Module: Tools.SwirlDumpProof
// Imports: public import Init public import Swirl.Protocol.Noninteractive.Wire.Raw
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
LEAN_EXPORT lean_object* l_List_mapTR_loop___at___00Tools_SwirlDumpProof_pvDigest_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* l_IO_println___at___00Tools_SwirlDumpProof_main_spec__0___boxed(lean_object*, lean_object*);
static lean_object* l_Tools_SwirlDumpProof_main___closed__4;
LEAN_EXPORT lean_object* _lean_main();
lean_object* l_Swirl_Protocol_Noninteractive_Wire_Raw_readRawPublicValues(lean_object*, lean_object*);
uint8_t lean_byte_array_fget(lean_object*, lean_object*);
static lean_object* l_Tools_SwirlDumpProof_digestToCsv___closed__0;
lean_object* lean_uint32_to_nat(uint32_t);
LEAN_EXPORT lean_object* l_Tools_SwirlDumpProof_parseThreeBlobs(lean_object*);
uint64_t lean_uint64_lor(uint64_t, uint64_t);
LEAN_EXPORT lean_object* l_Tools_SwirlDumpProof_parseThreeBlobs___boxed(lean_object*);
lean_object* l_ByteArray_extract(lean_object*, lean_object*, lean_object*);
static lean_object* l_Tools_SwirlDumpProof_main___closed__6;
LEAN_EXPORT lean_object* l_main___boxed(lean_object*);
static lean_object* l_Tools_SwirlDumpProof_main___closed__2;
lean_object* lean_string_push(lean_object*, uint32_t);
lean_object* lean_get_stdout();
lean_object* l_Nat_reprFast(lean_object*);
lean_object* l_IO_FS_Stream_readBinToEnd(lean_object*);
LEAN_EXPORT lean_object* l_Tools_SwirlDumpProof_proofDigest(lean_object*);
uint64_t lean_uint8_to_uint64(uint8_t);
lean_object* lean_uint64_to_nat(uint64_t);
lean_object* lean_array_to_list(lean_object*);
LEAN_EXPORT lean_object* l_Tools_SwirlDumpProof_main___boxed__const__1;
lean_object* l_Swirl_Protocol_Noninteractive_Wire_Raw_readRawProof(lean_object*);
lean_object* lean_get_stdin();
lean_object* lean_get_stderr();
LEAN_EXPORT uint32_t l_Tools_SwirlDumpProof_parseErrorExitCode(lean_object*);
LEAN_EXPORT lean_object* l_Tools_SwirlDumpProof_sliceBytes(lean_object*, lean_object*, lean_object*);
static lean_object* l_Tools_SwirlDumpProof_pvDigest___closed__0;
LEAN_EXPORT lean_object* l_IO_println___at___00Tools_SwirlDumpProof_main_spec__0(lean_object*);
lean_object* l_Swirl_Protocol_Noninteractive_Wire_Raw_readRawVk(lean_object*);
lean_object* l_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_toString(lean_object*);
LEAN_EXPORT lean_object* l_Tools_SwirlDumpProof_readU64LE___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* l_Tools_SwirlDumpProof_digestToCsv(lean_object*);
lean_object* l_outOfBounds___redArg(lean_object*);
extern uint8_t l_instInhabitedUInt8;
static lean_object* l_Tools_SwirlDumpProof_main___closed__0;
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
LEAN_EXPORT lean_object* l_Tools_SwirlDumpProof_main();
LEAN_EXPORT lean_object* l_Tools_SwirlDumpProof_vkDigest(lean_object*);
LEAN_EXPORT lean_object* l_Tools_SwirlDumpProof_main___boxed(lean_object*);
LEAN_EXPORT lean_object* l_Tools_SwirlDumpProof_fbbArrToCsv(lean_object*);
static lean_object* l_Tools_SwirlDumpProof_main___closed__3;
LEAN_EXPORT lean_object* l_Tools_SwirlDumpProof_readU64LE(lean_object*, lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
LEAN_EXPORT lean_object* l_Tools_SwirlDumpProof_parseErrorExitCode___boxed(lean_object*);
uint64_t lean_uint64_shift_left(uint64_t, uint64_t);
LEAN_EXPORT lean_object* l_IO_print___at___00IO_println___at___00Tools_SwirlDumpProof_main_spec__0_spec__0___boxed(lean_object*, lean_object*);
lean_object* l_String_intercalate(lean_object*, lean_object*);
LEAN_EXPORT lean_object* l_Tools_SwirlDumpProof_readU64LE___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* l_IO_print___at___00IO_println___at___00Tools_SwirlDumpProof_main_spec__0_spec__0(lean_object*);
LEAN_EXPORT uint8_t l_Tools_SwirlDumpProof_readU64LE___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* l_Tools_SwirlDumpProof_sliceBytes___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* l_Tools_SwirlDumpProof_pvDigest(lean_object*);
static lean_object* l_Tools_SwirlDumpProof_main___closed__5;
LEAN_EXPORT lean_object* l_Tools_SwirlDumpProof_main___boxed__const__2;
lean_object* l_IO_FS_Stream_putStrLn(lean_object*, lean_object*);
static lean_object* l_Tools_SwirlDumpProof_main___closed__7;
lean_object* lean_string_append(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
LEAN_EXPORT lean_object* l_List_mapTR_loop___at___00Tools_SwirlDumpProof_digestToCsv_spec__0(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_byte_array_size(lean_object*);
static lean_object* l_Tools_SwirlDumpProof_main___closed__1;
LEAN_EXPORT uint8_t l_Tools_SwirlDumpProof_readU64LE___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; uint8_t x_4; 
x_3 = lean_byte_array_size(x_1);
x_4 = lean_nat_dec_lt(x_2, x_3);
lean_dec(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* l_Tools_SwirlDumpProof_readU64LE(lean_object* x_1, lean_object* x_2) {
_start:
{
uint64_t x_3; uint64_t x_4; uint64_t x_5; uint64_t x_6; uint64_t x_7; uint64_t x_8; uint64_t x_9; uint8_t x_10; lean_object* x_36; lean_object* x_37; lean_object* x_38; uint8_t x_39; 
x_36 = lean_unsigned_to_nat(8u);
x_37 = lean_nat_add(x_2, x_36);
x_38 = lean_byte_array_size(x_1);
x_39 = lean_nat_dec_le(x_37, x_38);
lean_dec(x_38);
lean_dec(x_37);
if (x_39 == 0)
{
lean_object* x_40; 
x_40 = lean_box(0);
return x_40;
}
else
{
uint8_t x_41; uint64_t x_42; uint64_t x_43; uint64_t x_44; uint64_t x_45; uint64_t x_46; uint64_t x_47; uint8_t x_48; uint64_t x_58; uint64_t x_59; uint64_t x_60; uint64_t x_61; uint64_t x_62; uint8_t x_63; uint64_t x_73; uint64_t x_74; uint64_t x_75; uint64_t x_76; uint8_t x_77; uint64_t x_87; uint64_t x_88; uint64_t x_89; uint8_t x_90; uint64_t x_100; uint64_t x_101; uint8_t x_102; uint64_t x_112; uint8_t x_113; uint8_t x_123; uint8_t x_133; 
x_41 = l_instInhabitedUInt8;
x_133 = l_Tools_SwirlDumpProof_readU64LE___lam__0(x_1, x_2);
if (x_133 == 0)
{
lean_object* x_134; lean_object* x_135; uint8_t x_136; 
x_134 = lean_box(x_41);
x_135 = l_outOfBounds___redArg(x_134);
x_136 = lean_unbox(x_135);
x_123 = x_136;
goto block_132;
}
else
{
uint8_t x_137; 
x_137 = lean_byte_array_fget(x_1, x_2);
x_123 = x_137;
goto block_132;
}
block_57:
{
uint64_t x_49; lean_object* x_50; lean_object* x_51; uint8_t x_52; 
x_49 = lean_uint8_to_uint64(x_48);
x_50 = lean_unsigned_to_nat(7u);
x_51 = lean_nat_add(x_2, x_50);
x_52 = l_Tools_SwirlDumpProof_readU64LE___lam__0(x_1, x_51);
if (x_52 == 0)
{
lean_object* x_53; lean_object* x_54; uint8_t x_55; 
lean_dec(x_51);
x_53 = lean_box(x_41);
x_54 = l_outOfBounds___redArg(x_53);
x_55 = lean_unbox(x_54);
x_3 = x_43;
x_4 = x_42;
x_5 = x_49;
x_6 = x_44;
x_7 = x_45;
x_8 = x_46;
x_9 = x_47;
x_10 = x_55;
goto block_35;
}
else
{
uint8_t x_56; 
x_56 = lean_byte_array_fget(x_1, x_51);
lean_dec(x_51);
x_3 = x_43;
x_4 = x_42;
x_5 = x_49;
x_6 = x_44;
x_7 = x_45;
x_8 = x_46;
x_9 = x_47;
x_10 = x_56;
goto block_35;
}
}
block_72:
{
uint64_t x_64; lean_object* x_65; lean_object* x_66; uint8_t x_67; 
x_64 = lean_uint8_to_uint64(x_63);
x_65 = lean_unsigned_to_nat(6u);
x_66 = lean_nat_add(x_2, x_65);
x_67 = l_Tools_SwirlDumpProof_readU64LE___lam__0(x_1, x_66);
if (x_67 == 0)
{
lean_object* x_68; lean_object* x_69; uint8_t x_70; 
lean_dec(x_66);
x_68 = lean_box(x_41);
x_69 = l_outOfBounds___redArg(x_68);
x_70 = lean_unbox(x_69);
x_42 = x_59;
x_43 = x_58;
x_44 = x_60;
x_45 = x_61;
x_46 = x_62;
x_47 = x_64;
x_48 = x_70;
goto block_57;
}
else
{
uint8_t x_71; 
x_71 = lean_byte_array_fget(x_1, x_66);
lean_dec(x_66);
x_42 = x_59;
x_43 = x_58;
x_44 = x_60;
x_45 = x_61;
x_46 = x_62;
x_47 = x_64;
x_48 = x_71;
goto block_57;
}
}
block_86:
{
uint64_t x_78; lean_object* x_79; lean_object* x_80; uint8_t x_81; 
x_78 = lean_uint8_to_uint64(x_77);
x_79 = lean_unsigned_to_nat(5u);
x_80 = lean_nat_add(x_2, x_79);
x_81 = l_Tools_SwirlDumpProof_readU64LE___lam__0(x_1, x_80);
if (x_81 == 0)
{
lean_object* x_82; lean_object* x_83; uint8_t x_84; 
lean_dec(x_80);
x_82 = lean_box(x_41);
x_83 = l_outOfBounds___redArg(x_82);
x_84 = lean_unbox(x_83);
x_58 = x_78;
x_59 = x_73;
x_60 = x_74;
x_61 = x_75;
x_62 = x_76;
x_63 = x_84;
goto block_72;
}
else
{
uint8_t x_85; 
x_85 = lean_byte_array_fget(x_1, x_80);
lean_dec(x_80);
x_58 = x_78;
x_59 = x_73;
x_60 = x_74;
x_61 = x_75;
x_62 = x_76;
x_63 = x_85;
goto block_72;
}
}
block_99:
{
uint64_t x_91; lean_object* x_92; lean_object* x_93; uint8_t x_94; 
x_91 = lean_uint8_to_uint64(x_90);
x_92 = lean_unsigned_to_nat(4u);
x_93 = lean_nat_add(x_2, x_92);
x_94 = l_Tools_SwirlDumpProof_readU64LE___lam__0(x_1, x_93);
if (x_94 == 0)
{
lean_object* x_95; lean_object* x_96; uint8_t x_97; 
lean_dec(x_93);
x_95 = lean_box(x_41);
x_96 = l_outOfBounds___redArg(x_95);
x_97 = lean_unbox(x_96);
x_73 = x_87;
x_74 = x_88;
x_75 = x_89;
x_76 = x_91;
x_77 = x_97;
goto block_86;
}
else
{
uint8_t x_98; 
x_98 = lean_byte_array_fget(x_1, x_93);
lean_dec(x_93);
x_73 = x_87;
x_74 = x_88;
x_75 = x_89;
x_76 = x_91;
x_77 = x_98;
goto block_86;
}
}
block_111:
{
uint64_t x_103; lean_object* x_104; lean_object* x_105; uint8_t x_106; 
x_103 = lean_uint8_to_uint64(x_102);
x_104 = lean_unsigned_to_nat(3u);
x_105 = lean_nat_add(x_2, x_104);
x_106 = l_Tools_SwirlDumpProof_readU64LE___lam__0(x_1, x_105);
if (x_106 == 0)
{
lean_object* x_107; lean_object* x_108; uint8_t x_109; 
lean_dec(x_105);
x_107 = lean_box(x_41);
x_108 = l_outOfBounds___redArg(x_107);
x_109 = lean_unbox(x_108);
x_87 = x_100;
x_88 = x_103;
x_89 = x_101;
x_90 = x_109;
goto block_99;
}
else
{
uint8_t x_110; 
x_110 = lean_byte_array_fget(x_1, x_105);
lean_dec(x_105);
x_87 = x_100;
x_88 = x_103;
x_89 = x_101;
x_90 = x_110;
goto block_99;
}
}
block_122:
{
uint64_t x_114; lean_object* x_115; lean_object* x_116; uint8_t x_117; 
x_114 = lean_uint8_to_uint64(x_113);
x_115 = lean_unsigned_to_nat(2u);
x_116 = lean_nat_add(x_2, x_115);
x_117 = l_Tools_SwirlDumpProof_readU64LE___lam__0(x_1, x_116);
if (x_117 == 0)
{
lean_object* x_118; lean_object* x_119; uint8_t x_120; 
lean_dec(x_116);
x_118 = lean_box(x_41);
x_119 = l_outOfBounds___redArg(x_118);
x_120 = lean_unbox(x_119);
x_100 = x_114;
x_101 = x_112;
x_102 = x_120;
goto block_111;
}
else
{
uint8_t x_121; 
x_121 = lean_byte_array_fget(x_1, x_116);
lean_dec(x_116);
x_100 = x_114;
x_101 = x_112;
x_102 = x_121;
goto block_111;
}
}
block_132:
{
uint64_t x_124; lean_object* x_125; lean_object* x_126; uint8_t x_127; 
x_124 = lean_uint8_to_uint64(x_123);
x_125 = lean_unsigned_to_nat(1u);
x_126 = lean_nat_add(x_2, x_125);
x_127 = l_Tools_SwirlDumpProof_readU64LE___lam__0(x_1, x_126);
if (x_127 == 0)
{
lean_object* x_128; lean_object* x_129; uint8_t x_130; 
lean_dec(x_126);
x_128 = lean_box(x_41);
x_129 = l_outOfBounds___redArg(x_128);
x_130 = lean_unbox(x_129);
x_112 = x_124;
x_113 = x_130;
goto block_122;
}
else
{
uint8_t x_131; 
x_131 = lean_byte_array_fget(x_1, x_126);
lean_dec(x_126);
x_112 = x_124;
x_113 = x_131;
goto block_122;
}
}
}
block_35:
{
uint64_t x_11; uint64_t x_12; uint64_t x_13; uint64_t x_14; uint64_t x_15; uint64_t x_16; uint64_t x_17; uint64_t x_18; uint64_t x_19; uint64_t x_20; uint64_t x_21; uint64_t x_22; uint64_t x_23; uint64_t x_24; uint64_t x_25; uint64_t x_26; uint64_t x_27; uint64_t x_28; uint64_t x_29; uint64_t x_30; uint64_t x_31; uint64_t x_32; lean_object* x_33; lean_object* x_34; 
x_11 = lean_uint8_to_uint64(x_10);
x_12 = 8;
x_13 = lean_uint64_shift_left(x_4, x_12);
x_14 = lean_uint64_lor(x_7, x_13);
x_15 = 16;
x_16 = lean_uint64_shift_left(x_6, x_15);
x_17 = lean_uint64_lor(x_14, x_16);
x_18 = 24;
x_19 = lean_uint64_shift_left(x_8, x_18);
x_20 = lean_uint64_lor(x_17, x_19);
x_21 = 32;
x_22 = lean_uint64_shift_left(x_3, x_21);
x_23 = lean_uint64_lor(x_20, x_22);
x_24 = 40;
x_25 = lean_uint64_shift_left(x_9, x_24);
x_26 = lean_uint64_lor(x_23, x_25);
x_27 = 48;
x_28 = lean_uint64_shift_left(x_5, x_27);
x_29 = lean_uint64_lor(x_26, x_28);
x_30 = 56;
x_31 = lean_uint64_shift_left(x_11, x_30);
x_32 = lean_uint64_lor(x_29, x_31);
x_33 = lean_box_uint64(x_32);
x_34 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_34, 0, x_33);
return x_34;
}
}
}
LEAN_EXPORT lean_object* l_Tools_SwirlDumpProof_readU64LE___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = l_Tools_SwirlDumpProof_readU64LE___lam__0(x_1, x_2);
lean_dec(x_2);
lean_dec_ref(x_1);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* l_Tools_SwirlDumpProof_readU64LE___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = l_Tools_SwirlDumpProof_readU64LE(x_1, x_2);
lean_dec(x_2);
lean_dec_ref(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* l_Tools_SwirlDumpProof_sliceBytes(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; uint8_t x_6; 
x_4 = lean_nat_add(x_2, x_3);
x_5 = lean_byte_array_size(x_1);
x_6 = lean_nat_dec_le(x_4, x_5);
lean_dec(x_5);
if (x_6 == 0)
{
lean_object* x_7; 
lean_dec(x_4);
lean_dec(x_2);
x_7 = lean_box(0);
return x_7;
}
else
{
lean_object* x_8; lean_object* x_9; 
x_8 = l_ByteArray_extract(x_1, x_2, x_4);
lean_dec(x_4);
x_9 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_9, 0, x_8);
return x_9;
}
}
}
LEAN_EXPORT lean_object* l_Tools_SwirlDumpProof_sliceBytes___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = l_Tools_SwirlDumpProof_sliceBytes(x_1, x_2, x_3);
lean_dec(x_3);
lean_dec_ref(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* l_Tools_SwirlDumpProof_parseThreeBlobs(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_unsigned_to_nat(0u);
x_3 = l_Tools_SwirlDumpProof_readU64LE(x_1, x_2);
if (lean_obj_tag(x_3) == 0)
{
lean_object* x_4; 
x_4 = lean_box(0);
return x_4;
}
else
{
lean_object* x_5; lean_object* x_6; uint64_t x_7; lean_object* x_8; lean_object* x_9; 
x_5 = lean_ctor_get(x_3, 0);
lean_inc(x_5);
lean_dec_ref(x_3);
x_6 = lean_unsigned_to_nat(8u);
x_7 = lean_unbox_uint64(x_5);
lean_dec(x_5);
x_8 = lean_uint64_to_nat(x_7);
x_9 = l_Tools_SwirlDumpProof_sliceBytes(x_1, x_6, x_8);
if (lean_obj_tag(x_9) == 0)
{
lean_object* x_10; 
lean_dec(x_8);
x_10 = lean_box(0);
return x_10;
}
else
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_11 = lean_ctor_get(x_9, 0);
lean_inc(x_11);
lean_dec_ref(x_9);
x_12 = lean_nat_add(x_6, x_8);
lean_dec(x_8);
x_13 = l_Tools_SwirlDumpProof_readU64LE(x_1, x_12);
if (lean_obj_tag(x_13) == 0)
{
lean_object* x_14; 
lean_dec(x_12);
lean_dec(x_11);
x_14 = lean_box(0);
return x_14;
}
else
{
lean_object* x_15; lean_object* x_16; uint64_t x_17; lean_object* x_18; lean_object* x_19; 
x_15 = lean_ctor_get(x_13, 0);
lean_inc(x_15);
lean_dec_ref(x_13);
x_16 = lean_nat_add(x_12, x_6);
lean_dec(x_12);
x_17 = lean_unbox_uint64(x_15);
lean_dec(x_15);
x_18 = lean_uint64_to_nat(x_17);
lean_inc(x_16);
x_19 = l_Tools_SwirlDumpProof_sliceBytes(x_1, x_16, x_18);
if (lean_obj_tag(x_19) == 0)
{
lean_object* x_20; 
lean_dec(x_18);
lean_dec(x_16);
lean_dec(x_11);
x_20 = lean_box(0);
return x_20;
}
else
{
lean_object* x_21; lean_object* x_22; lean_object* x_23; 
x_21 = lean_ctor_get(x_19, 0);
lean_inc(x_21);
lean_dec_ref(x_19);
x_22 = lean_nat_add(x_16, x_18);
lean_dec(x_18);
lean_dec(x_16);
x_23 = l_Tools_SwirlDumpProof_readU64LE(x_1, x_22);
if (lean_obj_tag(x_23) == 0)
{
lean_object* x_24; 
lean_dec(x_22);
lean_dec(x_21);
lean_dec(x_11);
x_24 = lean_box(0);
return x_24;
}
else
{
lean_object* x_25; lean_object* x_26; uint64_t x_27; lean_object* x_28; lean_object* x_29; 
x_25 = lean_ctor_get(x_23, 0);
lean_inc(x_25);
lean_dec_ref(x_23);
x_26 = lean_nat_add(x_22, x_6);
lean_dec(x_22);
x_27 = lean_unbox_uint64(x_25);
lean_dec(x_25);
x_28 = lean_uint64_to_nat(x_27);
x_29 = l_Tools_SwirlDumpProof_sliceBytes(x_1, x_26, x_28);
lean_dec(x_28);
if (lean_obj_tag(x_29) == 0)
{
lean_object* x_30; 
lean_dec(x_21);
lean_dec(x_11);
x_30 = lean_box(0);
return x_30;
}
else
{
uint8_t x_31; 
x_31 = !lean_is_exclusive(x_29);
if (x_31 == 0)
{
lean_object* x_32; lean_object* x_33; lean_object* x_34; 
x_32 = lean_ctor_get(x_29, 0);
x_33 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_33, 0, x_21);
lean_ctor_set(x_33, 1, x_32);
x_34 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_34, 0, x_11);
lean_ctor_set(x_34, 1, x_33);
lean_ctor_set(x_29, 0, x_34);
return x_29;
}
else
{
lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; 
x_35 = lean_ctor_get(x_29, 0);
lean_inc(x_35);
lean_dec(x_29);
x_36 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_36, 0, x_21);
lean_ctor_set(x_36, 1, x_35);
x_37 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_37, 0, x_11);
lean_ctor_set(x_37, 1, x_36);
x_38 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_38, 0, x_37);
return x_38;
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* l_Tools_SwirlDumpProof_parseThreeBlobs___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = l_Tools_SwirlDumpProof_parseThreeBlobs(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* l_List_mapTR_loop___at___00Tools_SwirlDumpProof_digestToCsv_spec__0(lean_object* x_1, lean_object* x_2) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_3; 
x_3 = l_List_reverse___redArg(x_2);
return x_3;
}
else
{
uint8_t x_4; 
x_4 = !lean_is_exclusive(x_1);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; uint32_t x_7; lean_object* x_8; lean_object* x_9; 
x_5 = lean_ctor_get(x_1, 0);
x_6 = lean_ctor_get(x_1, 1);
x_7 = lean_unbox_uint32(x_5);
lean_dec(x_5);
x_8 = lean_uint32_to_nat(x_7);
x_9 = l_Nat_reprFast(x_8);
lean_ctor_set(x_1, 1, x_2);
lean_ctor_set(x_1, 0, x_9);
{
lean_object* _tmp_0 = x_6;
lean_object* _tmp_1 = x_1;
x_1 = _tmp_0;
x_2 = _tmp_1;
}
goto _start;
}
else
{
lean_object* x_11; lean_object* x_12; uint32_t x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; 
x_11 = lean_ctor_get(x_1, 0);
x_12 = lean_ctor_get(x_1, 1);
lean_inc(x_12);
lean_inc(x_11);
lean_dec(x_1);
x_13 = lean_unbox_uint32(x_11);
lean_dec(x_11);
x_14 = lean_uint32_to_nat(x_13);
x_15 = l_Nat_reprFast(x_14);
x_16 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_16, 0, x_15);
lean_ctor_set(x_16, 1, x_2);
x_1 = x_12;
x_2 = x_16;
goto _start;
}
}
}
}
static lean_object* _init_l_Tools_SwirlDumpProof_digestToCsv___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked(",", 1, 1);
return x_1;
}
}
LEAN_EXPORT lean_object* l_Tools_SwirlDumpProof_digestToCsv(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_2 = l_Tools_SwirlDumpProof_digestToCsv___closed__0;
x_3 = lean_array_to_list(x_1);
x_4 = lean_box(0);
x_5 = l_List_mapTR_loop___at___00Tools_SwirlDumpProof_digestToCsv_spec__0(x_3, x_4);
x_6 = l_String_intercalate(x_2, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* l_Tools_SwirlDumpProof_fbbArrToCsv(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_2 = l_Tools_SwirlDumpProof_digestToCsv___closed__0;
x_3 = lean_array_to_list(x_1);
x_4 = lean_box(0);
x_5 = l_List_mapTR_loop___at___00Tools_SwirlDumpProof_digestToCsv_spec__0(x_3, x_4);
x_6 = l_String_intercalate(x_2, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* l_Tools_SwirlDumpProof_proofDigest(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
lean_dec_ref(x_1);
x_3 = l_Tools_SwirlDumpProof_digestToCsv(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* l_Tools_SwirlDumpProof_vkDigest(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_ctor_get(x_1, 1);
lean_inc_ref(x_2);
lean_dec_ref(x_1);
x_3 = l_Tools_SwirlDumpProof_digestToCsv(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* l_List_mapTR_loop___at___00Tools_SwirlDumpProof_pvDigest_spec__0(lean_object* x_1, lean_object* x_2) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_3; 
x_3 = l_List_reverse___redArg(x_2);
return x_3;
}
else
{
uint8_t x_4; 
x_4 = !lean_is_exclusive(x_1);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_5 = lean_ctor_get(x_1, 0);
x_6 = lean_ctor_get(x_1, 1);
x_7 = l_Tools_SwirlDumpProof_fbbArrToCsv(x_5);
lean_ctor_set(x_1, 1, x_2);
lean_ctor_set(x_1, 0, x_7);
{
lean_object* _tmp_0 = x_6;
lean_object* _tmp_1 = x_1;
x_1 = _tmp_0;
x_2 = _tmp_1;
}
goto _start;
}
else
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_9 = lean_ctor_get(x_1, 0);
x_10 = lean_ctor_get(x_1, 1);
lean_inc(x_10);
lean_inc(x_9);
lean_dec(x_1);
x_11 = l_Tools_SwirlDumpProof_fbbArrToCsv(x_9);
x_12 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_12, 0, x_11);
lean_ctor_set(x_12, 1, x_2);
x_1 = x_10;
x_2 = x_12;
goto _start;
}
}
}
}
static lean_object* _init_l_Tools_SwirlDumpProof_pvDigest___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("|", 1, 1);
return x_1;
}
}
LEAN_EXPORT lean_object* l_Tools_SwirlDumpProof_pvDigest(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_2 = l_Tools_SwirlDumpProof_pvDigest___closed__0;
x_3 = lean_array_to_list(x_1);
x_4 = lean_box(0);
x_5 = l_List_mapTR_loop___at___00Tools_SwirlDumpProof_pvDigest_spec__0(x_3, x_4);
x_6 = l_String_intercalate(x_2, x_5);
return x_6;
}
}
LEAN_EXPORT uint32_t l_Tools_SwirlDumpProof_parseErrorExitCode(lean_object* x_1) {
_start:
{
switch (lean_obj_tag(x_1)) {
case 0:
{
uint32_t x_2; 
x_2 = 10;
return x_2;
}
case 1:
{
uint32_t x_3; 
x_3 = 11;
return x_3;
}
case 2:
{
uint32_t x_4; 
x_4 = 12;
return x_4;
}
default: 
{
uint32_t x_5; 
x_5 = 13;
return x_5;
}
}
}
}
LEAN_EXPORT lean_object* l_Tools_SwirlDumpProof_parseErrorExitCode___boxed(lean_object* x_1) {
_start:
{
uint32_t x_2; lean_object* x_3; 
x_2 = l_Tools_SwirlDumpProof_parseErrorExitCode(x_1);
lean_dec_ref(x_1);
x_3 = lean_box_uint32(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* l_IO_print___at___00IO_println___at___00Tools_SwirlDumpProof_main_spec__0_spec__0(lean_object* x_1) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lean_get_stdout();
x_4 = lean_ctor_get(x_3, 4);
lean_inc_ref(x_4);
lean_dec_ref(x_3);
x_5 = lean_apply_2(x_4, x_1, lean_box(0));
return x_5;
}
}
LEAN_EXPORT lean_object* l_IO_println___at___00Tools_SwirlDumpProof_main_spec__0(lean_object* x_1) {
_start:
{
uint32_t x_3; lean_object* x_4; lean_object* x_5; 
x_3 = 10;
x_4 = lean_string_push(x_1, x_3);
x_5 = l_IO_print___at___00IO_println___at___00Tools_SwirlDumpProof_main_spec__0_spec__0(x_4);
return x_5;
}
}
static lean_object* _init_l_Tools_SwirlDumpProof_main___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("swirl_dump_proof: stdin framing error (received ", 48, 48);
return x_1;
}
}
static lean_object* _init_l_Tools_SwirlDumpProof_main___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked(" bytes)", 7, 7);
return x_1;
}
}
static lean_object* _init_l_Tools_SwirlDumpProof_main___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("swirl_dump_proof: vk parse error: ", 34, 34);
return x_1;
}
}
static lean_object* _init_l_Tools_SwirlDumpProof_main___closed__3() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("swirl_dump_proof: proof parse error: ", 37, 37);
return x_1;
}
}
static lean_object* _init_l_Tools_SwirlDumpProof_main___closed__4() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("swirl_dump_proof: pv parse error: ", 34, 34);
return x_1;
}
}
static lean_object* _init_l_Tools_SwirlDumpProof_main___closed__5() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("vk: ", 4, 4);
return x_1;
}
}
static lean_object* _init_l_Tools_SwirlDumpProof_main___closed__6() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("proof: ", 7, 7);
return x_1;
}
}
static lean_object* _init_l_Tools_SwirlDumpProof_main___closed__7() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("pv: ", 4, 4);
return x_1;
}
}
static lean_object* _init_l_Tools_SwirlDumpProof_main___boxed__const__1() {
_start:
{
uint32_t x_1; lean_object* x_2; 
x_1 = 20;
x_2 = lean_box_uint32(x_1);
return x_2;
}
}
static lean_object* _init_l_Tools_SwirlDumpProof_main___boxed__const__2() {
_start:
{
uint32_t x_1; lean_object* x_2; 
x_1 = 0;
x_2 = lean_box_uint32(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* l_Tools_SwirlDumpProof_main() {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_get_stdin();
x_3 = l_IO_FS_Stream_readBinToEnd(x_2);
if (lean_obj_tag(x_3) == 0)
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
lean_dec_ref(x_3);
x_5 = lean_get_stderr();
x_6 = l_Tools_SwirlDumpProof_parseThreeBlobs(x_4);
if (lean_obj_tag(x_6) == 0)
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_7 = l_Tools_SwirlDumpProof_main___closed__0;
x_8 = lean_byte_array_size(x_4);
lean_dec(x_4);
x_9 = l_Nat_reprFast(x_8);
x_10 = lean_string_append(x_7, x_9);
lean_dec_ref(x_9);
x_11 = l_Tools_SwirlDumpProof_main___closed__1;
x_12 = lean_string_append(x_10, x_11);
x_13 = l_IO_FS_Stream_putStrLn(x_5, x_12);
if (lean_obj_tag(x_13) == 0)
{
uint8_t x_14; 
x_14 = !lean_is_exclusive(x_13);
if (x_14 == 0)
{
lean_object* x_15; lean_object* x_16; 
x_15 = lean_ctor_get(x_13, 0);
lean_dec(x_15);
x_16 = l_Tools_SwirlDumpProof_main___boxed__const__1;
lean_ctor_set(x_13, 0, x_16);
return x_13;
}
else
{
lean_object* x_17; lean_object* x_18; 
lean_dec(x_13);
x_17 = l_Tools_SwirlDumpProof_main___boxed__const__1;
x_18 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_18, 0, x_17);
return x_18;
}
}
else
{
uint8_t x_19; 
x_19 = !lean_is_exclusive(x_13);
if (x_19 == 0)
{
return x_13;
}
else
{
lean_object* x_20; lean_object* x_21; 
x_20 = lean_ctor_get(x_13, 0);
lean_inc(x_20);
lean_dec(x_13);
x_21 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_21, 0, x_20);
return x_21;
}
}
}
else
{
lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; 
lean_dec(x_4);
x_22 = lean_ctor_get(x_6, 0);
lean_inc(x_22);
lean_dec_ref(x_6);
x_23 = lean_ctor_get(x_22, 1);
lean_inc(x_23);
x_24 = lean_ctor_get(x_22, 0);
lean_inc(x_24);
lean_dec(x_22);
x_25 = lean_ctor_get(x_23, 0);
lean_inc(x_25);
x_26 = lean_ctor_get(x_23, 1);
lean_inc(x_26);
lean_dec(x_23);
x_27 = l_Swirl_Protocol_Noninteractive_Wire_Raw_readRawVk(x_24);
if (lean_obj_tag(x_27) == 0)
{
lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; 
lean_dec(x_26);
lean_dec(x_25);
x_28 = lean_ctor_get(x_27, 0);
lean_inc(x_28);
lean_dec_ref(x_27);
x_29 = l_Tools_SwirlDumpProof_main___closed__2;
lean_inc(x_28);
x_30 = l_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_toString(x_28);
x_31 = lean_string_append(x_29, x_30);
lean_dec_ref(x_30);
x_32 = l_IO_FS_Stream_putStrLn(x_5, x_31);
if (lean_obj_tag(x_32) == 0)
{
uint8_t x_33; 
x_33 = !lean_is_exclusive(x_32);
if (x_33 == 0)
{
lean_object* x_34; uint32_t x_35; lean_object* x_36; 
x_34 = lean_ctor_get(x_32, 0);
lean_dec(x_34);
x_35 = l_Tools_SwirlDumpProof_parseErrorExitCode(x_28);
lean_dec(x_28);
x_36 = lean_box_uint32(x_35);
lean_ctor_set(x_32, 0, x_36);
return x_32;
}
else
{
uint32_t x_37; lean_object* x_38; lean_object* x_39; 
lean_dec(x_32);
x_37 = l_Tools_SwirlDumpProof_parseErrorExitCode(x_28);
lean_dec(x_28);
x_38 = lean_box_uint32(x_37);
x_39 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_39, 0, x_38);
return x_39;
}
}
else
{
uint8_t x_40; 
lean_dec(x_28);
x_40 = !lean_is_exclusive(x_32);
if (x_40 == 0)
{
return x_32;
}
else
{
lean_object* x_41; lean_object* x_42; 
x_41 = lean_ctor_get(x_32, 0);
lean_inc(x_41);
lean_dec(x_32);
x_42 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_42, 0, x_41);
return x_42;
}
}
}
else
{
lean_object* x_43; lean_object* x_44; 
x_43 = lean_ctor_get(x_27, 0);
lean_inc(x_43);
lean_dec_ref(x_27);
x_44 = l_Swirl_Protocol_Noninteractive_Wire_Raw_readRawProof(x_25);
if (lean_obj_tag(x_44) == 0)
{
lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; 
lean_dec(x_43);
lean_dec(x_26);
x_45 = lean_ctor_get(x_44, 0);
lean_inc(x_45);
lean_dec_ref(x_44);
x_46 = l_Tools_SwirlDumpProof_main___closed__3;
lean_inc(x_45);
x_47 = l_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_toString(x_45);
x_48 = lean_string_append(x_46, x_47);
lean_dec_ref(x_47);
x_49 = l_IO_FS_Stream_putStrLn(x_5, x_48);
if (lean_obj_tag(x_49) == 0)
{
uint8_t x_50; 
x_50 = !lean_is_exclusive(x_49);
if (x_50 == 0)
{
lean_object* x_51; uint32_t x_52; lean_object* x_53; 
x_51 = lean_ctor_get(x_49, 0);
lean_dec(x_51);
x_52 = l_Tools_SwirlDumpProof_parseErrorExitCode(x_45);
lean_dec(x_45);
x_53 = lean_box_uint32(x_52);
lean_ctor_set(x_49, 0, x_53);
return x_49;
}
else
{
uint32_t x_54; lean_object* x_55; lean_object* x_56; 
lean_dec(x_49);
x_54 = l_Tools_SwirlDumpProof_parseErrorExitCode(x_45);
lean_dec(x_45);
x_55 = lean_box_uint32(x_54);
x_56 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_56, 0, x_55);
return x_56;
}
}
else
{
uint8_t x_57; 
lean_dec(x_45);
x_57 = !lean_is_exclusive(x_49);
if (x_57 == 0)
{
return x_49;
}
else
{
lean_object* x_58; lean_object* x_59; 
x_58 = lean_ctor_get(x_49, 0);
lean_inc(x_58);
lean_dec(x_49);
x_59 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_59, 0, x_58);
return x_59;
}
}
}
else
{
lean_object* x_60; lean_object* x_61; 
x_60 = lean_ctor_get(x_44, 0);
lean_inc(x_60);
lean_dec_ref(x_44);
lean_inc(x_43);
x_61 = l_Swirl_Protocol_Noninteractive_Wire_Raw_readRawPublicValues(x_43, x_26);
if (lean_obj_tag(x_61) == 0)
{
lean_object* x_62; lean_object* x_63; lean_object* x_64; lean_object* x_65; lean_object* x_66; 
lean_dec(x_60);
lean_dec(x_43);
x_62 = lean_ctor_get(x_61, 0);
lean_inc(x_62);
lean_dec_ref(x_61);
x_63 = l_Tools_SwirlDumpProof_main___closed__4;
lean_inc(x_62);
x_64 = l_Swirl_Protocol_Noninteractive_Wire_Raw_ParseError_toString(x_62);
x_65 = lean_string_append(x_63, x_64);
lean_dec_ref(x_64);
x_66 = l_IO_FS_Stream_putStrLn(x_5, x_65);
if (lean_obj_tag(x_66) == 0)
{
uint8_t x_67; 
x_67 = !lean_is_exclusive(x_66);
if (x_67 == 0)
{
lean_object* x_68; uint32_t x_69; lean_object* x_70; 
x_68 = lean_ctor_get(x_66, 0);
lean_dec(x_68);
x_69 = l_Tools_SwirlDumpProof_parseErrorExitCode(x_62);
lean_dec(x_62);
x_70 = lean_box_uint32(x_69);
lean_ctor_set(x_66, 0, x_70);
return x_66;
}
else
{
uint32_t x_71; lean_object* x_72; lean_object* x_73; 
lean_dec(x_66);
x_71 = l_Tools_SwirlDumpProof_parseErrorExitCode(x_62);
lean_dec(x_62);
x_72 = lean_box_uint32(x_71);
x_73 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_73, 0, x_72);
return x_73;
}
}
else
{
uint8_t x_74; 
lean_dec(x_62);
x_74 = !lean_is_exclusive(x_66);
if (x_74 == 0)
{
return x_66;
}
else
{
lean_object* x_75; lean_object* x_76; 
x_75 = lean_ctor_get(x_66, 0);
lean_inc(x_75);
lean_dec(x_66);
x_76 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_76, 0, x_75);
return x_76;
}
}
}
else
{
lean_object* x_77; lean_object* x_78; lean_object* x_79; lean_object* x_80; lean_object* x_81; 
lean_dec_ref(x_5);
x_77 = lean_ctor_get(x_61, 0);
lean_inc(x_77);
lean_dec_ref(x_61);
x_78 = l_Tools_SwirlDumpProof_main___closed__5;
x_79 = l_Tools_SwirlDumpProof_vkDigest(x_43);
x_80 = lean_string_append(x_78, x_79);
lean_dec_ref(x_79);
x_81 = l_IO_println___at___00Tools_SwirlDumpProof_main_spec__0(x_80);
if (lean_obj_tag(x_81) == 0)
{
lean_object* x_82; lean_object* x_83; lean_object* x_84; lean_object* x_85; 
lean_dec_ref(x_81);
x_82 = l_Tools_SwirlDumpProof_main___closed__6;
x_83 = l_Tools_SwirlDumpProof_proofDigest(x_60);
x_84 = lean_string_append(x_82, x_83);
lean_dec_ref(x_83);
x_85 = l_IO_println___at___00Tools_SwirlDumpProof_main_spec__0(x_84);
if (lean_obj_tag(x_85) == 0)
{
lean_object* x_86; lean_object* x_87; lean_object* x_88; lean_object* x_89; 
lean_dec_ref(x_85);
x_86 = l_Tools_SwirlDumpProof_main___closed__7;
x_87 = l_Tools_SwirlDumpProof_pvDigest(x_77);
x_88 = lean_string_append(x_86, x_87);
lean_dec_ref(x_87);
x_89 = l_IO_println___at___00Tools_SwirlDumpProof_main_spec__0(x_88);
if (lean_obj_tag(x_89) == 0)
{
uint8_t x_90; 
x_90 = !lean_is_exclusive(x_89);
if (x_90 == 0)
{
lean_object* x_91; lean_object* x_92; 
x_91 = lean_ctor_get(x_89, 0);
lean_dec(x_91);
x_92 = l_Tools_SwirlDumpProof_main___boxed__const__2;
lean_ctor_set(x_89, 0, x_92);
return x_89;
}
else
{
lean_object* x_93; lean_object* x_94; 
lean_dec(x_89);
x_93 = l_Tools_SwirlDumpProof_main___boxed__const__2;
x_94 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_94, 0, x_93);
return x_94;
}
}
else
{
uint8_t x_95; 
x_95 = !lean_is_exclusive(x_89);
if (x_95 == 0)
{
return x_89;
}
else
{
lean_object* x_96; lean_object* x_97; 
x_96 = lean_ctor_get(x_89, 0);
lean_inc(x_96);
lean_dec(x_89);
x_97 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_97, 0, x_96);
return x_97;
}
}
}
else
{
uint8_t x_98; 
lean_dec(x_77);
x_98 = !lean_is_exclusive(x_85);
if (x_98 == 0)
{
return x_85;
}
else
{
lean_object* x_99; lean_object* x_100; 
x_99 = lean_ctor_get(x_85, 0);
lean_inc(x_99);
lean_dec(x_85);
x_100 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_100, 0, x_99);
return x_100;
}
}
}
else
{
uint8_t x_101; 
lean_dec(x_77);
lean_dec(x_60);
x_101 = !lean_is_exclusive(x_81);
if (x_101 == 0)
{
return x_81;
}
else
{
lean_object* x_102; lean_object* x_103; 
x_102 = lean_ctor_get(x_81, 0);
lean_inc(x_102);
lean_dec(x_81);
x_103 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_103, 0, x_102);
return x_103;
}
}
}
}
}
}
}
else
{
uint8_t x_104; 
x_104 = !lean_is_exclusive(x_3);
if (x_104 == 0)
{
return x_3;
}
else
{
lean_object* x_105; lean_object* x_106; 
x_105 = lean_ctor_get(x_3, 0);
lean_inc(x_105);
lean_dec(x_3);
x_106 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_106, 0, x_105);
return x_106;
}
}
}
}
LEAN_EXPORT lean_object* l_IO_print___at___00IO_println___at___00Tools_SwirlDumpProof_main_spec__0_spec__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = l_IO_print___at___00IO_println___at___00Tools_SwirlDumpProof_main_spec__0_spec__0(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* l_IO_println___at___00Tools_SwirlDumpProof_main_spec__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = l_IO_println___at___00Tools_SwirlDumpProof_main_spec__0(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* l_Tools_SwirlDumpProof_main___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = l_Tools_SwirlDumpProof_main();
return x_2;
}
}
LEAN_EXPORT lean_object* _lean_main() {
_start:
{
lean_object* x_2; 
x_2 = l_Tools_SwirlDumpProof_main();
return x_2;
}
}
LEAN_EXPORT lean_object* l_main___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = _lean_main();
return x_2;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Swirl_Protocol_Noninteractive_Wire_Raw(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_Tools_SwirlDumpProof(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Swirl_Protocol_Noninteractive_Wire_Raw(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
l_Tools_SwirlDumpProof_digestToCsv___closed__0 = _init_l_Tools_SwirlDumpProof_digestToCsv___closed__0();
lean_mark_persistent(l_Tools_SwirlDumpProof_digestToCsv___closed__0);
l_Tools_SwirlDumpProof_pvDigest___closed__0 = _init_l_Tools_SwirlDumpProof_pvDigest___closed__0();
lean_mark_persistent(l_Tools_SwirlDumpProof_pvDigest___closed__0);
l_Tools_SwirlDumpProof_main___closed__0 = _init_l_Tools_SwirlDumpProof_main___closed__0();
lean_mark_persistent(l_Tools_SwirlDumpProof_main___closed__0);
l_Tools_SwirlDumpProof_main___closed__1 = _init_l_Tools_SwirlDumpProof_main___closed__1();
lean_mark_persistent(l_Tools_SwirlDumpProof_main___closed__1);
l_Tools_SwirlDumpProof_main___closed__2 = _init_l_Tools_SwirlDumpProof_main___closed__2();
lean_mark_persistent(l_Tools_SwirlDumpProof_main___closed__2);
l_Tools_SwirlDumpProof_main___closed__3 = _init_l_Tools_SwirlDumpProof_main___closed__3();
lean_mark_persistent(l_Tools_SwirlDumpProof_main___closed__3);
l_Tools_SwirlDumpProof_main___closed__4 = _init_l_Tools_SwirlDumpProof_main___closed__4();
lean_mark_persistent(l_Tools_SwirlDumpProof_main___closed__4);
l_Tools_SwirlDumpProof_main___closed__5 = _init_l_Tools_SwirlDumpProof_main___closed__5();
lean_mark_persistent(l_Tools_SwirlDumpProof_main___closed__5);
l_Tools_SwirlDumpProof_main___closed__6 = _init_l_Tools_SwirlDumpProof_main___closed__6();
lean_mark_persistent(l_Tools_SwirlDumpProof_main___closed__6);
l_Tools_SwirlDumpProof_main___closed__7 = _init_l_Tools_SwirlDumpProof_main___closed__7();
lean_mark_persistent(l_Tools_SwirlDumpProof_main___closed__7);
l_Tools_SwirlDumpProof_main___boxed__const__1 = _init_l_Tools_SwirlDumpProof_main___boxed__const__1();
lean_mark_persistent(l_Tools_SwirlDumpProof_main___boxed__const__1);
l_Tools_SwirlDumpProof_main___boxed__const__2 = _init_l_Tools_SwirlDumpProof_main___boxed__const__2();
lean_mark_persistent(l_Tools_SwirlDumpProof_main___boxed__const__2);
return lean_io_result_mk_ok(lean_box(0));
}
char ** lean_setup_args(int argc, char ** argv);
void lean_initialize_runtime_module();

  #if defined(WIN32) || defined(_WIN32)
  #include <windows.h>
  #endif

  int main(int argc, char ** argv) {
  #if defined(WIN32) || defined(_WIN32)
  SetErrorMode(SEM_FAILCRITICALERRORS);
  SetConsoleOutputCP(CP_UTF8);
  #endif
  lean_object* in; lean_object* res;
argv = lean_setup_args(argc, argv);
lean_initialize_runtime_module();
lean_set_panic_messages(false);
res = initialize_Tools_SwirlDumpProof(1 /* builtin */);
lean_set_panic_messages(true);
lean_io_mark_end_initialization();
if (lean_io_result_is_ok(res)) {
lean_dec_ref(res);
lean_init_task_manager();
res = _lean_main();
}
lean_finalize_task_manager();
if (lean_io_result_is_ok(res)) {
  int ret = lean_unbox_uint32(lean_io_result_get_value(res));
  lean_dec_ref(res);
  return ret;
} else {
  lean_io_result_show_error(res);
  lean_dec_ref(res);
  return 1;
}
}
#ifdef __cplusplus
}
#endif
