use eyre::Result;
use rvr_openvm_test_utils::{self as utils, ExecutionMode::Pure};

const RVTEST: &str = utils::RVTEST;

#[test]
fn test_rv32ui_simple() -> Result<()> {
    utils::run_and_compare(&format!("{RVTEST}/rv32ui-p-simple"), Pure)
}

#[test]
fn test_rv32ui_add() -> Result<()> {
    utils::run_and_compare(&format!("{RVTEST}/rv32ui-p-add"), Pure)
}

#[test]
fn test_rv32ui_addi() -> Result<()> {
    utils::run_and_compare(&format!("{RVTEST}/rv32ui-p-addi"), Pure)
}

#[test]
fn test_rv32ui_and() -> Result<()> {
    utils::run_and_compare(&format!("{RVTEST}/rv32ui-p-and"), Pure)
}

#[test]
fn test_rv32ui_andi() -> Result<()> {
    utils::run_and_compare(&format!("{RVTEST}/rv32ui-p-andi"), Pure)
}

#[test]
fn test_rv32ui_auipc() -> Result<()> {
    utils::run_and_compare(&format!("{RVTEST}/rv32ui-p-auipc"), Pure)
}

#[test]
fn test_rv32ui_beq() -> Result<()> {
    utils::run_and_compare(&format!("{RVTEST}/rv32ui-p-beq"), Pure)
}

#[test]
fn test_rv32ui_bge() -> Result<()> {
    utils::run_and_compare(&format!("{RVTEST}/rv32ui-p-bge"), Pure)
}

#[test]
fn test_rv32ui_bgeu() -> Result<()> {
    utils::run_and_compare(&format!("{RVTEST}/rv32ui-p-bgeu"), Pure)
}

#[test]
fn test_rv32ui_blt() -> Result<()> {
    utils::run_and_compare(&format!("{RVTEST}/rv32ui-p-blt"), Pure)
}

#[test]
fn test_rv32ui_bltu() -> Result<()> {
    utils::run_and_compare(&format!("{RVTEST}/rv32ui-p-bltu"), Pure)
}

#[test]
fn test_rv32ui_bne() -> Result<()> {
    utils::run_and_compare(&format!("{RVTEST}/rv32ui-p-bne"), Pure)
}

#[test]
fn test_rv32ui_fence_i() {
    let err = utils::run_and_compare(&format!("{RVTEST}/rv32ui-p-fence_i"), Pure).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("FENCE.I instruction is not supported")
            || msg.contains("couldn't parse the next instruction"),
        "unexpected error: {msg}"
    );
}

#[test]
fn test_rv32ui_jal() -> Result<()> {
    utils::run_and_compare(&format!("{RVTEST}/rv32ui-p-jal"), Pure)
}

#[test]
fn test_rv32ui_jalr() -> Result<()> {
    utils::run_and_compare(&format!("{RVTEST}/rv32ui-p-jalr"), Pure)
}

#[test]
fn test_rv32ui_lb() -> Result<()> {
    utils::run_and_compare(&format!("{RVTEST}/rv32ui-p-lb"), Pure)
}

#[test]
fn test_rv32ui_lbu() -> Result<()> {
    utils::run_and_compare(&format!("{RVTEST}/rv32ui-p-lbu"), Pure)
}

#[test]
fn test_rv32ui_lh() -> Result<()> {
    utils::run_and_compare(&format!("{RVTEST}/rv32ui-p-lh"), Pure)
}

#[test]
fn test_rv32ui_lhu() -> Result<()> {
    utils::run_and_compare(&format!("{RVTEST}/rv32ui-p-lhu"), Pure)
}

#[test]
fn test_rv32ui_lui() -> Result<()> {
    utils::run_and_compare(&format!("{RVTEST}/rv32ui-p-lui"), Pure)
}

#[test]
fn test_rv32ui_lw() -> Result<()> {
    utils::run_and_compare(&format!("{RVTEST}/rv32ui-p-lw"), Pure)
}

#[test]
fn test_rv32ui_ma_data() -> Result<()> {
    match utils::run_and_compare(&format!("{RVTEST}/rv32ui-p-ma_data"), Pure) {
        Ok(()) => Ok(()),
        Err(err) => {
            let msg = err.to_string();
            assert!(
                msg.contains("misaligned memory accesses are not supported")
                    || msg.contains("LoadSignExtend invalid shift amount"),
                "unexpected error: {msg}"
            );
            Ok(())
        }
    }
}

#[test]
fn test_rv32ui_or() -> Result<()> {
    utils::run_and_compare(&format!("{RVTEST}/rv32ui-p-or"), Pure)
}

#[test]
fn test_rv32ui_ori() -> Result<()> {
    utils::run_and_compare(&format!("{RVTEST}/rv32ui-p-ori"), Pure)
}

#[test]
fn test_rv32ui_sb() -> Result<()> {
    utils::run_and_compare(&format!("{RVTEST}/rv32ui-p-sb"), Pure)
}

#[test]
fn test_rv32ui_sh() -> Result<()> {
    utils::run_and_compare(&format!("{RVTEST}/rv32ui-p-sh"), Pure)
}

#[test]
fn test_rv32ui_sll() -> Result<()> {
    utils::run_and_compare(&format!("{RVTEST}/rv32ui-p-sll"), Pure)
}

#[test]
fn test_rv32ui_slli() -> Result<()> {
    utils::run_and_compare(&format!("{RVTEST}/rv32ui-p-slli"), Pure)
}

#[test]
fn test_rv32ui_slt() -> Result<()> {
    utils::run_and_compare(&format!("{RVTEST}/rv32ui-p-slt"), Pure)
}

#[test]
fn test_rv32ui_slti() -> Result<()> {
    utils::run_and_compare(&format!("{RVTEST}/rv32ui-p-slti"), Pure)
}

#[test]
fn test_rv32ui_sltiu() -> Result<()> {
    utils::run_and_compare(&format!("{RVTEST}/rv32ui-p-sltiu"), Pure)
}

#[test]
fn test_rv32ui_sltu() -> Result<()> {
    utils::run_and_compare(&format!("{RVTEST}/rv32ui-p-sltu"), Pure)
}

#[test]
fn test_rv32ui_sra() -> Result<()> {
    utils::run_and_compare(&format!("{RVTEST}/rv32ui-p-sra"), Pure)
}

#[test]
fn test_rv32ui_srai() -> Result<()> {
    utils::run_and_compare(&format!("{RVTEST}/rv32ui-p-srai"), Pure)
}

#[test]
fn test_rv32ui_srl() -> Result<()> {
    utils::run_and_compare(&format!("{RVTEST}/rv32ui-p-srl"), Pure)
}

#[test]
fn test_rv32ui_srli() -> Result<()> {
    utils::run_and_compare(&format!("{RVTEST}/rv32ui-p-srli"), Pure)
}

#[test]
fn test_rv32ui_sub() -> Result<()> {
    utils::run_and_compare(&format!("{RVTEST}/rv32ui-p-sub"), Pure)
}

#[test]
fn test_rv32ui_sw() -> Result<()> {
    utils::run_and_compare(&format!("{RVTEST}/rv32ui-p-sw"), Pure)
}

#[test]
fn test_rv32ui_xor() -> Result<()> {
    utils::run_and_compare(&format!("{RVTEST}/rv32ui-p-xor"), Pure)
}

#[test]
fn test_rv32ui_xori() -> Result<()> {
    utils::run_and_compare(&format!("{RVTEST}/rv32ui-p-xori"), Pure)
}

#[test]
fn test_rv32um_div() -> Result<()> {
    utils::run_and_compare(&format!("{RVTEST}/rv32um-p-div"), Pure)
}

#[test]
fn test_rv32um_divu() -> Result<()> {
    utils::run_and_compare(&format!("{RVTEST}/rv32um-p-divu"), Pure)
}

#[test]
fn test_rv32um_mul() -> Result<()> {
    utils::run_and_compare(&format!("{RVTEST}/rv32um-p-mul"), Pure)
}

#[test]
fn test_rv32um_mulh() -> Result<()> {
    utils::run_and_compare(&format!("{RVTEST}/rv32um-p-mulh"), Pure)
}

#[test]
fn test_rv32um_mulhsu() -> Result<()> {
    utils::run_and_compare(&format!("{RVTEST}/rv32um-p-mulhsu"), Pure)
}

#[test]
fn test_rv32um_mulhu() -> Result<()> {
    utils::run_and_compare(&format!("{RVTEST}/rv32um-p-mulhu"), Pure)
}

#[test]
fn test_rv32um_rem() -> Result<()> {
    utils::run_and_compare(&format!("{RVTEST}/rv32um-p-rem"), Pure)
}

#[test]
fn test_rv32um_remu() -> Result<()> {
    utils::run_and_compare(&format!("{RVTEST}/rv32um-p-remu"), Pure)
}
