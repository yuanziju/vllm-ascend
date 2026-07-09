//! cli — neutron 二进制（interface 的薄包装）

use std::env;
use std::fs;
use std::process;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("用法: neutron <input.onnx> [--target cuda|npu|cpu] [--opt 0|1|2|3] [--dump]");
        process::exit(1);
    }

    let input_path = &args[1];
    let mut target = common::Target::Cuda;
    let mut opt_level = common::OptLevel::O2;
    let mut dump_ir = false;

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--target" => {
                i += 1;
                if i < args.len() {
                    target = match args[i].as_str() {
                        "cuda" => common::Target::Cuda,
                        "npu" => common::Target::Npu,
                        "cpu" => common::Target::Cpu,
                        _ => {
                            eprintln!("未知 target: {}", args[i]);
                            process::exit(1);
                        }
                    };
                }
            }
            "--opt" => {
                i += 1;
                if i < args.len() {
                    opt_level = match args[i].as_str() {
                        "0" => common::OptLevel::O0,
                        "1" => common::OptLevel::O1,
                        "2" => common::OptLevel::O2,
                        "3" => common::OptLevel::O3,
                        _ => {
                            eprintln!("未知 opt level: {}", args[i]);
                            process::exit(1);
                        }
                    };
                }
            }
            "--dump" => {
                dump_ir = true;
            }
            _ => {
                eprintln!("未知参数: {}", args[i]);
                process::exit(1);
            }
        }
        i += 1;
    }

    let bytes = fs::read(input_path).unwrap_or_else(|e| {
        eprintln!("读取 {} 失败: {}", input_path, e);
        process::exit(1);
    });

    let config = common::Config {
        target,
        opt_level,
        dump_ir,
        trace_isel: false,
        algebra_unsafe_opts: false,
    };

    match interface::compile(interface::Input::Onnx(bytes), config) {
        Ok(out) => {
            if let Some(debug) = &out.debug {
                println!("{}", debug);
            }
            println!("target: {}", out.target);
            println!("instructions: {}", out.instructions.len());
        }
        Err(e) => {
            eprintln!("编译失败: {}", e);
            process::exit(1);
        }
    }
}
