//! cli — neutron 二进制（interface 的薄包装）

use std::env;
use std::fs;
use std::process;

const VERSION: &str = env!("CARGO_PKG_VERSION");

fn print_usage() {
    eprintln!("neutron v{} — ONNX → GPU/NPU kernel 编译器", VERSION);
    eprintln!();
    eprintln!("用法: neutron <input.onnx> [--target cuda|npu|cpu] [--opt 0|1|2|3] [--dump] [-o <file>] [--help] [--version]");
    eprintln!();
    eprintln!("选项:");
    eprintln!("  --target <t>   目标后端: cuda (NVIDIA), npu (昇腾), cpu (回退)");
    eprintln!("  --opt <n>      优化等级: 0=关闭, 1=基础, 2=默认, 3=激进");
    eprintln!("  --dump         输出 IR 调试信息到 stderr");
    eprintln!("  -o <file>      后端源码写入文件 (默认写 stdout)");
    eprintln!("  --help, -h     显示帮助");
    eprintln!("  --version, -V  显示版本号");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        print_usage();
        process::exit(2);
    }

    let input_path = &args[1];

    // flags
    if input_path == "--help" || input_path == "-h" {
        print_usage();
        process::exit(0);
    }
    if input_path == "--version" || input_path == "-V" {
        println!("neutron {}", VERSION);
        process::exit(0);
    }

    let mut target = common::Target::Cuda;
    let mut opt_level = common::OptLevel::O2;
    let mut dump_ir = false;
    let mut output_path: Option<String> = None;

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--target" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("--target 需要值 (可选: cuda, npu, cpu)");
                    process::exit(2);
                }
                target = match args[i].as_str() {
                    "cuda" => common::Target::Cuda,
                    "npu" => common::Target::Npu,
                    "cpu" => common::Target::Cpu,
                    _ => {
                        eprintln!("未知 target: {} (可选: cuda, npu, cpu)", args[i]);
                        process::exit(2);
                    }
                };
            }
            "--opt" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("--opt 需要值 (可选: 0, 1, 2, 3)");
                    process::exit(2);
                }
                opt_level = match args[i].as_str() {
                    "0" => common::OptLevel::O0,
                    "1" => common::OptLevel::O1,
                    "2" => common::OptLevel::O2,
                    "3" => common::OptLevel::O3,
                    _ => {
                        eprintln!("未知 opt level: {} (可选: 0, 1, 2, 3)", args[i]);
                        process::exit(2);
                    }
                };
            }
            "-o" | "--output" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("-o 需要值 (输出文件路径)");
                    process::exit(2);
                }
                output_path = Some(args[i].clone());
            }
            "--dump" => {
                dump_ir = true;
            }
            "--help" | "-h" => {
                print_usage();
                process::exit(0);
            }
            "--version" | "-V" => {
                println!("neutron {}", VERSION);
                process::exit(0);
            }
            _ => {
                eprintln!("未知参数: {}", args[i]);
                eprintln!("运行 neutron --help 查看用法");
                process::exit(2);
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
                eprintln!("{}", debug);
            }
            if let Some(src) = &out.backend_source {
                match &output_path {
                    Some(path) => fs::write(path, src).unwrap_or_else(|e| {
                        eprintln!("写入 {} 失败: {}", path, e);
                        process::exit(1);
                    }),
                    None => println!("{}", src),
                }
            }
            eprintln!("target: {}", out.target);
            eprintln!("instructions: {}", out.instructions.len());
        }
        Err(e) => {
            eprintln!("编译失败: {}", e);
            process::exit(1);
        }
    }
}
