//! common — 跨 crate 工具 + 共享数据结构 + 全局配置

use base::Graph;
use std::cell::Cell;

/// 优化的目标后端
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Target {
    #[default]
    Cuda,
    Npu,
    Cpu,
}

/// 优化等级
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum OptLevel {
    O0,
    O1,
    O2,
    O3,
}

/// ID 生成器（注意不与 Iterator::next 冲突，用 gen）
#[derive(Debug, Default)]
pub struct IdGen {
    next: Cell<u64>,
}

impl IdGen {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn gen(&self) -> u64 {
        let v = self.next.get();
        self.next.set(v + 1);
        v
    }
}

/// 简易 arena 分配器（用于需要稳定地址的场景）
#[derive(Debug, Default)]
pub struct Arena<T> {
    items: Vec<T>,
}

impl<T> Arena<T> {
    pub fn new() -> Self {
        Self { items: Vec::new() }
    }
    pub fn alloc(&mut self, v: T) -> usize {
        let i = self.items.len();
        self.items.push(v);
        i
    }
    pub fn get(&self, i: usize) -> Option<&T> {
        self.items.get(i)
    }
    pub fn get_mut(&mut self, i: usize) -> Option<&mut T> {
        self.items.get_mut(i)
    }
    pub fn len(&self) -> usize {
        self.items.len()
    }
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

/// 编译配置
#[derive(Debug, Clone)]
pub struct Config {
    pub target: Target,
    pub opt_level: OptLevel,
    pub dump_ir: bool,
    pub trace_isel: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            target: Target::Cuda,
            opt_level: OptLevel::O2,
            dump_ir: false,
            trace_isel: false,
        }
    }
}

/// 缩进打印机（用于 IR dump）
#[derive(Debug, Default)]
pub struct Printer {
    buf: String,
    indent: usize,
}

impl Printer {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn indent(&mut self) {
        self.indent += 1;
    }
    pub fn dedent(&mut self) {
        if self.indent > 0 {
            self.indent -= 1;
        }
    }
    pub fn line(&mut self, s: impl AsRef<str>) {
        for _ in 0..self.indent {
            self.buf.push_str("  ");
        }
        self.buf.push_str(s.as_ref());
        self.buf.push('\n');
    }
    pub fn finish(self) -> String {
        self.buf
    }
}

/// 把图 dump 成可读文本（调试/遗言用）
pub fn dump_graph(graph: &Graph) -> String {
    let mut p = Printer::new();
    p.line(format!("graph \"{}\" {{", graph.name));
    p.indent();
    p.line(format!(
        "// {} nodes, {} values, {} inputs, {} outputs",
        graph.node_count(),
        graph.value_count(),
        graph.inputs().len(),
        graph.outputs().len()
    ));
    for id in graph.node_ids() {
        match graph.node(id) {
            Ok(n) => {
                let ins: Vec<String> = n.inputs().iter().map(|v| format!("%{}", v)).collect();
                let outs: Vec<String> = n.outputs().iter().map(|v| format!("%{}", v)).collect();
                p.line(format!(
                    "  n{} = {:?}({}) -> [{}]",
                    n.id,
                    n.kind,
                    ins.join(", "),
                    outs.join(", ")
                ));
            }
            Err(e) => p.line(format!("  n{} = <err: {}>", id, e)),
        }
    }
    p.dedent();
    p.line("}");
    p.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn id_gen_sequential() {
        let g = IdGen::new();
        assert_eq!(g.gen(), 0);
        assert_eq!(g.gen(), 1);
        assert_eq!(g.gen(), 2);
    }

    #[test]
    fn opt_level_ordering() {
        assert!(OptLevel::O2 > OptLevel::O1);
    }
}
