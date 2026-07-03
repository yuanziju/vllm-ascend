use std::fmt;

/// 编译诊断
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagnostic {
    pub level: DiagnosticLevel,
    pub message: String,
    pub file: Option<String>,
    pub line: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagnosticLevel {
    Error,
    Warning,
    Note,
}

impl Diagnostic {
    pub fn error(msg: impl Into<String>) -> Self {
        Diagnostic {
            level: DiagnosticLevel::Error,
            message: msg.into(),
            file: None,
            line: None,
        }
    }
    pub fn warning(msg: impl Into<String>) -> Self {
        Diagnostic {
            level: DiagnosticLevel::Warning,
            message: msg.into(),
            file: None,
            line: None,
        }
    }
    pub fn note(msg: impl Into<String>) -> Self {
        Diagnostic {
            level: DiagnosticLevel::Note,
            message: msg.into(),
            file: None,
            line: None,
        }
    }
}

impl fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.level {
            DiagnosticLevel::Error => write!(f, "error: {}", self.message),
            DiagnosticLevel::Warning => write!(f, "warning: {}", self.message),
            DiagnosticLevel::Note => write!(f, "note: {}", self.message),
        }
    }
}