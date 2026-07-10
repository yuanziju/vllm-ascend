//! proto — 极简 protobuf wire-format 读取器（只读，无 schema 依赖）
//!
//! 设计哲学：不引入 prost/prost-build（重依赖 + protoc 代码生成），
//! 手写一个只够用的 protobuf 解码器。ONNX 是 protobuf 编码，我们只需
//! 解出 ModelProto → GraphProto → NodeProto 的关键字段（op_type / input /
//! output / name / initializer），足够构建架构无关图。
//!
//! Protobuf wire format 要点：
//! - 每个 field = (tag, wire_type)，varint 编码：tag<<3 | wire_type
//! - wire_type: 0=varint, 1=fixed64, 2=length-delimited(消息/string/bytes), 5=fixed32
//! - 我们主要用 0(varint) 和 2(length-delimited)

use base::Result;

/// protobuf wire types
const WT_VARINT: u8 = 0;
const WT_LEN: u8 = 2;
const WT_FIXED64: u8 = 1;
const WT_FIXED32: u8 = 5;

/// 光标：在字节切片上顺序读取
pub struct Cursor<'a> {
    pub data: &'a [u8],
    pub pos: usize,
}

impl<'a> Cursor<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }
    pub fn eof(&self) -> bool {
        self.pos >= self.data.len()
    }
    pub fn remaining(&self) -> &'a [u8] {
        &self.data[self.pos..]
    }

    /// 读一个 varint
    pub fn read_varint(&mut self) -> Result<u64> {
        let mut result: u64 = 0;
        let mut shift = 0u32;
        loop {
            if self.pos >= self.data.len() {
                return Err(base::NeutronError::Frontend("varint 意外结束".into()));
            }
            let b = self.data[self.pos];
            self.pos += 1;
            result |= ((b & 0x7F) as u64) << shift;
            if b & 0x80 == 0 {
                return Ok(result);
            }
            shift += 7;
            if shift >= 64 {
                return Err(base::NeutronError::Frontend("varint 过长".into()));
            }
        }
    }

    /// 读一个 length-delimited 字段（返回其字节切片）
    pub fn read_length_delimited(&mut self) -> Result<&'a [u8]> {
        let len = self.read_varint()? as usize;
        if self.pos + len > self.data.len() {
            return Err(base::NeutronError::Frontend("length-delimited 越界".into()));
        }
        let slice = &self.data[self.pos..self.pos + len];
        self.pos += len;
        Ok(slice)
    }

    /// 读一个 tag（field_number, wire_type）。返回 None 表示跳过。
    pub fn read_tag(&mut self) -> Result<(u32, u8)> {
        let v = self.read_varint()?;
        Ok(((v >> 3) as u32, (v & 0x07) as u8))
    }

    /// 跳过一个未知 wire_type 的字段
    pub fn skip_field(&mut self, wire_type: u8) -> Result<()> {
        match wire_type {
            WT_VARINT => {
                self.read_varint()?;
            }
            WT_LEN => {
                let len = self.read_varint()? as usize;
                if self.pos + len > self.data.len() {
                    return Err(base::NeutronError::Frontend("skip 越界".into()));
                }
                self.pos += len;
            }
            WT_FIXED64 => {
                self.pos += 8;
            }
            WT_FIXED32 => {
                self.pos += 4;
            }
            other => {
                return Err(base::NeutronError::Frontend(format!(
                    "未知 wire_type: {}",
                    other
                )));
            }
        }
        Ok(())
    }
}

/// 解码 UTF-8 字符串字段（已知是 length-delimited string）
pub fn read_string_field(buf: &[u8]) -> Result<String> {
    std::str::from_utf8(buf)
        .map(String::from)
        .map_err(|e| base::NeutronError::Frontend(format!("UTF-8 解码失败: {}", e)))
}

/// 解码 length-delimited 字段为子消息的光标
pub fn sub_cursor(buf: &[u8]) -> Cursor<'_> {
    Cursor::new(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_varint() {
        // 150 编码为 0x96 0x01
        let mut c = Cursor::new(&[0x96, 0x01]);
        assert_eq!(c.read_varint().unwrap(), 150);
    }

    #[test]
    fn reads_length_delimited() {
        // len=3, then "abc"
        let mut c = Cursor::new(&[0x03, b'a', b'b', b'c']);
        let s = c.read_length_delimited().unwrap();
        assert_eq!(s, b"abc");
    }

    #[test]
    fn reads_tag() {
        // field 1, wire_type 2 (LEN) = (1<<3)|2 = 10
        let mut c = Cursor::new(&[10]);
        let (field, wt) = c.read_tag().unwrap();
        assert_eq!(field, 1);
        assert_eq!(wt, WT_LEN);
    }

    #[test]
    fn skips_fixed64() {
        let mut c = Cursor::new(&[0u8; 8]);
        c.skip_field(WT_FIXED64).unwrap();
        assert!(c.eof());
    }
}
