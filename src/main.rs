#[cfg(not(target_pointer_width = "64"))]
compile_error!("64 bit is assumed");


use std::convert::TryFrom;
use std::collections::hash_map::{HashMap, DefaultHasher};
use std::hash::{Hash, Hasher};

mod parse;
mod ast;
mod sym;
mod types;
mod smallvec;

use ast::*;
use smallvec::*;
use sym::*;
use types::*;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub fn hash<T: Hash>(value: &T) -> u64 {
    let mut h = DefaultHasher::new();
    value.hash(&mut h);
    h.finish()
}

#[macro_export]
macro_rules! error {
    ($ctx: expr, $pos: expr, $($fmt: expr),*) => {{
        $ctx.error($pos, format!($($fmt),*))
    }}
}

#[allow(unused_macros)]
macro_rules! assert_implies {
    ($p:expr, $q:expr) => { assert!(!($p) || ($q)) }
}

#[allow(unused_macros)]
macro_rules! debug_assert_implies {
    ($p:expr, $q:expr) => { debug_assert!(!($p) || ($q)) }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Intern(u32);

struct InternBuf {
    buf: String,
    #[allow(dead_code)]
    next: Option<Box<InternBuf>>
}

struct Interns {
    bufs: *mut InternBuf,
    ids: HashMap<&'static str, Intern>,
    interns: Vec<&'static str>,
}

impl Interns {
    #[cfg(test)]
    const BUF_CAPACITY: usize = 12;
    #[cfg(not(test))]
    const BUF_CAPACITY: usize = 16*1024*1024;

    pub fn new() -> Interns {
        Interns { bufs: std::ptr::null_mut(), ids: HashMap::new(), interns: Vec::new() }
    }

    pub fn empty_string() -> Intern {
        Intern(0)
    }

    // The whole point of this is to pack interned strings tightly in memory
    // with very few allocations.
    //
    // Rust will give us a static lifetime to dynamically allocated strs through
    // `bufs`, which we can store in `ids`. Consequently, we need to ensure
    // `bufs.buf` never reallocates its internal storage.
    //
    // You can do this without unsafe by inserting `hash(str)` as keys, and
    // managing collisions yourself, but, eh
    fn push(&mut self, str: &str) -> &'static str {
        let (fits, next) = if self.bufs.is_null() {
            (false, None)
        } else {
            let buf = unsafe { &(*self.bufs).buf };
            let fits =  buf.len() + str.len() < buf.capacity();
            (fits, Some(self.bufs))
        };
        if !fits {
            // We assume str.len() <<<< BUF_CAPACITY, and that any leftover
            // space in the current buffer is too small to care about
            let cap = str.len().max(Self::BUF_CAPACITY);
            let link = Box::new(InternBuf {
                buf: String::with_capacity(cap),
                next: next.and_then(|bufs| unsafe { Some(Box::from_raw(bufs)) })
            });
            self.bufs = Box::leak(link);
        }
        let buf = unsafe { &mut (*self.bufs).buf };
        let old_len = buf.len();
        let old_cap = buf.capacity();
        // It would be nice to do something that cannot realloc even in
        // principle but miri doesn't like set_len-ing and copying into the
        // internal Vec<u8> ?_?
        buf.push_str(str);
        assert_eq!(old_cap, buf.capacity());
        &buf[old_len..]
    }

    pub fn put(&mut self, str: &str) -> Intern {
        match self.ids.get(str) {
            None => {
                assert!(self.interns.len() < u32::MAX as usize);
                let id = Intern(self.ids.len() as u32);
                let owned = self.push(str);
                self.interns.push(owned);
                self.ids.insert(owned, id);
                id
            },
            Some(id) => *id
        }
    }

    pub fn to_str(&self, id: Intern) -> Option<&str> {
        self.interns.get(id.0 as usize).copied()
    }
}

// If the compiler was a library we'd need this. As an application this just
// wastes time at exit
#[cfg(test)]
impl Drop for Interns {
    fn drop(&mut self) {
        unsafe { if self.bufs.is_null() == false { Box::from_raw(self.bufs); } }
    }
}

#[test]
fn interns() {
    let mut interns = Interns::new();

    let empty = interns.put("");
    let big = "d".repeat(Interns::BUF_CAPACITY * 3);

    let a = interns.put("a");
    let b = interns.put("b");
    let c = interns.put("c");
    let a2 = interns.put("a");
    let d = interns.put(&big);
    let e = interns.put("e");

    assert_eq!(a, Intern(1));
    assert_eq!(b, Intern(2));
    assert_eq!(c, Intern(3));
    assert_eq!(d, Intern(4));
    assert_eq!(e, Intern(5));
    assert_eq!(a, a2);

    assert_eq!(interns.to_str(empty), Some(""));
    assert_eq!(interns.to_str(a), Some("a"));
    assert_eq!(interns.to_str(b), Some("b"));
    assert_eq!(interns.to_str(c), Some("c"));
    assert_eq!(interns.to_str(d), Some(big.as_str()));
    assert_eq!(interns.to_str(e), Some("e"));

    assert_eq!(interns.ids.len(), interns.interns.len());
}

fn new_interns_with_keywords() -> Interns {
    let mut result = Interns::new();
    result.put("");
    for k in (1..).scan((), |(), v| parse::Keyword::from_u32(v)) {
        result.put(k.to_str());
    }
    result
}

#[test]
fn keywords() {
    let mut interns = new_interns_with_keywords();

    let mut i = Intern(1);
    for k in (1..).scan((), |(), v| parse::Keyword::from_u32(v)) {
        assert_eq!(interns.put(k.to_str()), i);
        i.0 += 1;
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum Op {
    Halt = 255,
    Noop = 0,
    Immediate,
    IntNeg,
    IntAdd,
    IntSub,
    IntMul,
    IntDiv,
    IntMod,
    IntLt,
    IntGt,
    IntEq,
    IntNEq,
    IntLtEq,
    IntGtEq,
    BitNeg,
    BitAnd,
    BitOr,
    BitXor,
    Not,
    CmpZero,
    LShift,
    RShift,
    LogicOr,
    LogicAnd,
    F32Neg,
    F32Add,
    F32Sub,
    F32Mul,
    F32Div,
    F32Lt,
    F32Gt,
    F32Eq,
    F32NEq,
    F32LtEq,
    F32GtEq,
    F64Neg,
    F64Add,
    F64Sub,
    F64Mul,
    F64Div,
    F64Lt,
    F64Gt,
    F64Eq,
    F64NEq,
    F64LtEq,
    F64GtEq,
    Move,
    MoveLower8,
    MoveLower16,
    MoveLower32,
    MoveAndSignExtendLower8,
    MoveAndSignExtendLower16,
    MoveAndSignExtendLower32,
    // StackRelativeStore64 is equivalent to Move
    StackRelativeStore8,
    StackRelativeStore16,
    StackRelativeStore32,
    // StackRelativeLoad64 is equivalent to Move
    StackRelativeLoad8,
    StackRelativeLoad16,
    StackRelativeLoad32,
    StackRelativeLoadAndSignExtend8,
    StackRelativeLoadAndSignExtend16,
    StackRelativeLoadAndSignExtend32,
    StackRelativeLoadBool,
    StackRelativeCopy,
    StackRelativeZero,
    IntToFloat32,
    IntToFloat64,
    Float32ToInt,
    Float32To64,
    Float64ToInt,
    Float64To32,
    Jump,
    JumpIfZero,
    JumpIfNotZero,
    Return,
    Call,
    CallIndirect,
    Store8,
    Store16,
    Store32,
    Store64,
    Load8,
    Load16,
    Load32,
    Load64,
    LoadAndSignExtend8,
    LoadAndSignExtend16,
    LoadAndSignExtend32,
    LoadBool,
    Copy,
    Zero,
    LoadStackAddress
}

fn requires_register_destination(op: Op) -> bool {
    match op {
        Op::StackRelativeStore8|Op::StackRelativeStore16|Op::StackRelativeStore32|Op::StackRelativeCopy|Op::StackRelativeZero|
        Op::Store8|Op::Store16|Op::Store32|Op::Copy|Op::Zero => false,
        _ => true
    }
}

#[derive(Debug)]
struct Instr {
    op: Op,
    dest: u8,
    left: u8,
    right: u8,
}

#[derive(Clone, Copy, Debug)]
struct FatInstr {
    op: Op,
    dest: u32,
    left: u32,
    right: u32,
}

impl FatInstr {
    const HALT: FatInstr = FatInstr { op: Op::Halt, dest: 0, left: 0, right: 0 };

    fn is_jump(&self) -> bool {
        use Op::*;
        matches!(self.op, Jump|JumpIfZero|JumpIfNotZero)
    }
}

#[derive(Clone, Copy)]
union RegValue {
    int: usize,
    sint: isize,
    wint: std::num::Wrapping<usize>,
    int32: (u32, u32),
    int16: (u16, u16, u16, u16),
    int8: (u8, u8, u8, u8, u8, u8, u8, u8),
    sint32: (i32, i32),
    sint16: (i16, i16, i16, i16),
    sint8: (i8, i8, i8, i8, i8, i8, i8, i8),
    float64: f64,
    float32: (f32, f32),
    b8: (bool, [bool; 7]),
}

impl RegValue {
    fn is_true(&self) -> bool {
        unsafe {
            debug_assert!(self.b8.0 == (self.int != 0));
            self.b8.0
        }
    }
}

impl std::fmt::Debug for RegValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        unsafe {
            write!(f, "RegValue {{ {:#016x} }}", self.int)
        }
    }
}

impl From<usize> for RegValue {
    fn from(value: usize) -> Self {
        RegValue { int: value }
    }
}

impl From<std::num::Wrapping<usize>> for RegValue {
    fn from(value: std::num::Wrapping<usize>) -> Self {
        RegValue { wint: value }
    }
}

impl From<isize> for RegValue {
    fn from(value: isize) -> Self {
        RegValue { sint: value }
    }
}

impl From<u32> for RegValue {
    fn from(value: u32) -> Self {
        RegValue { int: value as usize }
    }
}

impl From<i32> for RegValue {
    fn from(value: i32) -> Self {
        RegValue { int: value as usize }
    }
}

impl From<(u32, u32)> for RegValue {
    fn from(value: (u32, u32)) -> Self {
        RegValue { int32: value }
    }
}

impl From<f64> for RegValue {
    fn from(value: f64) -> Self {
        RegValue { float64: value }
    }
}

impl From<f32> for RegValue {
    fn from(value: f32) -> Self {
        RegValue { float32: (value, 0.0f32) }
    }
}

impl From<bool> for RegValue {
    fn from(value: bool) -> Self {
        RegValue { b8: (value, [false; 7]) }
    }
}

impl From<Intern> for RegValue {
    fn from(value: Intern) -> Self {
        RegValue { int32: (value.0, 0) }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Value {
    Int(isize),
    UInt(usize),
    F32(f32),
    F64(f64),
    Bool(bool),
    Ident(Intern)
}

impl Value {
    fn from(reg: RegValue, ty: Type) -> Value {
        match ty {
            Type::U8|Type::U16|Type::U32|Type::U64 => unsafe { Value::UInt(reg.int) }
            Type::F32  => unsafe { Value::F32(reg.float32.0) }
            Type::F64  => unsafe { Value::F64(reg.float64) }
            Type::Bool => unsafe { Value::Bool(reg.b8.0) }
            _ => unsafe { Value::Int(reg.sint) }
        }
    }
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Int(v) => v.fmt(f),
            Value::UInt(v) => v.fmt(f),
            Value::F32(v) => v.fmt(f),
            Value::F64(v) => v.fmt(f),
            Value::Bool(v) => v.fmt(f),
            Value::Ident(v) => v.0.fmt(f),
        }
    }
}

impl From<Value> for RegValue {
    fn from(value: Value) -> RegValue {
        match value {
            Value::Int(v) => v.into(),
            Value::UInt(v) => v.into(),
            Value::F32(v) => v.into(),
            Value::F64(v) => v.into(),
            Value::Bool(v) => v.into(),
            Value::Ident(v) => v.into(),
        }
    }
}

fn unary_op(op: parse::TokenKind, ty: Type) -> Option<(Op, Type)> {
    use parse::TokenKind;
    match (op, integer_promote(ty)) {
        (TokenKind::Sub,    Type::Int) => Some((Op::IntNeg, Type::Int)),
        (TokenKind::BitNeg, Type::Int) => Some((Op::BitNeg, Type::Int)),
        (TokenKind::Not,    _        ) => Some((Op::Not,    Type::Bool)),
        (TokenKind::Sub,    Type::F32) => Some((Op::F32Neg, Type::F32)),
        (TokenKind::Sub,    Type::F64) => Some((Op::F64Neg, Type::F64)),
        _ => None
    }
}

fn binary_op(op: parse::TokenKind, left: Type, right: Type) -> Option<(Op, Type)> {
    use parse::TokenKind;
    let left = integer_promote(left);
    let right = integer_promote(right);
    // TODO: eh
    if left != right && !(left.is_pointer() && right == Type::Int) {
        return None;
    }
    match (op, left) {
        (TokenKind::Add,      Type::Int) => Some((Op::IntAdd,  Type::Int)),
        (TokenKind::Sub,      Type::Int) => Some((Op::IntSub,  Type::Int)),
        (TokenKind::Mul,      Type::Int) => Some((Op::IntMul,  Type::Int)),
        (TokenKind::Div,      Type::Int) => Some((Op::IntDiv,  Type::Int)),
        (TokenKind::Mod,      Type::Int) => Some((Op::IntMod,  Type::Int)),
        (TokenKind::BitNeg,   Type::Int) => Some((Op::BitNeg,  Type::Int)),
        (TokenKind::BitAnd,   Type::Int) => Some((Op::BitAnd,  Type::Int)),
        (TokenKind::BitOr,    Type::Int) => Some((Op::BitOr,   Type::Int)),
        (TokenKind::BitXor,   Type::Int) => Some((Op::BitXor,  Type::Int)),
        (TokenKind::LShift,   Type::Int) => Some((Op::LShift,  Type::Int)),
        (TokenKind::RShift,   Type::Int) => Some((Op::RShift,  Type::Int)),
        (TokenKind::GtEq,     Type::Int) => Some((Op::IntGtEq, Type::Bool)),
        (TokenKind::Lt,       Type::Int) => Some((Op::IntLt,   Type::Bool)),
        (TokenKind::Gt,       Type::Int) => Some((Op::IntGt,   Type::Bool)),
        (TokenKind::Eq,       Type::Int) => Some((Op::IntEq,   Type::Bool)),
        (TokenKind::NEq,      Type::Int) => Some((Op::IntNEq,  Type::Bool)),
        (TokenKind::LtEq,     Type::Int) => Some((Op::IntLtEq, Type::Bool)),

        (TokenKind::LogicAnd, Type::Bool) => Some((Op::LogicAnd, Type::Bool)),
        (TokenKind::LogicOr,  Type::Bool) => Some((Op::LogicOr,  Type::Bool)),

        (TokenKind::Add,      Type::F32) => Some((Op::F32Add,  Type::F32)),
        (TokenKind::Sub,      Type::F32) => Some((Op::F32Sub,  Type::F32)),
        (TokenKind::Mul,      Type::F32) => Some((Op::F32Mul,  Type::F32)),
        (TokenKind::Div,      Type::F32) => Some((Op::F32Div,  Type::F32)),
        (TokenKind::Lt,       Type::F32) => Some((Op::F32Lt,   Type::Bool)),
        (TokenKind::Gt,       Type::F32) => Some((Op::F32Gt,   Type::Bool)),
        (TokenKind::Eq,       Type::F32) => Some((Op::F32Eq,   Type::Bool)),
        (TokenKind::NEq,      Type::F32) => Some((Op::F32NEq,  Type::Bool)),
        (TokenKind::LtEq,     Type::F32) => Some((Op::F32LtEq, Type::Bool)),
        (TokenKind::GtEq,     Type::F32) => Some((Op::F32GtEq, Type::Bool)),

        (TokenKind::Add,      Type::F64) => Some((Op::F64Add,  Type::F64)),
        (TokenKind::Sub,      Type::F64) => Some((Op::F64Sub,  Type::F64)),
        (TokenKind::Mul,      Type::F64) => Some((Op::F64Mul,  Type::F64)),
        (TokenKind::Div,      Type::F64) => Some((Op::F64Div,  Type::F64)),
        (TokenKind::Lt,       Type::F64) => Some((Op::F64Lt,   Type::Bool)),
        (TokenKind::Gt,       Type::F64) => Some((Op::F64Gt,   Type::Bool)),
        (TokenKind::Eq,       Type::F64) => Some((Op::F64Eq,   Type::Bool)),
        (TokenKind::NEq,      Type::F64) => Some((Op::F64NEq,  Type::Bool)),
        (TokenKind::LtEq,     Type::F64) => Some((Op::F64LtEq, Type::Bool)),
        (TokenKind::GtEq,     Type::F64) => Some((Op::F64GtEq, Type::Bool)),

        (TokenKind::Add,      Type::VoidPtr)             => Some((Op::IntAdd,  Type::VoidPtr)),
        (TokenKind::Sub,      Type::VoidPtr)             => Some((Op::IntSub,  Type::VoidPtr)),
        (TokenKind::Add,      Type::U8Ptr)               => Some((Op::IntAdd,  Type::U8Ptr)),
        (TokenKind::Sub,      Type::U8Ptr)               => Some((Op::IntSub,  Type::U8Ptr)),
        (TokenKind::GtEq,     Type::U8Ptr|Type::VoidPtr) => Some((Op::IntGtEq, Type::Bool)),
        (TokenKind::Lt,       Type::U8Ptr|Type::VoidPtr) => Some((Op::IntLt,   Type::Bool)),
        (TokenKind::Gt,       Type::U8Ptr|Type::VoidPtr) => Some((Op::IntGt,   Type::Bool)),
        (TokenKind::Eq,       Type::U8Ptr|Type::VoidPtr) => Some((Op::IntEq,   Type::Bool)),
        (TokenKind::NEq,      Type::U8Ptr|Type::VoidPtr) => Some((Op::IntNEq,  Type::Bool)),
        (TokenKind::LtEq,     Type::U8Ptr|Type::VoidPtr) => Some((Op::IntLtEq, Type::Bool)),

        _ => None
    }
}

fn convert_op(from: Type, to: Type) -> Option<Op> {
    let from = integer_promote(from);
    let op = if from == to {
        Op::Noop
    } else if from == Type::Int {
        match to {
            Type::Bool => Op::CmpZero,
            Type::U8   => Op::MoveLower8,
            Type::U16  => Op::MoveLower16,
            Type::U32  => Op::MoveLower32,
            Type::U64  => Op::Move,
            Type::I8   => Op::MoveAndSignExtendLower8,
            Type::I16  => Op::MoveAndSignExtendLower16,
            Type::I32  => Op::MoveAndSignExtendLower32,
            Type::I64  => Op::Move,
            Type::F32  => Op::IntToFloat32,
            Type::F64  => Op::IntToFloat64,
            _ if to.is_pointer() => Op::Noop,
            _ => return None
        }
    } else if from.is_pointer() {
        match to {
            Type::U64|Type::I64|Type::Int => Op::Noop,
            _ => return None
        }
    } else if to.is_integer() {
        match from {
            Type::Bool => Op::Noop,
            Type::F32  => Op::Float32ToInt,
            Type::F64  => Op::Float64ToInt,
            _ => return None
        }
    } else {
        match (from, to) {
            (Type::F32, Type::F64) => Op::Float32To64,
            (Type::F64, Type::F32) => Op::Float64To32,
            _ => return None
        }
    };
    Some(op)
}

fn store_op_stack(ty: Type) -> Option<Op> {
    match ty {
        Type::I8|Type::U8|Type::Bool            => Some(Op::StackRelativeStore8),
        Type::I16|Type::U16                     => Some(Op::StackRelativeStore16),
        Type::I32|Type::U32|Type::F32           => Some(Op::StackRelativeStore32),
        Type::Int|Type::I64|Type::U64|Type::F64 => Some(Op::Move),
        _ if ty.is_pointer()                    => Some(Op::Move),
        _ => None
    }
}

fn load_op_stack(ty: Type) -> Option<Op> {
    match ty {
        Type::Bool                    => Some(Op::StackRelativeLoadBool),
        Type::I8                      => Some(Op::StackRelativeLoadAndSignExtend8),
        Type::I16                     => Some(Op::StackRelativeLoadAndSignExtend16),
        Type::I32                     => Some(Op::StackRelativeLoadAndSignExtend32),
        Type::U8                      => Some(Op::StackRelativeLoad8),
        Type::U16                     => Some(Op::StackRelativeLoad16),
        Type::U32                     => Some(Op::StackRelativeLoad32),
        Type::Int|Type::I64|Type::U64 => Some(Op::Move),
        _ if ty.is_pointer()          => Some(Op::Move),
        _ => None
    }
}

fn store_op_pointer(ty: Type) -> Option<Op> {
    match ty {
        Type::I8|Type::U8|Type::Bool            => Some(Op::Store8),
        Type::I16|Type::U16                     => Some(Op::Store16),
        Type::I32|Type::U32|Type::F32           => Some(Op::Store32),
        Type::Int|Type::I64|Type::U64|Type::F64 => Some(Op::Store64),
        _ if ty.is_pointer()                    => Some(Op::Store64),
        _ => None
    }
}

fn load_op_pointer(ty: Type) -> Option<Op> {
    match ty {
        Type::Bool                    => Some(Op::LoadBool),
        Type::I8                      => Some(Op::LoadAndSignExtend8),
        Type::I16                     => Some(Op::LoadAndSignExtend16),
        Type::I32                     => Some(Op::LoadAndSignExtend32),
        Type::U8                      => Some(Op::Load8),
        Type::U16                     => Some(Op::Load16),
        Type::U32                     => Some(Op::Load32),
        Type::Int|Type::I64|Type::U64 => Some(Op::Load64),
        _ if ty.is_pointer()          => Some(Op::Load64),
        _ => None
    }
}

// TODO: right now, before calling either apply_unary_op or apply_binary_op, you
// need to check the type of the value isn't pointer, as the representation of
// constant pointers is not the same as their run time values (they are stack
// offsets until then). This sucks a little
fn apply_unary_op(op: Op, value: RegValue) -> RegValue {
    unsafe {
        match op {
            Op::Noop                     => value,
            Op::IntNeg                   => RegValue::from(-value.sint),
            Op::BitNeg                   => RegValue::from(!value.int),
            Op::Not                      => RegValue::from((value.int == 0) as usize),
            Op::CmpZero                  => RegValue::from((value.int != 0) as usize),
            Op::MoveLower8               => RegValue::from(value.int8.0     as usize),
            Op::MoveLower16              => RegValue::from(value.int16.0    as usize),
            Op::MoveLower32              => RegValue::from(value.int32.0    as usize),
            Op::MoveAndSignExtendLower8  => RegValue::from(value.sint8.0    as usize),
            Op::MoveAndSignExtendLower16 => RegValue::from(value.sint16.0   as usize),
            Op::MoveAndSignExtendLower32 => RegValue::from(value.sint32.0   as usize),
            Op::IntToFloat32             => RegValue::from(value.sint       as f32),
            Op::IntToFloat64             => RegValue::from(value.sint       as f64),
            Op::Float32ToInt             => RegValue::from(value.float32.0  as usize),
            Op::Float32To64              => RegValue::from(value.float32.0  as f64),
            Op::Float64ToInt             => RegValue::from(value.float64    as usize),
            Op::Float64To32              => RegValue::from(value.float64    as f32),
            Op::F32Neg                   => RegValue::from(-value.float32.0),
            Op::F64Neg                   => RegValue::from(-value.float64),
            _ => unreachable!()
        }
    }
}

fn apply_binary_op(op: Op, left: RegValue, right: RegValue) -> RegValue {
    unsafe {
        match op {
            Op::IntAdd   => RegValue::from(left.wint + right.wint),
            Op::IntSub   => RegValue::from(left.wint - right.wint),
            Op::IntMul   => RegValue::from(left.wint * right.wint),
            Op::IntDiv   => RegValue::from(left.wint / right.wint),
            Op::IntMod   => RegValue::from(left.wint % right.wint),
            Op::IntLt    => RegValue::from(left.int < right.int),
            Op::IntGt    => RegValue::from(left.int > right.int),
            Op::IntEq    => RegValue::from(left.int == right.int),
            Op::IntNEq   => RegValue::from(left.int != right.int),
            Op::IntLtEq  => RegValue::from(left.int <= right.int),
            Op::IntGtEq  => RegValue::from(left.int >= right.int),
            Op::BitAnd   => RegValue::from(left.int & right.int),
            Op::BitOr    => RegValue::from(left.int | right.int),
            Op::BitXor   => RegValue::from(left.int ^ right.int),

            Op::LShift   => RegValue::from(left.int << right.int),
            Op::RShift   => RegValue::from(left.int >> right.int),

            Op::LogicOr  => RegValue::from((left.b8.0 || right.b8.0) as usize),
            Op::LogicAnd => RegValue::from((left.b8.0 && right.b8.0) as usize),

            Op::F32Add   => RegValue::from(left.float32.0 + right.float32.0),
            Op::F32Sub   => RegValue::from(left.float32.0 - right.float32.0),
            Op::F32Mul   => RegValue::from(left.float32.0 * right.float32.0),
            Op::F32Div   => RegValue::from(left.float32.0 / right.float32.0),
            Op::F32Lt    => RegValue::from(left.float32.0 < right.float32.0),
            Op::F32Gt    => RegValue::from(left.float32.0 > right.float32.0),
            Op::F32Eq    => RegValue::from(left.float32.0 == right.float32.0),
            Op::F32NEq   => RegValue::from(left.float32.0 != right.float32.0),
            Op::F32LtEq  => RegValue::from(left.float32.0 <= right.float32.0),
            Op::F32GtEq  => RegValue::from(left.float32.0 >= right.float32.0),

            Op::F64Add   => RegValue::from(left.float64 + right.float64),
            Op::F64Sub   => RegValue::from(left.float64 - right.float64),
            Op::F64Mul   => RegValue::from(left.float64 * right.float64),
            Op::F64Div   => RegValue::from(left.float64 / right.float64),
            Op::F64Lt    => RegValue::from(left.float64 < right.float64),
            Op::F64Gt    => RegValue::from(left.float64 > right.float64),
            Op::F64Eq    => RegValue::from(left.float64 == right.float64),
            Op::F64NEq   => RegValue::from(left.float64 != right.float64),
            Op::F64LtEq  => RegValue::from(left.float64 <= right.float64),
            Op::F64GtEq  => RegValue::from(left.float64 >= right.float64),
            _ => unreachable!()
        }
    }
}

#[derive(Clone, Copy)]
pub struct Global {
    pub decl: Decl,
    pub value: Value,
    pub ty: Type
}

#[derive(Clone, Copy)]
struct Local {
    loc: Location,
    ty: Type,
}

impl Local {
    fn new(loc: Location, ty: Type) -> Local {
        let mut loc = loc;
        loc.can_assign = true;
        loc.can_access = true;
        Local { loc, ty }
    }
}

#[derive(Default)]
struct Locals {
    local_values: Vec<Local>,
    local_keys: Vec<Intern>,
    top_scope_index: usize,
}

impl Locals {
    fn assert_invariants(&self) {
        debug_assert!(self.local_keys.len() == self.local_values.len());
        debug_assert!(self.top_scope_index <= self.local_values.len());
    }

    fn insert(&mut self, ident: Intern, local: Local) {
        self.assert_invariants();
        self.local_keys.push(ident);
        self.local_values.push(local);
    }

    fn get(&self, ident: Intern) -> Option<&Local> {
        self.assert_invariants();
        self.local_keys.iter()
            .rposition(|v| *v == ident)
            .map(|i| unsafe { self.local_values.get_unchecked(i) })
    }

    fn push_scope(&mut self) -> usize {
        self.assert_invariants();
        let result = self.top_scope_index;
        self.top_scope_index = self.local_values.len();
        result
    }

    fn restore_scope(&mut self, mark: usize) {
        self.assert_invariants();
        debug_assert!(mark <= self.top_scope_index);
        self.local_values.truncate(self.top_scope_index);
        self.local_keys.truncate(self.top_scope_index);
        self.top_scope_index = mark;
    }
}

fn value_fits(value: Option<RegValue>, ty: Type) -> bool {
    unsafe {
        match (value, ty) {
            (Some(value), Type::I8)  =>  i8::MIN as isize <= value.sint && value.sint <=  i8::MAX as isize,
            (Some(value), Type::I16) => i16::MIN as isize <= value.sint && value.sint <= i16::MAX as isize,
            (Some(value), Type::I32) => i32::MIN as isize <= value.sint && value.sint <= i32::MAX as isize,
            (Some(value), Type::U8)  =>  u8::MIN as usize <= value.int  && value.int  <=  u8::MAX as usize,
            (Some(value), Type::U16) => u16::MIN as usize <= value.int  && value.int  <= u16::MAX as usize,
            (Some(value), Type::U32) => u32::MIN as usize <= value.int  && value.int  <= u32::MAX as usize,
            _ => true
        }
    }
}

/// Given a constant of type `value_ty`, determine if it is acceptable to store
/// it in a `want_ty`. This is distinct from the integer promotion rules. There
/// is no automatic coercion on writes, except for values known at compile-time
/// (through constant folding) to fit in the destination.
fn value_fits_or_types_match(value: Option<RegValue>, value_ty: Type, want_ty: Type) -> bool {
    value_ty == want_ty || (value_ty == Type::Int && value.is_some() && want_ty.is_integer() && value_fits(value, want_ty))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum LocationKind {
    None,
    Register,
    Stack,
    Based,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Location {
    base: isize,
    offset: isize,
    kind: LocationKind,
    can_assign: bool,
    can_access: bool
}

impl Location {
    const BAD_LOCATION: isize = isize::MAX;

    fn none() -> Location {
        Location { base: Self::BAD_LOCATION, offset: Self::BAD_LOCATION, kind: LocationKind::None, can_assign: false, can_access: false }
    }

    fn register(value: isize) -> Location {
        Location { base: Self::BAD_LOCATION, offset: value, kind: LocationKind::Register, can_assign: false, can_access: false }
    }

    fn stack(offset: isize) -> Location {
        Location { base: Self::BAD_LOCATION, offset: offset, kind: LocationKind::Stack, can_assign: true, can_access: false }
    }

    fn based(base: isize, offset: isize, can_assign: bool) -> Location {
        Location { base, offset, kind: LocationKind::Based, can_assign, can_access: false }
    }

    fn offset_by(&self, offset: isize) -> Location {
        debug_assert_implies!(offset > 0, matches!(self.kind, LocationKind::Stack|LocationKind::Based));
        let mut result = *self;
        result.offset += offset;
        result
    }
}

#[derive(Clone, Copy, Debug)]
struct ExprResult {
    addr: Location,
    ty: Type,
    value: Option<RegValue>,
}

impl ExprResult {
    fn is_ok(&self) -> bool {
        self.addr.kind != LocationKind::None || self.value.is_some()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Label(usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct InstrLocation(usize);

#[derive(Clone, Copy, Default, Debug)]
struct StmtContext {
    break_to: Option<Label>,
    continue_to: Option<Label>,
}

#[derive(Clone, Copy, Default, Debug)]
struct DestinationContext {
    ty: Option<Type>,
    dest: Option<Location>,
}

#[derive(Clone, Copy, Debug)]
enum PathContextState {
    ExpectAny,
    ExpectImplict,
    ExpectPaths,
    // ExpectIndices,
}

#[derive(Clone, Copy, Debug)]
struct PathContext {
    state: PathContextState,
    index: usize,
}

struct FatGen {
    code: Vec<FatInstr>,
    locals: Locals,
    reg_counter: isize,
    context: StmtContext,
    destination: DestinationContext,
    labels: Vec<InstrLocation>,
    patches: Vec<(Label, InstrLocation)>,
    return_type: Type,
    constant: bool,
    error: Option<IncompleteError>,
}

impl FatGen {
    const REGISTER_SIZE: usize = 8;

    fn new() -> FatGen {
        FatGen {
            code: Vec::new(),
            locals: Locals::default(),
            reg_counter: 0,
            context: StmtContext::default(),
            destination: DestinationContext::default(),
            labels: Vec::new(),
            patches: Vec::new(),
            return_type: Type::None,
            constant: false,
            error: None,
        }
    }

    fn error(&mut self, source_position: usize, msg: String) {
        if self.error.is_none() {
            self.error = Some(IncompleteError { source_position, msg });
        }
    }

    fn inc_bytes(&mut self, size: usize, align: usize) -> isize {
        debug_assert!(align.is_power_of_two());
        let result = (self.reg_counter as usize + (align - 1)) & !(align - 1);
        self.reg_counter += size as isize;
        // todo: error if size does not fit on the stack
        assert!(self.reg_counter < i32::MAX as isize);
        result as isize
    }

    fn inc_reg(&mut self) -> isize {
        self.inc_bytes(Self::REGISTER_SIZE, Self::REGISTER_SIZE)
    }

    fn put3(&mut self, op: Op, dest: isize, left: isize, right: isize) {
        if self.constant == false {
            let dest = i32::try_from(dest).unwrap_or_else(|_| { error!(self, 0, "stack is limited to range addressable by i32"); 0 }) as u32;
            let left = i32::try_from(left).unwrap_or_else(|_| { assert!(self.error.is_some()); 0 }) as u32;
            let right = i32::try_from(right).unwrap_or_else(|_| { assert!(self.error.is_some()); 0 }) as u32;
            self.code.push(FatInstr { op, dest, left, right });
        } else {
            // todo: expr position
            error!(self, 0, "non-constant expression");
        }
    }

    fn put(&mut self, op: Op, dest: isize, data: RegValue) {
        if self.constant == false {
            let dest = i32::try_from(dest).unwrap_or_else(|_| { error!(self, 0, "stack is limited to range addressable by i32"); 0 }) as u32;
            self.code.push(FatInstr { op, dest, left: unsafe { data.int32.0 }, right: unsafe { data.int32.1 }});
        } else {
            // todo: expr position
            error!(self, 0, "non-constant expression");
        }
    }

    fn put0(&mut self, op: Op) {
        self.put3(op, 0, 0, 0);
    }

    fn put2(&mut self, op: Op, dest: isize, left: isize) {
        self.put3(op, dest, left, 0);
    }

    fn put3p(&mut self, op: Op, dest: isize, leftright: isize) {
        self.put3(op, dest, leftright & 0xffffffff, leftright >> 32);
    }

    fn put_inc(&mut self, op: Op, data: RegValue) -> isize {
        let dest = self.inc_reg();
        self.put(op, dest, data);
        dest
    }

    fn put2_inc(&mut self, op: Op, left: isize) -> isize {
        let dest = self.inc_reg();
        self.put2(op, dest, left);
        dest
    }

    fn put3_inc(&mut self, op: Op, left: isize, right: isize) -> isize {
        let dest = self.inc_reg();
        self.put3(op, dest, left, right);
        dest
    }

    fn register(&mut self, expr: &ExprResult) -> isize {
        match expr.addr.kind {
            LocationKind::None => match expr.value {
                Some(v) if expr.ty.is_pointer() => self.put_inc(Op::LoadStackAddress, v),
                Some(v) => self.constant(v),
                None => Location::BAD_LOCATION
            },
            LocationKind::Register => expr.addr.offset,
            LocationKind::Stack => {
                if let Some(op) = load_op_stack(expr.ty) {
                    self.put2_inc(op, expr.addr.offset)
                } else {
                    // todo: overwritable error here. we expect the _next_ error
                    // will be more informative to the user. BAD_LOCATION can
                    // appear sometimes in non-error cases, so it would be nice
                    // to explicitly enter an error state, and generally keep
                    // BAD_LOCATION usage to a minimum
                    Location::BAD_LOCATION
                }
            }
            LocationKind::Based => {
                if let Some(op) = load_op_pointer(expr.ty) {
                    let ptr = self.location_register(&expr.addr);
                    self.put2_inc(op, ptr)
                } else {
                    // todo: weak (overwritable) error here, as above
                    Location::BAD_LOCATION
                }
            }
        }
    }

    fn location_register(&mut self, location: &Location) -> isize {
        assert!(location.kind == LocationKind::Based);
        if location.base != 0 {
            let offset_register = self.constant(location.offset.into());
            self.put3_inc(Op::IntAdd, location.base, offset_register)
        } else {
            location.base
        }
    }

    fn emit_constant(&mut self, expr: &ExprResult) -> ExprResult {
        if let Some(_) = expr.value {
            let mut expr = *expr;
            expr.addr = Location::register(self.register(&expr));
            expr
        } else {
            *expr
        }
    }

    fn put_jump(&mut self, label: Label) {
        self.patches.push((label, InstrLocation(self.code.len())));
        self.put2(Op::Jump, 0, label.0 as isize);
    }

    fn put_jump_zero(&mut self, cond_register: isize, label: Label) {
        self.patches.push((label, InstrLocation(self.code.len())));
        self.put2(Op::JumpIfZero, cond_register, label.0 as isize);
    }

    fn put_jump_nonzero(&mut self, cond_register: isize, label: Label) {
        self.patches.push((label, InstrLocation(self.code.len())));
        self.put2(Op::JumpIfNotZero, cond_register, label.0 as isize);
    }

    fn stack_alloc(&mut self, ctx: &Compiler, ty: Type) -> Location {
        let info = ctx.types.info(ty);
        Location::stack(self.inc_bytes(info.size, info.alignment))
    }

    fn constant(&mut self, value: RegValue) -> isize {
        // :)
        self.put_inc(Op::Immediate, value)
    }

    fn zero(&mut self, dest: &Location, size: isize) {
        match dest.kind {
            LocationKind::None => unreachable!(),
            LocationKind::Register => unreachable!(),
            LocationKind::Stack => {
                self.put2(Op::StackRelativeZero, dest.offset, size);
            }
            LocationKind::Based => {
                let dest_addr_register = self.location_register(dest);
                let size_register = self.constant(size.into());
                self.put2(Op::Zero, dest_addr_register, size_register);
            }
        }
    }

    fn copy(&mut self, ctx: &Compiler, destination_type: Type, dest: &Location, src: &ExprResult) {
        let size = ctx.types.info(destination_type).size as isize;
        let alignment = ctx.types.info(destination_type).alignment as isize;
        debug_assert_eq!((dest.offset) & (alignment - 1), 0);
        match dest.kind {
            LocationKind::None => unreachable!(),
            LocationKind::Register => {
                let result_register = self.register(src);
                self.put2(Op::Move, dest.offset, result_register);
            }
            LocationKind::Stack => {
                if let Some(op) = store_op_stack(destination_type) {
                    let result_register = self.register(src);
                    self.put2(op, dest.offset, result_register);
                } else {
                    self.put3(Op::StackRelativeCopy, dest.offset, src.addr.offset, size);
                }
            }
            LocationKind::Based => {
                let dest_addr_register = self.location_register(dest);
                let src = &self.emit_constant(src);

                match src.addr.kind {
                    LocationKind::None => (),
                    LocationKind::Register => {
                        if let Some(op) = store_op_pointer(destination_type) {
                            let src = self.register(src);
                            self.put2(op, dest_addr_register, src);
                        } else {
                            unreachable!();
                        }
                    }
                    LocationKind::Stack => {
                        let src_addr_register = self.put2_inc(Op::LoadStackAddress, src.addr.offset);
                        let size_register = self.constant(size.into());
                        self.put3(Op::Copy, dest_addr_register, src_addr_register, size_register);
                    }
                    LocationKind::Based => {
                        let src_addr_register = self.location_register(&src.addr);
                        let size_register = self.constant(size.into());
                        self.put3(Op::Copy, dest_addr_register, src_addr_register, size_register);
                    }
                }
            }
        }
    }

    fn copy_to_stack(&mut self, ctx: &Compiler, src: &ExprResult) -> Location {
        let info = ctx.types.info(src.ty);
        let offset = self.inc_bytes(info.size, info.alignment);
        let result = Location::stack(offset);
        self.copy(ctx, src.ty, &result, &src);
        result
    }

    fn label(&mut self) -> Label {
        if self.constant == false {
            let result = Label(self.labels.len());
            self.labels.push(InstrLocation(!0));
            result
        } else {
            // TODO: expr position
            error!(self, 0, "non-constant expression");
            return Label(0);
        }
    }

    fn label_here(&mut self) -> Label {
        let result = self.label();
        self.patch(result);
        result
    }

    fn break_label(&self) -> Option<Label> {
        self.context.break_to
    }

    fn continue_label(&self) -> Option<Label> {
        self.context.continue_to
    }

    fn push_loop_context(&mut self) -> (Label, Label, StmtContext) {
        let result = self.context;
        let break_to = self.label();
        let continue_to = self.label();
        self.context = StmtContext {
            break_to: Some(break_to),
            continue_to: Some(continue_to),
        };
        (break_to, continue_to, result)
    }

    fn push_break_context(&mut self) -> (Label, StmtContext) {
        let result = self.context;
        let break_to = self.label();
        self.context = StmtContext {
            break_to: Some(break_to),
            continue_to: result.continue_to,
        };
        (break_to, result)
    }

    fn restore(&mut self, mark: StmtContext) {
        self.context = mark;
    }

    fn patch(&mut self, label: Label) {
        assert!(self.labels[label.0] == InstrLocation(!0));
        self.labels[label.0] = InstrLocation(self.code.len());
    }

    fn apply_patches(&mut self) {
        if self.error.is_none() {
            for &(Label(label), InstrLocation(from)) in self.patches.iter() {
                let InstrLocation(to) = self.labels[label];
                assert!(to != !0);
                assert!(self.code[from].is_jump() && self.code[from].left == label as u32);
                let offset = (to as isize) - (from as isize) - 1;
                let offset = i32::try_from(offset).expect("function too large") as u32;
                self.code[from].left = offset;
            }
        }
        self.patches.clear();
    }

    fn type_expr(&mut self, ctx: &mut Compiler, expr: TypeExpr) -> Type {
        let mut result = Type::None;
        match ctx.ast.type_expr(expr) {
            TypeExprData::Infer => (unreachable!()),
            TypeExprData::Name(intern) => {
                result = sym::resolve_type(ctx, intern).ty;
                let resolve_error = std::mem::take(&mut ctx.error);
                let old_error = std::mem::take(&mut self.error);
                self.error = old_error.or(resolve_error);
            }
            TypeExprData::Expr(_) => unreachable!(),
            TypeExprData::List(intern, args) => {
                let mut args = args;
                if intern == ctx.intern("arr") {
                    if let Some(ty_expr) = args.next() {
                        let ty = self.type_expr(ctx, ty_expr);
                        if let Some(len_expr) = args.next() {
                            if let TypeExprData::Expr(len_expr) = ctx.ast.type_expr(len_expr) {
                                let len = self.constant_expr(ctx, len_expr);
                                if len.ty.is_integer() && len.value.is_some() {
                                    // TODO: This convert op is a roundabout way to check for the positive+fits condition
                                    // but should be more direct/less subtle
                                    if let Some(op) = convert_op(len.ty, Type::Int) {
                                        let value = apply_unary_op(op, len.value.unwrap());
                                        if unsafe { value.sint } > 0 {
                                            result = ctx.types.array(ty, unsafe { value.sint } as usize);
                                        } else {
                                            error!(self, 0, "arr length must be a positive integer and fit in a signed integer");
                                        }
                                    } else {
                                        unreachable!();
                                    }
                                } else {
                                    // todo: type expr location
                                    error!(self, 0, "arr length must be a constant integer")
                                }
                            } else {
                                error!(self, 0, "argument 2 of arr type must be a value expression")
                            }
                        } else /* Some(len_expr) != args.next() */ {
                            // infer?
                            todo!();
                        }
                    } else {
                        error!(self, 0, "type arr takes 2 arguments, base type and [length]")
                    }
                } else if intern == ctx.intern("ptr") {
                    if let Some(ty_expr) = args.next() {
                        let ty = self.type_expr(ctx, ty_expr);
                        if let Some(bound_expr) = args.next() {
                            if let TypeExprData::Expr(bound_expr) = ctx.ast.type_expr(bound_expr) {
                                let bound = self.constant_expr(ctx, bound_expr);
                                if bound.ty.is_integer() {
                                    todo!();
                                } else {
                                    // todo: type expr location
                                    error!(self, 0, "ptr bound must be a place")
                                }
                            } else {
                                error!(self, 0, "argument 2 of ptr type must be a value expression")
                            }
                        } else /* Some(len_expr) != args.next() */ {
                            // unbounded
                            result = ctx.types.pointer(ty);
                        }
                    } else {
                        // untyped
                        result = Type::VoidPtr;
                    }
                }
            }
        }
        result
    }

    fn path<'c>(&mut self, ctx: &Compiler, path_ctx: &mut PathContext, ty: Type, path: CompoundPath) -> Option<types::Item> {
        match path {
            CompoundPath::Implicit => {
                if matches!(path_ctx.state, PathContextState::ExpectAny) {
                    path_ctx.state = PathContextState::ExpectImplict;
                    assert!(path_ctx.index == 0);
                }
                if matches!(path_ctx.state, PathContextState::ExpectImplict) {
                    let result = ctx.types.index_info(ty, path_ctx.index);
                    if let None = result {
                        // Todo: expr position
                        error!(self, 0, "too many values in compund initializer for {}", ctx.type_str(ty));
                    }
                    path_ctx.index += 1;
                    result
                } else {
                    // Todo: expr position
                    error!(self, 0, "expected expression in item {} of compound initializer", path_ctx.index);
                    None
                }
            }
            CompoundPath::Path(path) => {
                if matches!(path_ctx.state, PathContextState::ExpectAny) {
                    path_ctx.state = PathContextState::ExpectPaths;
                }
                if matches!(path_ctx.state, PathContextState::ExpectPaths) {
                    if let ExprData::Name(name) = ctx.ast.expr(path) {
                        let result = ctx.types.item_info(ty, name);
                        if let None = result {
                            error!(self, ctx.ast.expr_source_position(path), "no field '{}' on type {}", ctx.str(name), ctx.type_str(ty));
                        }
                        result
                    } else {
                        // TODO: Support more.complicated[0].paths
                        error!(self, ctx.ast.expr_source_position(path), "bad path to struct field");
                        None
                    }
                } else {
                    error!(self, ctx.ast.expr_source_position(path), "expected path to struct field");
                    None
                }
            }
            CompoundPath::Index(_index) => {
                todo!();
            }
        }
    }

    fn constant_expr(&mut self, ctx: &mut Compiler, expr: Expr) -> ExprResult {
        // This is kind of a hack to keep constant and non-constant expressions
        // totally unified while things are still incomplete
        let code_len = self.code.len();
        let labels_len = self.labels.len();
        self.constant = true;
        let result = self.expr(ctx, expr);
        self.constant = false;
        if let None = result.value {
            // This means the constant expr has not evaluated to a constant. But
            // it might have evaluated to a memory location that the compiler
            // knows (e.g., a variables relative position on the stack), and we
            // want to support those in type exprs, so calling code has to expect this
        } else {
            debug_assert!(self.code.len() == code_len);
            debug_assert!(self.labels.len() == labels_len);
            debug_assert!(result.addr.offset == Location::BAD_LOCATION);
        }
        result
    }

    fn expr_with_destination_type(&mut self, ctx: &mut Compiler, expr: Expr, destination_type: Type) -> ExprResult {
        self.destination.ty = Some(destination_type);
        self.expr(ctx, expr)
    }

    fn expr_with_destination(&mut self, ctx: &mut Compiler, expr: Expr, destination_type: Type, dest: &Location) -> ExprResult {
        assert_ne!(dest.kind, LocationKind::None);
        self.destination.ty = Some(destination_type);
        self.destination.dest = Some(*dest);
        let mut result = self.expr(ctx, expr);
        self.destination.dest = None;
        if value_fits_or_types_match(result.value, result.ty, destination_type) {
            if result.addr != *dest {
                self.copy(ctx, destination_type, dest, &result);
                result.addr = *dest;
            }
        }
        result
    }

    fn expr(&mut self, ctx: &mut Compiler, expr: Expr) -> ExprResult {
        let mut result = ExprResult { addr: Location::none(), ty: Type::None, value: None };
        let destination_type = self.destination.ty.take();
        match ctx.ast.expr(expr) {
            ExprData::Int(value) => {
                result.value = Some(value.into());
                result.ty = Type::Int;
            }
            ExprData::Float32(value) => {
                result.value = Some(value.into());
                result.ty = Type::F32;
            }
            ExprData::Float64(value) => {
                result.value = Some(value.into());
                result.ty = Type::F64;
            }
            ExprData::Name(name) => {
                if let Some(local) = self.locals.get(name) {
                    result.addr = local.loc;
                    result.ty = local.ty;
                } else if let Some(&global) = ctx.globals.get(&name) {
                    result.value = Some(global.value.into());
                    result.ty = global.ty;
                } else {
                    error!(self, ctx.ast.expr_source_position(expr), "unknown identifier '{}'", ctx.str(name));
                }
            }
            ExprData::Compound(fields) => {
                if let Some(ty) = destination_type {
                    // Todo: check for duplicate fields. need to consider unions and fancy paths (both unimplemented)
                    let info = ctx.types.info(ty);
                    let base = self.destination.dest.unwrap_or_else(|| self.stack_alloc(ctx, ty));
                    let mut path_ctx = PathContext { state: PathContextState::ExpectAny, index: 0 };
                    if matches!(info.kind, TypeKind::Struct|TypeKind::Array) {
                        self.zero(&base, info.size as isize);
                        for field in fields {
                            let field = ctx.ast.compound_field(field);
                            if let Some(item) = self.path(ctx, &mut path_ctx, ty, field.path) {
                                let dest = base.offset_by(item.offset as isize);
                                let expr = self.expr_with_destination(ctx, field.value, item.ty, &dest);
                                if !expr.is_ok() {
                                    match path_ctx.state {
                                        PathContextState::ExpectPaths =>
                                            error!(self, ctx.ast.expr_source_position(field.value), "incompatible types (field '{}' is of type {}, found {})", ctx.str(item.name), ctx.type_str(item.ty), ctx.type_str(expr.ty)),
                                        PathContextState::ExpectImplict => //|PathContextState::ExpectIndices =>
                                            error!(self, ctx.ast.expr_source_position(field.value), "incompatible types (arr is of type {}, found {})", ctx.type_str(item.ty), ctx.type_str(expr.ty)),
                                        _ => todo!()
                                    }
                                }
                            }
                        }
                        result.addr = base;
                        result.ty = ty;
                    } else {
                        error!(self, ctx.ast.expr_source_position(expr), "compound initializer used for non-aggregate type");
                    }
                } else {
                    error!(self, ctx.ast.expr_source_position(expr), "untyped compound initializer");
                }
            }
            ExprData::Field(left_expr, field) => {
                let left = self.expr(ctx, left_expr);
                if left.ty.is_pointer() {
                    let ty = ctx.types.base_type(left.ty);
                    if let Some(item) = ctx.types.item_info(ty, field) {
                        match left.value {
                            Some(lv) => result.value = Some(unsafe { lv.int + item.offset }.into()),
                            None => {
                                let base = self.register(&left);
                                result.addr = Location::based(base, item.offset as isize, left.addr.can_assign);
                            }
                        }
                        // todo: can_assign/can_access?
                        result.ty = item.ty;
                    } else {
                        error!(self, ctx.ast.expr_source_position(left_expr), "no field '{}' on type {}", ctx.str(field), ctx.type_str(left.ty));
                    }
                } else if let Some(item) = ctx.types.item_info(left.ty, field) {
                    debug_assert_implies!(self.error.is_none(), matches!(left.addr.kind, LocationKind::Stack|LocationKind::Based));
                    result.addr = left.addr;
                    result.addr.offset += item.offset as isize;
                    result.ty = item.ty;
                } else {
                    error!(self, ctx.ast.expr_source_position(left_expr), "no field '{}' on type {}", ctx.str(field), ctx.type_str(left.ty));
                }
            }
            ExprData::Index(left_expr, index_expr) => {
                let left = self.expr(ctx, left_expr);
                if matches!(ctx.types.info(left.ty).kind, TypeKind::Array|TypeKind::Pointer) {
                    let base_type = ctx.types.base_type(left.ty);
                    let index = self.expr(ctx, index_expr);
                    if index.ty.is_integer() {
                        let element_size = ctx.types.info(base_type).size;
                        let addr = match index.value {
                            Some(value) => {
                                let index = unsafe { value.sint };
                                left.addr.offset_by(index * element_size as isize)
                            }
                            None => {
                                // This has a weird transposition where, if accessing an array member of a
                                // pointer, we update the base pointer, but leave the offset intact.
                                // In other words, we compute
                                //      a.b[i]
                                // as
                                //      (a +   i*sizeof(b[0])).b
                                //       ^ base                ^ offset
                                // instead of
                                //      (a.b + i*sizeof(b[0]))
                                //       ^base                 ^ offset (0)
                                // So we can keep accumulating static offsets at compile time.
                                let size_reg = self.constant(element_size.into());
                                let index_reg = self.register(&index);
                                let offset_reg = self.put3_inc(Op::IntMul, index_reg, size_reg);
                                // TODO: probably fold these 3 cases together and just decay arrays to pointers like an animal
                                if left.ty.is_pointer() {
                                    let (old_base_reg, offset) = if left.addr.kind == LocationKind::Based {
                                        (left.addr.base, left.addr.offset)
                                    } else {
                                        (self.register(&left), 0)
                                    };
                                    let base = self.put3_inc(Op::IntAdd, old_base_reg, offset_reg);
                                    Location::based(base, offset, left.addr.can_assign)
                                } else {
                                    match left.addr.kind {
                                        LocationKind::Stack => {
                                            let old_base_reg = self.put_inc(Op::LoadStackAddress, left.addr.offset.into());
                                            let base = self.put3_inc(Op::IntAdd, old_base_reg, offset_reg);
                                            Location { base, offset: 0, kind: LocationKind::Based, ..left.addr }
                                        }
                                        LocationKind::Based => {
                                            let old_base_reg = left.addr.base;
                                            let base = self.put3_inc(Op::IntAdd, old_base_reg, offset_reg);
                                            Location { base, ..left.addr }
                                        }
                                        _ => unreachable!(),
                                    }
                                }
                            }
                        };
                        result.addr = addr;
                        result.ty = base_type;
                    } else {
                        error!(self, ctx.ast.expr_source_position(index_expr), "index must be an integer (found {})", ctx.type_str(index.ty))
                    }
                } else {
                    error!(self, ctx.ast.expr_source_position(left_expr), "indexed type must be an array (found {}, a {:?})", ctx.type_str(left.ty), ctx.types.info(left.ty).kind)
                }
            },
            ExprData::Unary(op_token, right_expr) => {
                let right = if let (parse::TokenKind::BitAnd, Some(ty)) = (op_token, destination_type) {
                    self.expr_with_destination_type(ctx, right_expr, ty)
                } else {
                    self.expr(ctx, right_expr)
                };
                if matches!(op_token, parse::TokenKind::BitAnd|parse::TokenKind::Mul) {
                    if op_token == parse::TokenKind::BitAnd {
                        match right.addr.kind {
                            LocationKind::None =>
                                error!(self, ctx.ast.expr_source_position(right_expr), "cannot take address of this expression"),
                            LocationKind::Register if !right.addr.can_access =>
                                error!(self, ctx.ast.expr_source_position(right_expr), "cannot take address of this expression"),
                            LocationKind::Register|LocationKind::Stack => {
                                let base = right.addr.offset;
                                // todo: seems weird to be setting these with kind = None
                                result.addr.can_assign = right.addr.can_assign;
                                result.addr.can_access = right.addr.can_access;
                                result.value = Some(base.into());
                            }
                            LocationKind::Based => {
                                result.addr = Location::register(self.location_register(&right.addr));
                            }
                        }
                        result.ty = ctx.types.pointer(right.ty);
                    } else if op_token == parse::TokenKind::Mul {
                        if right.ty.is_pointer() {
                            match right.value {
                                Some(offset) => result.addr = Location::stack(unsafe { offset.sint }),
                                None => {
                                    let right_register = self.register(&right);
                                    result.addr = Location::based(right_register, 0, right.addr.can_assign);
                                }
                            }
                            result.ty = ctx.types.base_type(right.ty);
                        } else {
                            error!(self, ctx.ast.expr_source_position(right_expr), "cannot dereference {}", ctx.type_str(right.ty));
                        }
                    }
                } else if let Some((op, result_ty)) = unary_op(op_token, right.ty) {
                    match right.value {
                        Some(_) if right.ty.is_pointer() => result.addr = Location::register(self.register(&right)),
                        Some(rv) => result.value = Some(apply_unary_op(op, rv)),
                        None => {
                            let right_register = self.register(&right);
                            result.addr = Location::register(self.put2_inc(op, right_register));
                        }
                    }
                    result.ty = result_ty;
                } else {
                    error!(self, ctx.ast.expr_source_position(right_expr), "incompatible type {}{}", op_token, ctx.type_str(right.ty));
                }
            }
            ExprData::Binary(op_token, left_expr, right_expr) => {
                if op_token == parse::TokenKind::LogicAnd || op_token == parse::TokenKind::LogicOr {
                    // No constant folding here because it makes this code complicated for nothing in return
                    // TODO: want it for when both sides are constant, so constant expressions don't have this one weird edge case
                    let short_circuit = self.label();
                    let done = self.label();
                    let left = self.expr(ctx, left_expr);
                    let left_register = self.register(&left);
                    if op_token == parse::TokenKind::LogicAnd {
                        self.put_jump_zero(left_register, short_circuit);
                    } else {
                        self.put_jump_nonzero(left_register, short_circuit);
                    }
                    let right = self.expr(ctx, right_expr);
                    let right_register = self.register(&right);
                    if let Some((_, result_ty)) = binary_op(op_token, left.ty, right.ty) {
                        let dest_register = self.inc_reg();
                        self.put3p(Op::Move, dest_register, right_register);
                        self.put_jump(done);
                        self.patch(short_circuit);
                        self.put3p(Op::Move, dest_register, left_register);
                        self.patch(done);
                        result.addr = Location::register(dest_register);
                        result.ty = result_ty;
                    } else {
                        error!(self, ctx.ast.expr_source_position(left_expr), "incompatible types ({} {} {}), logical operators use booleans", ctx.type_str(left.ty), op_token, ctx.type_str(right.ty));
                    }
                } else {
                    let left = self.expr(ctx, left_expr);
                    let right = self.expr(ctx, right_expr);
                    if let Some((op, result_ty)) = binary_op(op_token, left.ty, right.ty) {
                        match (left.value, right.value) {
                            (Some(lv), Some(rv)) if !left.ty.is_pointer() && !right.ty.is_pointer() => result.value = Some(apply_binary_op(op, lv, rv)),
                            _ => {
                                let left_register = self.register(&left);
                                let right_register = self.register(&right);
                                let result_register = self.put3_inc(op, left_register, right_register);
                                result.addr = Location::register(result_register);
                            }
                        }
                        result.ty = result_ty;
                    } else {
                        error!(self, ctx.ast.expr_source_position(left_expr), "incompatible types ({} {} {})", ctx.type_str(left.ty), op_token, ctx.type_str(right.ty));
                    }
                }
            }
            ExprData::Ternary(cond_expr, left_expr, right_expr) => {
                let cond = self.expr(ctx, cond_expr);
                let cond_register = self.register(&cond);
                if cond.ty == Type::Bool {
                    if let Some(cv) = cond.value {
                        // Descend into both branches to type check, then discard the generated code.
                        {
                            let top = self.code.len();
                            self.expr(ctx, left_expr);
                            self.expr(ctx, right_expr);
                            self.code.truncate(top);
                        }
                        if cv.is_true() {
                            result = self.expr(ctx, left_expr);
                        } else {
                            result = self.expr(ctx, right_expr);
                        }
                    } else {
                        let right_branch = self.label();
                        let done = self.label();
                        self.put_jump_zero(cond_register, right_branch);
                        let left = self.expr(ctx, left_expr);
                        let left = self.emit_constant(&left);
                        self.put_jump(done);
                        self.patch(right_branch);
                        let right = self.expr_with_destination(ctx, right_expr, left.ty, &left.addr);
                        self.patch(done);
                        if left.ty == right.ty {
                            debug_assert!(left.addr == right.addr);
                            result = left;
                        } else {
                            error!(self, ctx.ast.expr_source_position(left_expr), "incompatible types (... ? {} :: {})", ctx.type_str(left.ty), ctx.type_str(right.ty));
                        }
                    }
                } else {
                    error!(self, ctx.ast.expr_source_position(cond_expr), "ternary expression requires a boolean condition");
                }
            }
            ExprData::Call(callable, args) => {
                let addr = self.expr(ctx, callable);
                let info = ctx.types.info(addr.ty);
                let return_type = info.items.last().expect("tried to generate code to call a func without return type").ty;
                if info.kind == TypeKind::Func {
                    let mut arg_gens = SmallVec::new();
                    for (i, expr) in args.enumerate() {
                        let info = ctx.types.info(addr.ty);
                        let item = info.items[i];
                        let gen = self.expr(ctx, expr);
                        if gen.ty != item.ty {
                            error!(self, ctx.ast.expr_source_position(expr), "argument {} is of type {}, found {}", i, ctx.type_str(item.ty), ctx.type_str(gen.ty));
                            break;
                        }
                        arg_gens.push(gen);
                    }
                    for &gen in arg_gens.iter() {
                        match gen.value {
                            Some(gv) => self.constant(gv),
                            _ => {
                                assert!(gen.addr.kind == LocationKind::Register, "todo");
                                self.put2_inc(Op::Move, gen.addr.offset)
                            }
                        };
                    }
                    result.addr.offset = match addr.value {
                        Some(func) => self.put_inc(Op::Call, func),
                        _ => todo!()
                    };
                    result.addr.kind = LocationKind::Register;
                    result.ty = return_type;
                } else {
                    // TODO: get a string representation of the whole `callable` expr for nicer error message
                    error!(self, ctx.ast.expr_source_position(callable), "cannot call a {:?}", info.kind);
                }
            }
            ExprData::Cast(expr, type_expr) => {
                let to_ty = self.type_expr(ctx, type_expr);
                let left = self.expr_with_destination_type(ctx, expr, to_ty);
                if let Some(op) = convert_op(left.ty, to_ty) {
                    match left.value {
                        Some(_) if left.ty.is_pointer() => result.addr = Location::register(self.register(&left)),
                        Some(lv) => result.value = Some(apply_unary_op(op, lv)),
                        None => {
                            if op != Op::Noop {
                                let reg = self.register(&left);
                                result.addr = Location::register(self.put2_inc(op, reg));
                            } else {
                                result = left;
                            }
                        }
                    }
                    if types_match_with_promotion(result.ty, to_ty) {
                        // We can send integer types back up the tree unmodified
                        // without doing the conversion here.
                    } else {
                        result.ty = to_ty;
                    }
                } else if let ExprData::Unary(parse::TokenKind::BitAnd, inner) = ctx.ast.expr(expr) {
                    if let ExprData::Compound(_) = ctx.ast.expr(inner) {
                        // The cast-on-the-right synax doesn't work great with taking addresses.
                        //
                        // Conceptually, this transposes
                        //      &{}:Struct
                        // to
                        //      &({}:Struct)
                        //
                        // We only do this if the above convert_op returns None, as otherwise we cannot distinguish between
                        //      &a:int / (&a):int
                        // and
                        //      &(a:int)
                        debug_assert!(left.value.is_some());
                        debug_assert!(left.ty.is_pointer());
                        result = ExprResult { ty: ctx.types.pointer(to_ty), ..left };
                    }
                } else {
                    error!(self, ctx.ast.expr_source_position(expr), "cannot cast from {} to {}", ctx.type_str(left.ty), ctx.type_str(to_ty));
                }
            }
        };
        debug_assert_implies!(self.error.is_none() && result.addr.offset == Location::BAD_LOCATION, result.addr.kind == LocationKind::None);
        debug_assert_implies!(self.error.is_none() && result.addr.offset != Location::BAD_LOCATION, result.addr.kind != LocationKind::None);
        debug_assert_implies!(self.error.is_none() && result.addr.base != Location::BAD_LOCATION, result.addr.kind == LocationKind::Based);
        result
    }

    fn stmt(&mut self, ctx: &mut Compiler, stmt: Stmt) -> Option<Type> {
        let mut return_type = None;
        match ctx.ast.stmt(stmt) {
            StmtData::Block(body) => {
                return_type = self.stmts(ctx, body)
            }
            StmtData::Return(Some(expr)) => {
                let ret_expr = self.expr(ctx, expr);
                if types_match_with_promotion(ret_expr.ty, self.return_type) {
                    let reg = self.register(&ret_expr);
                    self.put2(Op::Return, 0, reg);
                    return_type = Some(ret_expr.ty);
                } else {
                    error!(self, ctx.ast.expr_source_position(expr), "type mismatch: expected a return value of type {} (found {})", ctx.type_str(self.return_type), ctx.type_str(ret_expr.ty));
                }
            }
            StmtData::Return(None) => {
                if self.return_type == Type::None {
                    self.put0(Op::Return);
                } else {
                    // TODO: stmt source position
                    error!(self, 0, "empty return, expected a return value of type {}", ctx.type_str(self.return_type));
                }
            }
            StmtData::Break => {
                if let Some(label) = self.break_label() {
                    self.put_jump(label);
                } else {
                    // TODO: stmt source position
                    error!(self, 0, "break outside of loop/switch");
                }
            }
            StmtData::Continue => {
                if let Some(label) = self.continue_label() {
                    self.put_jump(label);
                } else {
                    // TODO: stmt source position
                    error!(self, 0, "continue outside of loop");
                }
            }
            StmtData::If(cond_expr, then_body, else_body) => {
                let cond = self.expr(ctx, cond_expr);
                if cond.ty == Type::Bool {
                    let cond_reg = self.register(&cond);
                    let else_branch = self.label();
                    self.put_jump_zero(cond_reg, else_branch);
                    let then_ret = self.stmts(ctx, then_body);
                    if else_body.is_nonempty() {
                        let skip_else = self.label();
                        self.put_jump(skip_else);
                        self.patch(else_branch);
                        let else_ret = self.stmts(ctx, else_body);
                        self.patch(skip_else);
                        return_type = then_ret.and(else_ret);
                    } else {
                        self.patch(else_branch);
                    }
                } else {
                    error!(self, ctx.ast.expr_source_position(cond_expr), "if statement requires a boolean condition");
                }
            }
            StmtData::For(pre_stmt, cond_expr, post_stmt, body) => {
                let (break_label, continue_label, gen_ctx) = self.push_loop_context();
                let mark = self.locals.push_scope();
                if let Some(stmt) = pre_stmt {
                    self.stmt(ctx, stmt);
                }
                if let Some(expr) = cond_expr {
                    let pre_cond = self.expr(ctx, expr);
                    let pre_cond_register = self.register(&pre_cond);
                    if pre_cond.ty == Type::Bool {
                        self.put_jump_zero(pre_cond_register, break_label);
                        let loop_entry = self.label_here();
                        return_type = self.stmts(ctx, body);
                        self.patch(continue_label);
                        if let Some(stmt) = post_stmt {
                            self.stmt(ctx, stmt);
                        }
                        let post_cond = self.expr(ctx, expr);
                        let post_cond_register = self.register(&post_cond);
                        self.put_jump_nonzero(post_cond_register, loop_entry);
                    } else {
                        error!(self, ctx.ast.expr_source_position(expr), "for statement requires a boolean condition");
                    }
                } else {
                    let loop_entry = self.label_here();
                    return_type = self.stmts(ctx, body);
                    self.patch(continue_label);
                    if let Some(stmt) = post_stmt {
                        self.stmt(ctx, stmt);
                    }
                    self.put_jump(loop_entry);
                }
                self.locals.restore_scope(mark);
                self.patch(break_label);
                self.restore(gen_ctx);
            }
            StmtData::While(cond_expr, body) => {
                let (break_label, continue_label, gen_ctx) = self.push_loop_context();
                let pre_cond = self.expr(ctx, cond_expr);
                if pre_cond.ty == Type::Bool {
                    let pre_cond_register = self.register(&pre_cond);
                    self.put_jump_zero(pre_cond_register, break_label);
                    let loop_entry = self.label_here();
                    return_type = self.stmts(ctx, body);
                    self.patch(continue_label);
                    let post_cond = self.expr(ctx, cond_expr);
                    let post_cond_register = self.register(&post_cond);
                    self.put_jump_nonzero(post_cond_register, loop_entry);
                } else {
                    error!(self, ctx.ast.expr_source_position(cond_expr), "while statement requires a boolean condition");
                }
                self.patch(break_label);
                self.restore(gen_ctx);
            }
            StmtData::Switch(control_expr, cases) => {
                // Totally naive switch implementation. Break by default,
                // fall-through unimplemented.
                let (end, gen_ctx) = self.push_break_context();
                let mut labels = SmallVec::new();
                let mut else_label = None;
                let control = self.expr(ctx, control_expr);
                let control_register = self.register(&control);
                for case in cases {
                    let block_label = self.label();
                    labels.push(block_label);
                    if let SwitchCaseData::Cases(exprs, _) = ctx.ast.switch_case(case) {
                        for case_expr in exprs {
                            let expr = self.expr(ctx, case_expr);
                            if let Some(_ev) = expr.value {
                                if let Some((op, ty)) = binary_op(parse::TokenKind::Eq, control.ty, expr.ty) {
                                    debug_assert_eq!(ty, Type::Bool);
                                    let expr_register = self.register(&expr);
                                    let matched = self.put3_inc(op, control_register, expr_register);
                                    self.put_jump_nonzero(matched, block_label);
                                } else {
                                    error!(self, ctx.ast.expr_source_position(case_expr), "type mismatch between switch control ({}) and case ({})", ctx.type_str(control.ty), ctx.type_str(expr.ty))
                                }
                            } else {
                                error!(self, ctx.ast.expr_source_position(case_expr), "non-constant switch case");
                            }
                        }
                    } else {
                        else_label = Some(block_label);
                    }
                }
                if let Some(label) = else_label {
                    self.put_jump(label);
                }
                let mut first_block = true;
                for (i, case) in cases.enumerate() {
                    let block = match ctx.ast.switch_case(case) {
                        SwitchCaseData::Cases(_, block) => block,
                        SwitchCaseData::Else(block) => block
                    };
                    let block_label = labels[i];
                    self.patch(block_label);
                    let ret = self.stmts(ctx, block);
                    if first_block {
                        return_type = ret;
                        first_block = false;
                    } else {
                        return_type = return_type.and(ret);
                    }
                    self.put_jump(end);
                }
                self.patch(end);
                self.restore(gen_ctx);
            }
            StmtData::Do(cond_expr, body) => {
                let (break_label, continue_label, gen) = self.push_loop_context();
                let loop_entry = self.label_here();
                return_type = self.stmts(ctx, body);
                self.patch(continue_label);
                let post_cond = self.expr(ctx, cond_expr);
                if post_cond.ty != Type::Bool {
                    error!(self, ctx.ast.expr_source_position(cond_expr), "do statement requires a boolean condition");
                }
                let post_cond_register = self.register(&post_cond);
                self.put_jump_nonzero(post_cond_register, loop_entry);
                self.patch(break_label);
                self.restore(gen);
            }
            StmtData::Expr(expr) => {
                if let ExprData::Cast(inner, _ty) = ctx.ast.expr(expr) {
                    if let ExprData::Name(_name) = ctx.ast.expr(inner) {
                        // This looks like an attempt to declare an uninitialised value,
                        //      name: ty;
                        // which will fail with a confusing error message if `name` is not declared,
                        // and be a noop otherwise. We disallow the latter case because it's confusing
                        // and provide a better error message for the former.
                        error!(self, ctx.ast.expr_source_position(expr), "cannot declare a value without initialising it");
                    }
                }
                self.expr(ctx, expr);
            }
            StmtData::VarDecl(ty_expr, left, right) => {
                if let ExprData::Name(var) = ctx.ast.expr(left) {
                    let expr;
                    let decl_type;
                    if matches!(ctx.ast.type_expr(ty_expr), TypeExprData::Infer) {
                        expr = self.expr(ctx, right);
                        decl_type = expr.ty;
                    } else {
                        decl_type = self.type_expr(ctx, ty_expr);
                        expr = self.expr_with_destination_type(ctx, right, decl_type);
                    }
                    let addr = if expr.addr.can_access {
                        self.copy_to_stack(ctx, &expr)
                    } else {
                        self.emit_constant(&expr).addr
                    };
                    if types_match_with_promotion(decl_type, expr.ty) {
                        if value_fits(expr.value, decl_type) {
                            self.locals.insert(var, Local::new(addr, decl_type));
                        } else {
                            error!(self, ctx.ast.expr_source_position(left), "constant expression does not fit in {}", ctx.type_str(decl_type));
                        }
                    } else {
                        error!(self, ctx.ast.expr_source_position(left), "type mismatch between declaration ({}) and value ({})", ctx.type_str(decl_type), ctx.type_str(expr.ty))
                    }
                } else {
                    error!(self, ctx.ast.expr_source_position(left), "cannot declare {} as a variable", ctx.ast.expr(left));
                }
            }
            StmtData::Assign(left, right) => {
                let lv = self.expr(ctx, left);
                if lv.addr.can_assign {
                    let value = self.expr_with_destination(ctx, right, lv.ty, &lv.addr);
                    if !value.is_ok() {
                        error!(self, ctx.ast.expr_source_position(left), "type mismatch between destination ({}) and value ({})", ctx.type_str(lv.ty), ctx.type_str(value.ty))
                    }
                } else {
                    error!(self, ctx.ast.expr_source_position(left), "cannot assign to {}", ctx.ast.expr(left));
                }
            }
        }
        return_type
    }

    fn stmts(&mut self, ctx: &mut Compiler, stmts: StmtList) -> Option<Type> {
        let mark = self.locals.push_scope();
        let mut return_type = None;
        for stmt in stmts {
            if ctx.have_error() {
                break;
            }
            return_type = self.stmt(ctx, stmt).or(return_type);
        }
        self.locals.restore_scope(mark);
        return_type
    }

    fn func(&mut self, ctx: &mut Compiler, signature: Type, decl: Decl) -> Func {
        let func = ctx.ast.callable(decl);
        let body = func.body;
        let sig = ctx.types.info(signature);
        assert!(self.code.len() == 0);
        assert!(sig.kind == TypeKind::Func);
        assert!(func.params.len() == sig.items.len() - 1, "the last element of a func's type signature's items must be the return type. it is not optional");
        self.return_type = sig.items.last().map(|i| i.ty).unwrap();
        self.reg_counter = 1 - sig.items.len() as isize;
        self.reg_counter *= Self::REGISTER_SIZE as isize;
        let param_names = func.params.map(|item| ctx.ast.item(item).name);
        let param_types = sig.items.iter().map(|i| i.ty);
        for (name, ty) in Iterator::zip(param_names, param_types) {
            let reg = self.inc_reg();
            self.locals.insert(name, Local::new(Location::register(reg), ty));
        }
        let return_register = self.inc_reg();
        assert_eq!(return_register, 0);
        let returned = self.stmts(ctx, body);
        if matches!(returned, None) {
            let func = ctx.ast.callable(decl);
            error!(self, func.pos, "func {}: not all control paths return a value", ctx.str(func.name))
        }
        self.apply_patches();
        assert!(self.patches.len() == 0);
        assert!(self.context.break_to.is_none());
        assert!(self.context.continue_to.is_none());
        Func { _signature: signature, code: std::mem::take(&mut self.code) }
    }
}

pub fn eval_type(ctx: &mut Compiler, ty: TypeExpr) -> Type {
    // Seems really weird to have to make one of these to evaluate type exprs!!!
    // See also FatGen::constant_expr
    let mut fg = FatGen::new();
    let result = fg.type_expr(ctx, ty);
    let error = std::mem::take(&mut ctx.error);
    ctx.error = error.or(fg.error);
    result
}

fn position_to_line_column(str: &str, pos: usize) -> (usize, usize) {
    let start_of_line = str[..pos].bytes().rposition(|c| c == b'\n').unwrap_or(0);
    let line = str[..start_of_line].lines().count() + 1;
    let col = str[start_of_line..pos].chars().count() + 1;
    (line, col)
}

#[derive(Debug)]
pub struct IncompleteError {
    source_position: usize,
    msg: String,
}

#[derive(Debug)]
pub struct Error {
    source_position: usize,
    msg: String
}

impl Error {
    fn new(err: IncompleteError, source: &str) -> Error {
        let (line, col) = position_to_line_column(source, err.source_position);
        Error {
            source_position: err.source_position,
            msg: format!("{}:{}: error: {}", line, col, err.msg),
        }
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.msg.fmt(f)
    }
}

impl std::error::Error for Error {}

#[derive(Default)]
pub struct Func {
    _signature: Type,
    code: Vec<FatInstr>
}

pub struct Compiler {
    interns: Interns,
    types: Types,
    symbols: HashMap<Intern, Symbol>,
    globals: HashMap<Intern, Global>,
    funcs: HashMap<Intern, Func>,
    error: Option<IncompleteError>,
    ast: Ast,
    entry_stub: [FatInstr; 2]
}

impl Compiler {
    fn new() -> Compiler {
        let mut result = Compiler {
            interns: new_interns_with_keywords(),
            types: Types::new(),
            symbols: HashMap::new(),
            globals: HashMap::new(),
            funcs: HashMap::new(),
            error: None,
            ast: Ast::new(),
            entry_stub: [FatInstr::HALT; 2]
        };
        types::builtins(&mut result);
        result
    }

    fn intern(&mut self, str: &str) -> Intern {
        self.interns.put(str)
    }

    fn str(&self, intern: Intern) -> &str {
        self.interns.to_str(intern).expect("Missing intern")
    }

    fn type_str(&self, ty: Type) -> &str {
        // TODO: String representation of anonymous types
        self.str(self.types.info(ty).name)
    }

    fn error(&mut self, source_position: usize, msg: String) {
        if self.error.is_none() {
            self.error = Some(IncompleteError { source_position, msg });
        }
    }

    fn have_error(&self) -> bool {
        self.error.is_some()
    }

    fn check_and_clear_error(&mut self, source: &str) -> Result<()> {
        match self.error.take() {
            None => Ok(()),
            Some(err) => Err(Error::new(err, source).into())
        }
    }

    fn run(&self) -> Value {
        let mut code = &self.entry_stub[..];
        let mut sp: usize = 0;
        let mut ip: usize = 0;
        let mut call_stack = vec![(code, ip, sp, 0)];

        // The VM allows taking (real!) pointers to addresses on the VM stack. These are
        // immediately invalidated by taking a _reference_ to the stack. Solution: never
        // take a reference to the stack (even one immediately turned into a pointer).

        // This just ensures the compiler knows the correct invariants involved, it
        // doesn't make the vm sound

        // MIRI shits the bed if this is on the stack
        let layout = std::alloc::Layout::array::<RegValue>(8192).unwrap();
        let stack: *mut [u8] = unsafe {
            let ptr = std::alloc::alloc(layout);
            std::slice::from_raw_parts_mut(ptr, layout.size())
        };

        // Access the stack by byte
        macro_rules! stack {
            ($addr:expr) => { (*stack)[$addr] }
        }

        // Access the stack by register. Register accesses must be aligned correctly (asserted below)
        macro_rules! reg {
            ($addr:expr) => { *(std::ptr::addr_of_mut!(stack![$addr]) as *mut RegValue) }
        }

        loop {
            let instr = &code[ip];
            let dest = sp.wrapping_add(sign_extend(instr.dest));
            let left = sp.wrapping_add(sign_extend(instr.left));
            let right = sp.wrapping_add(sign_extend(instr.right));

            assert_implies!(requires_register_destination(instr.op), (dest & (FatGen::REGISTER_SIZE - 1)) == 0);

            unsafe {
                 match instr.op {
                    Op::Halt       => { break; }
                    Op::Noop       => {}
                    Op::Immediate  => { reg![dest] = RegValue::from((instr.left, instr.right)); }
                    Op::IntNeg     => { reg![dest].sint = -reg![left].sint; }
                    Op::IntAdd     => { reg![dest].wint = reg![left].wint + reg![right].wint; }
                    Op::IntSub     => { reg![dest].wint = reg![left].wint - reg![right].wint; }
                    Op::IntMul     => { reg![dest].wint = reg![left].wint * reg![right].wint; }
                    Op::IntDiv     => { reg![dest].wint = reg![left].wint / reg![right].wint; }
                    Op::IntMod     => { reg![dest].wint = reg![left].wint % reg![right].wint; }
                    Op::IntLt      => { reg![dest].int = (reg![left].int < reg![right].int) as usize; }
                    Op::IntGt      => { reg![dest].int = (reg![left].int > reg![right].int) as usize; }
                    Op::IntEq      => { reg![dest].int = (reg![left].int == reg![right].int) as usize; }
                    Op::IntNEq     => { reg![dest].int = (reg![left].int != reg![right].int) as usize; }
                    Op::IntLtEq    => { reg![dest].int = (reg![left].int <= reg![right].int) as usize; }
                    Op::IntGtEq    => { reg![dest].int = (reg![left].int >= reg![right].int) as usize; }
                    Op::BitNeg     => { reg![dest].int = !reg![left].int }
                    Op::BitAnd     => { reg![dest].int = reg![left].int & reg![right].int; }
                    Op::BitOr      => { reg![dest].int = reg![left].int | reg![right].int; }
                    Op::BitXor     => { reg![dest].int = reg![left].int ^ reg![right].int; }

                    Op::LShift     => { reg![dest].int = reg![left].int << reg![right].int; }
                    Op::RShift     => { reg![dest].int = reg![left].int >> reg![right].int; }

                    Op::Not        => { reg![dest].int = (reg![left].int == 0) as usize; }
                    Op::CmpZero    => { reg![dest].int = (reg![left].int != 0) as usize; }
                    Op::LogicOr    => { reg![dest].int = (reg![left].b8.0 || reg![right].b8.0) as usize; }
                    Op::LogicAnd   => { reg![dest].int = (reg![left].b8.0 && reg![right].b8.0) as usize; }

                    Op::F32Neg     => { reg![dest].float32.0 = -reg![left].float32.0;
                                        reg![dest].float32.1 = 0.0; }
                    Op::F32Add     => { reg![dest].float32.0 = reg![left].float32.0 + reg![right].float32.0;
                                        reg![dest].float32.1 = 0.0; }
                    Op::F32Sub     => { reg![dest].float32.0 = reg![left].float32.0 - reg![right].float32.0;
                                        reg![dest].float32.1 = 0.0; }
                    Op::F32Mul     => { reg![dest].float32.0 = reg![left].float32.0 * reg![right].float32.0;
                                        reg![dest].float32.1 = 0.0; }
                    Op::F32Div     => { reg![dest].float32.0 = reg![left].float32.0 / reg![right].float32.0;
                                        reg![dest].float32.1 = 0.0; }
                    Op::F32Lt      => { reg![dest].int = (reg![left].float32.0 < reg![right].float32.0) as usize; }
                    Op::F32Gt      => { reg![dest].int = (reg![left].float32.0 > reg![right].float32.0) as usize; }
                    Op::F32Eq      => { reg![dest].int = (reg![left].float32.0 == reg![right].float32.0) as usize; }
                    Op::F32NEq     => { reg![dest].int = (reg![left].float32.0 != reg![right].float32.0) as usize; }
                    Op::F32LtEq    => { reg![dest].int = (reg![left].float32.0 <= reg![right].float32.0) as usize; }
                    Op::F32GtEq    => { reg![dest].int = (reg![left].float32.0 >= reg![right].float32.0) as usize; }

                    Op::F64Neg     => { reg![dest].float64 = -reg![left].float64; }
                    Op::F64Add     => { reg![dest].float64 = reg![left].float64 + reg![right].float64; }
                    Op::F64Sub     => { reg![dest].float64 = reg![left].float64 - reg![right].float64; }
                    Op::F64Mul     => { reg![dest].float64 = reg![left].float64 * reg![right].float64; }
                    Op::F64Div     => { reg![dest].float64 = reg![left].float64 / reg![right].float64; }
                    Op::F64Lt      => { reg![dest].int = (reg![left].float64 < reg![right].float64) as usize; }
                    Op::F64Gt      => { reg![dest].int = (reg![left].float64 > reg![right].float64) as usize; }
                    Op::F64Eq      => { reg![dest].int = (reg![left].float64 == reg![right].float64) as usize; }
                    Op::F64NEq     => { reg![dest].int = (reg![left].float64 != reg![right].float64) as usize; }
                    Op::F64LtEq    => { reg![dest].int = (reg![left].float64 <= reg![right].float64) as usize; }
                    Op::F64GtEq    => { reg![dest].int = (reg![left].float64 >= reg![right].float64) as usize; }

                    // dest and left are aligned to register size
                    Op::MoveLower8               => { reg![dest].int       = reg![left].int8.0    as usize; }
                    Op::MoveLower16              => { reg![dest].int       = reg![left].int16.0   as usize; }
                    Op::MoveLower32              => { reg![dest].int       = reg![left].int32.0   as usize; }
                    Op::MoveAndSignExtendLower8  => { reg![dest].int       = reg![left].sint8.0   as usize; }
                    Op::MoveAndSignExtendLower16 => { reg![dest].int       = reg![left].sint16.0  as usize; }
                    Op::MoveAndSignExtendLower32 => { reg![dest].int       = reg![left].sint32.0  as usize; }
                    Op::IntToFloat32             => { reg![dest].float32.0 = reg![left].sint      as f32;
                                                      reg![dest].float32.1 = 0.0; }
                    Op::IntToFloat64             => { reg![dest].float64   = reg![left].sint      as f64; }
                    Op::Float32ToInt             => { reg![dest].int       = reg![left].float32.0 as usize; }
                    Op::Float32To64              => { reg![dest].float64   = reg![left].float32.0 as f64; }
                    Op::Float64ToInt             => { reg![dest].int       = reg![left].float64   as usize; }
                    Op::Float64To32              => { reg![dest].float32.0 = reg![left].float64   as f32;
                                                      reg![dest].float32.1 = 0.0; }

                    // dest is aligned, left may not be. The stores are redundant with copy, but convenient; loads ensure high bits of registers are zeroed
                    Op::StackRelativeStore8  => { stack![dest] = stack![left]; }
                    Op::StackRelativeStore16 => { (*stack).copy_within(left..left+2, dest) }
                    Op::StackRelativeStore32 => { (*stack).copy_within(left..left+4, dest) }

                    Op::StackRelativeLoad8  => { reg![dest].int = stack![left] as usize; }
                    Op::StackRelativeLoad16 => { (*stack).copy_within(left..left+2, dest); reg![dest].int = reg![dest].int16.0 as usize }
                    Op::StackRelativeLoad32 => { (*stack).copy_within(left..left+4, dest); reg![dest].int = reg![dest].int32.0 as usize; }
                    Op::StackRelativeLoadAndSignExtend8  => { reg![dest].sint = stack![left] as isize; }
                    Op::StackRelativeLoadAndSignExtend16 => { (*stack).copy_within(left..left+2, dest); reg![dest].sint = reg![dest].sint16.0 as isize }
                    Op::StackRelativeLoadAndSignExtend32 => { (*stack).copy_within(left..left+4, dest); reg![dest].sint = reg![dest].sint32.0 as isize }
                    Op::StackRelativeLoadBool => { reg![dest].int = (stack![left] != 0) as usize }

                    Op::StackRelativeCopy =>  { (*stack).copy_within(left..left+(instr.right as usize), dest) }
                    Op::StackRelativeZero =>  { for i in 0..instr.left as usize { stack![dest+i] = 0 }}

                    Op::Move          => { reg![dest] = reg![left]; }
                    Op::Jump          => { ip = ip.wrapping_add(sign_extend(instr.left)); }
                    Op::JumpIfZero    => { if reg![dest].int == 0 { ip = ip.wrapping_add(sign_extend(instr.left)); } }
                    Op::JumpIfNotZero => { if reg![dest].int != 0 { ip = ip.wrapping_add(sign_extend(instr.left)); } }
                    Op::Return        => {
                        let ret = call_stack.pop().expect("Bad bytecode");
                        code = ret.0;
                        ip = ret.1;
                        sp = ret.2;
                        reg![ret.3] = reg![left];
                    },
                    Op::Call          => {
                        call_stack.push((code, ip, sp, dest));
                        let f = self.funcs.get(&Intern(instr.left as u32)).expect("Bad bytecode");
                        code = &f.code[..];
                        ip = !0;
                        sp = dest;
                    },
                    Op::CallIndirect  => { todo!() },

                    Op::Store8  => { *(reg![dest].int as *mut u8)  =   stack![left]; },
                    Op::Store16 => { *(reg![dest].int as *mut u16) = *(stack![left..].as_ptr() as *const u16); },
                    Op::Store32 => { *(reg![dest].int as *mut u32) = *(stack![left..].as_ptr() as *const u32); },
                    Op::Store64 => { *(reg![dest].int as *mut u64) = *(stack![left..].as_ptr() as *const u64); },
                    Op::Load8   => { reg![dest].int = *(reg![left].int as *const u8)  as usize; },
                    Op::Load16  => { reg![dest].int = *(reg![left].int as *const u16) as usize; },
                    Op::Load32  => { reg![dest].int = *(reg![left].int as *const u32) as usize; },
                    Op::Load64  => { reg![dest].int = *(reg![left].int as *const u64) as usize; },
                    Op::LoadAndSignExtend8  => { reg![dest].sint = *(reg![left].int as *const i8)  as isize; },
                    Op::LoadAndSignExtend16 => { reg![dest].sint = *(reg![left].int as *const i16) as isize; },
                    Op::LoadAndSignExtend32 => { reg![dest].sint = *(reg![left].int as *const i32) as isize; },
                    Op::LoadBool => { reg![dest].b8.0 = *(reg![left].int as *const u8) != 0; },
                    Op::Copy => { std::ptr::copy(reg![left].int as *const u8, reg![dest].int as *mut u8, reg![right].int); },
                    Op::Zero => { for b in std::slice::from_raw_parts_mut(reg![dest].int as *mut u8, reg![left].int) { *b = 0 }},
                    Op::LoadStackAddress => { reg![dest].int = std::ptr::addr_of_mut!(stack![left]) as usize },
                }
            }
            ip = ip.wrapping_add(1);
        }
        let result = unsafe { Value::from(reg![0 as usize], Type::Int) };
        unsafe { std::alloc::dealloc((*stack).as_mut_ptr(), layout); }
        result
    }
}

fn compile(str: &str) -> Result<Compiler> {
    let mut c = Compiler::new();
    c.ast = parse::parse(&mut c, str);

    c.check_and_clear_error(str)?;
    resolve_all(&mut c);

    let main = c.intern("main");
    if let Some(&main) = c.globals.get(&main) {
        let info = c.types.info(main.ty);
        if !matches!(info.arguments(), Some(&[])) || !matches!(info.return_type(), Some(Type::Int)) {
            let pos = c.ast.decl(main.decl).pos();
            error!(c, pos, "main must be a function that takes zero arguments and returns int");
        }
    }

    c.check_and_clear_error(str)?;

    let mut gen = FatGen::new();
    for decl in c.ast.decl_list() {
        if c.ast.is_callable(decl) {
            let name = c.ast.decl(decl).name();
            let sig = c.globals.get(&name).unwrap().ty;
            let func = gen.func(&mut c, sig, decl);
            c.funcs.insert(name, func);
        }
    }

    // :P
    c.error = gen.error.take();
    c.check_and_clear_error(str)?;

    if let Some(_func) = c.funcs.get(&main) {
        c.entry_stub[0] = FatInstr { op: Op::Call, dest: 0, left: main.0, right: 0 };
    }
    Ok(c)
}

fn compile_and_run(str: &str) -> Result<Value> {
    let c = compile(str)?;
    Ok(c.run())
}

fn sign_extend(val: u32) -> usize {
    val as i32 as usize
}

fn eval_expr(str: &str) -> Result<Value> {
    compile_and_run(&format!("func main(): int {{ return {}; }}", str))
}

fn repl() {
    use std::io;
    let mut line = String::new();
    println!("lang's bad repl");
    loop {
        match io::stdin().read_line(&mut line) {
            Ok(_) => match eval_expr(&line) {
               Ok(value) => println!("{}", value),
               Err(err) => println!("{}", err)
            },
            Err(_) => break
        }

        line.clear();
    }
}

fn main() {
    // arr();
    repl();
}

#[test]
fn expr() {
    macro_rules! eval {
        ($x: expr) => {
            let result = eval_expr(stringify!($x)).unwrap();
            assert_eq!(result, Value::Int($x), stringify!($x))
        };
        ($x: literal, $val: expr) => {
            let result = eval_expr($x).unwrap();
            assert_eq!(result, Value::Int($val), stringify!($x))
        };
    }
    eval!(1);
    eval!(1+1);
    eval!(1-1);
    eval!(-1-1);
    eval!(-1+1);
    eval!(-1+-1);
    eval!(-1);
    eval!(--1);
    eval!(---1);
    eval!(1 << 2);
    eval!(16 >> 3);
    eval!(24 / 4);
    eval!(3 * 3);
    eval!(2*3+2);
    eval!(2+3*2);
    eval!(1 * -9);
    eval!(2 | 4);
    eval!(9 & 7);
    eval!(9 & 7 | 3);
    eval!(84 ^ 23);
    eval!(42 % 23);
    eval!(1+2+3*4/4+(2*-4+2)*3-1);
    // << and >> have same associativity as * and /
    eval!("1 + 2 << 3 + 4",      1 + (2 << 3) + 4);
    eval!("1 * 2 << 3 * 4",      (1 * 2 << 3) * 4);
    eval!("(3 + 4) >> (1 + 2)",  3 + 4 >> 1 + 2);
    eval!("(3 * 4) >> (1 * 2)",  3 * 4 >> 1 * 2);
    eval!("!165: int", 0);
    eval!("!0: int", 1);
    eval!("~23", !23);
    eval!("(!!(2*3) || !!(3-3)) : int", 1);
    eval!("(!!(2*3) && !!(3-3)) : int", 0);
    eval!("256: i8", 0);
    eval!("-1: u8: int", 255);
    eval!("-1: i8: int", -1);
    eval!("1 == 1 ? 2 :: 3", 2);
    eval!("0 == 1 ? 2 :: 3", 3);
    eval!("(43 * 43) > 0 ? (1+1+1) :: (2 << 3)", 3);
    eval!("!!0 ? (!!1 ? 2 :: 3) :: (!!4 ? 5 :: 6)", 5);
    eval!("!!0 ?  !!1 ? 2 :: 3  ::  !!4 ? 5 :: 6",  5);
    eval!("!!1 ? (!!2 ? 3 :: 4) :: (!!5 ? 6 :: 7)", 3);
    eval!("!!1 ?  !!2 ? 3 :: 4  ::  !!5 ? 6 :: 7",  3);
}

#[test]
fn stmt() {
    let ok = [
        "a := 1; do { if (a > 2) { a = (a & 1) == 1 ? a * 2 :: a + 1; } else { a = a + 1; } } while (a < 10);",
        "a := 1.0; a = a + 1.0d:f32;",
        "a := !165: int;",
        "(!!(2*3)) || !!(3-3);",
        "a := (0: bool);",
        "a := 0; a := 1;",
        "if (1:bool) {}",
        "while (0:bool) {}",
        "do {} while (0:bool);",
    ];
    let err = [
        "a := -(0: bool);",
        "();"
    ];
    for str in ok.iter() {
        compile_and_run(&format!("func main(): int {{ {{ {} }} return 0; }}", str)).map_err(|e| { println!("input: \"{}\"", str); e }).unwrap();
    }
    for str in err.iter() {
        // Not checking it's the right error yet--maybe later
        compile_and_run(&format!("func main(): int {{ {{ {} }} return 0; }}", str)).map(|e| { println!("input: \"{}\"", str); e }).unwrap_err();
    }
}

#[test]
fn decls() {
    let ok = [
        "func add(a: int, b: int): int { return a + b; }",
        "func add'(a: int, b: int): int { return a + b; }",
        "struct V2 { x: f32, y: f32 }",
        "struct V1 { x: f32 } struct V2 { x: V1, y: V1 }",
    ];
    let err = [
        "func func(a: int, b: int): int { return a + b; }",
        "func dup_func(a: int, b: int): int { return a + b; } func dup_func(a: int, b: int): int { return a + b; }",
        "func dup_param(a: int, a: int): int { return a + a; }",
        "struct empty {}",
        "struct dup_struct { x: f32 } struct dup_struct { x: f32 }",
        "struct dup_field { x: f32, x: f32 }",
        "struct circular { x: circular }",
        "struct circular1 { x: circular2 } struct circular2 { x: circular1 }"
    ];
    for str in ok.iter() {
        compile_and_run(str).map_err(|e| { println!("input: \"{}\"", str); e }).unwrap();
    }
    for str in err.iter() {
        // Not checking it's the right error yet--maybe later
        compile_and_run(str).map(|v| { println!("input: \"{}\"", str); v }).unwrap_err();
    }
}

#[test]
fn control_paths() {
    let ok = [
        "func control_paths(a: int, b: int, c: bool): int { if (c) { return a + b; } else { return a - b; } }",
        "func control_paths(a: int, b: int, c: bool): int { while (c) { return a; } }",
        "func control_paths(a: int, b: int, c: bool): int { return a; while (c) { if (c) { return a; } } }",
    ];
    let err = [
        "func control_paths(a: int) {}",
        "func control_paths(a: int, b: int, c: bool): int { if (c) { return a + b; } }",
        "func control_paths(a: int, b: int, c: bool): int { if (c) { return a + b; } else { } }",
        "func control_paths(a: int, b: int, c: bool): int { if (c) {} else { return a - b; } }",
        "func control_paths(a: int, b: int, c: bool): int { while (c) { if (c) { return a; } } }",
    ];
    for str in ok.iter() {
        compile_and_run(str).map_err(|e| { println!("input: \"{}\"", str); e }).unwrap();
    }
    for str in err.iter() {
        // Not checking it's the right error yet--maybe later
        compile_and_run(str).map(|v| { println!("input: \"{}\"", str); v }).unwrap_err();
    }
}

#[test]
fn int_promotion() {
    let ok = [
        "func pro(): int { a: u8 = 1; b: u8 = 2; return a + b; }",
        "func pro(): int { a: u8 = 1; b: u16 = 2; return a + b; }",
        "func pro(): int { a: u8 = 1; b: i16 = 2; return a + b; }",
        "func pro(): int { a: u8 = 255:i16:u8; return a; }",
        "func pro(): int { a: u8 = 256:i16:u8; return a; }",
    ];
    let err = [
        "func pro(): u8 { a: u8 = 1; b: u8 = 2; return a + b; }",
        "func pro(): int { a: u16 = 256:i16:u8; return a; }",
    ];
    for str in ok.iter() {
        compile_and_run(str).map_err(|e| { println!("input: \"{}\"", str); e }).unwrap();
    }
    for str in err.iter() {
        // Not checking it's the right error yet--maybe later
        compile_and_run(str).map(|v| { println!("input: \"{}\"", str); v }).unwrap_err();
    }
}

#[test]
fn fits() {
    let err = [
        "func fits(): int { a: u8 = -1; return a; }",
        "func fits(): int { a: i8 = 128; return a; }",
        "func fits(): int { a: u8 = 256; return a; }",
        "func fits(): int { a: i16 = 32768; return a; }",
        "func fits(): int { a: u16 = 65536; return a; }",
        "func fits(): int { a: i32 = 2147483648; return a; }",
        "func fits(): int { a: u32 = 4294967296; return a; }",
        "func fits(): int { a := 18446744073709551616; return a; }",
    ];
    for str in err.iter() {
        // Not checking it's the right error yet--maybe later
        compile_and_run(str).map(|v| { println!("input: \"{}\"", str); v }).unwrap_err();
    }
}

#[test]
fn func() {
    let ok = [
(r#"
func madd(a: int, b: int, c: int): int {
    return a * b + c;
}

func main(): int {
    return madd(1, 2, 3);
}
"#, Value::Int(5)),
(r#"
func fib(n: int): int {
    if (n <= 1) {
        return n;
    }

    return fib(n - 2) + fib(n - 1);
}

func main(): int {
    return fib(6);
}
"#, Value::Int(8)),
(r#"
func even(n: int): bool {
    return n == 0 ? 1:bool :: odd(n - 1);
}

func odd(n: int): bool {
    return n == 0 ? 0:bool :: even(n - 1);
}

func main(): int {
    return even(10):int;
}
"#, Value::Int(1))
    ];
    for test in ok.iter() {
        let str = test.0;
        let expect = test.1;
        compile_and_run(str)
            .and_then(|ret| if ret == expect { Ok(ret) } else { Err(format!("expected {:?}, got {:?}", expect, ret).into())})
            .map_err(|e| { println!("input: \"{}\"", str); e })
            .unwrap();
    }
}

#[test]
fn control_structures() {
    let ok = [
(r#"
func main(): int {
    a := 0;
    for (i := 0; i < 10; i = i + 1) {
        a = a + 1;
    }
    return a;
}
"#, Value::Int(10)),
(r#"
func main(): int {
    a := 0;
    for (; a < 10; a = a + 2) {}
    return a;
}
"#, Value::Int(10)),
(r#"
func main(): int {
    a := 0;
    for (;;) {
        a = a + 1;
        if (a == 10) {
            break;
        }
        // continue;
    }
    return a;
}
"#, Value::Int(10)),
(r#"
//
func main(): int {
    a := 0;
    b := 0;
    for (; a < 10; a = a + 1) {
        if (a > 5) {
            continue;
        }
        b = a;
    }
    return a + b;
}
"#, Value::Int(15)),
(r#"
func main(): int {
    a := 10;
    switch (a) {
        1 {}
        2, 3 {
            return 1;
        }
        10 {
            return 100;
        }
    }
    return 0;
}
"#, Value::Int(100)),
(r#"
func main(): int {
    a := 3.0;
    b := 0;
    switch (a) {
        1.0 {
            b = 0;
        }
        else {
            b = 4;
        }
        2.0, 3.0 {
            b = 1;
        }
        10.0 {
            b = 100;
        }
    }
    return b;
}
"#, Value::Int(1)),
(r#"
func main(): int {
    a := 3.0;
    b := 0;
    switch (a) {
        1.0 {
            b = 2;
        }
        else {
            b = 4;
        }
        2.0 {
            b = 1;
        }
        10.0 {
            b = 100;
        }
    }
    return b;
}
"#, Value::Int(4)),
(r#"
func main(): int {
    a := 1;
    do {
        if (a > 2) {
            a = (a & 1) == 1 ? a * 2 :: a + 1;
        } else {
            a = a + 1;
        }
    } while (a < 10);
    return a;
}"#, Value::Int(14))
    ];
    for test in ok.iter() {
        let str = test.0;
        let expect = test.1;
        compile_and_run(str)
            .and_then(|ret| if ret == expect { Ok(ret) } else { Err(format!("expected {:?}, got {:?}", expect, ret).into())})
            .map_err(|e| { println!("input: \"{}\"", str); e })
            .unwrap();
    }
}

#[test]
fn structs() {
    let ok = [
(r#"
struct i32x4 {
    a: i32,
    b: i32,
    c: i32,
    d: i32,
}
func main(): int {
    a := {-1,-1,-1,-1}:i32x4;
    return a.d:int;
}
"#, Value::Int(-1)),
(r#"
struct RGBA {
    r: i8,
    g: i8,
    b: i8,
    a: i8,
}

func main(): int {
    c := { r = 1, g = 2, b = 3, a = 4 }:RGBA;
    d := { g = 5, r = 6, a = 7, b = 8, }:RGBA;
    e: RGBA = { 9, 10, 11, 12 };
    f: RGBA = { 13, 14, 15 };
    return (c.r << 24) + (d.g << 16) + (e.b << 8) + f.a;
}
"#, Value::Int(0x01050B00)),
(r#"
struct Outer {
    inner: Inner
}

struct Inner {
    ignored: i32,
    value: i32
}

func main(): int {
    a: Outer = { inner = { value = -1234 }};
    b: Outer = { inner = { value = -1234 }:Inner };
    c := { inner = { value = a.inner.value }:Inner }:Outer;
    d: Outer = { inner = { value = b.inner.value }:Inner }:Outer;
    e := { inner = d.inner }:Outer;
    f := { inner = d.inner:Inner }:Outer;
    g := {}:Outer;
    g.inner = f.inner;
    h: Outer = {};
    h.inner.value = f.inner.value;
    i: Outer = {}:Outer;
    i.inner = { value = d.inner.value };
    z: Outer = {};
    if ((a.inner.value == b.inner.value)
     && (b.inner.value == c.inner.value)
     && (c.inner.value == d.inner.value)
     && (d.inner.value == e.inner.value)
     && (e.inner.value == f.inner.value)
     && (f.inner.value == g.inner.value)
     && (g.inner.value == h.inner.value)
     && (h.inner.value != z.inner.value)) {
        return a.inner.value;
    }
    return 0;
}
"#, Value::Int(-1234)),
(r#"
struct V2 {
    x: i32,
    y: i32
}

func fn(): i32 {
    a := { x = 2 }:V2;
    b := { x = 3 }:V2;
    a = b;
    return b.x;
}

func main(): int {
    a := 0;
    b := 0;
    return fn():int;
}
"#, Value::Int(3)),
(r#"
struct V2 {
    x: i32,
    y: i32,
    z: i32,
    w: i32,
}

func main(): int {
    a := ({ x = -1 }:V2).x;
    b := (a == -1) ? { w = 3 }:V2 :: { w = 4 }:V2;
    return b.w:int;
}
"#, Value::Int(3)),
];
    let err = [r#"
func main(): int {
    v := {0}:int;
    return v;
}
"#, r#"
struct V2 { x: int, y: int }
func main(): int {
    v := {}:V2;
    v.a = 0;
    return v.x:int;
}
"#, r#"
struct V2 { x: int, y: int }
func main(): int {
    v := { a = 0 }:V2;
    return v.a;
},
struct V2 { x: int, y: int }
func main(): int {
    v := { x = 0, 1 }:V2;
    return v.a;
}
"#];
    for test in ok.iter() {
        let str = test.0;
        let expect = test.1;
        compile_and_run(str)
            .and_then(|ret| if ret == expect { Ok(ret) } else { Err(format!("expected {:?}, got {:?}", expect, ret).into())})
            .map_err(|e| { println!("input: \"{}\"", str); e })
            .unwrap();
    }
    for str in err.iter() {
        compile_and_run(str).map(|e| { println!("input: \"{}\"", str); e }).unwrap_err();
    }
}


#[test]
fn arr() {
    let ok = [
(r#"
struct asdf {
    arr: arr int [4],
}
func main(): int {
    asdf := {}:asdf;
    arr: arr int [4:i8] = {};
    asdf.arr[0] = 1;
    asdf.arr[1] = 2;
    asdf.arr[2] = 3;
    asdf.arr[3] = 4;
    arr[0] = 10;
    arr[1] = 20;
    arr[2] = 30;
    arr[3] = 40;
    more := { 100, 200, 300, 400, 500 }:arr int [5];
    return asdf.arr[256:i8] + arr[1] + more[2];
}
"#, Value::Int(321)),
(r#"
func main(): int {
    arr := {}:arr u8 [8];
    for (i := 0; i != 8; i = i + 1) {
        arr[i] = 1;
    }
    acc := 0;
    for (i := 0; i < 8; i = i + 1) {
        acc = acc + arr[i];
    }
    return acc;
}
"#, Value::Int(8)),
(r#"
struct V4 { padding: i16, c: arr i32 [4] }
struct M4 { padding: i16, r: arr V4 [4] }
func main(): int {
    a: M4 = {
        r = {
            { c = { 1, 2, 3, 4}},
            { c = { 5, 6, 7, 8}},
            { c = { 9,10,11,12}},
            { c = {13,14,15,16}}
        }
    };
    i := 3;
    j := 2;
    return a.r[i].c[j];
}
"#, Value::Int(15)),
];
    let err = ["r#
func main(): int {
    a := 2;
    arr: arr int [a] = {};
    return arr[0];
}
#",r#"
func main(): int {
    arr: arr int [1] = { 0 = 1 };
    return arr[0];
}
"#, r#"
func main(): int {
    arr: arr int [4.0] = {};
    return arr[0];
}
"#];
    for test in ok.iter() {
        let str = test.0;
        let expect = test.1;
        compile_and_run(str)
            .and_then(|ret| if ret == expect { Ok(ret) } else { Err(format!("expected {:?}, got {:?}", expect, ret).into())})
            .map_err(|e| { println!("input: \"{}\"", str); e })
            .unwrap();
    }
    for str in err.iter() {
        compile_and_run(str).map(|e| { println!("input: \"{}\"", str); e }).unwrap_err();
    }
}

#[test]
fn ptr() {
    let ok = [
(r#"
func main(): int {
    a := 1;
    b := &a;
    *b = 2;
    return *b;
}
"#, Value::Int(2)),
(r#"
func main(): int {
    a := 1;
    *&a = 2;
    b := &a:int;
    return a;
}
"#, Value::Int(2)),
(r#"
struct V2 { x: i32, y: i32 }
func main(): int {
    aa := {1, 2}:V2;
    bb := aa;
    a := &aa;
    b := &bb;
    c := &{1, 2}:V2;
    *a = {3, 4}:V2;
    *b = *a;
    c.x = b.x;
    c.y = b.y;
    if (a.x == aa.x && b.x == bb.x && a.x == b.x && b.x == c.x
     && a.y == aa.y && b.y == bb.y && a.y == b.y && b.y == c.y) {
        return aa.x;
    }
    return 0;
}
"#, Value::Int(3)),
(r#"
func main(): int {
    arr := {}:arr u8 [8];
    for (p := &arr[0]; p != &arr[8]; p = p + 1) {
        *p = 1;
    }
    arrp := &arr[0];
    acc := 0;
    for (i := 0; i < 8; i = i + 1) {
        acc = acc + arrp[i];
    }
    return acc;
}
"#, Value::Int(8)),
];
    let err = [r#""
func main(): int {
    a := 1;
    return *a;
}
"#,r#"
func main(): int {
    a := &-1:u8;
    return *a;
}
"#,r#"
func main(): int {
    a := -1;
    b := &a:f32;
    return *b;
}
"#];
    for test in ok.iter() {
        let str = test.0;
        let expect = test.1;
        compile_and_run(str)
            .and_then(|ret| if ret == expect { Ok(ret) } else { Err(format!("expected {:?}, got {:?}", expect, ret).into())})
            .map_err(|e| { println!("input: \"{}\"", str); e })
            .unwrap();
    }
    for str in err.iter() {
        compile_and_run(str).map(|e| { println!("input: \"{}\"", str); e }).unwrap_err();
    }
}
