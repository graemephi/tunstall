#[cfg(not(target_pointer_width = "64"))]
compile_error!("64 bit is assumed");


use std::convert::TryFrom;
use std::collections::hash_map::{HashMap, DefaultHasher};
use std::hash::{Hash, Hasher};
use std::fmt::Display;

mod parse;
mod ast;
mod sym;
mod types;
mod smallvec;

use ast::*;
use smallvec::*;
use sym::*;
use types::*;
use parse::Keytype;

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

    pub fn get(&self, str: &str) -> Option<Intern> {
        self.ids.get(str).copied()
    }

    pub fn to_str(&self, id: Intern) -> Option<&str> {
        self.interns.get(id.0 as usize).copied()
    }
}

// If the compiler were a library we'd need this. As an application this just
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
    let mut i = 1;
    for k in (1..).scan((), |(), v| parse::Keyword::from_u32(v)) {
        result.put(k.to_str());
        i += 1;
    }
    for k in (i..).scan((), |(), v| parse::Keytype::from_u32(v)) {
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
    for k in (i.0..).scan((), |(), v| parse::Keytype::from_u32(v)) {
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
    LoadStackAddress,
    Panic
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
    dest: i32,
    left: i32,
    right: i32,
}

impl FatInstr {
    const HALT: FatInstr = FatInstr { op: Op::Halt, dest: 0, left: 0, right: 0 };
    const PANIC: FatInstr = FatInstr { op: Op::Panic, dest: 0, left: 0, right: 0 };

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

fn ident(value: Option<RegValue>) -> Intern {
    value.map(|v| unsafe { Intern(v.int32.0) }).unwrap_or(Intern(0))
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

impl From<(i32, i32)> for RegValue {
    fn from(value: (i32, i32)) -> Self {
        RegValue { sint32: value }
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
        (TokenKind::Not,    _       ) => Some((Op::Not,    Type::Bool)),
        (TokenKind::Sub,    Type::F32) => Some((Op::F32Neg, Type::F32)),
        (TokenKind::Sub,    Type::F64) => Some((Op::F64Neg, Type::F64)),
        _ => None
    }
}

fn binary_op(op: parse::TokenKind, left: Type, right: Type) -> Option<(Op, Type)> {
    use parse::TokenKind;
    let pointers = left.is_pointer() && right.is_pointer();
    let left = integer_promote(left);
    let right = integer_promote(right);
    if left != right && !pointers {
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

        (TokenKind::GtEq, _)  if pointers => Some((Op::IntGtEq, Type::Bool)),
        (TokenKind::Lt,   _)  if pointers => Some((Op::IntLt,   Type::Bool)),
        (TokenKind::Gt,   _)  if pointers => Some((Op::IntGt,   Type::Bool)),
        (TokenKind::Eq,   _)  if pointers => Some((Op::IntEq,   Type::Bool)),
        (TokenKind::NEq,  _)  if pointers => Some((Op::IntNEq,  Type::Bool)),
        (TokenKind::LtEq, _)  if pointers => Some((Op::IntLtEq, Type::Bool)),

        _ => None
    }
}

fn is_address_computation(op: parse::TokenKind, left: Type, right: Type) -> bool {
    use parse::TokenKind::*;
    matches!(op, Add|Sub) && ((left.is_pointer() && right.is_integer()) || (left.is_integer() && right.is_pointer()))
}

fn is_offset_computation(ctx: &Compiler, op: parse::TokenKind, left: Type, right: Type) -> bool {
    use parse::TokenKind::*;
    // See... annoying
    matches!(op, Sub) && left.is_pointer() && right.is_pointer() && ctx.types.annoying_deep_eq(left, right)
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
            _ if to.is_pointer() => Op::Noop,
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
        Type::Bool                              => Some(Op::StackRelativeLoadBool),
        Type::I8                                => Some(Op::StackRelativeLoadAndSignExtend8),
        Type::I16                               => Some(Op::StackRelativeLoadAndSignExtend16),
        Type::I32                               => Some(Op::StackRelativeLoadAndSignExtend32),
        Type::U8                                => Some(Op::StackRelativeLoad8),
        Type::U16                               => Some(Op::StackRelativeLoad16),
        Type::U32|Type::F32                     => Some(Op::StackRelativeLoad32),
        Type::Int|Type::I64|Type::U64|Type::F64 => Some(Op::Move),
        _ if ty.is_pointer()                    => Some(Op::Move),
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
        Type::Bool                              => Some(Op::LoadBool),
        Type::I8                                => Some(Op::LoadAndSignExtend8),
        Type::I16                               => Some(Op::LoadAndSignExtend16),
        Type::I32                               => Some(Op::LoadAndSignExtend32),
        Type::U8                                => Some(Op::Load8),
        Type::U16                               => Some(Op::Load16),
        Type::U32|Type::F32                     => Some(Op::Load32),
        Type::Int|Type::I64|Type::U64|Type::F64 => Some(Op::Load64),
        _ if ty.is_pointer()                    => Some(Op::Load64),
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
        loc.is_mutable = true;
        loc.is_place = true;
        Local { loc, ty }
    }

    fn argument(loc: Location, ty: Type) -> Local {
        let mut loc = loc;
        debug_assert!(loc.is_mutable == false);
        loc.is_mutable = false;
        loc.is_place = true;
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

fn expr_integer_compatible_with_destination(value: &ExprResult, dest_ty: Type, dest: &Location) -> bool {
    if value.ty.is_integer() {
        if dest_ty == Type::Int {
            return true;
        }
        if value.value.is_some() && value_fits(value.value, dest_ty) {
            return true;
        }
    }
    if let LocationKind::Register = dest.kind {
        return value.ty.is_integer() && dest_ty.is_integer();
    }
    false
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum LocationKind {
    None,
    Control,
    Return,
    Register,
    Stack,
    Based,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Location {
    base: isize,
    offset: isize,
    kind: LocationKind,
    is_mutable: bool,
    is_place: bool
}

impl Location {
    const BAD: isize = isize::MAX;

    fn none() -> Location {
        Location { base: Self::BAD, offset: Self::BAD, kind: LocationKind::None, is_mutable: false, is_place: false }
    }

    fn control() -> Location {
        Location { base: Self::BAD, offset: Self::BAD, kind: LocationKind::Control, is_mutable: false, is_place: false }
    }

    fn register(value: isize) -> Location {
        Location { base: Self::BAD, offset: value, kind: LocationKind::Register, is_mutable: false, is_place: false }
    }

    fn ret() -> Location {
        Location { base: Self::BAD, offset: 0, kind: LocationKind::Return, is_mutable: false, is_place: false }
    }

    fn stack(offset: isize) -> Location {
        Location { base: Self::BAD, offset: offset, kind: LocationKind::Stack, is_mutable: true, is_place: true }
    }

    fn based(base: isize, offset: isize, is_mutable: bool) -> Location {
        Location { base, offset, kind: LocationKind::Based, is_mutable, is_place: true }
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
    value_is_stack_offset: bool,
    value: Option<RegValue>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Label(usize);

impl Label {
    const BAD: Label = Label(!0);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct InstrLocation(usize);

#[derive(Clone, Copy, Default, Debug)]
struct JumpContext {
    break_to: Option<Label>,
    continue_to: Option<Label>,
}

#[derive(Clone, Copy, Debug)]
struct Control {
    true_to: Label,
    false_to: Label,
    fallthrough_to: Label,
}

fn branch(true_to: Label, false_to: Label) -> Control {
    Control { true_to, false_to, fallthrough_to: Label::BAD }
}

fn fall_true(true_to: Label, false_to: Label) -> Control {
    Control { true_to, false_to, fallthrough_to: true_to }
}

fn fall_false(true_to: Label, false_to: Label) -> Control {
    Control { true_to, false_to, fallthrough_to: false_to }
}

fn jump(to: Label) -> Control {
    Control { true_to: to, false_to: to, fallthrough_to: to }
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
    jmp: JumpContext,
    labels: Vec<InstrLocation>,
    patches: Vec<(Label, InstrLocation)>,
    return_type: Type,
    constant: bool,
    generating_code_for_func: bool,
    error: Option<IncompleteError>,
}

impl FatGen {
    const REGISTER_SIZE: usize = 8;

    fn new() -> FatGen {
        FatGen {
            code: Vec::new(),
            locals: Locals::default(),
            reg_counter: 0,
            jmp: JumpContext::default(),
            labels: Vec::new(),
            patches: Vec::new(),
            return_type: Type::None,
            constant: false,
            generating_code_for_func: false,
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
        self.reg_counter = result as isize + size as isize;
        // todo: error if size does not fit on the stack
        assert!(self.reg_counter < i32::MAX as isize);
        result as isize
    }

    fn inc_reg(&mut self) -> isize {
        self.inc_bytes(Self::REGISTER_SIZE, Self::REGISTER_SIZE)
    }

    fn put3(&mut self, op: Op, dest: isize, left: isize, right: isize) {
        if self.constant == false {
            let dest = i32::try_from(dest).unwrap_or_else(|_| { error!(self, 0, "stack is limited to range addressable by i32"); 0 });
            let left = i32::try_from(left).unwrap_or_else(|_| { assert!(self.error.is_some()); 0 });
            let right = i32::try_from(right).unwrap_or_else(|_| { assert!(self.error.is_some()); 0 });
            self.code.push(FatInstr { op, dest, left, right });
        } else {
            // todo: expr position
            error!(self, 0, "non-constant expression");
        }
    }

    fn put(&mut self, op: Op, dest: isize, data: RegValue) {
        if self.constant == false {
            let dest = i32::try_from(dest).unwrap_or_else(|_| { error!(self, 0, "stack is limited to range addressable by i32"); 0 });
            self.code.push(FatInstr { op, dest, left: unsafe { data.sint32.0 }, right: unsafe { data.sint32.1 }});
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
                Some(v) if expr.value_is_stack_offset => { assert!(expr.ty.is_pointer()); self.put_inc(Op::LoadStackAddress, v) },
                Some(v) => self.constant(v),
                None => Location::BAD
            },
            LocationKind::Control => unreachable!(),
            LocationKind::Return => 0,
            LocationKind::Register => expr.addr.offset,
            LocationKind::Stack => {
                if let Some(op) = load_op_stack(expr.ty) {
                    self.put2_inc(op, expr.addr.offset)
                } else {
                    // todo: overwritable error here. we expect the _next_ error
                    // will be more informative to the user. BAD can
                    // appear sometimes in non-error cases, so it would be nice
                    // to explicitly enter an error state, and generally keep
                    // BAD usage to a minimum
                    Location::BAD
                }
            }
            LocationKind::Based => {
                if let Some(op) = load_op_pointer(expr.ty) {
                    let ptr = self.location_register(&expr.addr);
                    self.put2_inc(op, ptr)
                } else {
                    // todo: weak (overwritable) error here, as above
                    Location::BAD
                }
            }
        }
    }

    fn location_register(&mut self, location: &Location) -> isize {
        match location.kind {
            LocationKind::None => Location::BAD,
            LocationKind::Control => unreachable!(),
            LocationKind::Return => 0,
            LocationKind::Register => location.offset,
            LocationKind::Stack => {
                self.put_inc(Op::LoadStackAddress, location.offset.into())
            }
            LocationKind::Based => {
                if location.offset != 0 {
                    let offset_register = self.constant(location.offset.into());
                    self.put3_inc(Op::IntAdd, location.base, offset_register)
                } else {
                    location.base
                }
            }
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
            LocationKind::None|LocationKind::Control|LocationKind::Register => unreachable!(),
            LocationKind::Return|LocationKind::Stack => {
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
            LocationKind::None|LocationKind::Control => unreachable!(),
            LocationKind::Register => {
                let result_register = self.register(src);
                self.put2(Op::Move, dest.offset, result_register);
            }
            LocationKind::Return => {
                if destination_type.is_basic() {
                    assert_eq!(dest.offset, 0);
                    let result_register = self.register(src);
                    self.put2(Op::Move, 0, result_register);
                } else {
                    self.put3(Op::StackRelativeCopy, 0, src.addr.offset, size);
                }
            },
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
                    LocationKind::None|LocationKind::Control|LocationKind::Return => unreachable!(),
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
        let result = if src.ty.is_basic() {
            let reg = self.inc_reg();
            Location::register(reg)
        } else {
            let info = ctx.types.info(src.ty);
            let offset = self.inc_bytes(info.size, info.alignment);
            Location::stack(offset)
        };
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
        self.jmp.break_to
    }

    fn continue_label(&self) -> Option<Label> {
        self.jmp.continue_to
    }

    fn push_loop_context(&mut self) -> (Label, Label, JumpContext) {
        let result = self.jmp;
        let break_to = self.label();
        let continue_to = self.label();
        self.jmp = JumpContext {
            break_to: Some(break_to),
            continue_to: Some(continue_to),
        };
        (break_to, continue_to, result)
    }

    fn push_break_context(&mut self) -> (Label, JumpContext) {
        let result = self.jmp;
        let break_to = self.label();
        self.jmp = JumpContext {
            break_to: Some(break_to),
            continue_to: result.continue_to,
        };
        (break_to, result)
    }

    fn restore(&mut self, mark: JumpContext) {
        self.jmp = mark;
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
                assert!(self.code[from].is_jump() && self.code[from].left == label as i32);
                let offset = (to as isize) - (from as isize) - 1;
                let offset = i32::try_from(offset).expect("function too large");
                self.code[from].left = offset;
                assert!(offset != 0, "zero offset {:?}", self.code[from].op);
            }
        }
        self.patches.clear();
    }

    fn type_expr(&mut self, ctx: &mut Compiler, expr: TypeExpr) -> Type {
        let mut result = Type::None;
        match *ctx.ast.type_expr(expr) {
            TypeExprData::Infer => (unreachable!()),
            TypeExprData::Name(name) => {
                if name.0 == Keytype::Ptr as u32 {
                    result = Type::VoidPtr;
                } else {
                    result = sym::resolve_type(ctx, name);
                }
            }
            TypeExprData::Expr(_) => todo!(),
            TypeExprData::Items(items) => result = sym::resolve_anonymous_struct(ctx, items),
            TypeExprData::List(args) => {
                let mut args = args;
                if let Some(name) = args.next() {
                    match ctx.ast.type_expr_keytype(name) {
                        Some(Keytype::Arr) => {
                            if let Some(ty_expr) = args.next() {
                                let ty = self.type_expr(ctx, ty_expr);
                                if let Some(len_expr) = args.next() {
                                    if let TypeExprData::Expr(len_expr) = *ctx.ast.type_expr(len_expr) {
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
                        }
                        Some(Keytype::Ptr) => {
                            if let Some(ty_expr) = args.next() {
                                let ty = self.type_expr(ctx, ty_expr);
                                if let Some(bound_expr) = args.next() {
                                    if let TypeExprData::Expr(bound_expr) = *ctx.ast.type_expr(bound_expr) {
                                        let bound = self.constant_expr(ctx, bound_expr);
                                        if bound.ty.is_integer() {
                                            todo!();
                                        } else {
                                            // todo: type expr location
                                            error!(self, 0, "ptr bound must be an integer place")
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
                        Some(Keytype::Struct) => {
                            if let Some(items) = ctx.ast.type_expr_items(expr) {
                                result = sym::resolve_anonymous_struct(ctx, items);
                            } else {
                                error!(self, 0, "missing fields in struct declaration");
                            }
                        }
                        _ => error!(self, 0, "expected keytype (one of func, proc, struct, arr, ptr)")
                    }
                }
            }
        }
        let resolve_error = std::mem::take(&mut ctx.error);
        let old_error = std::mem::take(&mut self.error);
        self.error = old_error.or(resolve_error);
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

    fn compute_address(&mut self, ctx: &mut Compiler, base: &ExprResult, index: &ExprResult) -> Location {
        let base_type = ctx.types.base_type(base.ty);
        let element_size = ctx.types.info(base_type).size;
        let addr = match index.value {
            Some(value) => {
                let index = unsafe { value.sint };
                let offset = index * element_size as isize;
                if base.ty.is_pointer() {
                    // mostly the same code as below
                    let offset_reg = self.constant(offset.into());
                    let (old_base_reg, old_offset) = if base.addr.kind == LocationKind::Based {
                        (base.addr.base, base.addr.offset)
                    } else {
                        (self.register(&base), 0)
                    };
                    let base_reg = self.put3_inc(Op::IntAdd, old_base_reg, offset_reg);
                    Location::based(base_reg, old_offset, base.addr.is_mutable)
                } else {
                    base.addr.offset_by(offset)
                }
            }
            None => {
                // This has a transposition where, if accessing an array member of a
                // pointer, we update the base pointer, but leave the offset intact.
                // In other words, we compute
                //      a.b[i]
                // as
                //      (a +   i*sizeof(b[0])).b
                //       ^ base                ^ offset
                // instead of
                //      (a.b + i*sizeof(b[0]))
                //       ^base                 ^ offset (0)
                // So we can keep accumulating static offsets without emitting code.
                let size_reg = self.constant(element_size.into());
                let index_reg = self.register(&index);
                let offset_reg = self.put3_inc(Op::IntMul, index_reg, size_reg);
                // TODO: probably fold these 3 cases together and just decay arrays to pointers like an animal
                if base.ty.is_pointer() {
                    let (old_base_reg, old_offset) = if base.addr.kind == LocationKind::Based {
                        (base.addr.base, base.addr.offset)
                    } else {
                        (self.register(&base), 0)
                    };
                    let base_reg = self.put3_inc(Op::IntAdd, old_base_reg, offset_reg);
                    Location::based(base_reg, old_offset, true)
                } else {
                    match base.addr.kind {
                        LocationKind::Stack => {
                            let old_base_reg = self.put_inc(Op::LoadStackAddress, base.addr.offset.into());
                            let base_reg = self.put3_inc(Op::IntAdd, old_base_reg, offset_reg);
                            Location { base: base_reg, offset: 0, kind: LocationKind::Based, ..base.addr }
                        }
                        LocationKind::Based => {
                            let old_base_reg = base.addr.base;
                            let base_reg = self.put3_inc(Op::IntAdd, old_base_reg, offset_reg);
                            Location { base: base_reg, ..base.addr }
                        }
                        _ => unreachable!(),
                    }
                }
            }
        };
        assert!(addr.is_place == true);
        addr
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
            debug_assert!(result.addr.offset == Location::BAD);
        }
        result
    }

    fn expr(&mut self, ctx: &mut Compiler, expr: Expr) -> ExprResult {
        self.expr_with_destination_and_optional_control(ctx, expr, Type::None, &Location::none(), None)
    }

    fn expr_with_control(&mut self, ctx: &mut Compiler, expr: Expr, control: Control) -> ExprResult {
        self.expr_with_destination_and_optional_control(ctx, expr, Type::None, &Location::none(), Some(control))
    }

    fn expr_with_destination_type(&mut self, ctx: &mut Compiler, expr: Expr, destination_type: Type) -> ExprResult {
        self.expr_with_destination_and_optional_control(ctx, expr, destination_type, &Location::none(), None)
    }

    fn expr_with_destination(&mut self, ctx: &mut Compiler, expr: Expr, destination_type: Type, dest: &Location) -> ExprResult {
        self.expr_with_destination_and_optional_control(ctx, expr, destination_type, dest, None)
    }

    fn expr_with_destination_and_control(&mut self, ctx: &mut Compiler, expr: Expr, destination_type: Type, dest: &Location, control: Control) -> ExprResult {
        self.expr_with_destination_and_optional_control(ctx, expr, destination_type, dest, Some(control))
    }

    fn expr_with_destination_and_optional_control(&mut self, ctx: &mut Compiler, expr: Expr, destination_type: Type, dest: &Location, control: Option<Control>) -> ExprResult {
        let mut control = control;
        let mut result = ExprResult { addr: Location::none(), ty: Type::None, value_is_stack_offset: false, value: None };
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
                if destination_type != Type::None {
                    // Todo: check for duplicate fields. need to consider unions and fancy paths (both unimplemented)
                    let info = ctx.types.info(destination_type);
                    if matches!(info.kind, TypeKind::Struct|TypeKind::Array) {
                        // TODO: We dump everything to the stack and copy over because we don't know yet
                        // if the compound initializers are loading from the destination it is storing
                        // to. But this seems like a rare case to me, and we ought to be able to write
                        // directly to the destination in the common case, even without optimisations?
                        let base;
                        if fields.is_empty() && dest.kind != LocationKind::None {
                            base = *dest;
                        } else {
                            base = self.stack_alloc(ctx, destination_type);
                        }
                        let mut path_ctx = PathContext { state: PathContextState::ExpectAny, index: 0 };
                        self.zero(&base, info.size as isize);
                        for field in fields {
                            let field = ctx.ast.compound_field(field);
                            if let Some(item) = self.path(ctx, &mut path_ctx, destination_type, field.path) {
                                let dest = base.offset_by(item.offset as isize);
                                let expr = self.expr_with_destination(ctx, field.value, item.ty, &dest);
                                if expr.ty != item.ty {
                                    match path_ctx.state {
                                        PathContextState::ExpectPaths =>
                                            error!(self, ctx.ast.expr_source_position(field.value), "incompatible types (field '{}' is of type {}, found {})", ctx.str(item.name), ctx.type_str(item.ty), ctx.type_str(expr.ty)),
                                        PathContextState::ExpectImplict => //|PathContextState::ExpectIndices =>
                                            error!(self, ctx.ast.expr_source_position(field.value), "incompatible types (slot is of type {}, found {})", ctx.type_str(item.ty), ctx.type_str(expr.ty)),
                                        _ => todo!("PathContextState::ExpectIndices")
                                    }
                                }
                            }
                        }
                        result.addr = base;
                        result.ty = destination_type;
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
                            Some(lv) => {
                                result.addr.is_place = true;
                                result.value = Some(unsafe { lv.int + item.offset }.into());
                            }
                            None => {
                                let base = self.register(&left);
                                result.addr = Location::based(base, item.offset as isize, true);
                            }
                        }
                        result.ty = item.ty;
                        if left.ty.is_immutable_pointer() {
                            result.addr.is_mutable = false;
                            result.ty = ctx.types.immutable(result.ty);
                        }
                    } else {
                        error!(self, ctx.ast.expr_source_position(left_expr), "no field '{}' on type {}", ctx.str(field), ctx.type_str(left.ty));
                    }
                } else if let Some(item) = ctx.types.item_info(left.ty, field) {
                    debug_assert_implies!(self.error.is_none(), matches!(left.addr.kind, LocationKind::Stack|LocationKind::Based));
                    debug_assert_implies!(self.error.is_none(), left.addr.is_place);
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
                    let index = self.expr(ctx, index_expr);
                    if index.ty.is_integer() {
                        result.addr = self.compute_address(ctx, &left, &index);
                        result.ty = ctx.types.base_type(left.ty);
                        if left.ty.is_immutable_pointer() {
                            result.addr.is_mutable = false;
                            result.ty = ctx.types.immutable(result.ty);
                        }
                    } else {
                        error!(self, ctx.ast.expr_source_position(index_expr), "index must be an integer (found {})", ctx.type_str(index.ty))
                    }
                } else {
                    error!(self, ctx.ast.expr_source_position(left_expr), "indexed type must be an array (found {}, a {:?})", ctx.type_str(left.ty), ctx.types.info(left.ty).kind)
                }
            },
            ExprData::Unary(op_token, right_expr) => {
                let right = if op_token == parse::TokenKind::BitAnd && destination_type != Type::None {
                    // Address/cast/compound transposition
                    self.expr_with_destination_type(ctx, right_expr, ctx.types.base_type(destination_type))
                } else {
                    self.expr(ctx, right_expr)
                };
                use parse::TokenKind::*;
                match op_token {
                    BitAnd => {
                        if right.addr.is_place {
                            match right.addr.kind {
                                LocationKind::None|LocationKind::Return =>
                                    // This happens when doing offset-of type address calculations off of constant 0 pointers
                                    result = right,
                                LocationKind::Control => unreachable!(),
                                LocationKind::Register|LocationKind::Stack => {
                                    let base = right.addr.offset;
                                    // todo: seems weird to be setting these with kind == None
                                    result.addr.is_mutable = right.addr.is_mutable;
                                    result.addr.is_place = right.addr.is_place;
                                    result.value_is_stack_offset = true;
                                    result.value = Some(base.into());
                                }
                                LocationKind::Based => {
                                    assert!(right.addr.is_place);
                                    result.addr = Location::register(self.location_register(&right.addr));
                                }
                            }
                            result.addr.is_mutable = right.addr.is_mutable;
                            result.ty = ctx.types.pointer(right.ty);
                        } else {
                            error!(self, ctx.ast.expr_source_position(right_expr), "cannot take address of this expression");
                        }
                    }
                    Mul => {
                        if right.ty.is_pointer() {
                            match right.value {
                                Some(offset) => result.addr = Location::stack(unsafe { offset.sint }),
                                None => {
                                    let right_register = self.register(&right);
                                    result.addr = Location::based(right_register, 0, right.addr.is_mutable);
                                }
                            }
                            result.addr.is_mutable = !right.ty.is_immutable_pointer();
                            result.addr.is_place = true;
                            result.ty = ctx.types.base_type(right.ty);
                            if right.ty.is_immutable_pointer() {
                                result.ty = ctx.types.immutable(result.ty);
                            }
                        } else {
                            error!(self, ctx.ast.expr_source_position(right_expr), "cannot dereference {}", ctx.type_str(right.ty));
                        }
                    }
                    _ => {
                        if let Some((op, result_ty)) = unary_op(op_token, right.ty) {
                            match right.value {
                                Some(rv) => result.value = Some(apply_unary_op(op, rv)),
                                None if control.is_some() && op_token == Not => {
                                    if let Some(c) = control {
                                        if c.true_to == c.fallthrough_to {
                                            control = Some(Control { true_to: c.false_to, false_to: c.true_to, fallthrough_to: c.true_to });
                                        } else {
                                            control = Some(Control { true_to: c.false_to, false_to: c.true_to, fallthrough_to: c.false_to });
                                        }
                                    }
                                    let right_register = self.register(&right);
                                    result.addr = Location::register(right_register);
                                },
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
                }
            }
            ExprData::Binary(op_token, left_expr, right_expr) => {
                let (left, right, emit);
                if let Some(c) = control {
                    match op_token {
                        parse::TokenKind::LogicAnd => {
                            let next = self.label();
                            left = self.expr_with_control(ctx, left_expr, fall_true(next, c.false_to));
                            self.patch(next);
                            right = self.expr_with_control(ctx, right_expr, c);
                            emit = false;
                        }
                        parse::TokenKind::LogicOr  => {
                            let next = self.label();
                            left = self.expr_with_control(ctx, left_expr, fall_false(c.true_to, next));
                            self.patch(next);
                            right = self.expr_with_control(ctx, right_expr, c);
                            emit = false;
                        }
                        _ => {
                            left = self.expr(ctx, left_expr);
                            right = self.expr(ctx, right_expr);
                            emit = true;
                        }
                    }
                } else {
                    left = self.expr(ctx, left_expr);
                    right = self.expr(ctx, right_expr);
                    emit = true;
                }
                if let Some((op, result_ty)) = binary_op(op_token, left.ty, right.ty) {
                    match (left.value, right.value) {
                        _ if !emit => result.addr = Location::control(),
                        (Some(lv), Some(rv)) => { result.value = Some(apply_binary_op(op, lv, rv)) },
                        _ => {
                            let left_register = self.register(&left);
                            let right_register = self.register(&right);
                            let result_register = self.put3_inc(op, left_register, right_register);
                            result.addr = Location::register(result_register);
                        }
                    }
                    result.ty = result_ty;
                } else if is_address_computation(op_token, left.ty, right.ty) {
                    let (ptr, offset) = if left.ty.is_pointer() { assert!(right.ty.is_integer()); (&left, &right) } else { assert!(left.ty.is_integer()); (&right, &left) };
                    let where_u_at = self.compute_address(ctx, ptr, offset);
                    result.addr = Location::register(self.location_register(&where_u_at));
                    result.ty = ptr.ty;
                } else if is_offset_computation(ctx, op_token, left.ty, right.ty) {
                    let size = ctx.types.info(ctx.types.base_type(left.ty)).size;
                    match (left.value, right.value) {
                        (Some(lv), Some(rv)) => result.value = Some(unsafe { (lv.sint - rv.sint) / size as isize }.into()),
                        _ => {
                            let left_register = self.register(&left);
                            let right_register = self.register(&right);
                            let size_register = self.constant(size.into());
                            let diff_bytes_register = self.put3_inc(Op::IntSub, left_register, right_register);
                            let result_register = self.put3_inc(Op::IntDiv, diff_bytes_register, size_register);
                            result.addr = Location::register(result_register);
                            result.ty = Type::Int;
                        }
                    }
                } else {
                    error!(self, ctx.ast.expr_source_position(left_expr), "incompatible types ({} {} {})", ctx.type_str(left.ty), op_token, ctx.type_str(right.ty));
                }
            }
            ExprData::Ternary(cond_expr, left_expr, right_expr) => {
                let left = self.label();
                let right = self.label();
                let cond = self.expr_with_control(ctx, cond_expr, fall_true(left, right));
                if cond.ty == Type::Bool {
                    let exit = self.label();
                    self.patch(left);
                    let left = self.expr_with_destination_and_control(ctx, left_expr, destination_type, dest, jump(exit));
                    self.patch(right);
                    let right = self.expr_with_destination(ctx, right_expr, left.ty, &left.addr);
                    self.patch(exit);
                    if left.ty == right.ty {
                        debug_assert!(left.addr == right.addr);
                        result = left;
                    } else {
                        error!(self, ctx.ast.expr_source_position(left_expr), "incompatible types (... ? {} :: {})", ctx.type_str(left.ty), ctx.type_str(right.ty));
                    }
                } else {
                    error!(self, ctx.ast.expr_source_position(cond_expr), "ternary expression requires a boolean condition");
                }
            }
            ExprData::Call(callable, args) => {
                let addr = self.expr(ctx, callable);
                let info = ctx.types.info(addr.ty);
                if info.kind == TypeKind::Callable {
                    if !(self.generating_code_for_func && info.mutable) {
                        let return_type = info.items.last().expect("tried to generate code to call a func without return type").ty;
                        let mut arg_gens = SmallVec::new();
                        for (i, expr) in args.enumerate() {
                            let info = ctx.types.info(addr.ty);
                            let item = info.items[i];
                            let gen = self.expr_with_destination_type(ctx, expr, item.ty);
                            if !ctx.types.annoying_deep_eq(gen.ty, item.ty) {
                                error!(self, ctx.ast.expr_source_position(expr), "argument {} of {} is of type {}, found {}", i, ctx.callable_str(ident(addr.value)), ctx.type_str(item.ty), ctx.type_str(gen.ty));
                                break;
                            }
                            let reg = self.location_register(&gen.addr);
                            arg_gens.push((gen, reg));
                        }
                        for &(gen, reg) in arg_gens.iter() {
                            match gen.value {
                                Some(_) if gen.ty.is_pointer() => self.register(&gen),
                                Some(gv) => self.constant(gv),
                                _ => self.put2_inc(Op::Move, reg)
                            };
                        }
                        result.addr = match addr.value {
                            Some(func) => {
                                if return_type.is_basic() {
                                    let dest = self.inc_reg();
                                    self.put(Op::Call, dest, func);
                                    Location::register(dest)
                                } else {
                                    let (return_size, return_alignment) = {
                                        let info = ctx.types.info(return_type);
                                        (info.size, info.alignment)
                                    };
                                    let dest = self.inc_bytes(return_size, return_alignment);
                                    self.put(Op::Call, dest, func);
                                    Location::stack(dest)
                                }
                            },
                            _ => todo!("indirect call")
                        };
                        result.ty = return_type;
                    } else {
                        error!(self, ctx.ast.expr_source_position(callable), "cannot call proc '{}' from within a func", ctx.callable_str(ident(addr.value)));
                    }
                } else {
                    // TODO: get a string representation of the whole `callable` expr for nicer error message
                    error!(self, ctx.ast.expr_source_position(callable), "cannot call a {}", ctx.type_str(addr.ty));
                }
            }
            ExprData::Cast(expr, type_expr) => {
                let to_ty = self.type_expr(ctx, type_expr);
                let left = if to_ty == destination_type {
                    self.expr_with_destination(ctx, expr, destination_type, dest)
                } else {
                    self.expr_with_destination_type(ctx, expr, to_ty)
                };
                let epxression_transposable = || {
                    if let ExprData::Unary(parse::TokenKind::BitAnd, inner) = ctx.ast.expr(expr) {
                        if let ExprData::Compound(_) = ctx.ast.expr(inner) {
                            return true;
                        }
                    }
                    false
                };
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
                    if ctx.types.types_match_with_promotion(result.ty, to_ty) {
                        // We can send integer types back up the tree unmodified
                        // without doing the conversion here.
                    } else {
                        result.ty = ctx.types.copy_mutability(to_ty, left.ty);
                    }
                } else if epxression_transposable() {
                    // The cast-on-the-right synax doesn't work great with taking addresses.
                    //
                    // Conceptually, this transposes
                    //      &{}:Struct
                    // to
                    //      &({}:Struct)
                    //
                    // We only do this if the above convert_op returns None, as otherwise we cannot distinguish between
                    //      &a:int (which is equivalent to (&a):int)
                    // and
                    //      &(a:int)
                    debug_assert!(left.value.is_some());
                    debug_assert!(left.ty.is_pointer());
                    result = ExprResult { ty: ctx.types.pointer(to_ty), ..left };
                } else {
                    error!(self, ctx.ast.expr_source_position(expr), "cannot cast from {} to {}", ctx.type_str(left.ty), ctx.type_str(to_ty));
                }
            }
        };

        #[cfg(debug_assertions)]
        if self.error.is_none() {
            debug_assert_implies!(result.addr.offset == Location::BAD, matches!(result.addr.kind, LocationKind::None|LocationKind::Control));
            debug_assert_implies!(result.addr.offset != Location::BAD, result.addr.kind != LocationKind::None);
            debug_assert_implies!(result.addr.base != Location::BAD, result.addr.kind == LocationKind::Based);
            debug_assert_implies!(result.ty != Type::None, result.value.is_some() || result.addr.kind != LocationKind::None);
        }

        if destination_type != Type::None {
            let result_info = ctx.types.info(result.ty);
            if destination_type.is_pointer()
            && result_info.kind == TypeKind::Array
            && result_info.base_type == ctx.types.base_type(destination_type) {
                // Decay array to pointer
                result.addr = Location::based(self.location_register(&result.addr), 0, result.addr.is_mutable);
                result.ty = destination_type;
            }
        }

        if destination_type != Type::None && dest.kind != LocationKind::None {
            let compatible = expr_integer_compatible_with_destination(&result, destination_type, dest) ;
            if compatible || ctx.types.annoying_deep_eq(destination_type, result.ty) {
                if result.addr != *dest {
                    self.copy(ctx, destination_type, dest, &result);
                    result.addr = *dest;
                }

                if compatible {
                    result.ty = destination_type;
                }
            }
        }

        if let Some(control) = control {
            if result.addr.kind != LocationKind::Control {
                if result.ty == Type::Bool && control.true_to != control.false_to {
                    match result.value {
                        None => {
                            if control.true_to == control.fallthrough_to {
                                let cond_register = self.register(&result);
                                self.put_jump_zero(cond_register, control.false_to);
                            } else if control.false_to == control.fallthrough_to {
                                let cond_register = self.register(&result);
                                self.put_jump_nonzero(cond_register, control.true_to);
                            } else if control.fallthrough_to != Label::BAD {
                                assert!(control.true_to == control.false_to);
                                self.put_jump(control.fallthrough_to);
                            } else {
                                let cond_register = self.register(&result);
                                self.put_jump_zero(cond_register, control.false_to);
                                self.put_jump(control.true_to);
                            }
                        },
                        Some(v) => {
                            if control.true_to == control.fallthrough_to {
                                if !v.is_true() {
                                    self.put_jump(control.false_to);
                                }
                            } else if control.false_to == control.fallthrough_to {
                                if v.is_true() {
                                    self.put_jump(control.true_to);
                                }
                            } else if control.fallthrough_to != Label::BAD {
                                assert!(control.true_to == control.false_to);
                                self.put_jump(control.fallthrough_to);
                            } else {
                                if v.is_true() {
                                    self.put_jump(control.true_to);
                                } else {
                                    self.put_jump(control.false_to);
                                }
                            }
                        }
                    }
                } else if control.true_to == control.false_to {
                    // Unconditional jump; if we have a constant value, we need to
                    // emit it. Otherwise, the caller can't place it behind the jump
                    // destination
                    result = self.emit_constant(&result);
                    self.put_jump(control.true_to);
                }
            }
        }
        result
    }

    fn stmt(&mut self, ctx: &mut Compiler, stmt: Stmt) -> Option<Type> {
        let mut return_type = None;
        match ctx.ast.stmt(stmt) {
            StmtData::Block(body) => {
                return_type = self.stmts(ctx, body)
            }
            StmtData::Return(Some(expr)) => {
                let ret_expr = self.expr_with_destination(ctx, expr, self.return_type, &Location::ret());
                if ret_expr.ty == self.return_type {
                    self.put0(Op::Return);
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
                let then_branch = self.label();
                let else_branch = self.label();
                let exit = self.label();
                let cond = self.expr_with_control(ctx, cond_expr, fall_true(then_branch, else_branch));
                if cond.ty == Type::Bool {
                    self.patch(then_branch);
                    let then_ret = self.stmts(ctx, then_body);
                    if else_body.is_nonempty() {
                        self.put_jump(exit);
                        self.patch(else_branch);
                        let else_ret = self.stmts(ctx, else_body);
                        return_type = then_ret.and(else_ret);
                    } else {
                        self.patch(else_branch);
                    }
                    self.patch(exit);
                } else {
                    error!(self, ctx.ast.expr_source_position(cond_expr), "if statement requires a boolean condition");
                }
            }
            StmtData::For(pre_stmt, cond_expr, post_stmt, body_stmts) => {
                let (break_label, continue_label, gen_ctx) = self.push_loop_context();
                let body = self.label();
                let mark = self.locals.push_scope();
                if let Some(stmt) = pre_stmt {
                    self.stmt(ctx, stmt);
                }
                if let Some(expr) = cond_expr {
                    let pre_cond = self.expr_with_control(ctx, expr, fall_true(body, break_label));
                    if pre_cond.ty == Type::Bool {
                        self.patch(body);
                        return_type = self.stmts(ctx, body_stmts);
                        self.patch(continue_label);
                        if let Some(stmt) = post_stmt {
                            self.stmt(ctx, stmt);
                        }
                        self.expr_with_control(ctx, expr, fall_false(body, break_label));
                    } else {
                        error!(self, ctx.ast.expr_source_position(expr), "for statement requires a boolean condition");
                    }
                } else {
                    self.patch(body);
                    return_type = self.stmts(ctx, body_stmts);
                    self.patch(continue_label);
                    if let Some(stmt) = post_stmt {
                        self.stmt(ctx, stmt);
                    }
                    self.put_jump(body);
                }
                self.locals.restore_scope(mark);
                self.patch(break_label);
                self.restore(gen_ctx);
            }
            StmtData::While(cond_expr, body_stmts) => {
                let (break_label, continue_label, gen_ctx) = self.push_loop_context();
                if body_stmts.is_nonempty() {
                    let body = self.label();
                    let pre_cond = self.expr_with_control(ctx, cond_expr, fall_true(body, break_label));
                    if pre_cond.ty == Type::Bool {
                        self.patch(body);
                        return_type = self.stmts(ctx, body_stmts);
                        self.patch(continue_label);
                        self.expr_with_control(ctx, cond_expr, fall_false(body, break_label));
                    } else {
                        error!(self, ctx.ast.expr_source_position(cond_expr), "while statement requires a boolean condition");
                    }
                } else {
                    let cond = self.label_here();
                    self.expr_with_control(ctx, cond_expr, fall_false(cond, break_label));
                }
                self.patch(break_label);
                self.restore(gen_ctx);
            }
            StmtData::Switch(control_expr, cases) => {
                // Totally naive switch implementation. Break by default,
                // fall-through unimplemented.
                let (break_label, gen_ctx) = self.push_break_context();
                let mut labels = SmallVec::new();
                let mut else_label = None;
                let control = self.expr(ctx, control_expr);
                let control_register = self.register(&control);
                for case in cases {
                    let block_label = self.label();
                    labels.push(block_label);
                    if let SwitchCaseData::Cases(_, exprs) = ctx.ast.switch_case(case) {
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
                    if !first_block {
                        self.put_jump(break_label);
                    }
                    let block = match ctx.ast.switch_case(case) {
                        SwitchCaseData::Cases(block, _) => block,
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
                }
                self.patch(break_label);
                self.restore(gen_ctx);
            }
            StmtData::Do(cond_expr, body_stmts) => {
                let (break_label, continue_label, gen) = self.push_loop_context();
                let body = self.label_here();
                return_type = self.stmts(ctx, body_stmts);
                self.patch(continue_label);
                let post_cond = self.expr_with_control(ctx, cond_expr, fall_false(body, break_label));
                if post_cond.ty != Type::Bool {
                    error!(self, ctx.ast.expr_source_position(cond_expr), "do statement requires a boolean condition");
                }
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
                    let addr = if expr.addr.is_place {
                        self.copy_to_stack(ctx, &expr)
                    } else {
                        self.emit_constant(&expr).addr
                    };
                    if ctx.types.types_match_with_promotion(decl_type, expr.ty) {
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
                if lv.addr.is_mutable {
                    let value = self.expr_with_destination(ctx, right, lv.ty, &lv.addr);
                    if lv.ty != value.ty {
                        error!(self, ctx.ast.expr_source_position(left), "type mismatch between destination ({}) and value ({})", ctx.type_str(lv.ty), ctx.type_str(value.ty))
                    }
                } else {
                    // todo: terrible error message
                    error!(self, ctx.ast.expr_source_position(left), "{} is not mutable", ctx.ast.expr(left));
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

    fn callable(&mut self, ctx: &mut Compiler, signature: Type, decl: Decl) -> Func {
        let callable = ctx.ast.callable(decl);
        let body = callable.body;
        let sig = ctx.types.info(signature);
        let kind = ctx.ast.type_expr_keytype(callable.expr).expect("tried to generate code for callable without keytype");
        let params = ctx.ast.type_expr_items(callable.expr).expect("tried to generate code for callable without parameters");
        self.generating_code_for_func = ctx.ast.type_expr_keytype(callable.expr) == Some(Keytype::Func);
        assert!(self.code.len() == 0);
        assert_eq!(sig.kind, TypeKind::Callable, "sig.kind == TypeKind::Callable");
        assert!(params.len() == sig.items.len() - 1, "the last element of a callable's type signature's items must be the return type");
        self.return_type = sig.items.last().map(|i| i.ty).unwrap();
        assert_implies!(self.generating_code_for_func, self.return_type != Type::None);
        self.reg_counter = (Self::REGISTER_SIZE as isize) * (1 - sig.items.len() as isize);
        let param_names = params.map(|item| ctx.ast.item(item).name);
        let param_types = sig.items.iter().map(|i| i.ty);
        let top = self.locals.push_scope();
        for (name, ty) in Iterator::zip(param_names, param_types) {
            let reg = self.inc_reg();
            let loc = if ty.is_basic() {
                Location::register(reg)
            } else {
                Location::based(reg, 0, false)
            };
            if self.generating_code_for_func {
                let const_ty = ctx.types.immutable(ty);
                self.locals.insert(name, Local::argument(loc, const_ty));
            } else {
                self.locals.insert(name, Local::argument(loc, ty));
            }
        }
        let (return_size, return_alignment) = {
            let info = ctx.types.info(self.return_type);
            (info.size, info.alignment)
        };
        let return_dest = self.inc_bytes(return_size, return_alignment);
        assert_eq!(return_dest, 0);
        let returned = self.stmts(ctx, body);
        if self.return_type != Type::None {
            if matches!(returned, None) {
                let callable = ctx.ast.callable(decl);
                error!(self, callable.pos, "{} {}: not all control paths return a value", kind, ctx.str(callable.name))
            }
        } else {
            self.put0(Op::Return);
        }
        self.apply_patches();
        self.locals.restore_scope(top);
        assert!(self.patches.len() == 0);
        assert!(self.jmp.break_to.is_none());
        assert!(self.jmp.continue_to.is_none());
        assert!(self.locals.top_scope_index == 0);
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

struct TypeStrFormatter<'a> {
    ctx: &'a Compiler,
    ty: Type,
    callable_name: Intern,
}

impl Display for TypeStrFormatter<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::result::Result<(), std::fmt::Error> {
        let info = self.ctx.types.info(self.ty);
        let have_name = self.callable_name != Intern(0);
        let base_type = self.ctx.types.base_type(self.ty);
        let needs_parens = self.ctx.types.base_type(base_type) != base_type;
        match info.kind {
            // todo: string of func signature, anonymous structs
            TypeKind::Callable if have_name   => write!(f, "{} {}", if info.mutable { "proc" } else { "func" }, self.ctx.str(self.callable_name)),
            TypeKind::Callable                => write!(f, "{}", if info.mutable { "proc" } else { "func" }),
            TypeKind::Struct if have_name     => write!(f, "{}", self.ctx.str(info.name)),
            TypeKind::Struct                  => write!(f, "anonymous struct"),
            TypeKind::Array if needs_parens   => write!(f, "arr ({}) [{}]", self.ctx.type_str(info.base_type), info.num_array_elements),
            TypeKind::Array                   => write!(f, "arr {} [{}]", self.ctx.type_str(info.base_type), info.num_array_elements),
            TypeKind::Pointer if needs_parens => write!(f, "ptr ({})", self.ctx.type_str(info.base_type)),
            TypeKind::Pointer                 => write!(f, "ptr {}", self.ctx.type_str(info.base_type)),
            _ => self.ctx.str(info.name).fmt(f)
        }
    }
}

pub struct Compiler {
    interns: Interns,
    types: Types,
    symbols: Symbols,
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
            symbols: Default::default(),
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

    fn callable_str(&self, name: Intern) -> TypeStrFormatter {
        TypeStrFormatter { ctx: self, ty: sym::lookup_value(self, name).map(|sym| sym.ty).unwrap_or(Type::None), callable_name: name }
    }

    fn type_str(&self, ty: Type) -> TypeStrFormatter {
        TypeStrFormatter { ctx: self, ty, callable_name: Intern(0) }
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

    fn run(&self) -> Result<Value> {
        let mut code = &self.entry_stub[..];
        let mut sp: usize = 0;
        let mut ip: usize = 0;
        let mut call_stack = vec![(code, ip, sp, Intern(0))];

        // The VM allows taking (real!) pointers to addresses on the VM stack. These are
        // immediately invalidated by taking a & reference to the stack. Solution: never
        // take a reference to the stack (even one immediately turned into a pointer).

        // This just ensures the compiler knows the correct invariants involved, it
        // doesn't make the vm sound

        // MIRI shits the bed if this is on the stack
        let layout = std::alloc::Layout::array::<RegValue>(8192).unwrap();
        let stack: *mut [u8] = unsafe {
            let ptr = std::alloc::alloc(layout);
            // Zero the first register so we don't return junk if the entry stub immediately halts
            *(ptr as *mut usize) = 0;
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

        // terrible
        macro_rules! panic_message {
            () => {
                if Some(call_stack[call_stack.len() - 2].3) == self.interns.get("assert") {
                    format!("assertion failed in {}", self.str(call_stack[call_stack.len() - 3].3))
                } else {
                    format!("panic in {}", self.str(call_stack[call_stack.len() - 2].3))
                }
            }
        }

        loop {
            let instr = &code[ip];
            let dest = sp.wrapping_add(instr.dest as usize);
            let left = sp.wrapping_add(instr.left as usize);
            let right = sp.wrapping_add(instr.right as usize);

            assert_implies!(requires_register_destination(instr.op), (dest & (FatGen::REGISTER_SIZE - 1)) == 0);

            unsafe {
                 match instr.op {
                    Op::Halt       => { break; }
                    Op::Panic      => { return Err(panic_message!().into()) }
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
                    Op::Jump          => { ip = ip.wrapping_add(instr.left as usize); }
                    Op::JumpIfZero    => { if reg![dest].int == 0 { ip = ip.wrapping_add(instr.left as usize); } }
                    Op::JumpIfNotZero => { if reg![dest].int != 0 { ip = ip.wrapping_add(instr.left as usize); } }
                    Op::Return        => {
                        let ret = call_stack.pop().expect("Bad bytecode");
                        code = ret.0;
                        ip = ret.1;
                        sp = ret.2;
                    },
                    Op::Call          => {
                        let name = Intern(instr.left as u32);
                        call_stack.push((code, ip, sp, name));
                        let f = self.funcs.get(&name).expect("Bad bytecode");
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
        Ok(result)
    }
}

fn compile(str: &str) -> Result<Compiler> {
    let mut c = Compiler::new();

    parse::parse(&mut c, r#"
panic: func () int {
    // built-in
}

assert: func (condition: bool) int {
    if (!condition) {
        panic();
    }
    return 0;
}
"#);

    parse::parse(&mut c, str);

    c.check_and_clear_error(str)?;
    resolve_all(&mut c);

    for decl in c.ast.decl_list() {
        if c.ast.is_callable(decl) {
            let decl_data = c.ast.decl(decl);
            let name = decl_data.name();
            let pos = decl_data.pos();
            if let Some(sym) = sym::lookup_value(&c, name) {
                let ty = sym.ty;
                match c.globals.entry(name) {
                    std::collections::hash_map::Entry::Occupied(_) => { error!(c, pos, "{} {} has already been defined", c.ast.decl(decl), c.str(name)); }
                    std::collections::hash_map::Entry::Vacant(entry) => { entry.insert(Global { decl: decl, value: Value::Ident(name), ty: ty }); }
                }
            }
        }
    }

    c.check_and_clear_error(str)?;

    let main = c.intern("main");
    if let Some(&main) = c.globals.get(&main) {
        let info = c.types.info(main.ty);
        if info.mutable
        && matches!(info.arguments(), Some(&[]))
        && matches!(info.return_type(), Some(Type::Int)) {
            // main's ok
        } else {
            let pos = c.ast.decl(main.decl).pos();
            error!(c, pos, "main must be a procedure that takes zero arguments and returns int");
        }
    }

    let panic = c.intern("panic");
    if let Some(&panic_def) = c.globals.get(&panic) {
        let info = c.types.info(panic_def.ty);
        assert!(matches!(info.arguments(), Some(&[])) && matches!(info.return_type(), Some(Type::Int)));
        c.funcs.insert(panic, Func { _signature: panic_def.ty, code: vec![FatInstr::PANIC] });
    }

    c.check_and_clear_error(str)?;

    let mut gen = FatGen::new();
    for decl in c.ast.decl_list().skip(1) {
        if c.ast.is_callable(decl) {
            let name = c.ast.decl(decl).name();
            let sig = c.globals.get(&name).unwrap().ty;
            let func = gen.callable(&mut c, sig, decl);
            c.funcs.insert(name, func);
        }
    }

    // :P
    c.error = gen.error.take();
    c.check_and_clear_error(str)?;

    if let Some(_func) = c.funcs.get(&main) {
        c.entry_stub[0] = FatInstr { op: Op::Call, dest: 0, left: main.0 as i32, right: 0 };
    }
    Ok(c)
}

fn compile_and_run(str: &str) -> Result<Value> {
    let c = compile(str)?;
    c.run()
}

fn eval_expr(str: &str) -> Result<Value> {
    compile_and_run(&format!("main: proc () int {{ return {}; }}", str))
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

    let ok = [
        ("add': func (a: int, b: int) int { return a + b; }", Value::Int(0)),
        ("add: func (a: int, b: int) -> int { return a + b; }", Value::Int(0)),
        ("add: (a: int, b: int) -> int { return a + b; }", Value::Int(0)),
        ("V2: struct (x: f32, y: f32);", Value::Int(0)),
        ("V2: struct (x, y: f32);", Value::Int(0)),
        ("func: func (func, proc: int) int { return func + proc; }", Value::Int(0)),
        ("V1: struct (x: f32); V2: struct (x: V1, y: V1);", Value::Int(0)),
        ("struct: struct (struct: struct (struct: int));", Value::Int(0)),
        ("struct: (struct: (struct: int));", Value::Int(0)),
        ("int: (int: int) int { return 0; }", Value::Int(0)),
];
    for test in ok.iter() {
        let str = test.0;
        let expect = test.1;
        compile_and_run(str)
            .and_then(|ret| if ret == expect { Ok(ret) } else { Err(format!("expected {:?}, got {:?}", expect, ret).into())})
            .map_err(|e| { println!("input: \"{}\"", str); e })
            .unwrap();
    }
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
        "a := 1.0; a = a + (1.0:f32);",
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
        compile_and_run(&format!("main: proc () int {{ {{ {} }} return 0; }}", str)).map_err(|e| { println!("input: \"{}\"", str); e }).unwrap();
    }
    for str in err.iter() {
        // Not checking it's the right error yet--maybe later
        compile_and_run(&format!("main: proc () int {{ {{ {} }} return 0; }}", str)).map(|e| { println!("input: \"{}\"", str); e }).unwrap_err();
    }
}

#[test]
fn decls() {
    let ok = [
        "add: func (a: int, b: int) int { return a + b; }",
        "add': func (a: int, b: int) int { return a + b; }",
        "add: func (a: int, b: int) -> int { return a + b; }",
        "add: (a: int, b: int) -> int { return a + b; }",
        "V2: struct (x: f32, y: f32);",
        "V2: struct (x, y: f32);",
        "func: func (func, proc: int) int { return func + proc; }",
        "V1: struct (x: f32); V2: struct (x: V1, y: V1);",
        "struct: struct (struct: struct (struct: int));",
        "struct: (struct: (struct: int));",
        "int: (int: int) int { return int; }"
    ];
    let err = [
        "dup_func: func (a: int, b: int) int { return a + b; } dup_func: func (a: int, b: int) int { return a + b; }",
        "dup_param: func (a: int, a: int) int { return a + a; }",
        "empty: struct ();",
        "empty: ();",
        "dup: struct (x: f32); dup: struct (y: f32);",
        "dup_: struct{: struct (: f32) dup_: struct{: struct (: f32);",
        "dup_field: struct (x: f32, x: f32);",
        "circular: struct (x: circular);",
        "circular1: struct (x: circular2); circular2: struct (x: circular1);",
        "struct: struct (struct: struct);",
        "struct: (struct: (struct: struct));",
        "struct: (struct: struct) struct {}"
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
        "control_paths: func (a: int, b: int, c: bool) int { if (c) { return a + b; } else { return a - b; } }",
        "control_paths: func (a: int, b: int, c: bool) int { while (c) { return a; } }",
        "control_paths: func (a: int, b: int, c: bool) int { return a; while (c) { if (c) { return a; } } }",
    ];
    let err = [
        "control_paths: func (a: int) {}",
        "control_paths: func (a: int, b: int, c: bool) int { if (c) { return a + b; } }",
        "control_paths: func (a: int, b: int, c: bool) int { if (c) { return a + b; } else { } }",
        "control_paths: func (a: int, b: int, c: bool) int { if (c) {} else { return a - b; } }",
        "control_paths: func (a: int, b: int, c: bool) int { while (c) { if (c) { return a; } } }",
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
        "pro: func () int { a: u8 = 1; b: u8 = 2; return a + b; }",
        "pro: func () int { a: u8 = 1; b: u16 = 2; return a + b; }",
        "pro: func () int { a: u8 = 1; b: i16 = 2; return a + b; }",
        "pro: func () int { a: u8 = 255:i16:u8; return a; }",
        "pro: func () int { a: u8 = 256:i16:u8; return a; }",
    ];
    let err = [
        "pro: func () u8 { a: u8 = 1; b: u8 = 2; return a + b; }",
        "pro: func () int { a: u16 = 256:i16:u8; return a; }",
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
        "fits: func () int { a: u8 = -1; return a; }",
        "fits: func () int { a: i8 = 128; return a; }",
        "fits: func () int { a: u8 = 256; return a; }",
        "fits: func () int { a: i16 = 32768; return a; }",
        "fits: func () int { a: u16 = 65536; return a; }",
        "fits: func () int { a: i32 = 2147483648; return a; }",
        "fits: func () int { a: u32 = 4294967296; return a; }",
        "fits: func () int { a := 18446744073709551616; return a; }",
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
madd: func (a: int, b: int, c: int) int {
    return a * b + c;
}

main: proc () int {
    return madd(1, 2, 3);
}
"#, Value::Int(5)),
(r#"
fib: func (n: int) int {
    if (n <= 1) {
        return n;
    }

    return fib(n - 2) + fib(n - 1);
}

main: proc () int {
    return fib(6);
}
"#, Value::Int(8)),
(r#"
even: func (n: int) bool {
    return n == 0 ? 1:bool :: odd(n - 1);
}

odd: func (n: int) bool {
    return n == 0 ? 0:bool :: even(n - 1);
}

main: proc () int {
    return even(10):int;
}
"#, Value::Int(1)),
(r#"
V2: struct (x: f32, y: f32);

dot: func (a: V2, b: V2) f32 {
    return a.x*b.x + a.y*b.y;
}

main: proc () int {
    return dot({ 1.0, 2.0 }, { 3.0, 4.0 }):i32;
}
"#, Value::Int(3+8)),
(r#"
V2: struct (x, y: i32);

add: func (a: V2, b: V2) V2 {
    return { (a.x + b.x):i32, (a.y + b.y):i32 };
}

main: proc () int {
    v := add({1, 2}, {3, 4});
    return v.y;
}
"#, Value::Int(2+4)),
(r#"
// identical codegen
sum: func (a: arr i32 [3]) int {
    acc := 0;
    for (i := 0; i < 3; i = i + 1) {
        acc = acc + a[i];
    }
    return acc;
}
sum2: func (a: ptr (arr i32 [3])) int {
    acc := 0;
    for (i := 0; i < 3; i = i + 1) {
        acc = acc + (*a)[i];
    }
    return acc;
}
sum3: func (a: ptr i32) int {
    acc := 0;
    for (i := 0; i < 3; i = i + 1) {
        acc = acc + a[i];
    }
    return acc;
}

main: proc () int {
    return sum({2,3,1}) + sum2(&{2,3,1}) + sum3({2,3,1}:arr i32 [3]);
}
"#, Value::Int(18)),
(r#"
does_something_without_return_value: proc (a: ptr i32) {
    if (*a == 0) {
        *a = 1;
        return;
    }

    *a = 2;
}

main: proc () int {
    a: i32 = 1;
    does_something_without_return_value(&a);
    return a;
}
"#, Value::Int(2)),
    ];
    let err = [r#"
even: proc (n: int) bool {
    return n == 0 ? 1:bool :: odd(n - 1);
}

odd: func (n: int) bool {
    return n == 0 ? 0:bool :: even(n - 1);
}

main: proc () int {
    return even(10):int;
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
fn control_structures() {
    let ok = [
(r#"
main: proc () int {
    a := 0;
    for (i := 0; i < 10; i = i + 1) {
        a = a + 1;
    }
    return a;
}
"#, Value::Int(10)),
(r#"
main: proc () int {
    a := 0;
    for (; a < 10; a = a + 2) {}
    return a;
}
"#, Value::Int(10)),
(r#"
main: proc () int {
    a := 1:bool;
    b := 1:bool;
    c := 1:bool;
    d := 0:bool;
    while (!(a && b && c && d)) {
        d = 1:bool;
    }
    return d:int;
}
"#, Value::Int(1)),
(r#"
main: proc () int {
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
main: proc () int {
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
main: proc () int {
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
main: proc () int {
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
main: proc () int {
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
main: proc () int {
    a := 1;
    do {
        if (a > 2) {
            a = (a & 1) == 1 ? a * 2 :: a + 1;
        } else {
            a = a + 1;
        }
    } while (a < 10);
    return a;
}"#, Value::Int(14)),
(r#"
main: proc () int {
    a := 1;
    b := 2;
    c := 3;
    (a == 1 ? b :: c) = 4;
    return b;
}"#, Value::Int(4)),
(r#"
inc: proc (p: ptr int) bool {
    *p = *p + 2;
    return 0;
}
main: proc () int {
    t := !!1;
    f := !!0;
    v := 1;
    if ((t || f) && (f || f)) {
        return 9;
    } else if ((t && f) || (t && t) || (f && inc(&v))) {
        return v;
    }
    return 0;
}"#, Value::Int(1))
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
i32x4: struct (
    a: i32,
    b: i32,
    c: i32,
    d: i32,
);

main: proc () int {
    a := {-1,-1,-1,-1}:i32x4;
    return a.d:int;
}
"#, Value::Int(-1)),
(r#"
RGBA: struct (
    r: i8,
    g: i8,
    b: i8,
    a: i8,
);

main: proc () int {
    c := { r = 1, g = 2, b = 3, a = 4 }:RGBA;
    d := { g = 5, r = 6, a = 7, b = 8, }:RGBA;
    e: RGBA = { 9, 10, 11, 12 };
    f: RGBA = { 13, 14, 15 };
    return (c.r << 24) + (d.g << 16) + (e.b << 8) + f.a;
}
"#, Value::Int(0x01050B00)),
(r#"
Outer: struct (
    inner: Inner
);

Inner: struct (
    ignored: i32,
    value: i32
);

main: proc () int {
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
V2: struct (
    x: i32,
    y: i32
);

fn: func () i32 {
    a := { x = 2 }:V2;
    b := { x = 3 }:V2;
    a = b;
    return b.x;
}

main: proc () int {
    a := 0;
    b := 0;
    return fn():int;
}
"#, Value::Int(3)),
(r#"
V4: (
    x: i32,
    y: i32,
    z: i32,
    w: i32,
);

main: proc () int {
    a := ({ x = -1 }:V4).x;
    b := (a == -1) ? { w = 3 }:V4 :: { w = 4 }:V4;
    return b.w:int;
}
"#, Value::Int(3)),
(r#"
add: func (a: struct (x, y: int), b: (x, y: int)) (x, y: int) {
    return { a.x+b.x, a.y+b.y };
}

main: proc () int {
    a := add({1,2},{3,4});
    return a.x + a.y;
}
"#, Value::Int(10)),
];
    let err = [r#"
main: proc () int {
    v := {0}:int;
    return v;
}
"#, r#"
V2: struct (x: int, y: int);
main: proc () int {
    v := {}:V2;
    v.a = 0;
    return v.x:int;
}
"#, r#"
V2: struct (x: int, y: int);
main: proc () int {
    v := { a = 0 }:V2;
    return v.a;
},
V2: struct (x: int, y: int);
main: proc () int {
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
asdf: struct (
    arr: arr int [4],
);
main: proc () int {
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
main: proc () int {
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
V4: struct (padding: i16, c: arr i32 [4]);
M4: struct (padding: i16, r: arr V4 [4]);
main: proc () int {
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
main: proc () int {
    a := 2;
    arr: arr int [a] = {};
    return arr[0];
}
#",r#"
main: proc () int {
    arr: arr int [1] = { 0 = 1 };
    return arr[0];
}
"#, r#"
main: proc () int {
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
main: proc () int {
    a := 1;
    b := &a;
    *b = 2;
    return *b;
}
"#, Value::Int(2)),
(r#"
main: proc () int {
    a := 1;
    *&a = 2;
    b := &a:int;
    return a;
}
"#, Value::Int(2)),
(r#"
V2: struct (x: i32, y: i32);
main: proc () int {
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
main: proc () int {
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
 (r#"
main: proc () int {
    arr := {}:arr u16 [8];
    for (p := &arr[0]; p != &arr[8]; p = p + 1) {
        *p = p - &arr[0] : u16;
    }
    arrp := &arr[0];
    acc := 0;
    for (i := 0; i < 8; i = i + 1) {
        acc = acc + *(arrp + (&arrp[i] - arrp));
    }
    return acc;
}
"#, Value::Int(0+1+2+3+4+5+6+7)),
(r#"
Buf: struct (
    buf: ptr u8,
    len: int
);
main: proc () int {
    arr := {}:arr u8 [8];
    arr[1] = 1;
    buf: Buf = { &arr[0], 8 };
    return buf.buf[1];
}
"#, Value::Int(1)),
(r#"
A: struct (
    padding: arr i16 [9],
    b: ptr B
);
B: struct (
    padding: arr i16 [3],
    c: ptr C
);
C: struct (
    padding: i16,
    d: ptr i32
);
main: proc () int {
    d := -1234;
    c: C = {d=&d:ptr(i32)};
    b: B = {c=&c};
    a: A = {b=&b};
    return *a.b.c.d;
}
"#, Value::Int(-1234)),
(r#"
V2: struct (x: i32, y: i32);
rot:(proc)(v:ptr(V2))(int) {
    *v = { -v.y:i32, v.x };
    return 0;
}
main: proc () int {
    v: V2 = {0,1};
    rot(&v);
    return v.x;
}
"#, Value::Int(-1)),
(r#"
V2:(struct (x: i32, y: i32));
main: (proc () int) {
    return (&(0:ptr V2).y:ptr) - (0:ptr);
}
"#, Value::Int(4)),
(r#"
V2: struct (x: i32, y: i32);
rot22: proc(vv: ptr -> ptr V2) {
    **vv = { -vv[0].y:i32, vv[0].x };
}
rot12: proc(v: ptr V2) {
    vv := &v;
    rot22(vv);
}
rot21: proc(vv: ptr -> ptr V2) {
    vv[0].x = -1;
}
rot11: proc(v: ptr V2) {
    vv := &v;
    rot21(vv);
}
main: proc () int {
    v1: V2 = {0,1};
    v2: V2 = {0,2};
    rot11(&v1);
    rot12(&v2);
    assert(v1.x == -1);
    assert(v2.x == -2);
    return v1.x;
}
"#, Value::Int(-1)),
];
    let err = [r#""
main: proc () int {
    a := 1;
    return *a;
}
"#,r#"
main: proc () -> int {
    a := &-1:u8;
    return *a;
}
"#,r#"
main: proc () int {
    a := -1;
    b := &a:f32;
    return *b;
}
"#,r#"
V2: struct (x: i32, y: i32);
rot: (v: ptr V2) int {
    *v = { -v.y:i32, v.x };
    return 0;
}
main: proc () int {
    v: V2 = {0,1};
    rot(&v);
    return v.x;
}
"#, r#"
main: proc () int {
    pp := 0:ptr u8;
    pq := 0:ptr u16;
    a := pq - pp;
    return 0;
}
"#, r#"
main: proc () int {
    a := &3;
    return 0;
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
