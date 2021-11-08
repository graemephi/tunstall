use crate::{
    align_up, assert_implies,
    debug_assert_implies, error,
    Code, Compiler, Error,
};
use crate::sym;
use crate::types::{self, integer_promote, bare, Bound, TypeKind, TypeInfo};
use crate::ast::*;

use crate::parse::{Keytype, TokenKind};
use crate::Intern;
use crate::{BareType, Type};
use crate::smallvec::*;

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
    Panic
}

pub fn requires_register_destination(op: Op) -> bool {
    match op {
        Op::Store8|Op::Store16|Op::Store32|Op::Copy|Op::Zero => false,
        _ => true
    }
}

pub fn disallowed_in_constant_expression(op: Op) -> bool {
    use Op::*;
    matches!(op, Call|CallIndirect|Store8|Store16|Store32|Store64|Copy|Zero|Halt|Panic)
}

#[derive(Clone, Copy, Debug)]
pub struct FatInstr {
    pub op: Op,
    pub dest: i32,
    pub left: i32,
    pub right: i32,
}

impl FatInstr {
    pub const HALT: FatInstr = FatInstr { op: Op::Halt, dest: 0, left: 0, right: 0 };

    pub fn is_jump(&self) -> bool {
        use Op::*;
        matches!(self.op, Jump|JumpIfZero|JumpIfNotZero)
    }

    pub fn call(intern: Intern) -> FatInstr {
        FatInstr { op: Op::Call, dest: 0, left: intern.0 as i32, right: 0 }
    }
}

#[derive(Clone, Copy)]
pub union RegValue {
    pub int: usize,
    pub sint: isize,
    pub wint: std::num::Wrapping<usize>,
    pub int32: (u32, u32),
    pub int16: (u16, u16, u16, u16),
    pub int8: (u8, u8, u8, u8, u8, u8, u8, u8),
    pub sint32: (i32, i32),
    pub sint16: (i16, i16, i16, i16),
    pub sint8: (i8, i8, i8, i8, i8, i8, i8, i8),
    pub float64: f64,
    pub float32: (f32, f32),
    pub b8: (bool, [bool; 7]),
}

impl RegValue {
    fn is_true(&self) -> bool {
        unsafe {
            debug_assert!(self.b8.0 == (self.int != 0));
            self.b8.0
        }
    }

    fn to_nonnegative_isize(&self, ty: Type) -> Option<isize> {
        // Todo: This convert op is a roundabout way to check for the
        // positive+fits condition but should be more direct/less subtle.
        // Really, a non-sign extended integer should never reach here, so this
        // is superfluous, but putting it off
        if let Some(op) = convert_op(ty, Type::Int) {
            let value = apply_unary_op(op, *self);
            if unsafe { value.sint } >= 0 {
                return unsafe { Some(value.sint) };
            }
        }
        None
    }

    fn to_positive_isize(&self, ty: Type) -> Option<isize> {
        if let Some(op) = convert_op(ty, Type::Int) {
            let value = apply_unary_op(op, *self);
            if unsafe { value.sint } > 0 {
                return unsafe { Some(value.sint) };
            }
        }
        None
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
    pub fn from(reg: RegValue, ty: Type) -> Value {
        match bare(ty) {
            BareType::U8|BareType::U16|BareType::U32|BareType::U64 => unsafe { Value::UInt(reg.int) }
            BareType::F32  => unsafe { Value::F32(reg.float32.0) }
            BareType::F64  => unsafe { Value::F64(reg.float64) }
            BareType::Bool => unsafe { Value::Bool(reg.b8.0) }
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

fn unary_op(op: TokenKind, ty: Type) -> Option<(Op, Type)> {
    match (op, bare(integer_promote(ty))) {
        (TokenKind::Sub,    BareType::Int) => Some((Op::IntNeg, Type::Int)),
        (TokenKind::BitNeg, BareType::Int) => Some((Op::BitNeg, Type::Int)),
        (TokenKind::Not,    _            ) => Some((Op::Not,    Type::Bool)),
        (TokenKind::Sub,    BareType::F32) => Some((Op::F32Neg, Type::F32)),
        (TokenKind::Sub,    BareType::F64) => Some((Op::F64Neg, Type::F64)),
        _ => None
    }
}

fn binary_op(op: TokenKind, left: Type, right: Type) -> Option<(Op, Type)> {
    let pointers = left.is_pointer() && right.is_pointer();
    let left = integer_promote(left);
    let right = integer_promote(right);
    if left != right && !pointers {
        return None;
    }
    match (op, bare(left)) {
        (TokenKind::Add,      BareType::Int) => Some((Op::IntAdd,  Type::Int)),
        (TokenKind::Sub,      BareType::Int) => Some((Op::IntSub,  Type::Int)),
        (TokenKind::Mul,      BareType::Int) => Some((Op::IntMul,  Type::Int)),
        (TokenKind::Div,      BareType::Int) => Some((Op::IntDiv,  Type::Int)),
        (TokenKind::Mod,      BareType::Int) => Some((Op::IntMod,  Type::Int)),
        (TokenKind::BitNeg,   BareType::Int) => Some((Op::BitNeg,  Type::Int)),
        (TokenKind::BitAnd,   BareType::Int) => Some((Op::BitAnd,  Type::Int)),
        (TokenKind::BitOr,    BareType::Int) => Some((Op::BitOr,   Type::Int)),
        (TokenKind::BitXor,   BareType::Int) => Some((Op::BitXor,  Type::Int)),
        (TokenKind::LShift,   BareType::Int) => Some((Op::LShift,  Type::Int)),
        (TokenKind::RShift,   BareType::Int) => Some((Op::RShift,  Type::Int)),
        (TokenKind::GtEq,     BareType::Int) => Some((Op::IntGtEq, Type::Bool)),
        (TokenKind::Lt,       BareType::Int) => Some((Op::IntLt,   Type::Bool)),
        (TokenKind::Gt,       BareType::Int) => Some((Op::IntGt,   Type::Bool)),
        (TokenKind::Eq,       BareType::Int) => Some((Op::IntEq,   Type::Bool)),
        (TokenKind::NEq,      BareType::Int) => Some((Op::IntNEq,  Type::Bool)),
        (TokenKind::LtEq,     BareType::Int) => Some((Op::IntLtEq, Type::Bool)),

        (TokenKind::LogicAnd, BareType::Bool) => Some((Op::LogicAnd, Type::Bool)),
        (TokenKind::LogicOr,  BareType::Bool) => Some((Op::LogicOr,  Type::Bool)),

        (TokenKind::Add,      BareType::F32) => Some((Op::F32Add,  Type::F32)),
        (TokenKind::Sub,      BareType::F32) => Some((Op::F32Sub,  Type::F32)),
        (TokenKind::Mul,      BareType::F32) => Some((Op::F32Mul,  Type::F32)),
        (TokenKind::Div,      BareType::F32) => Some((Op::F32Div,  Type::F32)),
        (TokenKind::Lt,       BareType::F32) => Some((Op::F32Lt,   Type::Bool)),
        (TokenKind::Gt,       BareType::F32) => Some((Op::F32Gt,   Type::Bool)),
        (TokenKind::Eq,       BareType::F32) => Some((Op::F32Eq,   Type::Bool)),
        (TokenKind::NEq,      BareType::F32) => Some((Op::F32NEq,  Type::Bool)),
        (TokenKind::LtEq,     BareType::F32) => Some((Op::F32LtEq, Type::Bool)),
        (TokenKind::GtEq,     BareType::F32) => Some((Op::F32GtEq, Type::Bool)),

        (TokenKind::Add,      BareType::F64) => Some((Op::F64Add,  Type::F64)),
        (TokenKind::Sub,      BareType::F64) => Some((Op::F64Sub,  Type::F64)),
        (TokenKind::Mul,      BareType::F64) => Some((Op::F64Mul,  Type::F64)),
        (TokenKind::Div,      BareType::F64) => Some((Op::F64Div,  Type::F64)),
        (TokenKind::Lt,       BareType::F64) => Some((Op::F64Lt,   Type::Bool)),
        (TokenKind::Gt,       BareType::F64) => Some((Op::F64Gt,   Type::Bool)),
        (TokenKind::Eq,       BareType::F64) => Some((Op::F64Eq,   Type::Bool)),
        (TokenKind::NEq,      BareType::F64) => Some((Op::F64NEq,  Type::Bool)),
        (TokenKind::LtEq,     BareType::F64) => Some((Op::F64LtEq, Type::Bool)),
        (TokenKind::GtEq,     BareType::F64) => Some((Op::F64GtEq, Type::Bool)),

        (TokenKind::GtEq, _)  if pointers    => Some((Op::IntGtEq, Type::Bool)),
        (TokenKind::Lt,   _)  if pointers    => Some((Op::IntLt,   Type::Bool)),
        (TokenKind::Gt,   _)  if pointers    => Some((Op::IntGt,   Type::Bool)),
        (TokenKind::Eq,   _)  if pointers    => Some((Op::IntEq,   Type::Bool)),
        (TokenKind::NEq,  _)  if pointers    => Some((Op::IntNEq,  Type::Bool)),
        (TokenKind::LtEq, _)  if pointers    => Some((Op::IntLtEq, Type::Bool)),

        _ => None
    }
}

fn is_address_computation(op: TokenKind, left: Type, right: Type) -> bool {
    use TokenKind::*;
    matches!(op, Add|Sub) && ((left.is_pointer() && right.is_integer()) || (left.is_integer() && right.is_pointer()))
}

fn is_offset_computation(op: TokenKind, left: Type, right: Type) -> bool {
    use TokenKind::*;
    matches!(op, Sub) && left == right && left.is_pointer() && right.is_pointer()
}

fn convert_op(from: Type, to: Type) -> Option<Op> {
    let from = integer_promote(from);
    let op = if from == to {
        Op::Noop
    } else if from == Type::Int {
        match bare(to) {
            BareType::Bool => Op::CmpZero,
            BareType::U8   => Op::MoveLower8,
            BareType::U16  => Op::MoveLower16,
            BareType::U32  => Op::MoveLower32,
            BareType::U64  => Op::Move,
            BareType::I8   => Op::MoveAndSignExtendLower8,
            BareType::I16  => Op::MoveAndSignExtendLower16,
            BareType::I32  => Op::MoveAndSignExtendLower32,
            BareType::I64  => Op::Move,
            BareType::F32  => Op::IntToFloat32,
            BareType::F64  => Op::IntToFloat64,
            _ if to.is_pointer() => Op::Noop,
            _ => return None
        }
    } else if from.is_pointer() {
        match bare(to) {
            BareType::U64|BareType::I64|BareType::Int => Op::Noop,
            _ if to.is_pointer() => Op::Noop,
            _ => return None
        }
    } else if to.is_integer() {
        match bare(from) {
            BareType::Bool => Op::Noop,
            BareType::F32  => Op::Float32ToInt,
            BareType::F64  => Op::Float64ToInt,
            _ => return None
        }
    } else {
        match (bare(from), bare(to)) {
            (BareType::F32, BareType::F64) => Op::Float32To64,
            (BareType::F64, BareType::F32) => Op::Float64To32,
            _ => return None
        }
    };
    Some(op)
}

fn store_op(ty: Type) -> Option<Op> {
    match bare(ty) {
        BareType::I8|BareType::U8|BareType::Bool                => Some(Op::Store8),
        BareType::I16|BareType::U16                             => Some(Op::Store16),
        BareType::I32|BareType::U32|BareType::F32               => Some(Op::Store32),
        BareType::Int|BareType::I64|BareType::U64|BareType::F64 => Some(Op::Store64),
        _ if ty.is_pointer()                                    => Some(Op::Store64),
        _ => None
    }
}

fn load_op(ty: Type) -> Option<Op> {
    match bare(ty) {
        BareType::Bool                                          => Some(Op::LoadBool),
        BareType::I8                                            => Some(Op::LoadAndSignExtend8),
        BareType::I16                                           => Some(Op::LoadAndSignExtend16),
        BareType::I32                                           => Some(Op::LoadAndSignExtend32),
        BareType::U8                                            => Some(Op::Load8),
        BareType::U16                                           => Some(Op::Load16),
        BareType::U32|BareType::F32                             => Some(Op::Load32),
        BareType::Int|BareType::I64|BareType::U64|BareType::F64 => Some(Op::Load64),
        _ if ty.is_pointer()                                    => Some(Op::Load64),
        _ => None
    }
}

// right now, before calling either apply_unary_op or apply_binary_op, you
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

#[derive(Clone, Copy, Debug)]
struct Local {
    loc: Location,
    ty: Type,
    bound_context: BoundContext
}

impl Local {
    fn new(loc: Location, ty: Type, bound_context: BoundContext) -> Local {
        let mut loc = loc;
        loc.is_place = true;
        Local { loc, ty, bound_context }
    }

    fn argument(loc: Location, ty: Type, number_of_arguments: isize) -> Local {
        let mut loc = loc;
        loc.is_place = true;
        Local { loc, ty, bound_context: BoundContext::Mark(Mark{ mode: Mode::RestrictedScope, index: number_of_arguments }) }
    }

    fn item(loc: Location, ty: Type, bound_context: BoundContext) -> Local {
        let mut loc = loc;
        loc.is_place = true;
        Local { loc, ty, bound_context }
    }
}

#[derive(Clone, Copy, Eq, PartialEq, Debug)]
enum Mode {
    All,
    Type,
    RestrictedScope,
}
impl Default for Mode { fn default() -> Mode { Mode::All }}

#[derive(Clone, Copy, Eq, PartialEq, Default, Debug)]
struct Mark {
    mode: Mode,
    index: isize
}

#[derive(Default)]
struct Locals {
    mark: Mark,
    local_values: Vec<Local>,
    local_keys: Vec<Intern>,
}

impl Locals {
    fn assert_invariants(&self) {
        debug_assert!(self.local_keys.len() == self.local_values.len());
        debug_assert!(self.mark.index <= self.local_values.len() as isize);
    }

    fn insert(&mut self, ident: Intern, local: Local) {
        self.assert_invariants();
        self.local_keys.push(ident);
        self.local_values.push(local);
    }

    fn get(&self, ident: Intern) -> Option<&Local> {
        self.assert_invariants();
        let start = if self.mark.mode == Mode::Type {
            self.mark.index as usize
        }  else {
            0
        };
        let end = if self.mark.mode == Mode::RestrictedScope {
            self.mark.index as usize
        } else {
            isize::MAX as usize
        };
        self.local_keys.iter()
            .take(end)
            .skip(start)
            .rposition(|v| *v == ident)
            .map(|i| &self.local_values[start + i])
    }

    fn push_scope(&mut self) -> Mark {
        self.assert_invariants();
        let result = self.mark;
        self.mark = self.top();
        result
    }

    fn mark_for_type(&self) -> Mark {
        let mut result = self.top();
        result.mode = Mode::Type;
        result
    }

    fn push_type(&mut self, ctx: &Compiler, ty: Type, base: &Location) -> Mark {
        self.assert_invariants();
        let result = self.push_scope();
        self.mark = self.mark_for_type();
        let info = ctx.types.info(ty);
        if info.kind == TypeKind::Callable {
            for item in &info.items {
                let loc = if item.ty.is_basic() {
                    base.offset_by(item.offset as isize)
                } else {
                    Location::pointer(base.base + item.offset as isize, 0)
                };
                self.insert(item.name, Local::item(loc, item.ty, BoundContext::Mark(self.mark)));
            }
        } else {
            assert!(info.kind == TypeKind::Struct);
            for item in &info.items {
                let loc = base.offset_by(item.offset as isize);
                self.insert(item.name, Local::item(loc, item.ty, BoundContext::Mark(self.mark)));
            }
        }
        result
    }

    fn restore_scope(&mut self, mark: Mark) {
        self.assert_invariants();
        debug_assert!(mark.index <= self.mark.index);
        self.local_values.truncate(self.mark.index as usize);
        self.local_keys.truncate(self.mark.index as usize);
        self.mark = mark;
    }

    fn restrict_to(&mut self, mark: Mark) -> Mark {
        self.assert_invariants();
        let result = self.mark;
        if mark.mode == Mode::Type {
            self.mark = mark;
        } else {
            assert!(mark.mode == Mode::All || mark.mode == Mode::RestrictedScope);
            self.mark = Mark { mode: Mode::RestrictedScope, ..mark };
        }
        result
    }

    fn clear_restriction(&mut self, mark: Mark) {
        self.assert_invariants();
        debug_assert!(mark.index <= self.mark.index);
        self.mark = mark;
    }

    fn transition_restriction_to_type_scope(&mut self, restriction: Mark) -> Mark {
        self.assert_invariants();
        assert!(self.mark.mode == Mode::RestrictedScope);
        self.mark.mode = Mode::Type;
        restriction
    }

    fn top(&self) -> Mark {
        Mark { mode: Mode::All, index: self.local_values.len() as isize }
    }
}

fn value_fits(value: Option<RegValue>, ty: Type) -> bool {
    unsafe {
        match (value, bare(ty)) {
            (Some(value), BareType::I8)  =>  i8::MIN as isize <= value.sint && value.sint <=  i8::MAX as isize,
            (Some(value), BareType::I16) => i16::MIN as isize <= value.sint && value.sint <= i16::MAX as isize,
            (Some(value), BareType::I32) => i32::MIN as isize <= value.sint && value.sint <= i32::MAX as isize,
            (Some(value), BareType::U8)  =>  u8::MIN as usize <= value.int  && value.int  <=  u8::MAX as usize,
            (Some(value), BareType::U16) => u16::MIN as usize <= value.int  && value.int  <= u16::MAX as usize,
            (Some(value), BareType::U32) => u32::MIN as usize <= value.int  && value.int  <= u32::MAX as usize,
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
        return dest.offset != Location::RETURN_REGISTER && value.ty.is_integer() && dest_ty.is_integer();
    }
    false
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum LocationKind {
    None,
    Control,
    Register,
    Based,
    Rip,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Location {
    base: isize,
    offset: isize,
    kind: LocationKind,
    is_place: bool
}

impl Location {
    pub const BAD: isize = isize::MIN;
    pub const RETURN_REGISTER: isize = 0;
    pub const RETURN_ADDRESS_REGISTER: isize = 1 * FatGen::REGISTER_SIZE as isize;
    pub const FP_REGISTER: isize = 2 * FatGen::REGISTER_SIZE as isize;
    pub const SP_REGISTER: isize = 3 * FatGen::REGISTER_SIZE as isize;
    pub const IP_REGISTER: isize = 4 * FatGen::REGISTER_SIZE as isize;

    fn none() -> Location {
        Location { base: Self::BAD, offset: Self::BAD, kind: LocationKind::None, is_place: false }
    }

    fn control() -> Location {
        Location { base: Self::BAD, offset: Self::BAD, kind: LocationKind::Control, is_place: false }
    }

    fn register(value: isize) -> Location {
        Location { base: Self::BAD, offset: value, kind: LocationKind::Register, is_place: false }
    }

    fn ret(is_basic: bool) -> Location {
        if is_basic {
            Location { base: Self::BAD, offset: Self::RETURN_REGISTER, kind: LocationKind::Register, is_place: false }
        } else {
            Location { base: Self::RETURN_REGISTER, offset: 0, kind: LocationKind::Based, is_place: false }
        }
    }

    fn stack(offset: isize) -> Location {
        Location { base: Self::SP_REGISTER, offset: offset, kind: LocationKind::Based, is_place: true }
    }

    fn rip(location: isize) -> Location {
        // Morally, base is Self::IP_REGISTER. But we don't use LocationKind::Based as
        // `offset` is interpreted differently for it, as we only know the value of RIP
        // when we actually emit code to acess this location. Using BAD here will make
        // things blow up (rather than emit bad loads) on misuse.
        Location { base: Self::BAD, offset: location, kind: LocationKind::Rip, is_place: true }
    }

    fn pointer(base: isize, offset: isize) -> Location {
        Location { base, offset, kind: LocationKind::Based, is_place: true }
    }

    fn offset_by(&self, offset: isize) -> Location {
        let mut result = *self;
        result.offset += offset;
        result
    }
}

#[derive(Clone, Copy, Debug)]
enum EvaluatedBound {
    Unconditional,
    Constant(RegValue),
    Location(Type, Location)
}

#[derive(Clone, Copy, Debug)]
enum BoundContext {
    None,
    Mark(Mark),
    Type(Type, Location),
}

#[derive(Clone, Copy, Debug)]
struct Destination {
    addr: Location,
    bound_context: BoundContext,
    ty: Type,
    // functionally identical, but set for different reasons: for calls and
    // compound inits, we emit bound checks after; for casts, we don't check.
    // This means calls/compounds "speculatively load" past bounds, which seems
    // bad, but is it actually?
    deferred_bound_checks: bool,
    supressed_bound_checks: bool,
}

impl Destination {
    fn none() -> Destination {
        Destination { ty: Type::None, addr: Location::none(), bound_context: BoundContext::None, deferred_bound_checks: false, supressed_bound_checks: false }
    }

    fn for_type(ty: Type, bound_context: BoundContext) -> Destination {
        Destination { ty, addr: Location::none(), bound_context, deferred_bound_checks: false, supressed_bound_checks: false }
    }

    fn for_location(ty: Type, addr: Location) -> Destination {
        Destination { ty, addr, bound_context: BoundContext::None, deferred_bound_checks: false, supressed_bound_checks: false }
    }

    fn for_item(ty: Type) -> Destination {
        Destination { ty, addr: Location::none(), bound_context: BoundContext::None, deferred_bound_checks: true, supressed_bound_checks: false }
    }

    fn for_item_with_location(ty: Type, addr: Location) -> Destination {
        Destination { ty, addr, bound_context: BoundContext::None, deferred_bound_checks: true, supressed_bound_checks: false }
    }

    fn for_return_type(ty: Type, bound_context: BoundContext) -> Destination {
        Destination { ty, addr: Location::ret(ty.is_basic()), bound_context, deferred_bound_checks: false, supressed_bound_checks: false }
    }

    fn of_expr_result(e: &ExprResult) -> Destination {
        Destination { ty: e.ty, addr: e.addr, bound_context: e.bound_context, deferred_bound_checks: false, supressed_bound_checks: false }
    }
}

#[derive(Clone, Copy, Debug)]
struct ExprResult {
    addr: Location,
    bound_context: BoundContext,
    ty: Type,
    expr: Expr,
    value_is_register: bool,
    is_call_result: bool,
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CodeGenTarget {
    Func,
    Proc,
    GlobalExpr,
    TypeExpr,
    TypeExprIgnoreBounds
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PanicReason {
    AssertionFailed,
    GlobalNotInitialized,
    IndexOutOfBounds
}

pub struct FatGen {
    pub code: Vec<FatInstr>,
    locals: Locals,
    reg_counter: isize,
    jmp: JumpContext,
    labels: Vec<InstrLocation>,
    patches: Vec<(Label, InstrLocation)>,
    generating_code_for: Type,
    return_type: Type,
    arguments_location: Location,
    arguments_mark: Mark,
    target: CodeGenTarget,
    panics: Vec<(Label, PanicReason)>,
    error: Option<Error>,
}

impl FatGen {
    pub const REGISTER_SIZE: usize = 8;
    pub fn new(code: Vec<FatInstr>) -> FatGen {
        FatGen {
            code,
            locals: Locals::default(),
            reg_counter: 0,
            jmp: JumpContext::default(),
            labels: Vec::new(),
            patches: Vec::new(),
            generating_code_for: Type::None,
            return_type: Type::None,
            arguments_location: Location::none(),
            arguments_mark: Mark::default(),
            target: CodeGenTarget::Func,
            panics: Vec::new(),
            error: None,
        }
    }

    fn can_emit_code(&self) -> bool {
        matches!(self.target, CodeGenTarget::Func|CodeGenTarget::Proc|CodeGenTarget::GlobalExpr)
    }

    fn error(&mut self, source_position: usize, msg: String) {
        if self.error.is_none() {
            self.error = Some(Error { source_position, msg });
        }
    }

    fn inc_bytes(&mut self, size: usize, align: usize) -> isize {
        debug_assert!(align.is_power_of_two());
        let result = align_up(self.reg_counter as usize, align) as isize;
        self.reg_counter = result + size as isize;
        // Todo: error if size does not fit on the stack
        assert!(self.reg_counter < i32::MAX as isize);
        result
    }

    fn inc_reg(&mut self) -> isize {
        self.inc_bytes(Self::REGISTER_SIZE, Self::REGISTER_SIZE)
    }

    fn put3(&mut self, op: Op, dest: isize, left: isize, right: isize) {
        if self.can_emit_code() {
            assert_implies!(requires_register_destination(op), (dest as usize & (Self::REGISTER_SIZE - 1)) == 0);
            let dest = i32::try_from(dest).unwrap_or_else(|_| { error!(self, 0, "stack is limited to range addressable by i32"); 0 });
            let left = i32::try_from(left).unwrap_or_else(|_| { assert!(self.error.is_some()); 0 });
            let right = i32::try_from(right).unwrap_or_else(|_| { assert!(self.error.is_some()); 0 });
            self.code.push(FatInstr { op, dest, left, right });
        } else if disallowed_in_constant_expression(op) {
            // Todo: expr position
            error!(self, 0, "non-constant expression");
        }
    }

    fn put(&mut self, op: Op, dest: isize, data: RegValue) {
        if self.can_emit_code() {
            assert_implies!(requires_register_destination(op), (dest as usize & (Self::REGISTER_SIZE - 1)) == 0);
            let dest = i32::try_from(dest).unwrap_or_else(|_| { error!(self, 0, "stack is limited to range addressable by i32"); 0 });
            self.code.push(FatInstr { op, dest, left: unsafe { data.sint32.0 }, right: unsafe { data.sint32.1 }});
        } else if disallowed_in_constant_expression(op) {
            // Todo: expr position
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

    fn location_address(&mut self, location: &Location) -> isize {
        match location.kind {
            LocationKind::None => Location::BAD,
            LocationKind::Control => Location::BAD,
            LocationKind::Register => location.offset,
            LocationKind::Based => {
                if location.offset != 0 {
                    let offset_register = self.constant(location.offset.into());
                    self.put3_inc(Op::IntAdd, location.base, offset_register)
                } else {
                    location.base
                }
            }
            LocationKind::Rip => {
                // Add 1 to include the immediate load. If self.constant ever
                // tracks available constant registers then this will break
                let offset = location.offset - ((self.code.len() + 1) * std::mem::size_of::<FatInstr>()) as isize;
                let offset_register = self.constant(offset.into());
                self.put3_inc(Op::IntAdd, Location::IP_REGISTER, offset_register)
            }
        }
    }

    #[allow(unreachable_code)]
    fn location_base_offset(&mut self, location: &Location) -> (isize, isize) {
        if location.offset > i32::MAX as isize {
            panic!("untested");
            return (self.location_address(location), 0);
        }
        match location.kind {
            LocationKind::Based => {
                (location.base, location.offset)
            }
            LocationKind::Rip => {
                let offset = location.offset - (self.code.len() * std::mem::size_of::<FatInstr>()) as isize;
                (Location::IP_REGISTER, offset)
            }
            _ => unreachable!()
        }
    }

    fn location_value(&mut self, ty: Type, addr: &Location) -> isize {
        match addr.kind {
            LocationKind::None => Location::BAD,
            LocationKind::Control => unreachable!(),
            LocationKind::Register => addr.offset,
            LocationKind::Based|LocationKind::Rip => {
                if let Some(op) = load_op(ty) {
                    let (ptr, offset) = self.location_base_offset(&addr);
                    self.put3_inc(op, ptr, offset)
                } else {
                    Location::BAD
                }
            }
        }
    }

    fn register(&mut self, expr: &ExprResult) -> isize {
        // Todo: apparently, it's load bearing whether we check the addr first
        // or the constant value first. Why???
        match expr.addr.kind {
            LocationKind::None => match expr.value {
                Some(v) if expr.value_is_register => {
                    assert!(expr.ty.is_pointer());
                    let v_register = self.constant(v);
                    self.put3_inc(Op::IntAdd, Location::SP_REGISTER, v_register)
                },
                Some(v) => self.constant(v),
                None => Location::BAD
            },
            _ => self.location_value(expr.ty, &expr.addr)
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
            LocationKind::Based|LocationKind::Rip => {
                let dest_addr_register = self.location_address(dest);
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
            LocationKind::Based|LocationKind::Rip => {
                let src = &self.emit_constant(src);

                match src.addr.kind {
                    LocationKind::None|LocationKind::Control => unreachable!(),
                    LocationKind::Register => {
                        if let Some(op) = store_op(destination_type) {
                            let (dest_addr_register, offset) = self.location_base_offset(dest);
                            let src = self.register(src);
                            self.put3(op, dest_addr_register, src, offset);
                        } else {
                            unreachable!();
                        }
                    }
                    LocationKind::Based|LocationKind::Rip => {
                        let dest_addr_register = self.location_address(dest);
                        let src_addr_register = self.location_address(&src.addr);
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
        if self.can_emit_code() {
            let result = Label(self.labels.len());
            self.labels.push(InstrLocation(!0));
            result
        } else {
            // Todo: expr position
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
                debug_assert!(offset != 0, "zero offset {:?}", self.code[from].op);
            }
        }
        self.patches.clear();
    }

    fn type_expr(&mut self, ctx: &mut Compiler, expr: TypeExpr) -> Type {
        self.type_expr_really(ctx, expr, true, false)
    }

    fn type_expr_ptr(&mut self, ctx: &mut Compiler, expr: TypeExpr) -> Type {
        self.type_expr_really(ctx, expr, false, true)
    }

    fn type_expr_really(&mut self, ctx: &mut Compiler, expr: TypeExpr, outermost: bool, ptr: bool) -> Type {
        let stashed_target = self.target;
        if !matches!(self.target, CodeGenTarget::TypeExpr|CodeGenTarget::TypeExprIgnoreBounds) {
            self.target = CodeGenTarget::TypeExpr;
        }
        let mut result = Type::None;
        match *ctx.ast.type_expr(expr) {
            TypeExprData::Infer => unreachable!(),
            TypeExprData::Name(name) => {
                if name.0 == Keytype::Ptr as u32 {
                    result = Type::VoidPtr;
                } else if ptr {
                    result = sym::touch_type(ctx, name);
                } else {
                    result = sym::resolve_type(ctx, name);
                }
            }
            TypeExprData::Expr(_) => todo!(),
            TypeExprData::Items(items) => result = sym::resolve_anonymous_struct(ctx, items),
            TypeExprData::List(_) => {
                match ctx.ast.type_expr_keytype(expr) {
                    Some(Keytype::Arr) => {
                        let ty_expr = ctx.ast.type_expr_base_type(expr);
                        let len_expr = ctx.ast.type_expr_bound(expr);
                        // Todo: infer if ty_expr = None?
                        if let (Some(ty_expr), Some(len_expr)) = (ty_expr, len_expr) {
                            let ty = self.type_expr_really(ctx, ty_expr, false, ptr);
                            if let TypeExprData::Expr(len_expr) = *ctx.ast.type_expr(len_expr) {
                                let len = self.constant_expr(ctx, len_expr);
                                if len.ty.is_integer() && len.value.is_some() {
                                    if let Some(value) = len.value.and_then(|v| v.to_positive_isize(len.ty)) {
                                        result = ctx.types.array(ty, value as usize);
                                    } else {
                                        error!(self, 0, "arr length must be positive integer and fit in a signed integer");
                                    }
                                } else {
                                    // Todo: type expr location
                                    error!(self, 0, "arr length must be a constant integer")
                                }
                            } else {
                                error!(self, 0, "argument 2 of arr type must be a value expression")
                            }
                        } else {
                            error!(self, 0, "type arr takes 2 arguments, base type and [length]")
                        }
                        if ctx.ast.type_expr_len(expr) >= 4 {
                            error!(ctx, 0, "type arr takes 2 arguments, base type and [length]");
                        }
                    }
                    Some(Keytype::Ptr) => {
                        if let Some(ty_expr) = ctx.ast.type_expr_base_type(expr) {
                            let ty = self.type_expr_ptr(ctx, ty_expr);
                            if let Some(bound_expr) = ctx.ast.type_expr_bound(expr) {
                                if !outermost {
                                    error!(self, 0, "ptr bounds apply only to the outermost type");
                                } else if self.target == CodeGenTarget::TypeExpr {
                                    if let TypeExprData::Expr(bound_expr) = *ctx.ast.type_expr(bound_expr) {
                                        let expr_result = self.constant_expr(ctx, bound_expr);
                                        if expr_result.ty.is_integer() {
                                            assert!(expr_result.value_is_register == false, "unimplemented/untested");
                                            let mut bound = Bound::Constant(1);
                                            match expr_result.value {
                                                Some(v) => {
                                                    if let Some(v) = v.to_nonnegative_isize(expr_result.ty) {
                                                        bound = Bound::Constant(v as usize)
                                                    } else {
                                                        // We could lift this restriction, but why?
                                                        error!(ctx, 0, "constant ptr bounds must be nonnegative");
                                                    }
                                                }
                                                None => {
                                                    bound = Bound::Expr(bound_expr);
                                                }
                                            }
                                            result = ctx.types.bound_pointer(ty, bound)
                                        } else {
                                            // Todo: type expr location
                                            error!(self, 0, "ptr bound must be an integer")
                                        }
                                    } else {
                                        error!(self, 0, "argument 2 of ptr type must be a value expression")
                                    }
                                } else {
                                    assert!(self.target == CodeGenTarget::TypeExprIgnoreBounds);
                                    result = ctx.types.pointer(ty);
                                    result.bound = Bound::Compiling;
                                }
                            } else {
                                // unbound
                                result = ctx.types.pointer(ty);
                            }
                        } else {
                            // untyped
                            result = Type::VoidPtr;
                        }
                        if ctx.ast.type_expr_len(expr) >= 4 {
                            error!(ctx, 0, "too many parameters for keytype ptr");
                        }
                    }
                    Some(Keytype::Struct) => {
                        if let Some(items) = ctx.ast.type_expr_items(expr) {
                            result = sym::resolve_anonymous_struct(ctx, items);
                        } else {
                            error!(self, 0, "struct has no fields");
                        }
                        if ctx.ast.type_expr_len(expr) >= 3 {
                            error!(ctx, 0, "too many parameters for keytype struct");
                        }
                    }
                    Some(Keytype::Func)|Some(Keytype::Proc) => error!(self, 0, "func/proc pointers are not implemented"),
                    None => error!(self, 0, "expected keytype (one of func, proc, struct, arr, ptr)")
                }
            }
        }
        self.target = stashed_target;
        result
    }

    fn path(&mut self, ctx: &Compiler, path_ctx: &mut PathContext, ty: Type, path: CompoundPath) -> Option<types::Item> {
        match path {
            CompoundPath::Implicit => {
                if let PathContextState::ExpectAny = path_ctx.state {
                    path_ctx.state = PathContextState::ExpectImplict;
                    assert!(path_ctx.index == 0);
                }
                if let PathContextState::ExpectImplict = path_ctx.state {
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
                if let PathContextState::ExpectAny = path_ctx.state {
                    path_ctx.state = PathContextState::ExpectPaths;
                }
                if let PathContextState::ExpectPaths = path_ctx.state {
                    if let ExprData::Name(name) = ctx.ast.expr(path) {
                        let result = ctx.types.item_info(ty, name);
                        if let None = result {
                            error!(self, ctx.ast.expr_source_position(path), "no field '{}' on type {}", ctx.str(name), ctx.type_str(ty));
                        }
                        result
                    } else {
                        // Todo: Support more.complicated[0].paths
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

    fn bound(&mut self, ctx: &mut Compiler, bound: Bound, bound_context: &BoundContext) -> EvaluatedBound {
        if let Bound::Compiling = bound {
            assert!(matches!(self.target, CodeGenTarget::TypeExpr|CodeGenTarget::TypeExprIgnoreBounds));
            return EvaluatedBound::Unconditional;
        }
        if let BoundContext::None = bound_context {
            return EvaluatedBound::Unconditional;
        }

        match bound {
            Bound::Compiling => EvaluatedBound::Unconditional,
            Bound::Single => EvaluatedBound::Constant(1.into()),
            Bound::Constant(n) => EvaluatedBound::Constant(n.into()),
            Bound::Expr(expr) => match bound_context {
                BoundContext::None => unreachable!(),
                BoundContext::Mark(mark) => {
                    let restriction = self.locals.restrict_to(*mark);
                    let bound = self.expr(ctx, expr);
                    assert_implies!(self.error.is_none(), bound.ty.is_integer());
                    assert!(bound.value.is_none());
                    self.locals.clear_restriction(restriction);
                    EvaluatedBound::Location(bound.ty, bound.addr)
                },
                BoundContext::Type(ty, addr) => {
                    let mark = if ty.is_pointer() {
                        let base_ty = ctx.types.base_type(*ty);
                        let base_addr = Location::pointer(self.location_address(&addr), 0);
                        self.locals.push_type(ctx, base_ty, &base_addr)
                    } else {
                        self.locals.push_type(ctx, *ty, addr)
                    };
                    let bound = self.expr(ctx, expr);
                    assert_implies!(self.error.is_none(), bound.ty.is_integer());
                    assert!(bound.value.is_none());
                    self.locals.restore_scope(mark);
                    EvaluatedBound::Location(bound.ty, bound.addr)
                }
            }
        }
    }

    fn destination_bounds_check(&mut self, ctx: &mut Compiler, destination: &Destination, expr: &ExprResult) {
        match destination.ty.bound {
            Bound::Compiling => unreachable!(),
            Bound::Single => (),
            Bound::Constant(dest_n) => {
                match self.bound(ctx, expr.ty.bound, &expr.bound_context) {
                    EvaluatedBound::Unconditional => (),
                    EvaluatedBound::Constant(v) => {
                        let expr_n = unsafe { v.int };
                        if dest_n > expr_n {
                            // todo: string repr of the expr, dest
                            error!(self, ctx.ast.expr_source_position(expr.expr), "bounds do not match: have {}:{}, but destination wants {}:{}", expr_n, ctx.type_str(expr.ty), dest_n, ctx.type_str(destination.ty))
                        }
                    }
                    EvaluatedBound::Location(ty, addr) => {
                        let dest = self.constant(dest_n.into());
                        let src = self.location_value(ty, &addr);
                        let in_bounds = self.put3_inc(Op::IntLtEq, dest, src);
                        self.assert(in_bounds, PanicReason::IndexOutOfBounds);
                    }
                }
            }
            Bound::Expr(_) => {
                let dest = self.bound(ctx, destination.ty.bound, &destination.bound_context);
                let src = self.bound(ctx, expr.ty.bound, &expr.bound_context);
                if let EvaluatedBound::Location(dest_ty, dest_addr) = dest {
                    if let EvaluatedBound::Location(src_ty, src_addr) = src {
                        if dest_ty == src_ty && dest_addr == src_addr {
                            return;
                        }
                    }
                }
                let dest = match dest {
                    EvaluatedBound::Unconditional => unreachable!(),
                    EvaluatedBound::Constant(v) => self.constant(v),
                    EvaluatedBound::Location(ty, addr) => self.location_value(ty, &addr),
                };
                let src = match src {
                    EvaluatedBound::Unconditional => unreachable!(),
                    EvaluatedBound::Constant(v) => self.constant(v),
                    EvaluatedBound::Location(ty, addr) => self.location_value(ty, &addr),
                };
                assert!(dest != src);
                let in_bounds = self.put3_inc(Op::IntLtEq, dest, src);
                self.assert(in_bounds, PanicReason::IndexOutOfBounds);
            }
        }
    }

    fn bounds_check(&mut self, index_register: isize, bound_register: isize) {
        if self.can_emit_code() {
            let in_bounds = self.put3_inc(Op::IntLt, index_register, bound_register);
            self.assert(in_bounds, PanicReason::IndexOutOfBounds);
        }
    }

    fn compute_address(&mut self, ctx: &mut Compiler, base: &ExprResult, index: &ExprResult, bound_context: &BoundContext) -> Location {
        let base_type = ctx.types.base_type(base.ty);
        let element_size = ctx.types.info(base_type).size;
        let bound = self.bound(ctx, base.ty.bound, bound_context);
        let addr = match index.value {
            Some(value) => {
                let index = unsafe { value.sint };
                match bound {
                    EvaluatedBound::Unconditional => (),
                    EvaluatedBound::Constant(v) => {
                        let bound = unsafe { v.sint };
                        if bound < 0 {
                            panic!("negative bound should not have reached here?");
                        } else if index < 0 {
                            // todo: index position and text representation
                            error!(self, 0, "index is less than zero. array accesses using [] must be non-negative");
                        } else if !(index < bound) {
                            // todo: index position and text representation
                            error!(self, 0, "constant index {} is out of bound {}", index, bound)
                        }
                    },
                    EvaluatedBound::Location(ty, addr) => {
                        let index_register = self.constant(value);
                        let bound_register = self.location_value(ty, &addr);
                        self.bounds_check(index_register, bound_register);
                    }
                }
                let offset = index * element_size as isize;
                if base.ty.is_pointer() {
                    let ptr = self.register(&base);
                    Location::pointer(ptr, offset)
                } else {
                    base.addr.offset_by(offset)
                }
            }
            None => {
                let index_register = self.register(&index);
                match bound {
                    EvaluatedBound::Unconditional => (),
                    EvaluatedBound::Constant(value) => {
                        let bound_register = self.constant(value);
                        self.bounds_check(index_register, bound_register);
                    },
                    EvaluatedBound::Location(ty, addr) => {
                        let bound_register = self.location_value(ty, &addr);
                        self.bounds_check(index_register, bound_register);
                    }
                }
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
                let size_register = self.constant(element_size.into());
                let offset_register = self.put3_inc(Op::IntMul, index_register, size_register);
                if base.ty.is_pointer() {
                    let ptr = self.register(&base);
                    let base_reg = self.put3_inc(Op::IntAdd, ptr, offset_register);
                    Location::pointer(base_reg, 0)
                } else {
                    match base.addr.kind {
                        LocationKind::Based => {
                            let old_base_reg = base.addr.base;
                            let base_reg = self.put3_inc(Op::IntAdd, old_base_reg, offset_register);
                            // As per comment above, we keep base.addr.offset
                            Location { base: base_reg, ..base.addr }
                        }
                        LocationKind::Rip => {
                            todo!();
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
        //
        // Constant expressions will not emit code when evaluated here for type
        // checking, but may be evaluated later in a non-constant context where
        // they can emit code. For example
        //      len := 3;
        //      p: ptr u8 [len + 1] = ...;
        // When generating code for the declaration of p, we don't emit code to
        // compute len + 1, but when p is accessed and bound checked, we will.
        //
        // But we don't track whether this re-evaluation is necessary. So,
        // calling code can't trust the ExprResult to have a valid location that
        // can be used for codegen. But currently we don't use constant
        // expressions for anything but type exprs, which don't use ExprResult
        // locations that way.
        assert!(matches!(self.target, CodeGenTarget::TypeExpr|CodeGenTarget::TypeExprIgnoreBounds));

        let code_len = self.code.len();
        let labels_len = self.labels.len();
        let result = self.expr(ctx, expr);
        if let None = result.value {
            // This means the constant expr has not evaluated to a constant. But
            // it might have evaluated to a memory location that the compiler
            // knows (e.g., a variables relative position on the stack), and we
            // want to support those in type exprs, so calling code has to expect this
        } else {
            debug_assert!(self.code.len() == code_len);
            debug_assert!(self.labels.len() == labels_len);
        }
        result
    }


    fn assert(&mut self, condition_register: isize, reason: PanicReason) {
        let label = self.label();
        self.panics.push((label, reason));
        self.put_jump_zero(condition_register, label);
    }

    fn expr(&mut self, ctx: &mut Compiler, expr: Expr) -> ExprResult {
        self.expr_with_destination_and_optional_control(ctx, expr, &Destination::none(), None)
    }

    fn expr_with_control(&mut self, ctx: &mut Compiler, expr: Expr, control: Control) -> ExprResult {
        self.expr_with_destination_and_optional_control(ctx, expr, &Destination::none(), Some(control))
    }

    fn expr_with_destination_type(&mut self, ctx: &mut Compiler, expr: Expr, destination_type: Type) -> ExprResult {
        self.expr_with_destination_and_optional_control(ctx, expr, &Destination::for_type(destination_type, BoundContext::Mark(self.locals.top())), None)
    }

    fn expr_with_destination(&mut self, ctx: &mut Compiler, expr: Expr, destination: &Destination) -> ExprResult {
        self.expr_with_destination_and_optional_control(ctx, expr, destination, None)
    }

    fn expr_with_destination_and_control(&mut self, ctx: &mut Compiler, expr: Expr, destination: &Destination, control: Control) -> ExprResult {
        self.expr_with_destination_and_optional_control(ctx, expr, destination, Some(control))
    }

    fn expr_with_destination_and_optional_control(&mut self, ctx: &mut Compiler, expr: Expr, destination: &Destination, control: Option<Control>) -> ExprResult {
        let mut control = control;
        let mut result = ExprResult { addr: Location::none(), bound_context: BoundContext::None, ty: Type::None, expr, value_is_register: false, is_call_result: true, value: None };
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
                    result.ty = local.ty;
                    result.bound_context = local.bound_context;
                    result.addr = local.loc;
                } else if let Some(sym) = sym::lookup_value(ctx, name) {
                    result.ty = sym.ty;
                    result.bound_context = BoundContext::Mark(Mark { mode: Mode::RestrictedScope, index: self.arguments_mark.index });
                    match ctx.types.info(sym.ty).kind {
                        TypeKind::Callable => result.value = Some(sym.name.into()),
                        _ => {
                            // Global variable access. Our strategy for dealing with these is:
                            // 1) Storage is allocated in the declaration type resolution phase
                            // 2 a) Code to compute every global variable is dumped out linearly,
                            //      this emits writes to the allocation for each global.
                            // 2 b) inside global expressions, if we hit a global variable we
                            //      haven't generated code for yet, immediately switch to generating
                            //      code for that.
                            // 3) inside functions and procedures, which can be called before main, emit
                            //    checks that they have been initialized. The downside of this is that
                            //    indeterminism means illegal accesses can be missed if they are
                            //    branched over. Instead, this could can be done as a scan over the
                            //    bytecode. I don't want to implement logic for that until I know the
                            //    bytecode representation is ok.
                            // 4) Stick a call to main at the end of the code from 2. This is the compiled
                            //    program.
                            let (ty, loc) = (sym.ty, sym.location);
                            match sym.state {
                                sym::State::Resolved => {
                                    assert!(matches!(self.target, CodeGenTarget::GlobalExpr));
                                    if let DeclData::Var(VarDecl { value, .. }) = *ctx.ast.decl(sym.decl) {
                                        debug_assert!(self.reg_counter != 0);
                                        sym::compiling(ctx, name);
                                        self.global_expr(ctx, ty, value, loc, self.reg_counter);
                                        sym::compiled(ctx, name);
                                    } else {
                                        unreachable!();
                                    }
                                }
                                sym::State::Compiling|sym::State::Circular => {
                                    error!(self, ctx.ast.expr_source_position(expr), "definition of {} is circular", ctx.str(name));
                                }
                                sym::State::Compiled => {
                                    // Code to init variable has already been emit, so we don't have to do anything
                                },
                                _ => unreachable!()
                            }

                            if let CodeGenTarget::Func|CodeGenTarget::Proc = self.target {
                                // We emit this check every time we access any global, because if the function is called
                                // in a global expression it may be uninitialized. This is a heavy hammer that lets us
                                // avoid doing any reachability analysis.
                                let (initialized_flag_ptr, offset) = self.location_base_offset(&Location::rip(loc - 4));
                                let initialized_flag = self.put3_inc(Op::Load32, initialized_flag_ptr, offset);
                                self.assert(initialized_flag, PanicReason::GlobalNotInitialized);
                            }

                            result.addr = Location::rip(loc);
                        }
                    }
                } else {
                    error!(self, ctx.ast.expr_source_position(expr), "unknown identifier '{}'", ctx.str(name));
                }
            }
            ExprData::Compound(fields) => {
                if destination.ty != Type::None {
                    // Todo: check for duplicate fields. need to consider unions and fancy paths (both unimplemented)
                    let info = ctx.types.info(destination.ty);
                    let kind = info.kind;
                    if let TypeKind::Struct|TypeKind::Array = kind {
                        let base;
                        if fields.is_empty() && destination.addr.kind != LocationKind::None {
                            base = destination.addr;
                        } else {
                            // Todo: We dump everything to the stack and copy over because we don't know yet
                            // if the compound initializer is loading from the destination it is storing
                            // to. But this seems like a rare case to me, and we ought to be able to write
                            // directly to the destination in the common case, even without optimisations?
                            base = self.stack_alloc(ctx, destination.ty);
                        }
                        let mut path_ctx = PathContext { state: PathContextState::ExpectAny, index: 0 };
                        let mut field_exprs = SmallVec::new();
                        self.zero(&base, info.size as isize);
                        for field in fields {
                            let field = ctx.ast.compound_field(field);
                            if let Some(item) = self.path(ctx, &mut path_ctx, destination.ty, field.path) {
                                let dest = base.offset_by(item.offset as isize);
                                let expr = self.expr_with_destination(ctx, field.value, &Destination::for_item_with_location(item.ty, dest));
                                if expr.ty == item.ty {
                                    field_exprs.push((item.name, expr));

                                    if let Bound::Expr(_) = item.ty.bound {
                                        assert!(expr.ty.bound != item.ty.bound, "original bound for expression has been lost, but we're deferring evaluation");
                                    }
                                } else {
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
                        if let TypeKind::Struct = kind {
                            let mark = self.locals.push_type(ctx, destination.ty, &base);
                            for &(item, expr) in field_exprs.iter() {
                                let local = self.locals.get(item).expect("local scope is missing item in bound context");
                                let dest = &Destination::for_type(local.ty, local.bound_context);
                                self.destination_bounds_check(ctx, &dest, &expr);
                            }
                            self.locals.restore_scope(mark);
                        }
                        result.addr = base;
                        result.ty = destination.ty;
                    } else {
                        error!(self, ctx.ast.expr_source_position(expr), "compound initializer used for non-aggregate type");
                    }
                } else {
                    error!(self, ctx.ast.expr_source_position(expr), "untyped compound initializer");
                }
            }
            ExprData::Field(left_expr, field) => {
                let left = self.expr(ctx, left_expr);
                result.bound_context = BoundContext::Type(left.ty, left.addr);
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
                                result.addr = Location::pointer(base, item.offset as isize);
                            }
                        }
                        result.ty = item.ty;
                    } else {
                        error!(self, ctx.ast.expr_source_position(left_expr), "no field '{}' on type {}", ctx.str(field), ctx.type_str(left.ty));
                    }
                } else if let Some(item) = ctx.types.item_info(left.ty, field) {
                    debug_assert_implies!(self.error.is_none(), matches!(left.addr.kind, LocationKind::Based|LocationKind::Rip));
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
                if let TypeKind::Array|TypeKind::Pointer = ctx.types.info(left.ty).kind {
                    if left.ty.bound != Bound::Single {
                        let index = self.expr(ctx, index_expr);
                        if index.ty.is_integer() {
                            result.addr = self.compute_address(ctx, &left, &index, &left.bound_context);
                            result.ty = ctx.types.base_type(left.ty);
                        } else {
                            error!(self, ctx.ast.expr_source_position(index_expr), "index must be an integer (found {})", ctx.type_str(index.ty))
                        }
                    } else {
                        error!(self, ctx.ast.expr_source_position(left_expr), "cannot index unbounded pointers. You can cast to a bound pointer or use pointer arithmetic and explicit dereferences");
                    }
                } else {
                    error!(self, ctx.ast.expr_source_position(left_expr), "indexed type must be an array or pointer (found {}, a {:?})", ctx.type_str(left.ty), ctx.types.info(left.ty).kind)
                }
            },
            ExprData::Unary(op_token, right_expr) => {
                let right = if op_token == TokenKind::BitAnd && destination.ty != Type::None {
                    // Address/cast/compound transposition
                    self.expr_with_destination_type(ctx, right_expr, ctx.types.base_type(destination.ty))
                } else {
                    self.expr(ctx, right_expr)
                };
                use TokenKind::*;
                match op_token {
                    BitAnd => {
                        if right.addr.is_place {
                            match right.addr.kind {
                                LocationKind::None =>
                                    // This happens when doing offset-of type address calculations off of constant 0 pointers
                                    result = right,
                                LocationKind::Control => unreachable!(),
                                LocationKind::Register => {
                                    let base = right.addr.offset;
                                    result.value_is_register = true;
                                    result.value = Some(base.into());
                                    // Todo: seems weird to be setting these with kind == None
                                    result.addr.is_place = right.addr.is_place;
                                }
                                LocationKind::Based|LocationKind::Rip => {
                                    debug_assert!(right.addr.is_place);
                                    result.addr = Location::register(self.location_address(&right.addr));
                                }
                            }
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
                                    result.addr = Location::pointer(right_register, 0);
                                }
                            }
                            result.addr.is_place = true;
                            result.ty = ctx.types.base_type(right.ty);
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
                match (control, op_token) {
                    (Some(c), TokenKind::LogicAnd) => {
                        let next = self.label();
                        left = self.expr_with_control(ctx, left_expr, fall_true(next, c.false_to));
                        self.patch(next);
                        right = self.expr_with_control(ctx, right_expr, c);
                        emit = false;
                    }
                    (Some(c), TokenKind::LogicOr) => {
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
                    let where_u_at = self.compute_address(ctx, ptr, offset, &BoundContext::None);
                    result.addr = Location::register(self.location_address(&where_u_at));
                    result.ty = ptr.ty;
                } else if is_offset_computation(op_token, left.ty, right.ty) {
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
                    if left.ty != right.ty {
                        error!(self, ctx.ast.expr_source_position(left_expr), "incompatible types ({} {} {})", ctx.type_str(left.ty), op_token, ctx.type_str(right.ty));
                    } else {
                        error!(self, ctx.ast.expr_source_position(left_expr), "cannot use the {} operator with {}", op_token, ctx.type_str(right.ty));
                    }
                }
            }
            ExprData::Ternary(cond_expr, left_expr, right_expr) => {
                let left = self.label();
                let right = self.label();
                let cond = self.expr_with_control(ctx, cond_expr, fall_true(left, right));
                if cond.ty == Type::Bool {
                    let exit = self.label();
                    self.patch(left);
                    let left = self.expr_with_destination_and_control(ctx, left_expr, destination, jump(exit));
                    self.patch(right);
                    let right = self.expr_with_destination(ctx, right_expr, &Destination::of_expr_result(&left));
                    self.patch(exit);
                    if left.ty == right.ty {
                        debug_assert!(left.addr == right.addr);
                        result = left;
                    } else {
                        error!(self, ctx.ast.expr_source_position(left_expr), "incompatible types (... ? {}, {})", ctx.type_str(left.ty), ctx.type_str(right.ty));
                    }
                } else {
                    error!(self, ctx.ast.expr_source_position(cond_expr), "ternary expression requires a boolean condition");
                }
            }
            ExprData::Call(callable, args) => {
                let addr = self.expr(ctx, callable);
                let info = ctx.types.info(addr.ty);
                if info.kind == TypeKind::Callable {
                    if info.items.len() == args.len() + 1 {
                        let not_calling_proc_from_func = !(self.target == CodeGenTarget::Func && info.mutable);
                        if not_calling_proc_from_func {
                            let return_type = info.return_type().expect("tried to generate code to call a func without return type");
                            let dest_location = if return_type.is_basic() == false {
                                if return_type == destination.ty
                                && destination.addr.kind != LocationKind::None {
                                    // For aggregate types, if we already have a destination, we can just pass the
                                    // pointer in as the return destination directly
                                    destination.addr
                                } else {
                                    // Otherwise, use the stack
                                    let info = ctx.types.info(return_type);
                                    let dest = self.inc_bytes(info.size, info.alignment);
                                    Location::stack(dest)
                                }
                            } else {
                                Location::none()
                            };
                            let dest_ptr = if return_type.is_basic() {
                                Location::BAD
                            } else {
                                self.location_address(&dest_location)
                            };
                            let mut arg_exprs = SmallVec::new();
                            let isolated_scope_for_bound_checks = self.locals.restrict_to(self.locals.top());
                            let items_mark = self.locals.mark_for_type();
                            for (i, expr) in args.enumerate() {
                                let info = ctx.types.info(addr.ty);
                                let item = info.items[i];
                                let compiled_expr = self.expr_with_destination(ctx, expr, &Destination::for_item(item.ty));
                                if compiled_expr.ty != item.ty {
                                    error!(self, ctx.ast.expr_source_position(expr), "argument {} of {} is of type {}, found {}", i, ctx.callable_str(ident(addr.value)), ctx.type_str(item.ty), ctx.type_str(compiled_expr.ty));
                                    break;
                                }
                                let reg = if compiled_expr.ty.is_basic() {
                                    if compiled_expr.value.is_none() {
                                        self.register(&compiled_expr)
                                    } else {
                                        Location::BAD
                                    }
                                } else {
                                    self.location_address(&compiled_expr.addr)
                                };
                                arg_exprs.push((compiled_expr, item.name, reg));

                                self.locals.insert(item.name, Local::item(compiled_expr.addr, item.ty, BoundContext::Mark(items_mark)));
                            }
                            let mark = self.locals.transition_restriction_to_type_scope(isolated_scope_for_bound_checks);
                            for &(expr, item, _) in arg_exprs.iter() {
                                let local = self.locals.get(item).expect("local scope is missing item in bound context");
                                let dest = &Destination::for_type(local.ty, local.bound_context);
                                self.destination_bounds_check(ctx, &dest, &expr);
                            }
                            self.locals.restore_scope(mark);
                            let arguments_start = self.reg_counter;
                            for &(expr, _, reg) in arg_exprs.iter() {
                                match expr.value {
                                    Some(_) if expr.ty.is_pointer() => self.register(&expr),
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
                                        assert!(dest_ptr != Location::BAD);
                                        let dest = self.put2_inc(Op::Move, dest_ptr);
                                        self.put(Op::Call, dest, func);
                                        dest_location
                                    }
                                },
                                _ => todo!("indirect call")
                            };
                            result.ty = return_type;
                            result.bound_context = BoundContext::Type(addr.ty, Location::stack(arguments_start));
                            result.is_call_result = true;
                        } else {
                            error!(self, ctx.ast.expr_source_position(callable), "cannot call proc '{}' from within a func", ctx.callable_str(ident(addr.value)));
                        }
                    } else {
                        error!(self, ctx.ast.expr_source_position(callable), "{} arguments passed to {}, which takes {} arguments", args.len(), ctx.callable_str(ident(addr.value)), info.items.len() - 1);
                    }
                } else {
                    // Todo: get a string representation of the whole `callable` expr for nicer error message
                    error!(self, ctx.ast.expr_source_position(callable), "cannot call a {}", ctx.type_str(addr.ty));
                }
            }
            ExprData::Cast(expr, type_expr) => {
                let to_ty = self.type_expr(ctx, type_expr);
                let mut dest = Destination::for_type(to_ty, BoundContext::Mark(self.locals.top()));
                if to_ty == destination.ty {
                    dest.addr = destination.addr;
                }
                dest.supressed_bound_checks = true;
                let left = self.expr_with_destination(ctx, expr, &dest);
                let expression_transposable = || {
                    if let ExprData::Unary(TokenKind::BitAnd, inner) = ctx.ast.expr(expr) {
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
                        result.ty = to_ty;
                    }
                } else if expression_transposable() {
                    // The cast-on-the-right synax doesn't work great with taking addresses.
                    //
                    // Conceptually, this transposes
                    //      &{}:Struct   (never legal)
                    // to
                    //      &({}:Struct) (allocates Struct on the stack and takes its address)
                    //
                    // We only do this if the above convert_op returns None, as otherwise we as distinguish between
                    //      &a:int       (equivalent to (&a):int)
                    // and
                    //      &(a:int)     (never legal, but would appear if transposed)
                    debug_assert!(left.ty.is_pointer());
                    result = ExprResult { ty: ctx.types.pointer(to_ty), ..left };
                } else {
                    error!(self, ctx.ast.expr_source_position(expr), "cannot cast from {} to {}", ctx.type_str(left.ty), ctx.type_str(to_ty));
                }
            }
        };

        #[cfg(debug_assertions)]
        if self.error.is_none() && self.can_emit_code() {
            debug_assert_implies!(result.addr.offset == Location::BAD, matches!(result.addr.kind, LocationKind::None|LocationKind::Control));
            debug_assert_implies!(result.addr.offset != Location::BAD, result.addr.kind != LocationKind::None);
            debug_assert_implies!(result.addr.base != Location::BAD, matches!(result.addr.kind, LocationKind::Based|LocationKind::Rip));
            debug_assert_implies!(result.ty != Type::None, result.value.is_some() || result.addr.kind != LocationKind::None);
        }

        // Decay array to pointer
        if destination.ty.is_pointer() {
            let result_info = ctx.types.info(result.ty);
            if result_info.kind == TypeKind::Array
            && result_info.base_type == ctx.types.base_type(destination.ty) {
                result.addr = Location::register(self.location_address(&result.addr));
                result.ty = destination.ty.with_bound_of(result.ty);
            }
        }

        // Check bounds are compatible if specified on destination.
        // So, constant bounds are fine if they fit, expression bounds have to emit a check
        if destination.ty != Type::None {
            if destination.deferred_bound_checks {
                // Pass up the original bound for type checking later
            } else if destination.supressed_bound_checks {
                // No bounds-checking, update the bound
                result.ty.bound = destination.ty.bound;
            } else {
                self.destination_bounds_check(ctx, destination, &result);
                result.ty.bound = destination.ty.bound;
            }
        }

        // Emit copy to destination and, if the copy changes the representation of an integer, update the type
        if destination.ty != Type::None && destination.addr.kind != LocationKind::None {
            let compatible = expr_integer_compatible_with_destination(&result, destination.ty, &destination.addr);
            if compatible || (destination.ty == result.ty) {
                if result.addr != destination.addr {
                    self.copy(ctx, destination.ty, &destination.addr, &result);
                    result.addr = destination.addr;
                }

                // Changing the result type here will never change the base type of a pointer.
                // In that case, the ptr bound may no longer accurately reflect the size of the
                // underyling allocation. In other words, the following must hold:
                if result.ty != destination.ty && result.ty.is_pointer() {
                    let rb = ctx.types.info(result.ty).base_type;
                    let db = ctx.types.info(destination.ty).base_type;
                    assert!(ctx.types.info(rb).size == ctx.types.info(db).size);
                }

                result.ty = destination.ty.with_bound_of(result.ty);
            }
        }

        // Emit jumps if the expression was for a control value (in if conditions, for example)
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

        // If we don't have a bound context, use the current scope
        if let BoundContext::None = result.bound_context {
            result.bound_context = BoundContext::Mark(self.locals.top());
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
                let bound_context = BoundContext::Type(self.generating_code_for, self.arguments_location);
                let ret_expr = self.expr_with_destination(ctx, expr, &Destination::for_return_type(self.return_type, bound_context));
                // TODO: iron out the story for constant pointers. i dont want any explicit
                // annotations, except possibly for pointers into write-protected pages (which
                // is mostly out of scope for this compiler). but for constness to leak out of
                // the return type we need some analysis
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
                    // Todo: stmt source position
                    error!(self, 0, "empty return, expected a return value of type {}", ctx.type_str(self.return_type));
                }
            }
            StmtData::Break => {
                if let Some(label) = self.break_label() {
                    self.put_jump(label);
                } else {
                    // Todo: stmt source position
                    error!(self, 0, "break outside of loop/switch");
                }
            }
            StmtData::Continue => {
                if let Some(label) = self.continue_label() {
                    self.put_jump(label);
                } else {
                    // Todo: stmt source position
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
                    if let SwitchCaseData::Cases(block, exprs) = ctx.ast.switch_case(case) {
                        for case_expr in exprs {
                            let expr = self.expr(ctx, case_expr);
                            if let Some(_ev) = expr.value {
                                if let Some((op, ty)) = binary_op(TokenKind::Eq, control.ty, expr.ty) {
                                    debug_assert_eq!(ty, Type::Bool);
                                    let expr_register = self.register(&expr);
                                    let matched = self.put3_inc(op, control_register, expr_register);
                                    let label = if block.is_empty() { break_label } else { block_label };
                                    self.put_jump_nonzero(matched, label);
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
                } else {
                    self.put_jump(break_label);
                }
                let mut first_block = true;
                for (i, case) in cases.enumerate() {
                    let block = match ctx.ast.switch_case(case) {
                        SwitchCaseData::Cases(block, _) => block,
                        SwitchCaseData::Else(block) => block
                    };
                    if !first_block && !block.is_empty() {
                        self.put_jump(break_label);
                    }
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
                        error!(self, ctx.ast.expr_source_position(expr), "cannot declare a value without initializing it");
                    }
                }
                self.expr(ctx, expr);
            }
            StmtData::VarDecl(ty_expr, left, right) => {
                if let ExprData::Name(var) = ctx.ast.expr(left) {
                    let expr;
                    let decl_type;
                    let bound_context;
                    if let TypeExprData::Infer = ctx.ast.type_expr(ty_expr) {
                        expr = self.expr(ctx, right);
                        // We strip the expr bound so bounds are not tracked if ptrs are taken off their
                        // parent structure and store them on the stack. We _could_ do that, but maybe
                        // it would be confusing? Or not that useful?
                        decl_type = if expr.is_call_result { expr.ty } else { expr.ty.strip_expr_bound() };
                        bound_context = expr.bound_context;
                    } else {
                        decl_type = self.type_expr(ctx, ty_expr);
                        expr = self.expr_with_destination_type(ctx, right, decl_type);
                        bound_context = BoundContext::Mark(self.locals.top());
                    }
                    let addr = if expr.addr.is_place {
                        self.copy_to_stack(ctx, &expr)
                    } else {
                        self.emit_constant(&expr).addr
                    };
                    if ctx.types.types_match_with_promotion(decl_type, expr.ty) {
                        if value_fits(expr.value, decl_type) {
                            self.locals.insert(var, Local::new(addr, decl_type, bound_context));
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
                let value = self.expr_with_destination(ctx, right, &Destination::of_expr_result(&lv));
                if lv.ty != value.ty {
                    if value_fits(value.value, lv.ty) {
                        error!(self, ctx.ast.expr_source_position(left), "type mismatch between destination ({}) and value ({})", ctx.type_str(lv.ty), ctx.type_str(value.ty))
                    } else {
                        error!(self, ctx.ast.expr_source_position(left), "constant expression does not fit in {}", ctx.type_str(lv.ty));
                    }
                }
            }
        }
        return_type
    }

    fn stmts(&mut self, ctx: &mut Compiler, stmts: StmtList) -> Option<Type> {
        let mark = self.locals.push_scope();
        let mut return_type = None;
        for stmt in stmts {
            if self.error.is_some() {
                break;
            }
            return_type = self.stmt(ctx, stmt).or(return_type);
        }
        self.locals.restore_scope(mark);
        return_type
    }

    fn prolog(&mut self) {
        // No code is emitted here.
        let return_reg = self.inc_reg();
        let _ret = self.inc_reg();
        let _frame = self.inc_reg();
        let _stack = self.inc_reg();
        let _instr = self.inc_reg();
        assert_eq!(return_reg, 0);
    }

    pub fn callable(&mut self, ctx: &mut Compiler, signature: Type, decl: Decl) -> Code {
        let callable = ctx.ast.callable(decl);
        let body = callable.body;
        let sig = ctx.types.info(signature);
        assert_eq!(sig.kind, TypeKind::Callable, "sig.kind == TypeKind::Callable");

        let kind = ctx.ast.type_expr_keytype(callable.expr).expect("tried to generate code for callable without keytype");
        let params = ctx.ast.type_expr_items(callable.expr).expect("tried to generate code for callable without parameters");
        assert!(params.len() == sig.items.len() - 1, "the last element of a callable's type signature's items must be the return type");

        let is_func = ctx.ast.type_expr_keytype(callable.expr) == Some(Keytype::Func);
        self.target = if is_func { CodeGenTarget::Func } else { CodeGenTarget::Proc };
        self.return_type = sig.items.last().map(|i| i.ty).unwrap();
        self.reg_counter = (Self::REGISTER_SIZE as isize) * (1 - sig.items.len() as isize);
        self.generating_code_for = signature;
        assert_implies!(is_func, self.return_type != Type::None);

        let param_names = params.map(|item| ctx.ast.item(item).name);
        let param_types = sig.items.iter().map(|i| i.ty);
        let number_of_arguments = params.len() as isize;
        let top = self.locals.push_scope();
        self.arguments_location = Location::stack(self.reg_counter);
        self.arguments_mark = top;
        for (name, ty) in Iterator::zip(param_names, param_types) {
            let reg = self.inc_reg();
            let loc = if ty.is_basic() {
                Location::register(reg)
            } else {
                Location::pointer(reg, 0)
            };
            self.locals.insert(name, Local::argument(loc, ty, number_of_arguments));
        }

        let addr = self.code.len();
        self.prolog();
        let returned = self.stmts(ctx, body);
        if self.return_type != Type::None {
            if let Some(returned) = returned {
                assert!(self.return_type == returned);
            } else {
                let callable = ctx.ast.callable(decl);
                error!(self, callable.pos, "{} {}: not all control paths return a value", kind, ctx.str(callable.name))
            }
        } else {
            self.put0(Op::Return);
        }

        let panic: RegValue = ctx.intern("panic").into();
        let mut panics = std::mem::take(&mut self.panics);
        for &(label, reason) in &panics {
            self.patch(label);
            let c = self.constant((reason as u32).into());
            self.put(Op::Call, c, panic);
        }
        self.panics = { panics.clear(); panics };

        self.apply_patches();
        self.locals.restore_scope(top);

        assert!(self.patches.len() == 0);
        assert!(self.jmp.break_to.is_none());
        assert!(self.jmp.continue_to.is_none());
        assert!(self.locals.mark.index == 0);

        self.error.take().map(|e| ctx.error(e.source_position, e.msg));
        Code { signature, addr }
    }

    pub fn global_expr(&mut self, ctx: &mut Compiler, ty: Type, right: Expr, location: isize, reg_counter: isize) {
        self.target = CodeGenTarget::GlobalExpr;
        self.generating_code_for = Type::None;
        self.reg_counter = reg_counter;
        if reg_counter == 0 {
            self.prolog();
        }
        let expr = self.expr_with_destination(ctx, right, &Destination::for_location(ty, Location::rip(location)));
        let flag = self.constant(1.into());
        let (ip, offset) = self.location_base_offset(&Location::rip(location - 4));
        self.put3(Op::Store32, ip, flag, offset);
        if ctx.types.types_match_with_promotion(ty, expr.ty) {
            if value_fits(expr.value, ty) {
                assert!(expr.addr.kind == LocationKind::Rip);
                assert!(expr.addr.offset == location);
            } else {
                error!(self, ctx.ast.expr_source_position(right), "constant expression does not fit in {}", ctx.type_str(ty));
            }
        } else {
            error!(self, ctx.ast.expr_source_position(right), "type mismatch between declaration ({}) and value ({})", ctx.type_str(ty), ctx.type_str(expr.ty))
        }
        self.error.take().map(|e| ctx.error(e.source_position, e.msg));
    }
}

pub fn eval_type(ctx: &mut Compiler, ty: TypeExpr) -> Type {
    // re: pop/push: This function is called recursively. It's trivial to make fg reentryable
    // but borrow checker doesn't like it if it lives on ctx, and I've thought
    // about it too long already
    let mut fg = ctx.fgs.pop().unwrap_or_else(|| FatGen::new(Vec::new()));
    fg.target = CodeGenTarget::TypeExprIgnoreBounds;
    let result = fg.type_expr(ctx, ty);
    fg.error.take().map(|e| ctx.errors.push(e));
    ctx.fgs.push(fg);
    result
}

pub fn allocate_global_var(ctx: &mut Compiler, ty: Type) -> isize {
    let top = ctx.data.len();
    let initialized_flag = align_up(top, std::mem::align_of::<u32>());
    ctx.data.resize_with(initialized_flag + std::mem::size_of::<u32>(), Default::default);

    let top = ctx.data.len();
    let TypeInfo { size, alignment, .. } = *ctx.types.info(ty);
    let result = align_up(top, alignment);
    ctx.data.resize_with(result+size, Default::default);
    result as isize
}

pub fn eval_item_bounds(ctx: &mut Compiler, ty: BareType, items: ItemList, returns: Option<TypeExpr>) {
    // Same song and dance as eval_type
    let mut fg = ctx.fgs.pop().unwrap_or_else(|| FatGen::new(Vec::new()));
    fg.target = CodeGenTarget::TypeExpr;
    let locals_mark = fg.locals.push_type(ctx, ty.into(), &Location::none());

    for (i, item) in items.enumerate() {
        let info = ctx.ast.item(item);
        let item_ty = fg.type_expr(ctx, info.expr);
        ctx.types.set_item_bound(ty, i, info.name, item_ty);
    }

    if let Some(returns) = returns {
        let item_ty = fg.type_expr(ctx, returns);
        let items = &ctx.types.info(ty.into()).items;
        let i = items.len() - 1;
        let name = items[i].name;
        ctx.types.set_item_bound(ty, i, name, item_ty);
    }

    fg.locals.restore_scope(locals_mark);
    fg.error.take().map(|e| ctx.errors.push(e));
    ctx.fgs.push(fg);
}
