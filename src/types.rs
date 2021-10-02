use std::collections::hash_map::HashMap;
use std::convert::TryFrom;

use crate::Compiler;
use crate::ast;
use crate::Intern;
use crate::hash;

use crate::sym;

use crate::smallvec::*;

pub fn builtins(ctx: &mut Compiler) {
    let builtins = [
        ("_",    TypeKind::None, Bound::Constant(0), 1),
        ("int",  TypeKind::Int,  Bound::Single, 8),
        ("i8",   TypeKind::I8,   Bound::Single, 1),
        ("i16",  TypeKind::I16,  Bound::Single, 2),
        ("i32",  TypeKind::I32,  Bound::Single, 4),
        ("i64",  TypeKind::I64,  Bound::Single, 8),
        ("u8",   TypeKind::U8,   Bound::Single, 1),
        ("u16",  TypeKind::U16,  Bound::Single, 2),
        ("u32",  TypeKind::U32,  Bound::Single, 4),
        ("u64",  TypeKind::U64,  Bound::Single, 8),
        ("f32",  TypeKind::F32,  Bound::Single, 4),
        ("f64",  TypeKind::F64,  Bound::Single, 8),
        ("bool", TypeKind::Bool, Bound::Single, 1),
    ];

    for (i, &(str, kind, bound, size)) in builtins.iter().enumerate() {
        assert!(i == kind as usize);
        let name = ctx.interns.put(str);
        let ty = Type { id: i as u32, bound };
        let items = Vec::new();
        ctx.types.types.push(TypeInfo { kind, name, size: size, alignment: size.max(1), items, base_type: bare(ty), mutable: true, num_array_elements: 0 });
        assert!(size == 0 || size.is_power_of_two());
        sym::builtin(ctx, name, ty);
    }

    let voidptr = ctx.types.pointer(Type::None);
    let u8ptr = ctx.types.pointer(Type::U8);
    assert_eq!(voidptr, Type::VoidPtr);
    assert_eq!(u8ptr, Type::U8Ptr);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Bound {
    Single,
    Constant(usize),
    Expr(ast::Expr)
}

impl Default for Bound {
    fn default() -> Self {
        Bound::Single
    }
}

#[derive(Copy, Clone, Default, PartialEq, Eq, Debug)]
pub struct BareType {
    id: u32
}

pub const fn bare(ty: Type) -> BareType {
    BareType { id: ty.id }
}

#[allow(non_upper_case_globals, dead_code)]
impl BareType {
    pub const None: BareType = bare(Type::None);
    pub const Int:  BareType = bare(Type::Int);
    pub const I8:   BareType = bare(Type::I8);
    pub const I16:  BareType = bare(Type::I16);
    pub const I32:  BareType = bare(Type::I32);
    pub const I64:  BareType = bare(Type::I64);
    pub const U8:   BareType = bare(Type::U8);
    pub const U16:  BareType = bare(Type::U16);
    pub const U32:  BareType = bare(Type::U32);
    pub const U64:  BareType = bare(Type::U64);
    pub const F32:  BareType = bare(Type::F32);
    pub const F64:  BareType = bare(Type::F64);
    pub const Bool: BareType = bare(Type::Bool);
    pub const VoidPtr: BareType = bare(Type::VoidPtr);
    pub const U8Ptr:   BareType = bare(Type::U8Ptr);
}

#[derive(Copy, Clone, Default, Eq, Debug)]
pub struct Type {
    pub id: u32,
    pub bound: Bound
}

impl PartialEq for Type {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl PartialEq<BareType> for Type {
    fn eq(&self, other: &BareType) -> bool {
        self.id == other.id
    }
}

impl PartialEq<Type> for BareType {
    fn eq(&self, other: &Type) -> bool {
        self.id == other.id
    }
}

impl From<BareType> for Type {
    fn from(ty: BareType) -> Type {
        Type { id: ty.id, bound: Bound::Single }
    }
}

#[allow(non_upper_case_globals)]
impl Type {
    // Pointer types have the last bit set on the handle. This is so pointers get
    // folded into "base types" that can be reasoned about by the handle alone
    // without having to look up information from the handle. It's for convenience,
    // more than anything else.
    const POINTER_BIT: u32 = 1 << (u32::BITS - 1);
    // Likewise, we always put immutable and mutable pointers to the same base type
    // adjacent in the big type array, with immutable pointers in the odd indices.
    // Immutable pointer types always come after mutable pointers in the array.
    // Non-pointer types are never immutable so the immutable bit is free to vary.
    const IMMUTABLE_BIT: u32 = 1;

    pub const None: Type = Type { id: TypeKind::None as u32, bound: Bound::Constant(0) };
    pub const Int:  Type = Type { id: TypeKind::Int as u32, bound: Bound::Single };
    pub const I8:   Type = Type { id: TypeKind::I8 as u32, bound: Bound::Single };
    pub const I16:  Type = Type { id: TypeKind::I16 as u32, bound: Bound::Single };
    pub const I32:  Type = Type { id: TypeKind::I32 as u32, bound: Bound::Single };
    pub const I64:  Type = Type { id: TypeKind::I64 as u32, bound: Bound::Single };
    pub const U8:   Type = Type { id: TypeKind::U8 as u32, bound: Bound::Single };
    pub const U16:  Type = Type { id: TypeKind::U16 as u32, bound: Bound::Single };
    pub const U32:  Type = Type { id: TypeKind::U32 as u32, bound: Bound::Single };
    pub const U64:  Type = Type { id: TypeKind::U64 as u32, bound: Bound::Single };
    pub const F32:  Type = Type { id: TypeKind::F32 as u32, bound: Bound::Single };
    pub const F64:  Type = Type { id: TypeKind::F64 as u32, bound: Bound::Single };
    pub const Bool: Type = Type { id: TypeKind::Bool as u32, bound: Bound::Single };
    pub const VoidPtr: Type = Type { id: (TypeKind::Bool as u32 + 2) | Self::POINTER_BIT, bound: Bound::Single };
    pub const U8Ptr:   Type = Type { id: (TypeKind::Bool as u32 + 4) | Self::POINTER_BIT, bound: Bound::Single };

    pub fn is_integer(self) -> bool {
        Type::Int.id <= self.id && self.id <= Type::U64.id
    }

    pub fn is_basic(self) -> bool {
        self.id <= TypeKind::Bool as u32 || self.is_pointer()
    }

    pub fn is_pointer(self) -> bool {
        (self.id & Self::POINTER_BIT) == Self::POINTER_BIT
    }

    pub fn is_immutable_pointer(self) -> bool {
        self.is_pointer() && (self.id & Self::IMMUTABLE_BIT) == Self::IMMUTABLE_BIT
    }

    pub fn to_mutable(self) -> Type {
        if self.is_pointer() {
            return Type { id: self.id & !Self::IMMUTABLE_BIT, bound: self.bound };
        }
        self
    }

    pub fn to_immutable(self) -> Type {
        if self.is_pointer() {
            return Type { id: self.id | Self::IMMUTABLE_BIT, bound: self.bound };
        }
        self
    }

    pub fn with_mutability_of(self, other: Type) -> Type {
        if other.is_immutable_pointer() {
            return self.to_immutable();
        }
        self.to_mutable()
    }
}

pub fn integer_promote(ty: Type) -> Type {
    if ty.is_integer() { Type::Int } else { ty }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum TypeKind {
    None,
    Int,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F32,
    F64,
    Bool,
    Callable,
    Struct,
    Array,
    Pointer,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Item {
    pub name: Intern,
    pub ty: Type,
    pub offset: usize,
}

// Type doesn't get the derive trait because whenever you hash it you need to
// decide if you want to include the bound. For an item, we do.
impl std::hash::Hash for Item {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.name.hash(state);
        self.ty.id.hash(state);
        self.ty.bound.hash(state);
        self.offset.hash(state);
    }
}

#[derive(Clone, Debug)]
pub struct TypeInfo {
    pub kind: TypeKind,
    pub name: Intern,
    pub size: usize,
    pub alignment: usize,
    pub items: Vec<Item>,
    // Types without a base_type are their own base_type
    pub base_type: BareType,
    // Two kinds of type can be mutable
    //  pointers: appear as immutable when passed into funcs
    //  callables: procs are "mutable" in that they allow side-effects
    pub mutable: bool,
    pub num_array_elements: usize,
}

impl TypeInfo {
    pub fn arguments(&self) -> Option<&[Item]> {
        if self.kind == TypeKind::Callable {
            let end = if self.items.len() == 0 { 0 } else { self.items.len() - 1 };
            Some(&self.items[..end])
        } else {
            None
        }
    }

    pub fn return_type(&self) -> Option<Type> {
        if self.kind == TypeKind::Callable {
            self.items.last().map(|i| i.ty)
        } else {
            None
        }
    }

    pub fn bare_return_type(&self) -> Option<BareType> {
        self.return_type().map(bare)
    }
}

pub struct Types {
    types: Vec<TypeInfo>,
    type_by_hash: HashMap<u64, SmallVecN<Type, 5>>,
    freelist: Vec<u32>
}

impl Types {
    pub fn new() -> Types {
        Types {
            types: Vec::new(),
            type_by_hash: HashMap::new(),
            freelist: Vec::new()
        }
    }

    pub fn info(&self, ty: Type) -> &TypeInfo {
        &self.types[(ty.id & !Type::POINTER_BIT) as usize]
    }

    fn info_mut(&mut self, ty: BareType) -> &mut TypeInfo {
        &mut self.types[(ty.id & !Type::POINTER_BIT) as usize]
    }

    pub fn item_info(&self, ty: Type, name: Intern) -> Option<Item> {
        self.info(ty).items.iter().find(|it| it.name == name).copied()
    }

    pub fn index_info(&self, ty: Type, index: usize) -> Option<Item> {
        let info = self.info(ty);
        match info.kind {
            TypeKind::Struct => {
                info.items.iter().nth(index).copied()
            }
            TypeKind::Array => {
                if index < info.num_array_elements {
                    let base_type = Type::from(info.base_type);
                    let element_size = self.info(base_type).size;
                    Some(Item {
                        name: Intern(0),
                        ty: base_type,
                        offset: index * element_size,
                    })
                } else {
                    None
                }
            }
            _ => None
        }
    }

    fn make(&mut self, kind: TypeKind) -> BareType {
        let idx = self.freelist.pop().or_else(|| u32::try_from(self.types.len()).ok()).expect("Program too big!!");
        let mut result = BareType { id: idx };
        let idx = idx as usize;
        if result.id >= Type::POINTER_BIT {
            todo!("Program too big!!");
        }
        if kind == TypeKind::Pointer {
            result.id |= Type::POINTER_BIT;
        }
        let info = TypeInfo {
            kind: kind,
            name: Intern(0),
            size: 0,
            alignment: 1,
            items: Vec::new(),
            base_type: result,
            mutable: true,
            num_array_elements: 0,
        };
        if idx == self.types.len() {
            self.types.push(info)
        } else {
            let mut items = std::mem::take(&mut self.info_mut(result).items);
            items.clear();
            *self.info_mut(result) = TypeInfo { items, ..info };
        }
        result
    }

    pub fn sorry(&mut self, ty: BareType) {
        self.freelist.push(ty.id & !Type::POINTER_BIT);
    }

    pub fn callable(&mut self, name: Intern, mutable: bool) -> BareType {
        #[cfg(debug_assertions)] {
            let hash = hash(&(TypeKind::Callable, name));
            if let Some(types) = self.type_by_hash.get(&hash) {
                for &ty in types.iter() {
                    let info = self.info(ty);
                    if info.kind == TypeKind::Callable && info.name == name {
                        panic!("callable made twice");
                    }
                }
            }
        }
        let ty = self.make(TypeKind::Callable);
        self.info_mut(ty).name = name;
        self.info_mut(ty).mutable = mutable;
        ty
    }

    pub fn anonymous(&mut self, kind: TypeKind) -> BareType {
        self.make(kind)
    }

    pub fn strukt(&mut self, name: Intern) -> BareType {
        #[cfg(debug_assertions)] {
            let hash = hash(&(TypeKind::Struct, name));
            if let Some(types) = self.type_by_hash.get(&hash) {
                for &ty in types.iter() {
                    let info = self.info(ty);
                    if info.kind == TypeKind::Struct && info.name == name {
                        panic!("struct made twice: calling code must check for cycles");
                    }
                }
            }
        }
        let result = self.make(TypeKind::Struct);
        self.info_mut(result).name = name;
        result
    }

    // Traps if base_type.size * num_array_elements overflows
    pub fn array(&mut self, base_type: Type, num_array_elements: usize) -> Type {
        assert!(num_array_elements > 0);
        let hash = hash(&(TypeKind::Array, base_type.id, num_array_elements));
        if let Some(types) = self.type_by_hash.get(&hash) {
            for &ty in types.iter() {
                let info = self.info(ty);
                if info.kind == TypeKind::Array && info.base_type == base_type && info.num_array_elements == num_array_elements {
                    return ty;
                }
            }
        }
        let ty = self.make(TypeKind::Array);
        let (base_size, alignment) = {
            let base = self.info(base_type);
            (base.size, base.alignment)
        };
        let arr = self.info_mut(ty);
        arr.size = base_size.checked_mul(num_array_elements).unwrap();
        arr.alignment = alignment;
        arr.base_type = BareType { id: base_type.id };
        arr.num_array_elements = num_array_elements;
        let ty = Type { id: ty.id, bound: Bound::Constant(num_array_elements) };
        self.type_by_hash.entry(hash).or_insert_with(|| SmallVecN::new()).push(ty);
        ty
    }

    pub fn pointer(&mut self, base_type: Type) -> Type {
        let hash = hash(&(TypeKind::Pointer, base_type.id));
        if let Some(types) = self.type_by_hash.get(&hash) {
            for &ty in types.iter() {
                let info = self.info(ty);
                if info.kind == TypeKind::Pointer && info.base_type == base_type {
                    return ty;
                }
            }
        }

        // Make mutable and immutable variants of the pointer. This........ means we can
        // go to and from mutable and immutable pointers without a mutable reference to
        // self........

        // lowest friction way to achieve this but feels stupid. we want ty < imm_ty and
        // ty on the odd index. but entries in the freelist are probably not contiguous,
        // and the easiest way to deal with that is to hide it from .make()
        let dont_use_this = std::mem::take(&mut self.freelist);
        let ty = self.make(TypeKind::Pointer);
        let immutable_ty = self.make(TypeKind::Pointer);
        let (ty, immutable_ty) = if ty.id & 1 == 0 {
            self.freelist = dont_use_this;
            (ty, immutable_ty)
        } else {
            let shift = self.make(TypeKind::Pointer);
            self.freelist = dont_use_this;
            self.sorry(ty);
            (immutable_ty, shift)
        };
        assert!((ty.id & Type::IMMUTABLE_BIT) == 0);
        assert!((immutable_ty.id & Type::IMMUTABLE_BIT) == Type::IMMUTABLE_BIT);
        assert!(ty.id + 1 == immutable_ty.id);

        let ptr = self.info_mut(ty);
        ptr.size = std::mem::size_of::<*const u8>();
        ptr.alignment = ptr.size;
        ptr.base_type = bare(base_type);
        ptr.mutable = true;

        let ty = Type::from(ty);
        *self.info_mut(immutable_ty) = TypeInfo { mutable: false, items: Vec::new(), ..*self.info(ty) };

        self.type_by_hash.entry(hash).or_insert_with(|| SmallVecN::new()).push(ty);
        ty
    }

    pub fn bound_pointer(&mut self, ty: Type, bound: Bound) -> Type {
        let ptr_ty = self.pointer(ty);
        Type { id: ptr_ty.id, bound }
    }

    pub fn base_type(&self, ty: Type) -> Type {
        Type { id: self.info(ty).base_type.id, bound: ty.bound }
    }

    pub fn add_item_to_type(&mut self, ty: BareType, name: Intern, item: Type) {
        let (size, alignment) = {
            let ty_info = self.info_mut(ty);
            match ty_info.kind {
                TypeKind::Callable => (std::mem::size_of::<*const u8>(), std::mem::align_of::<*const u8>()),
                TypeKind::Struct => {
                    let item_info = self.info(item);
                    (item_info.size, item_info.alignment)
                },
                _ => unreachable!()
            }
        };
        let info = self.info_mut(ty);
        let offset = (info.size + (alignment - 1)) & !(alignment - 1);
        assert!(offset & !(alignment - 1) == offset);
        info.size = offset + size;
        info.alignment = info.alignment.max(alignment);
        info.items.push(Item { name, ty: item, offset });
    }

    pub fn set_item_bound(&mut self, ty: BareType, index: usize, name: Intern, type_with_bound: Type) {
        let info = self.info_mut(ty);
        assert!(info.items[index].name == name);
        assert!(info.items[index].ty.id == type_with_bound.id || type_with_bound == Type::None);
        info.items[index].ty = type_with_bound;
    }

    pub fn complete(&mut self, ty: BareType) -> Type {
        let info = self.info_mut(ty);
        let alignment = info.alignment;
        info.size = (info.size + (alignment - 1)) & !(alignment - 1);
        assert!(info.size & !(alignment - 1) == info.size);
        assert!(info.size as isize >= 0);
        let result = Type::from(ty);
        if info.name == Intern(0) {
            let hash = hash(&(info.kind, &info.items));
            if let Some(types) = self.type_by_hash.get(&hash) {
                for &other_ty in types.iter() {
                    let info = self.info(result);
                    let other = self.info(other_ty);
                    if info.kind == other.kind && info.items.iter().eq(other.items.iter()) {
                        // Deduplicate. Is this even worth doing? What if we
                        // just define equality on hashes?
                        self.sorry(ty);
                        return other_ty;
                    }
                }
            }
            self.type_by_hash.entry(hash).or_insert_with(|| SmallVecN::new()).push(result);
        }
        result
    }

    pub fn annoying_deep_eq_ignoring_mutability(&self, a: Type, b: Type) -> bool {
        // We need to do this because const-ness is encoded in the type. This
        // lets us catch some writes thru constant pointers without doing
        // analysis, but can't catch all of them. SO: we can, and should, not do this,
        // and also, there are currently programs permitted that shouldnt be
        if a.to_mutable() == b.to_mutable() {
            return true;
        }

        if a.is_pointer() && b.is_pointer() {
            let a_base = self.base_type(a);
            let b_base = self.base_type(b);
            assert!(a_base != a && b_base != b);

            return self.annoying_deep_eq_ignoring_mutability(a_base, b_base);
        }

        false
    }

    pub fn types_match_with_promotion(&self, promotable: Type, other: Type) -> bool {
        self.annoying_deep_eq_ignoring_mutability(promotable, other) || integer_promote(promotable) == other
    }
}

#[test]
fn types() {
    let code = r#"
V2: struct (
    x: f32,
    y: f32
);

Rect: struct (
    top_left: V2,
    bottom_right: V2
);

Rect2: struct (
    verts: arr f32 [4],
    value: f32
);

Arr: struct (
    arr1: arr (arr i32 [3]) [7],
    arr2: arr (arr i32 [3]) [7]
);

Padding: struct (
    a: i16,
    b: i32,
    c: i64,
    d: i8
);

Padding2: struct (
    a: i16,
    b: Padding,
    c: i16,
);

Ptr: struct (
    a: ptr,
    b: ptr u8 [2],
    c: ptr int [bound],
    d: ptr Ptr [b[4]],
    bound: int
);
    "#;

    let mut c = crate::compile(&code).unwrap();
    let v2 = c.intern("V2");
    let rect = c.intern("Rect");
    let rect2 = c.intern("Rect2");
    let arr = c.intern("Arr");
    let ptr = c.intern("Ptr");
    let padding = c.intern("Padding");
    let padding2 = c.intern("Padding2");

    let int_ptr = c.types.pointer(Type::Int);
    let ptr_ptr = c.types.pointer(sym::lookup_type(&c, ptr).map(|sym| sym.ty).unwrap());

    let get = |name: Intern| sym::lookup_type(&c, name).map(|sym| c.types.info(sym.ty)).unwrap();
    let v2 = get(v2);
    let rect = get(rect);
    let rect2 = get(rect2);
    let arr = get(arr);
    let ptr = get(ptr);
    let padding = get(padding);
    let padding2 = get(padding2);

    assert_eq!(v2.size, 8);
    assert_eq!(v2.alignment, 4);
    assert_eq!(v2.items[0].offset, 0);
    assert_eq!(v2.items[1].offset, 4);

    assert_eq!(rect.size, 16);
    assert_eq!(rect.alignment, 4);
    assert_eq!(rect.items[0].offset, 0);
    assert_eq!(rect.items[1].offset, 8);

    assert_eq!(rect2.size, 20);
    assert_eq!(rect2.alignment, 4);
    assert_eq!(rect2.items[0].offset, 0);
    assert_eq!(rect2.items[1].offset, 16);

    let verts = c.types.info(rect2.items[0].ty);
    assert_eq!(verts.kind, TypeKind::Array);
    assert_eq!(verts.size, 16);
    assert_eq!(verts.alignment, 4);
    assert_eq!(verts.base_type, Type::F32);
    assert_eq!(verts.num_array_elements, 4);

    assert_eq!(arr.size, 3*7*4*2);
    assert_eq!(arr.alignment, 4);
    assert_eq!(arr.items[0].ty, arr.items[1].ty);
    assert_eq!(c.types.info(arr.items[0].ty).base_type, c.types.info(arr.items[1].ty).base_type);

    let aoa = c.types.info(arr.items[0].ty);
    let aoi = c.types.info(aoa.base_type.into());

    assert_eq!(aoa.kind, TypeKind::Array);
    assert_eq!(aoa.size, aoi.size * aoa.num_array_elements);
    assert_eq!(aoa.alignment, 4);
    assert_eq!(aoa.num_array_elements, 7);

    assert_eq!(aoi.kind, TypeKind::Array);
    assert_eq!(aoi.size, 3*4);
    assert_eq!(aoi.alignment, 4);
    assert_eq!(aoi.base_type, Type::I32);
    assert_eq!(aoi.num_array_elements, 3);

    assert_eq!(ptr.kind, TypeKind::Struct);
    assert_eq!(ptr.size, 40);
    assert_eq!(ptr.alignment, 8);
    assert_eq!(ptr.items[0].ty, Type::VoidPtr);
    assert_eq!(ptr.items[1].ty, Type::U8Ptr);
    assert_eq!(ptr.items[2].ty, int_ptr);
    assert_eq!(ptr.items[3].ty, ptr_ptr);
    assert_eq!(ptr.items[0].ty.bound, Bound::Single);
    assert_eq!(ptr.items[1].ty.bound, Bound::Constant(2));
    // assert_eq!(ptr.items[2].ty.bound, ???);
    // assert_eq!(ptr.items[3].ty.bound, ???);
    assert_eq!(c.types.info(c.types.base_type(ptr_ptr)) as *const _, ptr as *const _);

    assert_eq!(padding.size, 24);
    assert_eq!(padding.alignment, 8);
    assert_eq!(padding.items[0].offset, 0);
    assert_eq!(padding.items[1].offset, 4);
    assert_eq!(padding.items[2].offset, 8);
    assert_eq!(padding.items[3].offset, 16);

    assert_eq!(padding2.size, 40);
    assert_eq!(padding2.alignment, 8);
    assert_eq!(padding2.items[0].offset, 0);
    assert_eq!(padding2.items[1].offset, 8);
    assert_eq!(padding2.items[2].offset, 32);
}
