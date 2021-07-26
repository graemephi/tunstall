use std::collections::hash_map::HashMap;
use std::convert::TryFrom;

use crate::Compiler;
use crate::Intern;
use crate::hash;

use crate::sym;

use crate::smallvec::*;

pub fn builtins(ctx: &mut Compiler) {
    let builtins = [
        ("_",    TypeKind::None, 1),
        ("int",  TypeKind::Int,  8),
        ("i8",   TypeKind::I8,   1),
        ("i16",  TypeKind::I16,  2),
        ("i32",  TypeKind::I32,  4),
        ("i64",  TypeKind::I64,  8),
        ("u8",   TypeKind::U8,   1),
        ("u16",  TypeKind::U16,  2),
        ("u32",  TypeKind::U32,  4),
        ("u64",  TypeKind::U64,  8),
        ("f32",  TypeKind::F32,  4),
        ("f64",  TypeKind::F64,  8),
        ("bool", TypeKind::Bool, 1),
    ];

    for (i, &(str, kind, size)) in builtins.iter().enumerate() {
        assert!(i == kind as usize);
        let name = ctx.interns.put(str);
        let ty = Type(i as u32);
        let items = Vec::new();
        ctx.types.types.push(TypeInfo { kind, name, size: size, alignment: size.max(1), items, base_type: ty, mutable: true, num_array_elements: 0 });
        assert!(size == 0 || size.is_power_of_two());
        sym::builtin(ctx, name, ty);
    }

    let voidptr = ctx.types.pointer(Type::None);
    let u8ptr = ctx.types.pointer(Type::U8);
    assert_eq!(voidptr, Type::VoidPtr);
    assert_eq!(u8ptr, Type::U8Ptr);
}

#[derive(Copy, Clone, Default, PartialEq, Eq, Debug, Hash)]
pub struct Type(u32);

#[allow(non_upper_case_globals)]
impl Type {
    // Pointer types have the last bit set on the handle. This is so pointers get
    // folded into "base types" that can be reasoned about by the handle alone
    // without having to look up information from the handle. It's for convenience,
    // more than anything else.
    const POINTER_BIT: u32 = 1 << (u32::BITS - 1);
    // Likewise, we always put immutable and mutable pointers to the same base type
    // adjacent in the big type array, with immutable pointers in the odd indices.
    // But we still pack them tight, so, if a pointer is immutable, its mutable dual
    // might be in either of the two neighbouring slots. (Non-pointer types are
    // never immutable so the immutable bit is free to vary).
    const IMMUTABLE_BIT: u32 = 1;

    pub const None: Type = Type(TypeKind::None as u32);
    pub const Int:  Type = Type(TypeKind::Int as u32);
    pub const I8:   Type = Type(TypeKind::I8 as u32);
    pub const I16:  Type = Type(TypeKind::I16 as u32);
    pub const I32:  Type = Type(TypeKind::I32 as u32);
    pub const I64:  Type = Type(TypeKind::I64 as u32);
    pub const U8:   Type = Type(TypeKind::U8 as u32);
    pub const U16:  Type = Type(TypeKind::U16 as u32);
    pub const U32:  Type = Type(TypeKind::U32 as u32);
    pub const U64:  Type = Type(TypeKind::U64 as u32);
    pub const F32:  Type = Type(TypeKind::F32 as u32);
    pub const F64:  Type = Type(TypeKind::F64 as u32);
    pub const Bool: Type = Type(TypeKind::Bool as u32);
    pub const VoidPtr: Type = Type((TypeKind::Bool as u32 + 2) | Self::POINTER_BIT);
    pub const U8Ptr:   Type = Type((TypeKind::Bool as u32 + 4) | Self::POINTER_BIT);

    pub fn is_integer(self) -> bool {
        Type::Int.0 <= self.0 && self.0 <= Type::U64.0
    }

    pub fn is_basic(self) -> bool {
        self.0 <= TypeKind::Bool as u32 || self.is_pointer()
    }

    pub fn is_pointer(self) -> bool {
        (self.0 & Self::POINTER_BIT) == Self::POINTER_BIT
    }

    pub fn is_immutable_pointer(self) -> bool {
        self.is_pointer() && (self.0 & Self::IMMUTABLE_BIT) == Self::IMMUTABLE_BIT
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

#[derive(Clone, Copy, Debug)]
pub enum Bound {
    Constant(isize),
    Offset(isize),
    Indirect(isize, isize)
}

#[derive(Clone, Copy, Debug)]
enum BoundType {
    Free(Type),
    Bound(Type, Bound)
}

#[derive(Clone, Copy, Debug)]
pub struct Item {
    pub name: Intern,
    pub ty: Type,
    pub offset: usize,
}

#[derive(Clone, Debug)]
pub struct TypeInfo {
    pub kind: TypeKind,
    pub name: Intern,
    pub size: usize,
    pub alignment: usize,
    pub items: Vec<Item>,
    // Types without a base_type are their own base_type
    pub base_type: Type,
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
}

pub struct Types {
    types: Vec<TypeInfo>,
    type_by_hash: HashMap<u64, SmallVecN<Type, 5>>,
}

impl Types {
    pub fn new() -> Types {
        Types {
            types: Vec::new(),
            type_by_hash: HashMap::new()
        }
    }

    pub fn info(&self, ty: Type) -> &TypeInfo {
        &self.types[(ty.0 & !Type::POINTER_BIT) as usize]
    }

    fn info_mut(&mut self, ty: Type) -> &mut TypeInfo {
        &mut self.types[(ty.0 & !Type::POINTER_BIT) as usize]
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
                    let element_size = self.info(info.base_type).size;
                    Some(Item {
                        name: Intern(0),
                        ty: info.base_type,
                        offset: index * element_size,
                    })
                } else {
                    None
                }
            }
            _ => None
        }
    }

    pub fn next(&mut self) -> Type {
        Type(u32::try_from(self.types.len()).expect("Program too big!!"))
    }

    pub fn make(&mut self, kind: TypeKind) -> Type {
        let mut result = self.next();
        if result.0 >= Type::POINTER_BIT {
            todo!("Program too big!!");
        }
        if kind == TypeKind::Pointer {
            result.0 |= Type::POINTER_BIT;
        }
        self.types.push(TypeInfo {
            kind: kind,
            name: Intern(0),
            size: 0,
            alignment: 1,
            items: Vec::new(),
            base_type: result,
            mutable: true,
            num_array_elements: 0,
        });
        result
    }

    pub fn signature(&mut self, kind: TypeKind, items: &[Type], mutable: bool) -> Type {
        let hash = hash(&(kind, items, mutable));
        if let Some(types) = self.type_by_hash.get(&hash) {
            for &ty in types.iter() {
                let info = self.info(ty);
                if info.mutable == mutable
                && info.items.iter().map(|item| &item.ty).eq(items.iter()) {
                    return ty;
                }
            }
        }
        let ty = self.make(kind);
        for &item in items {
            self.add_item_to_type(ty, Intern(0), item);
        }
        self.complete_type(ty);
        self.info_mut(ty).mutable = mutable;
        self.type_by_hash.entry(hash).or_insert_with(|| SmallVecN::new()).push(ty);
        ty
    }

    pub fn tuple(&mut self, kind: TypeKind, items: &[(Intern, Type)]) -> Type {
        let hash = hash(&(kind, items));
        if let Some(types) = self.type_by_hash.get(&hash) {
            for &ty in types.iter() {
                let info = self.info(ty);
                if items.iter().copied().eq(info.items.iter().map(|item| (item.name, item.ty))) {
                    return ty;
                }
            }
        }
        let ty = self.make(kind);
        for &(name, item_ty) in items {
            self.add_item_to_type(ty, name, item_ty);
        }
        self.complete_type(ty);
        self.type_by_hash.entry(hash).or_insert_with(|| SmallVecN::new()).push(ty);
        ty
    }

    pub fn strukt(&mut self, name: Intern) -> Type {
        let hash = hash(&(TypeKind::Struct, name));
        if let Some(types) = self.type_by_hash.get(&hash) {
            for &ty in types.iter() {
                let info = self.info(ty);
                if info.kind == TypeKind::Array && info.name == name {
                    panic!("struct made twice: calling code must check for cycles");
                }
            }
        }
        let result = self.make(TypeKind::Struct);
        self.info_mut(result).name = name;
        result
    }

    // Callers responsibility to make sure base_type.size * num_array_elements doesn't overflow (and trap).
    pub fn array(&mut self, base_type: Type, num_array_elements: usize) -> Type {
        assert!(num_array_elements > 0);
        let hash = hash(&(TypeKind::Array, base_type, num_array_elements));
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
        arr.base_type = base_type;
        arr.num_array_elements = num_array_elements;
        self.type_by_hash.entry(hash).or_insert_with(|| SmallVecN::new()).push(ty);
        ty
    }

    pub fn pointer(&mut self, base_type: Type) -> Type {
        let mut result = Type::None;
        let hash = hash(&(TypeKind::Pointer, base_type));
        if let Some(types) = self.type_by_hash.get(&hash) {
            for &ty in types.iter() {
                let info = self.info(ty);
                if info.kind == TypeKind::Pointer && info.base_type == base_type {
                    return ty;
                }
            }
        }
        {
            let ty = self.make(TypeKind::Pointer);
            let ptr = self.info_mut(ty);
            ptr.size = std::mem::size_of::<*const u8>();
            ptr.alignment = ptr.size;
            ptr.base_type = base_type;
            ptr.mutable = (ty.0 & 1) == 0;
            if ptr.mutable {
                self.type_by_hash.entry(hash).or_insert_with(|| SmallVecN::new()).push(ty);
                result = ty;
            }
        }
        {
            // Immediately make an immutable variant of the pointer.
            // This........ means self.immutable() does not have to take
            // a mutable reference to self. Not sorry
            let ty = self.make(TypeKind::Pointer);
            let ptr = self.info_mut(ty);
            ptr.size = std::mem::size_of::<*const u8>();
            ptr.alignment = ptr.size;
            ptr.base_type = base_type;
            ptr.mutable = (ty.0 & 1) == 0;
            if ptr.mutable {
                self.type_by_hash.entry(hash).or_insert_with(|| SmallVecN::new()).push(ty);
                result = ty;
            }
        }
        assert!(result != Type::None);
        result
    }

    pub fn immutable(&self, ty: Type) -> Type {
        if !ty.is_pointer() {
            return ty;
        }
        let info = self.info(ty);
        if !info.mutable {
            return ty;
        }
        let left = Type(ty.0 - 1);
        let left_info = self.info(left);
        if left_info.base_type == info.base_type
        && left_info.kind == TypeKind::Pointer {
            assert!(left_info.mutable == false);
            assert!(left.is_immutable_pointer());
            return left;
        }
        let right = Type(ty.0 + 1);
        let right_info = self.info(right);
        if right_info.base_type == info.base_type
        && right_info.kind == TypeKind::Pointer {
            assert!(right_info.mutable == false);
            assert!(right.is_immutable_pointer());
            return right;
        }
        unreachable!();
    }

    pub fn mutable(&self, ty: Type) -> Type {
        if !ty.is_pointer() {
            return ty;
        }
        let info = self.info(ty);
        if info.mutable {
            return ty;
        }
        let left = Type(ty.0 - 1);
        let left_info = self.info(left);
        if left_info.base_type == info.base_type
        && left_info.kind == TypeKind::Pointer {
            assert!(left_info.mutable == true);
            assert!(!left.is_immutable_pointer());
            return left;
        }
        let right = Type(ty.0 + 1);
        let right_info = self.info(right);
        if right_info.base_type == info.base_type
        && right_info.kind == TypeKind::Pointer {
            assert!(right_info.mutable == true);
            assert!(!right.is_immutable_pointer());
            return right;
        }
        unreachable!();
    }

    pub fn copy_mutability(&self, dest: Type, src: Type) -> Type {
        if src.is_immutable_pointer() {
            self.immutable(dest)
        } else {
            self.mutable(dest)
        }
    }

    pub fn base_type(&self, ty: Type) -> Type {
        self.info(ty).base_type
    }

    pub fn add_item_to_type(&mut self, ty: Type, name: Intern, item: Type) {
        let (size, alignment) = {
            let item_info = self.info(item);
            (item_info.size, item_info.alignment)
        };
        let info = self.info_mut(ty);
        let offset = (info.size + (alignment - 1)) & !(alignment - 1);
        assert!(offset & !(alignment - 1) == offset);
        info.size = offset + size;
        info.alignment = info.alignment.max(alignment);
        info.items.push(Item { name, ty: item, offset });
    }

    pub fn complete_type(&mut self, ty: Type) {
        let info = self.info_mut(ty);
        let alignment = info.alignment;
        info.size = (info.size + (alignment - 1)) & !(alignment - 1);
        assert!(info.size & !(alignment - 1) == info.size);
        assert!(info.size as isize >= 0);
    }

    pub fn annoying_deep_eq(&self, a: Type, b: Type) -> bool {
        // This is bad side of encoding mutability in the pointer type.
        // We could do a kind of half-baked provenance thing while
        // generating code, but I don't know else how to handle cases like
        //
        // a = mutable local ptr;
        // b = immutable ptr from func args;
        // ab: SomeStruct = { a, b };
        // ab.a.v = ...; // allow
        // ab.b.v = ...; // disallow
        //
        // without actual "analysis"

        if a == b {
            return true;
        }

        if a.is_pointer() && b.is_pointer() {
            if self.immutable(a) == b {
                return true;
            }

            let a_base = self.base_type(a);
            let b_base = self.base_type(b);
            if a_base != a && b_base != b {
                // Should be very shallow in practice
                return self.annoying_deep_eq(a_base, b_base);
            }
        }

        false
    }

    pub fn types_match_with_promotion(&self, promotable: Type, other: Type) -> bool {
        self.annoying_deep_eq(promotable, other) || integer_promote(promotable) == other
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
    b: ptr u8,
    c: ptr int,
    d: ptr Ptr
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
    let aoi = c.types.info(aoa.base_type);

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
    assert_eq!(ptr.size, 32);
    assert_eq!(ptr.alignment, 8);
    assert_eq!(ptr.items[0].ty, Type::VoidPtr);
    assert_eq!(ptr.items[1].ty, Type::U8Ptr);
    assert_eq!(ptr.items[2].ty, int_ptr);
    assert_eq!(ptr.items[3].ty, ptr_ptr);
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
