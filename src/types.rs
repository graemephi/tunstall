use std::collections::hash_map::HashMap;
use std::convert::TryFrom;

use crate::Compiler;
use crate::Intern;
use crate::hash;

use crate::sym;

use crate::smallvec::*;

pub fn builtins(ctx: &mut Compiler) {
    let builtins = [
        ("_",    TypeKind::None, 0),
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
        ctx.types.types.push(TypeInfo { kind, name, size: size, alignment: size.max(1), items, base_type: Type::None, num_array_elements: 0 });
        assert!(size == 0 || size.is_power_of_two());
        sym::builtin(ctx, name, ty);
    }
}

#[derive(Copy, Clone, Default, PartialEq, Eq, Debug, Hash)]
pub struct Type(u32);

#[allow(non_upper_case_globals)]
impl Type {
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

    pub fn is_integer(self) -> bool {
        matches!(self, Type::I8|Type::I16|Type::I32|Type::I64|Type::U8|Type::U16|Type::U32|Type::U64|Type::Int)
    }

    pub fn is_basic(self) -> bool {
        self.0 <= TypeKind::Bool as u32
    }
}

pub fn integer_promote(ty: Type) -> Type {
    match ty {
        Type::I8|Type::I16|Type::I32|Type::I64|Type::U8|Type::U16|Type::U32|Type::U64 => Type::Int,
        ty => ty
    }
}

pub fn types_match_with_promotion(promotable: Type, other: Type) -> bool {
    promotable == other || integer_promote(promotable) == other
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
    Func,
    Struct,
    Array,
}

#[derive(Clone, Copy, Debug)]
pub struct Item {
    pub name: Intern,
    pub ty: Type,
    pub offset: usize
}

#[derive(Clone, Debug)]
pub struct TypeInfo {
    pub kind: TypeKind,
    pub name: Intern,
    pub size: usize,
    pub alignment: usize,
    pub items: Vec<Item>,
    pub base_type: Type,
    pub num_array_elements: usize,
}

impl TypeInfo {
    pub fn arguments(&self) -> Option<&[Item]> {
        if self.kind == TypeKind::Func {
            let end = if self.items.len() == 0 { 0 } else { self.items.len() - 1 };
            Some(&self.items[..end])
        } else {
            None
        }
    }

    pub fn return_type(&self) -> Option<Type> {
        if self.kind == TypeKind::Func {
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
        &self.types[ty.0 as usize]
    }

    fn info_mut(&mut self, ty: Type) -> &mut TypeInfo {
        &mut self.types[ty.0 as usize]
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
                        offset: index * element_size
                    })
                } else {
                    None
                }
            }
            _ => None
        }
    }

    pub fn make(&mut self, kind: TypeKind, name: Intern) -> Type {
        let result = u32::try_from(self.types.len()).expect("Program too big!!");
        self.types.push(TypeInfo {
            kind: kind,
            name: name,
            size: 0,
            alignment: 1,
            items: Vec::new(),
            base_type: Type::None,
            num_array_elements: 0,
        });
        Type(result)
    }

    pub fn make_anonymous(&mut self, kind: TypeKind, items: &[Type]) -> Type {
        let hash = hash(&(kind, items));
        if let Some(types) = self.type_by_hash.get(&hash) {
            for &ty in types.iter() {
                let info = self.info(ty);
                if info.items.iter().map(|item| &item.ty).eq(items.iter()) {
                    return ty;
                }
            }
        }
        let ty = self.make(kind, Intern(0));
        for &item in items {
            self.add_item_to_type(ty, Intern(0), item);
        }
        self.type_by_hash.entry(hash).or_insert_with(|| SmallVecN::new()).push(ty);
        ty
    }

    pub fn make_array(&mut self, base_type: Type, num_array_elements: usize) -> Type {
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
        let ty = self.make(TypeKind::Array, Intern(0));
        let (base_size, alignment) = {
            let base = self.info(base_type);
            (base.size, base.alignment)
        };
        let arr = self.info_mut(ty);
        arr.size = base_size * num_array_elements;
        arr.alignment = alignment;
        arr.base_type = base_type;
        arr.num_array_elements = num_array_elements;
        self.type_by_hash.entry(hash).or_insert_with(|| SmallVecN::new()).push(ty);
        ty
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
    }
}

#[test]
fn types() {
    let code = r#"
struct V2 {
    x: f32,
    y: f32
}

struct Rect {
    top_left: V2,
    bottom_right: V2
}

struct Rect2 {
    verts: arr f32 [4],
    value: f32
}

struct Arr {
    arr1: arr (arr i32 [3]) [7],
    arr2: arr (arr i32 [3]) [7]
}

struct Padding {
    a: i16,
    b: i32,
    c: i64,
    d: i8
}

struct Padding2 {
    a: i16,
    b: Padding,
    c: i16,
}

func main(): int { return 0; }
    "#;

    let mut c = crate::compile(&code).unwrap();
    let v2 = c.intern("V2");
    let rect = c.intern("Rect");
    let rect2 = c.intern("Rect2");
    let arr = c.intern("Arr");
    let padding = c.intern("Padding");
    let padding2 = c.intern("Padding2");

    let get = |name: Intern| c.symbols.get(&name).map(|sym| c.types.info(sym.ty)).unwrap();
    let v2 = get(v2);
    let rect = get(rect);
    let rect2 = get(rect2);
    let arr = get(arr);
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
