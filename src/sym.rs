use std::collections::hash_map::HashMap;
use std::collections::hash_map::Entry;

use crate::Compiler;
use crate::Type;
use crate::TypeKind;
use crate::Intern;

use crate::eval_type;
use crate::allocate_global_var;

use crate::error;

use crate::ast::*;
use crate::smallvec::*;
use crate::parse::Keytype;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Kind {
    Type,
    Value
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum State {
    Declared,
    Touched,
    Resolving,
    Resolved,
    Compiling,
    Compiled,
    Circular
}

#[derive(Clone, Copy, Debug)]
pub struct Symbol {
    pub state: State,
    pub kind: Kind,
    pub name: Intern,
    pub decl: Decl,
    pub expr: TypeExpr,
    pub ty: Type,
    pub location: isize,
}

#[derive(Debug, Default)]
pub struct Symbols {
    table: HashMap<(Kind, Intern), Symbol>,
}

impl Symbols {
    fn builtin(&mut self, name: Intern, ty: Type) {
        self.table.insert((Kind::Type, name), Symbol { state: State::Resolved, kind: Kind::Type, name, decl: Decl::BUILTIN, expr: TypeExpr::Infer, ty, location: 0 });
    }

    fn declared(&mut self, kind: Kind, name: Intern, decl: Decl) -> Decl {
        match self.table.entry((kind, name)) {
            Entry::Occupied(e) => {
                e.get().decl
            },
            Entry::Vacant(e) => {
                e.insert(Symbol { state: State::Declared, kind, name, decl, expr: TypeExpr::Infer, ty: Type::None, location: 0 });
                decl
            }
        }
    }

    fn touched(&mut self, name: Intern, ty: Type) {
        match self.table.entry((Kind::Type, name)) {
            Entry::Occupied(mut e) => {
                let sym = e.get_mut();
                assert_eq!(sym.state, State::Declared);
                sym.ty = ty;
                sym.state = State::Touched;
            },
            Entry::Vacant(_) => unreachable!()
        }
    }

    // The symbol may or may not have a type assigned to it; assign ty if none has been set. Returns the set type.
    fn resolving(&mut self, kind: Kind, name: Intern, ty: Type) -> Type {
        match self.table.entry((kind, name)) {
            Entry::Occupied(mut e) => {
                let sym = e.get_mut();
                if sym.state == State::Declared {
                    assert!(sym.ty == Type::None);
                    sym.ty = ty;
                } else {
                    assert!(sym.ty != Type::None);
                }
                sym.state = State::Resolving;
                sym.ty
            },
            Entry::Vacant(_) => unreachable!(),
        }
    }

    fn resolved(&mut self, kind: Kind, name: Intern, ty: Type, location: isize) -> &Symbol {
        match self.table.entry((kind, name)) {
            Entry::Occupied(mut e) => {
                let sym = e.get_mut();
                assert!(sym.state == State::Resolving);
                sym.state = State::Resolved;
                sym.ty = ty;
                sym.location = location;
                e.into_mut()
            },
            Entry::Vacant(_) => unreachable!(),
        }
    }

    fn get(&self, kind: Kind, name: Intern) -> Option<&Symbol> {
        self.table.get(&(kind, name))
    }

    fn get_mut(&mut self, kind: Kind, name: Intern) -> Option<&mut Symbol> {
        self.table.get_mut(&(kind, name))
    }
}

fn find_first_duplicate(ast: &Ast, items: ItemList) -> Option<Intern> {
    let items = ast.items(items);
    for i in 0..items.len() {
        for j in i+1..items.len() {
            if items[i].name == items[j].name {
                return Some(items[i].name);
            }
        }
    }
    None
}

fn resolve_callable(ctx: &mut Compiler, name: Intern, expr: TypeExpr) -> &Symbol {
    ctx.symbols.resolving(Kind::Value, name, Type::None);
    let mut decl_types = SmallVec::new();
    let kind = ctx.ast.type_expr_keytype(expr);
    let params = ctx.ast.type_expr_items(expr);
    let returns = ctx.ast.type_expr_returns(expr);
    if let Some(Keytype::Func)|Some(Keytype::Proc) = kind {
        if let Some(params) = params {
            for param in params {
                let item = ctx.ast.item(param);
                let ty = eval_type(ctx, item.expr);
                decl_types.push(ty);
            }

            if let Some(returns) = returns {
                let return_type = eval_type(ctx, returns);
                decl_types.push(return_type);
            } else {
                decl_types.push(Type::None);

                if let Some(Keytype::Func) = kind {
                    // todo: type expr pos
                    error!(ctx, 0, "func without return type");
                }
            }
            if let Some(dup) = find_first_duplicate(&ctx.ast, params) {
                error!(ctx, 0, "duplicate parameter name {} in func {}", ctx.str(dup), ctx.str(name));
            }
        } else {
            error!(ctx, 0, "{} without parameter list", kind.unwrap())
        }
    } else {
        panic!("non-callable type in callable decl");
    }
    let ty = ctx.types.signature(TypeKind::Callable, &decl_types, kind == Some(Keytype::Proc));
    ctx.symbols.resolved(Kind::Value, name, ty, 0)
}

fn resolve_var(ctx: &mut Compiler, name: Intern, expr: TypeExpr) -> &Symbol {
    ctx.symbols.resolving(Kind::Value, name, Type::None);
    let ty = eval_type(ctx, expr);
    let location = allocate_global_var(ctx, ty);
    ctx.symbols.resolved(Kind::Value, name, ty, location)
}

fn resolve_struct(ctx: &mut Compiler, name: Intern, expr: TypeExpr) -> &Symbol {
    let tentative_ty = ctx.types.next();
    let ty = ctx.symbols.resolving(Kind::Type, name, tentative_ty);
    if ty == tentative_ty {
        ctx.types.strukt(name);
    }
    if let Some(fields) = ctx.ast.type_expr_items(expr) {
        for field in fields {
            let item = ctx.ast.item(field);
            let item_type = eval_type(ctx, item.expr);
            ctx.types.add_item_to_type(ty, item.name, item_type);
        }
        if fields.len() == 0 {
            // todo: type expr pos
            error!(ctx, 0, "struct {} has no fields", ctx.str(name));
        }
        if let Some(dup) = find_first_duplicate(&ctx.ast, fields) {
            error!(ctx, 0, "{} has already been defined on struct {}", ctx.str(dup), ctx.str(name));
        }
    } else {
        error!(ctx, 0, "struct {} has no fields", ctx.str(name));
    }
    if ctx.ast.type_expr_len(expr) >= 3 {
        error!(ctx, 0, "struct {} has too many parameters for keytype struct", ctx.str(name));
    }
    ctx.types.complete_type(ty);
    ctx.symbols.resolved(Kind::Type, name, ty, 0)
}

pub fn resolve_anonymous_struct(ctx: &mut Compiler, fields: ItemList) -> Type {
    // Anonymous structs do not have a symbol, but we still have to resolve their fields.
    let mut decl_types = SmallVec::new();
    if fields.is_nonempty() {
        for field in fields {
            let item = ctx.ast.item(field);
            let item_type = eval_type(ctx, item.expr);
            decl_types.push((item.name, item_type));
        }
        if fields.len() == 0 {
            // todo: type expr pos
            error!(ctx, 0, "anonymous struct has no fields");
        }
        if let Some(dup) = find_first_duplicate(&ctx.ast, fields) {
            error!(ctx, 0, "{} has already been defined on anonymous struct", ctx.str(dup));
        }
    } else {
        error!(ctx, 0, "anonymous struct has no fields");
    }
    ctx.types.tuple(TypeKind::Struct, &decl_types)
}

fn resolve(ctx: &mut Compiler, kind: Kind, name: Intern) -> Option<&Symbol> {
    let mut decl_to_resolve = None;
    if let Some(sym) = ctx.symbols.get(kind, name) {
        match sym.state {
            State::Declared|State::Touched => decl_to_resolve = Some(sym.decl),
            State::Resolving|State::Circular => error!(ctx, 0, "{} is circular", ctx.str(name)),
            State::Resolved|State::Compiling|State::Compiled => {
                // MIRI don't care so i don't care
                return unsafe { Some(&*(sym as *const Symbol)) };
            }
        }
    } else {
        error!(ctx, 0, "cannot find {}", ctx.str(name));
    }
    if let Some(decl) = decl_to_resolve {
        return match *ctx.ast.decl(decl) {
            DeclData::Callable(func) => Some(resolve_callable(ctx, func.name, func.expr)),
            DeclData::Struct(strukt) => Some(resolve_struct(ctx, strukt.name, strukt.expr)),
            DeclData::Var(var) =>  Some(resolve_var(ctx, var.name, var.expr)),
        };
    }
    None
}

pub fn resolve_decls(ctx: &mut Compiler) {
    for decl in ctx.ast.decl_list() {
        let (kind, &name) = match ctx.ast.decl(decl) {
            DeclData::Struct(StructDecl { name, .. }) => (Kind::Type, name),
            DeclData::Callable(CallableDecl { name, .. }) => (Kind::Value, name),
            DeclData::Var(VarDecl { name, .. }) => (Kind::Value, name)
        };

        let declared = ctx.symbols.declared(kind, name, decl);
        if declared != decl {
            error!(ctx, ctx.ast.decl(decl).pos(), "{} is already been defined", ctx.str(name));
            return;
        }
    }
    for decl in ctx.ast.decl_list() {
        let (kind, &name) = match ctx.ast.decl(decl) {
            DeclData::Struct(StructDecl { name, .. }) => (Kind::Type, name),
            DeclData::Callable(CallableDecl { name, .. }) => (Kind::Value, name),
            DeclData::Var(VarDecl { name, .. }) => (Kind::Value, name)
        };

        resolve(ctx, kind, name);
    }
}

pub fn resolve_type(ctx: &mut Compiler, name: Intern) -> Type {
    if let Some(&Symbol{ kind: Kind::Type, ty, .. }) = resolve(ctx, Kind::Type, name) {
        return ty;
    }
    assert!(ctx.have_error());
    Type::None
}

pub fn builtin(ctx: &mut Compiler, name: Intern, ty: Type) {
    ctx.symbols.builtin(name, ty);
}

#[allow(dead_code)]
pub fn lookup_type(ctx: &Compiler, name: Intern) -> Option<&Symbol> {
    ctx.symbols.get(Kind::Type, name)
}

pub fn lookup_type_mut(ctx: &mut Compiler, name: Intern) -> Option<&mut Symbol> {
    ctx.symbols.get_mut(Kind::Type, name)
}

pub fn touch_type(ctx: &mut Compiler, name: Intern) -> Type {
    lookup_type_mut(ctx, name).map(|sym| (sym.state, sym.ty)).map(|(state, ty)| {
        if state == State::Declared {
            // We don't know what this is yet. But we don't have anything it could otherwise be (enum, union, ?).
            assert!(ty == Type::None);
            let ty = ctx.types.strukt(name);
            ctx.symbols.touched(name, ty);
            ty
        } else {
            assert!(ty != Type::None);
            ty
        }
    }).unwrap_or_else(|| {
        error!(ctx, 0, "could not find type '{}'", ctx.str(name));
        Type::None
    })
}

pub fn lookup_value(ctx: &Compiler, name: Intern) -> Option<&Symbol> {
    ctx.symbols.get(Kind::Value, name)
}

pub fn compiling(ctx: &mut Compiler, name: Intern) -> &Symbol {
    let sym = ctx.symbols.get_mut(Kind::Value, name).expect("tried to compile name without symbol");
    if sym.state == State::Resolved {
        sym.state = State::Compiling;
    } else if sym.state == State::Compiling {
        sym.state = State::Circular;
    } else {
        assert!(sym.state == State::Compiled);
    }
    sym
}

pub fn compiled(ctx: &mut Compiler, name: Intern) {
    let sym = ctx.symbols.get_mut(Kind::Value, name).expect("tried to compile name without symbol");
    assert!(sym.state == State::Compiling);
    sym.state = State::Compiled;
}
