# Tunstall

This is an incomplete prototype for a C-like language with automatic bounds checking.

## The one interesting idea

You tell the compiler where to find the bounds on pointers. From `todo.txt` (but is implemented):

```
Buf: struct (
    buf: ptr u8 [cap],
    len: int,
    cap: int
);

buf_copy: proc (buf: ptr Buf, dest: ptr u8 [buf.len]) {
    // checks buf.len < buf.cap
    memcpy(dest, buf.buf, buf.len);
}

// bounds are checked at call site
memcpy: proc (dest: ptr u8 [len], src: ptr u8 [len], len: int) {
    // dest and src shared a bound so no bounds checking
}
```

This is most interesting when ptrs share a bound, for example:
```
dot: proc (a: ptr f32 [len], b: ptr f32 [len], len: int) -> f32 {
    sum := 0.0;
    for (i := 0; i < len; i++) {
        sum += a[i] * b[i];
    }
    return sum;
} 
```

Here, `i < len` is always true so this function can be compiled without bound checks (this isn't implemented). Compared to slices, slices come in with their own lengths and you end up relying on the optimizer and carefully written iterators to get good codegen. This is a "[10% of the effort for 70% of the performance](https://c9x.me/compile/)" ethos language.

## Status

`todo.txt` compiles and runs as one of the tests. Many more test cases in `src/main.rs`. Everything compiles to bytecode and run in a VM; the language is hermetically sealed and has no way to communicate with the outside world.

### Todo

- Some story for real code spread over multiple files
- Real error messages
- Constants and enums
- Full set of signed/unsigned integer operators. (In this language, all `int`s are treated as being register sized, and the `u8`/`i8`/etc types are for 1) loading and sign extending integers read out of structs into `int`s, and 2) convenience for truncating and sign extending. Consequently, you also need to cast to truncate integers when writing to structs. This last bit might be a bad tradeoff: if, for example, you widen the width of a struct member, you are now over-truncating on already existing assignments. Mandating that assignments exactly match the integer width might be better)
- Probably C output
- Do more with tagging types with expressions that evaluate to values/memory locations at use-time
