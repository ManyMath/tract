fn main() {
    // wasm32-unknown-unknown links a 1 MiB stack by default. ONNX protobuf
    // decoding and tract's graph optimizer traverse deep expression trees,
    // and a stack overflow traps the WASM instance, so reserve 8 MiB of stack.
    if std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("wasm32") {
        println!("cargo:rustc-link-arg-cdylib=-zstack-size={}", 8 << 20);
    }
}
