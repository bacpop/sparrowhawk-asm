#[cfg(not(target_family = "wasm"))]
fn main() {
    sparrowhawk_asm::run_cli();
}
#[cfg(target_family = "wasm")]
fn main() {
    panic!("The WebAssembly build is used through AssemblyHelper");
}
