#[cfg(target_arch = "wasm32")]
fn main() {
    return;    
}


#[cfg(not(target_arch = "wasm32"))]
#[tokio::main]
async fn main() {
    sparrowhawk::main().await;
}
