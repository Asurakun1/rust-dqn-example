// build.rs
fn main() {
    // Instruct the linker to retain the CUDA warp_size symbol
    println!("cargo:rustc-link-arg=/INCLUDE:?warp_size@cuda@at@@YAHXZ");
}