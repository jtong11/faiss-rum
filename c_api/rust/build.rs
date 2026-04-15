fn main() {
    println!("cargo:rerun-if-env-changed=FAISS_C_LIB_PATH");
}
