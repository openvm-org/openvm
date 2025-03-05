use quote::quote;

pub mod air;
pub mod execution;
pub mod folder1;

struct Bing(usize, usize);

#[derive(Clone, Copy)]
struct A {
    b: &'static B,
}

struct B {
    a: A,
}

fn main() {
    println!("Hello, world!");
    /*let bing = Bing(1, 2);
    let mut x = 4;
    let mut y = 5;
    Bing(x, y) = bing;

    let Bing(a, y) = Bing(3, 4);

    let x = [] as [usize; 0];

    let a = [1, 2, 3, 4, 5];
    let b: [i32; 3];
    //b = a[1..4];*/

    let array = quote! { array_name };
    let indices = 0..3;
    let slice = quote! {
        [#(#array[#indices]),*]
    };
    println!("{}", slice);
}
