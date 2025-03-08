// use crate::parser::LanguageParser;

pub mod air;
pub mod execution;
pub mod folder1;
// pub mod parser;

struct Bing(usize, usize);

/*#[derive(Clone, Copy)]
struct A {
    b: &'static mut B,
}

struct B {
    a: A,
}

enum C {
    D(usize, usize),
}

struct E {
    f: usize,
    g: usize,
}

impl E {
    fn assign(&mut self, c: C) {
        match c {
            C::D(a, b) => {}
        }
    }
}*/

use pest::Parser;
use pest_derive::Parser;

#[derive(Parser)]
#[grammar_inline = r#"
// your grammar here
a = { "a" }
number_literal = @{ "-"? ~ '0'..'9'+ }
plus = { "+" }
minus = { "-" }
times = { "*" }
expr = { number_literal ~ ((plus | minus | times) ~ expr)? }

"#]
struct MyParser;

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

    /*let array = quote! { array_name };
    let indices = 0..3;
    let slice = quote! {
        [#(#array[#indices]),*]
    };
    println!("{}", slice);*/

    /*let not_a_program = "wef wbwoeif";
    let parse_result = LanguageParser::parse(Rule::program, not_a_program);
    println!("{:?}", parse_result);*/

    let parse_result = MyParser::parse(Rule::expr, "1+2*3");
    println!("{:?}", parse_result);
}
