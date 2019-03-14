pub mod ida;
pub mod ida_nls;
pub mod linear;
pub mod lorenz63;
pub mod nonlinear;
pub mod traits;

mod constants;
mod error;
mod norm_rms;

#[cfg(test)]
mod tests {}

mod tester {

    trait NLPT {
        fn crap(&mut self) -> ();
    }
    struct NLP {
        cj: usize,
    }
    impl NLPT for NLP {
        fn crap(&mut self) -> () {
            self.cj += 1;
        }
    }

    trait NLST {
        fn solve(&mut self, nlp: &mut impl NLPT) -> ();
    }
    struct NLS {}
    impl NLST for NLS {
        fn solve(&mut self, nlp: &mut impl NLPT) -> () {
            nlp.crap();
        }
    }

    struct IDA {
        nlp: NLP,
        nls: NLS,

        x: usize,
    }
    impl NLPT for IDA {
        fn crap(&mut self) -> () {
            self.x += 1;
        }
    }

    impl IDA {
        fn test<'a>(&'a mut self) {
            self.nls.solve(&mut self.nlp);
        }
    }

    fn test() {
        let mut ida = IDA {
            nlp: NLP { cj: 0 },
            nls: NLS {},
            x: 0,
        };

        ida.test();
    }
}
