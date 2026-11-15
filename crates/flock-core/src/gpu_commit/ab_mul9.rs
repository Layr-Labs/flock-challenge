mod ab_mul9 {
    use std::borrow::Cow;
    use std::ffi::OsStr;

    const PRIMITIVE: &str = include_str!("ab_mul9.metal");

    // Select once during the existing library build; dispatches use the
    // resulting pipelines directly, without a multiplier selector.
    pub(super) fn source(original: &'static str) -> Cow<'static, str> {
        source_for(original, std::env::var_os("FLOCK_NO_AB_GPU_MUL9").as_deref())
    }

    fn source_for(original: &'static str, control: Option<&OsStr>) -> Cow<'static, str> {
        if control == Some(OsStr::new("1")) {
            return Cow::Borrowed(original);
        }
        let start = original
            .find("static inline ulong clmul32(")
            .expect("AB multiplier");
        let end = start
            + original[start..]
                .find("\nstruct U128k")
                .expect("AB product limbs");
        let mut source = String::with_capacity(original.len() + PRIMITIVE.len());
        source.push_str(&original[..start]);
        source.push_str(PRIMITIVE);
        source.push_str(&original[end..]);
        Cow::Owned(source)
    }

    #[cfg(test)]
    mod tests {
        use super::super::*;
        use super::*;

        #[test]
        fn ab_mul9_source_selection() {
            let off = source_for(ZC_AB_MSL_SOURCE, Some(OsStr::new("1")));
            assert!(matches!(off, Cow::Borrowed(_)));
            assert_eq!(off.as_ref(), ZC_AB_MSL_SOURCE);
            let suffix = ZC_AB_MSL_SOURCE.find("\nstruct U128k").unwrap();
            for control in [None, Some(""), Some("0"), Some("01"), Some("true")] {
                let on = source_for(ZC_AB_MSL_SOURCE, control.map(OsStr::new));
                assert!(on.contains(PRIMITIVE));
                assert!(on.ends_with(&ZC_AB_MSL_SOURCE[suffix..]));
                assert_eq!(on.matches("static inline ulong clmul32(").count(), 1);
            }
        }

        include!("ab_mul9_tests.rs");
    }
}
