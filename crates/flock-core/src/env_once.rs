//! Environment reads kept per call site for the life of the process.
//!
//! [`env_var_once!`](crate::env_var_once) stands in for `std::env::var_os` at
//! a call site that runs on every proof: the site reads the environment on its
//! first call and keeps the value in a static of its own, so later calls do
//! not read the environment. It returns the same `Option<OsString>`.
//!
//! The cached value is exact. Production code never sets or removes an
//! environment variable after start-up: every `set_var` / `remove_var` in
//! these crates is inside a test module, and a test build of the calling crate
//! reads the environment on every call, so those tests still see their own
//! changes. A variable that a test in another crate changes at run time keeps
//! a plain `std::env::var_os` read, because that crate's tests build this one
//! without `cfg(test)`.
//!
//! Kill switch: [`ENV_NO_ENV_SITE_CACHE`].

/// Kill switch for [`env_var_once!`](crate::env_var_once)
/// (`FLOCK_NO_ENV_SITE_CACHE`, any value): every site reads the environment on
/// each call, as before. Read once per process.
pub const ENV_NO_ENV_SITE_CACHE: &str = "FLOCK_NO_ENV_SITE_CACHE";

/// Whether [`env_var_once!`](crate::env_var_once) sites return their cached
/// value ([`ENV_NO_ENV_SITE_CACHE`] unset). Read once per process.
pub fn enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os(ENV_NO_ENV_SITE_CACHE).is_none());
    *ON
}

/// `std::env::var_os($name)`, read once per process at this call site (see
/// [`env_once`](crate::env_once)). `$name` must be a constant: a string literal
/// or a `const` item. A test build of the calling crate, or a process with
/// [`ENV_NO_ENV_SITE_CACHE`](crate::env_once::ENV_NO_ENV_SITE_CACHE) set,
/// reads the environment on every call.
#[macro_export]
macro_rules! env_var_once {
    ($name:expr) => {{
        static ENV_ONCE_VALUE: ::std::sync::LazyLock<::std::option::Option<::std::ffi::OsString>> =
            ::std::sync::LazyLock::new(|| ::std::env::var_os($name));
        if ::core::cfg!(test) || !$crate::env_once::enabled() {
            ::std::env::var_os($name)
        } else {
            ::std::clone::Clone::clone(&*ENV_ONCE_VALUE)
        }
    }};
}
