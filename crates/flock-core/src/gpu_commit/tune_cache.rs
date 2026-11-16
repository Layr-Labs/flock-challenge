//! Exact warm-up configuration header. The compact source fingerprint is
//! supplemental: distinct tuning configurations must never alias through it.
const MAGIC: &[u8; 8] = b"FLKCFG02";
pub(super) type Identity = [u64; 9];
const HEADER_BYTES: usize = MAGIC.len() + 9 * core::mem::size_of::<u64>();
const MAX_POLICY_BYTES: usize = 65536;

pub(super) fn environment_policy(
    entries: impl IntoIterator<Item = (std::ffi::OsString, std::ffi::OsString)>,
) -> Option<Vec<u8>> {
    // Raw encoded bytes preserve presence, empty values and non-Unicode
    // overrides. Sort and frame each name/value, so concatenations cannot alias.
    let mut selected = Vec::new();
    let mut total = 0usize;
    for (name, value) in entries {
        let name = name.as_encoded_bytes();
        if !name.starts_with(b"FLOCK_") && name != b"RAYON_NUM_THREADS" {
            continue;
        }
        let value = value.as_encoded_bytes();
        total = total.checked_add(16)?.checked_add(name.len())?.checked_add(value.len())?;
        if total > MAX_POLICY_BYTES {
            return None;
        }
        selected.push((name.to_vec(), value.to_vec()));
    }
    selected.sort_unstable();
    let mut output = Vec::with_capacity(total);
    for (name, value) in selected {
        output.extend_from_slice(&(name.len() as u64).to_le_bytes());
        output.extend_from_slice(&name);
        output.extend_from_slice(&(value.len() as u64).to_le_bytes());
        output.extend_from_slice(&value);
    }
    Some(output)
}

pub(super) fn append_environment(output: &mut Vec<u8>, policy: &[u8]) {
    output.extend_from_slice(&(policy.len() as u64).to_le_bytes());
    output.extend_from_slice(policy);
}

pub(super) fn strip_environment<'a>(bytes: &'a [u8], expected: &[u8]) -> Option<&'a [u8]> {
    let len = usize::try_from(u64::from_le_bytes(bytes.get(..8)?.try_into().ok()?)).ok()?;
    if len > MAX_POLICY_BYTES || len != expected.len() {
        return None;
    }
    let end = 8usize.checked_add(len)?;
    if bytes.get(8..end)? != expected {
        return None;
    }
    bytes.get(end..)
}

pub(super) fn split_compatible(
    split: usize,
    default: usize,
    pinned: bool,
    autotune: bool,
    fine: bool,
    gpu_suffix: bool,
) -> bool {
    if pinned || !(split < 16 || (split < 256 && split % 16 != 0)) {
        return false;
    }
    if !autotune && split != default {
        return false;
    }
    split < 16 || (fine && gpu_suffix)
}

pub(super) fn reuse_eligible(lead: bool, measured_alignment: bool) -> bool {
    // Probe registration/success can differ even under identical environment
    // settings. Never import or publish that process-local measured decision.
    !(lead && measured_alignment)
}

pub(super) fn append_identity(output: &mut Vec<u8>, identity: Identity) {
    output.extend_from_slice(MAGIC);
    for value in identity {
        output.extend_from_slice(&value.to_le_bytes());
    }
}

pub(super) fn strip_identity(bytes: &[u8], expected: Identity) -> Option<&[u8]> {
    if bytes.len() < HEADER_BYTES {
        return None;
    }
    if bytes.get(..MAGIC.len())? != MAGIC {
        return None;
    }
    let mut offset = MAGIC.len();
    for wanted in expected {
        let actual = u64::from_le_bytes(bytes.get(offset..offset + 8)?.try_into().ok()?);
        if actual != wanted {
            return None;
        }
        offset += 8;
    }
    bytes.get(offset..)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_supported_override_changes_the_environment_policy() {
        let baseline = environment_policy([]).unwrap();
        for name in [
            "FLOCK_NO_HYBRID_FINE", "FLOCK_NO_HYBRID_AUTOTUNE",
            "FLOCK_NO_HYBRID_GPU_SUFFIX_HASH", "FLOCK_NO_GPU_SUFFIX_COALESCE",
            "FLOCK_NO_TUNE_FIT", "FLOCK_TUNE_FIT_FINE_SHARES",
            "FLOCK_NO_TUNE_CB3_LAT", "FLOCK_HYBRID_TIE_BAND",
            "FLOCK_HYBRID_KEEP_DEFAULT_BAND", "FLOCK_NO_TUNE_PRE_CB2",
            "FLOCK_HYBRID_SCORE_JOIN_WALL", "FLOCK_FUTURE_POLICY",
            "RAYON_NUM_THREADS",
        ] {
            let changed = environment_policy([(name.into(), "1".into())]).unwrap();
            assert_ne!(baseline, changed, "{name}");
        }
        let sorted = environment_policy([("FLOCK_B".into(), "x".into()), ("FLOCK_A".into(), "y".into())]).unwrap();
        let reverse = environment_policy([("FLOCK_A".into(), "y".into()), ("FLOCK_B".into(), "x".into())]).unwrap();
        assert_eq!(sorted, reverse);
    }

    #[test]
    fn different_suffix_modes_do_not_share_a_policy() {
        let cpu = environment_policy([("FLOCK_NO_HYBRID_GPU_SUFFIX_HASH".into(), "1".into())]).unwrap();
        let gpu = environment_policy([("FLOCK_NO_GPU_SUFFIX_COALESCE".into(), "1".into())]).unwrap();
        assert_ne!(cpu, gpu);
    }

    #[test]
    fn policy_header_rejects_other_environments() {
        let mut bytes = Vec::new();
        append_environment(&mut bytes, b"FLOCK_A=1");
        bytes.extend_from_slice(b"payload");
        assert_eq!(strip_environment(&bytes, b"FLOCK_A=1"), Some(b"payload".as_slice()));
        assert_eq!(strip_environment(&bytes, b"FLOCK_A=0"), None);
    }

    #[test]
    fn cached_split_must_obey_current_execution_policy() {
        assert!(!split_compatible(182, 5, false, true, false, true));
        assert!(!split_compatible(182, 5, false, true, true, false));
        assert!(!split_compatible(7, 5, false, false, true, true));
        assert!(!split_compatible(5, 5, true, true, true, true));
        assert!(split_compatible(5, 5, false, false, false, false));
        assert!(split_compatible(182, 5, false, true, true, true));
    }

    #[test]
    fn unobserved_probe_success_or_failure_cannot_import_a_split() {
        assert!(!reuse_eligible(true, true));
        assert!(reuse_eligible(true, false));
        assert!(reuse_eligible(false, true));
    }

    #[test]
    fn framed_policy_preserves_empty_and_distinct_values() {
        let absent = environment_policy([]).unwrap();
        let empty = environment_policy([("FLOCK_A".into(), "".into())]).unwrap();
        assert_ne!(absent, empty);
        let a = environment_policy([("FLOCK_A".into(), "bc".into())]).unwrap();
        let b = environment_policy([("FLOCK_Ab".into(), "c".into())]).unwrap();
        assert_ne!(a, b);
        assert_eq!(environment_policy([("PATH".into(), "ignored".into())]).unwrap(), absent);
    }

    #[test]
    fn excessive_policy_and_truncated_length_are_rejected() {
        assert!(environment_policy([("FLOCK_A".into(), "x".repeat(MAX_POLICY_BYTES).into())]).is_none());
        for end in 0..8 {
            assert!(strip_environment(&[0u8; 8][..end], &[]).is_none());
        }
        assert!(strip_environment(&u64::MAX.to_le_bytes(), &[]).is_none());
        let mut encoded = Vec::new();
        append_environment(&mut encoded, b"abc");
        assert!(strip_environment(&encoded[..encoded.len()-1], b"abc").is_none());
    }

    #[cfg(unix)]
    #[test]
    fn non_unicode_override_values_remain_distinct() {
        use std::os::unix::ffi::OsStringExt;
        let a = environment_policy([("FLOCK_A".into(), std::ffi::OsString::from_vec(vec![0xff]))]).unwrap();
        let b = environment_policy([("FLOCK_A".into(), std::ffi::OsString::from_vec(vec![0xfe]))]).unwrap();
        assert_ne!(a, b);
    }

    fn identity(phi: f64, bias: u64) -> Identity {
        [2, 1, 1, phi.to_bits(), 1, bias, 0, 1, 10]
    }

    #[test]
    fn equal_compact_fingerprints_cannot_reuse_other_configuration() {
        let phi0 = 0.5f64;
        let phi1 = f64::from_bits(phi0.to_bits() + 1);
        // Exact counterexample from the independent review.
        assert_eq!(phi0.to_bits().rotate_left(7), phi1.to_bits().rotate_left(7) ^ 128);
        let mut encoded = Vec::new();
        append_identity(&mut encoded, identity(phi0, 0));
        encoded.extend_from_slice(b"same compact fingerprint and cached split");
        assert_eq!(strip_identity(&encoded, identity(phi0, 0)), Some(b"same compact fingerprint and cached split".as_slice()));
        assert_eq!(strip_identity(&encoded, identity(phi1, 128)), None);
    }

    #[test]
    fn each_identity_word_is_checked() {
        let expected = identity(0.5, 0);
        let mut encoded = Vec::new();
        append_identity(&mut encoded, expected);
        encoded.extend_from_slice(b"payload");
        for word in 0..expected.len() {
            let mut changed = expected;
            changed[word] ^= 1;
            assert_eq!(strip_identity(&encoded, changed), None, "word {word}");
        }
    }

    #[test]
    fn truncated_or_corrupt_headers_are_rejected() {
        let expected = identity(0.5, 0);
        let mut encoded = Vec::new();
        append_identity(&mut encoded, expected);
        for end in 0..HEADER_BYTES {
            assert_eq!(strip_identity(&encoded[..end], expected), None);
        }
        encoded[0] ^= 1;
        assert_eq!(strip_identity(&encoded, expected), None);
    }

    #[test]
    fn payload_is_not_rewritten_or_truncated() {
        let expected = identity(0.5000000000000001, 255);
        let payload: Vec<u8> = (0..=255).collect();
        let mut encoded = Vec::new();
        append_identity(&mut encoded, expected);
        encoded.extend_from_slice(&payload);
        assert_eq!(strip_identity(&encoded, expected), Some(payload.as_slice()));
    }
}
