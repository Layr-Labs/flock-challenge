use crate::challenger::Challenger;
use crate::field::F128;
use crate::lincheck::{self, QuirkyPoint};
use crate::pcs::{self, Commitment};
use crate::r1cs::BlockR1cs;
use crate::zerocheck;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct R1csProofLigerito {
    pub zerocheck: zerocheck::ZerocheckProof,
    pub lincheck: lincheck::LincheckProof,
    pub pcs_open: pcs::BatchOpeningProofLigerito,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ZClaim {
    pub point: QuirkyPoint,
    pub value: F128,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct R1csClaim {
    pub ab: ZClaim,

    pub c: ZClaim,
}

pub fn bind_statement<Ch: Challenger>(
    challenger: &mut Ch,
    r1cs: &BlockR1cs,
    commitment: &Commitment,
) {
    challenger.observe_label(b"flock-r1cs-v0");
    challenger.observe_bytes(&r1cs.statement_digest());
    challenger.observe_bytes(&commitment.root);
}
