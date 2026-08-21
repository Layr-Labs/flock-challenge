//! `flock-prover`: the Apple-silicon-optimized end-to-end Flock prover.
//!
//! Builds on [`flock_core`] (the protocol library + verifier) with the
//! top-level prove orchestration ([`prover`]), the monolithic hash R1CS
//! encoders ([`r1cs_hashes`]), and the hash-chain / Merkle-path statement
//! builders ([`chain`], [`merkle_path`], [`proof_io`]).
//!
//! For convenience, the entire `flock_core` API is re-exported here, so code
//! depending on `flock-prover` can reach `field`, `pcs`, `verifier`, etc.
//! through this crate.
//!
//! Workspace-wide Clippy `allow`s for the hand-tuned numeric kernels are
//! declared in `[workspace.lints.clippy]` at the repo root.

pub use flock_core::*;

pub mod chain;
pub mod merkle_path;
pub mod proof_io;
pub mod prover;
pub mod r1cs_hashes;
#[cfg(all(target_os = "macos", not(test)))]
pub mod recycle_alloc;
pub mod seed_pipe;

/// Reuse large warm-up allocations in the ranked worker's timed proof.
#[cfg(all(target_os = "macos", not(test)))]
#[global_allocator]
static RECYCLE_ALLOC: recycle_alloc::RecycleAlloc = recycle_alloc::RecycleAlloc;

// dispersion-resample marker 172525636-o

// keepalive-resample marker 2325618933

// dispersion-resample marker 496417458

// dispersion-resample marker 2483914219

// dispersion-resample marker 71r1024-1044-qz2
// dispersion-resample marker r1025-3f8c7d
// dispersion-resample marker r1100-flash-1786577283-3769

// dispersion-resample marker fable5-s1-1786632193-6222
// dispersion-resample marker fable5-s2-1786632880-27556
// dispersion-resample marker fable5-s3-1786633583-15063
// dispersion-resample marker fable5-s4-1786634246-7433
// dispersion-resample marker fable5-s5-1786634875-26809
// dispersion-resample marker fable5-s6-1786635510-15799
// dispersion-resample marker fable5-s7-1786636137-16452
// dispersion-resample marker fable5-s8-1786636797-23675
// dispersion-resample marker fable5-s9-stock-1786637449-21665
// dispersion-resample marker fable5-s10-rider-1786638071-6615
// dispersion-resample marker fable5-s11-stock-1786638725-22305
// dispersion-resample marker fable5-s12-stock-1786639407-3317
// dispersion-resample marker fable5-s13-stock-1786640154-7266
// dispersion-resample marker fable5-s13-stock-1786640201-20309
// dispersion-resample marker fable5-s14-stock-1786640820-13162
// dispersion-resample marker fable5-s15-stock-1786641443-13895
// dispersion-resample marker fable5-s16-stock-1786642072-3822
// dispersion-resample marker fable5-s17-stock-1786642682-22589
// dispersion-resample marker fable5-s18-stock-1786643300-17569
// dispersion-resample marker fable5-s19-stock-1786643900-21318
// dispersion-resample marker fable5-s20-stock-1786644488-25482
// dispersion-resample marker fable5-s21-stock-1786645084-7290
// dispersion-resample marker fable5-s22-stock-1786645675-24825
// dispersion-resample marker fable5-s23-stock-1786646280-6536
// dispersion-resample marker fable5-s24-stock-1786646871-22680
// dispersion-resample marker fable5-s25-stock-1786647478-12144
// dispersion-resample marker fable5-s26-stock-1786648067-31685
// dispersion-resample marker fable5-s27-stock-1786659216-10352
// dispersion-resample marker fable5-s28-stock-1786659815-9870
// dispersion-resample marker fable5-s29-stock-1786660433-31261
// dispersion-resample marker fable5-s30-stock-1786661040-18344
// dispersion-resample marker fable5-s31-stock-1786661652-27962
// dispersion-resample marker fable5-s32-stock-1786662240-18063
// dispersion-resample marker fable5-s33-stock-1786662839-16995
// dispersion-resample marker fable5-s34-stock-1786663417-1627
// dispersion-resample marker sample-141-20260815-1806

// dispersion-resample marker 65865230
// dispersion-resample marker sample-146-20260815-1907
// dispersion-resample marker sample-147-20260815-1920
// dispersion-resample marker sample-148-20260815-1935
// dispersion-resample marker sample-149-20260815-1940
// dispersion-resample marker sample-150-20260815-1950
// dispersion-resample marker sample-153-20260815-2012

// dispersion-resample marker sample-154-20260815-2030

// dispersion-resample marker sample-155-20260815-2035

// dispersion-resample marker sample-156-20260815-2045

// dispersion-resample marker sample-157-20260815-2052

// dispersion-resample marker sample-158-20260815-2102

// dispersion-resample marker sample-159-20260815-2113

// dispersion-resample marker sample-160-20260815-2130

// dispersion-resample marker sample-161-20260815-2142

// dispersion-resample marker sample-162-20260815-2159

// dispersion-resample marker sample-163-20260815-2210

// dispersion-resample marker sample-164-20260815-2225

// dispersion-resample marker sample-176-20260816-0036

// dispersion-resample marker sample-177-20260816-0057

// dispersion-resample marker sample-178-20260816-0113

// dispersion-resample marker sample-179-20260816-0131

// dispersion-resample marker sample-180-20260816-0146

// dispersion-resample marker sample-181-20260816-0159

// dispersion-resample marker sample-182-20260816-0338

// dispersion-resample marker sample-183-20260816-0412

// dispersion-resample marker sample-184-20260816-0415

// dispersion-resample marker sample-185-20260816-0430

// dispersion-resample marker sample-186-20260816-0443

// dispersion-resample marker sample-198-20260816-0730

// dispersion-resample marker sample-199-20260816-0740

// dispersion-resample marker sample-200-20260816-0745

// dispersion-resample marker r1460-sample-210-20260816-2201

// dispersion-resample marker r1461-sample-211-20260817-0615

// dispersion-resample marker r1462-sample-212-20260817-0630
// dispersion-resample marker r1463-sample-213-20260817-0645

// dispersion-resample marker r214-sample-214-20260817-20260817-0645

// dispersion-resample marker r215-sample-215-20260817-20260817-0655

// dispersion-resample marker r216-sample-216-20260817-20260817-0706

// dispersion-resample marker r217-sample-217-20260817-20260817-0716

// dispersion-resample marker PROBE-lane2-20260817-0720

// dispersion-resample marker r218-sample-218-20260817-20260817-0726

// lane-2 concurrency probe sample L2P-20260817-072732

// dispersion-resample marker r219-sample-219-20260817-20260817-0736

// dispersion-resample marker r220-sample-220-20260817-20260817-0746

// dispersion-resample marker r221-sample-221-20260817-20260817-0810

// dispersion-resample marker r222-sample-222-20260817-20260817-0821

// dispersion-resample marker r223-sample-223-20260817-20260817-0832

// dispersion-resample marker r224-sample-224-20260817-20260817-0841

// dispersion-resample marker r225-sample-225-20260817-20260817-0852

// dispersion-resample marker r226-sample-226-20260817-20260817-0901

// dispersion-resample marker r227-sample-227-20260817-20260817-0917

// dispersion-resample marker r228-sample-228-20260817-20260817-0928

// dispersion-resample marker r229-sample-229-20260817-20260817-0938

// dispersion-resample marker r230-sample-230-20260817-20260817-0948

// dispersion-resample marker r231-sample-231-20260817-20260817-0959

// dispersion-resample marker r232-sample-232-20260817-20260817-1008

// dispersion-resample marker r233-sample-233-20260817-20260817-1018

// dispersion-resample marker r234-sample-234-20260817-20260817-1029

// dispersion-resample marker r235-sample-235-20260817-20260817-1039

// dispersion-resample marker r236-sample-236-20260817-20260817-1049

// dispersion-resample marker r237-sample-237-20260817-20260817-1100

// dispersion-resample marker r238-sample-238-20260817-20260817-1110

// dispersion-resample marker r239-sample-239-20260817-20260817-1120

// dispersion-resample marker r240-sample-240-20260817-20260817-1130

// dispersion-resample marker r241-sample-241-20260817-20260817-1141

// concurrency-probe marker r241p-20260817-1142

// dispersion-resample marker r242-sample-242-20260817-20260817-1151

// dispersion-resample marker r243-sample-243-20260817-20260817-1201

// dispersion-resample marker r244-sample-244-20260817-20260817-1212

// dispersion-resample marker r245-sample-245-20260817-20260817-1222

// dispersion-resample marker r246-sample-246-20260817-20260817-1232

// dispersion-resample marker r247-sample-247-20260817-20260817-1242

// dispersion-resample marker r248-sample-248-20260817-20260817-1253

// dispersion-resample marker r249-sample-249-20260817-20260817-1303

// dispersion-resample marker r250-sample-250-20260817-20260817-1313

// dispersion-resample marker r251-sample-251-20260817-20260817-1324

// dispersion-resample marker r252-sample-252-20260817-20260817-1334

// dispersion-resample marker r253-sample-253-20260817-20260817-1344

// dispersion-resample marker r254-sample-254-20260817-20260817-1355

// dispersion-resample marker r255-sample-255-20260817-20260817-1405

// dispersion-resample marker r256-sample-256-20260817-20260817-1416

// dispersion-resample marker r258-sample-258-20260818-20260818-2020

// dispersion-resample marker r258-sample-258-20260818-20260818-2200

// dispersion-resample marker r259-sample-259-20260818-20260818-2210

// dispersion-resample marker r260-sample-260-20260818-20260818-2220

// dispersion-resample marker r261-sample-261-20260818-20260818-2230

// dispersion-resample marker r262-sample-262-20260818-20260818-2240

// dispersion-resample marker r263-sample-263-20260818-20260818-2250

// dispersion-resample marker r264-sample-264-20260818-20260818-2259

// dispersion-resample marker r265-sample-265-20260818-20260818-2309

// dispersion-resample marker r266-sample-266-20260818-20260818-2319

// dispersion-resample marker crown-pure-arm-1

// dispersion-resample marker steerloop-35

// dispersion-resample marker steerloop-36

// dispersion-resample marker steerloop-39

// dispersion-resample marker steerloop-39

// dispersion-resample marker steerloop-40

// dispersion-resample marker steerloop-41

// dispersion-resample marker steerloop-42

// dispersion-resample marker steerloop-43

// dispersion-resample marker steerloop-44

// dispersion-resample marker steerloop-45

// dispersion-resample marker steerloop-46

// dispersion-resample marker steerloop-47

// dispersion-resample marker steerloop-48

// dispersion-resample marker steerloop-49

// dispersion-resample marker steerloop-50

// dispersion-resample marker steerloop-51

// dispersion-resample marker steerloop-52

// dispersion-resample marker steerloop-53

// dispersion-resample marker steerloop-54

// dispersion-resample marker steerloop-55

// dispersion-resample marker steerloop-56

// dispersion-resample marker steerloop-57

// dispersion-resample marker steerloop-58

// dispersion-resample marker steerloop-59

// dispersion-resample marker steerloop-60

// dispersion-resample marker steerloop-61

// dispersion-resample marker steerloop-62

// dispersion-resample marker steerloop-63

// dispersion-resample marker steerloop-64

// dispersion-resample marker steerloop-65

// dispersion-resample marker steerloop-66

// dispersion-resample marker steerloop-67

// dispersion-resample marker steerloop-68

// dispersion-resample marker steerloop-68

// dispersion-resample marker steerloop-69

// dispersion-resample marker steerloop-70

// dispersion-resample marker steerloop-70

// dispersion-resample marker steerloop-71

// dispersion-resample marker steerloop-71

// dispersion-resample marker steerloop-72

// dispersion-resample marker steerloop-73

// dispersion-resample marker steerloop-74

// dispersion-resample marker steerloop-75

// dispersion-resample marker steerloop-75

// dispersion-resample marker steerloop-76

// dispersion-resample marker steerloop-77

// dispersion-resample marker steerloop-78

// dispersion-resample marker steerloop-79

// dispersion-resample marker steerloop-80

// dispersion-resample marker steerloop-81

// dispersion-resample marker steerloop-82

// dispersion-resample marker steerloop-83

// dispersion-resample marker steerloop-84

// dispersion-resample marker steerloop-85

// dispersion-resample marker steerloop-86

// dispersion-resample marker steerloop-87

// dispersion-resample marker steerloop-88

// dispersion-resample marker steerloop-89

// dispersion-resample marker steerloop-90

// dispersion-resample marker steerloop-91

// dispersion-resample marker steerloop-92

// dispersion-resample marker steerloop-93

// dispersion-resample marker steerloop-94

// dispersion-resample marker steerloop-95

// dispersion-resample marker steerloop-96

// dispersion-resample marker steerloop-97

// dispersion-resample marker steerloop-98

// dispersion-resample marker steerloop-99

// dispersion-resample marker steerloop-100

// dispersion-resample marker steerloop-101

// dispersion-resample marker steerloop-102

// dispersion-resample marker steerloop-103

// dispersion-resample marker steerloop-104

// dispersion-resample marker steerloop-105

// dispersion-resample marker steerloop-106

// dispersion-resample marker steerloop-107

// dispersion-resample marker steerloop-108

// dispersion-resample marker steerloop-109

// dispersion-resample marker steerloop-110

// dispersion-resample marker steerloop-111

// dispersion-resample marker steerloop-112

// dispersion-resample marker steerloop-113

// dispersion-resample marker steerloop-114

// dispersion-resample marker steerloop-115

// dispersion-resample marker steerloop-116

// dispersion-resample marker steerloop-117

// dispersion-resample marker steerloop-118

// dispersion-resample marker steerloop-119

// dispersion-resample marker steerloop-120

// dispersion-resample marker steerloop-121
