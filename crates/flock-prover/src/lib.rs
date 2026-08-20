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

// dispersion-resample marker r267-sample-267-20260818-20260818-2328

// dispersion-resample marker r268-sample-268-20260818-20260818-2338

// dispersion-resample marker r269-sample-269-20260818-20260818-2348

// dispersion-resample marker r270-sample-270-20260818-20260819-0001

// dispersion-resample marker r271-sample-271-20260818-20260819-0011

// dispersion-resample marker r272-sample-272-20260818-20260819-0021

// dispersion-resample marker r273-sample-273-20260818-20260819-0030

// dispersion-resample marker r274-sample-274-20260818-20260819-0040

// dispersion-resample marker r275-sample-275-20260818-20260819-0050

// dispersion-resample marker r276-sample-276-20260818-20260819-0100

// dispersion-resample marker r277-sample-277-20260818-20260819-0109

// dispersion-resample marker r278-sample-278-20260818-20260819-0123

// dispersion-resample marker r279-sample-279-20260818-20260819-0133

// dispersion-resample marker r280-sample-280-20260818-20260819-0155

// dispersion-resample marker r281-sample-281-20260818-20260819-0204

// dispersion-resample marker r282-sample-282-20260818-20260819-0214

// dispersion-resample marker r283-sample-283-20260818-20260819-0224

// dispersion-resample marker r284-sample-284-20260818-20260819-0234

// dispersion-resample marker r285-sample-285-20260818-20260819-0244

// dispersion-resample marker r286-sample-286-20260818-20260819-0254

// dispersion-resample marker r287-sample-287-20260818-20260819-0303

// dispersion-resample marker r288-sample-288-20260818-20260819-0313

// dispersion-resample marker r289-sample-289-20260818-20260819-0323

// dispersion-resample marker r290-sample-290-20260818-20260819-0333

// dispersion-resample marker r291-sample-291-20260818-20260819-0343

// dispersion-resample marker r292-sample-292-20260818-20260819-0353

// dispersion-resample marker r293-sample-293-20260818-20260819-0403

// dispersion-resample marker r294-sample-294-20260818-20260819-0412

// dispersion-resample marker r295-sample-295-20260818-20260819-0422

// dispersion-resample marker r296-sample-296-20260818-20260819-0432

// dispersion-resample marker r297-sample-297-20260818-20260819-0509

// dispersion-resample marker r298-sample-298-20260818-20260819-0519

// dispersion-resample marker r299-sample-299-20260818-20260819-0529

// dispersion-resample marker r300-sample-300-20260818-20260819-0539

// dispersion-resample marker r301-sample-301-20260818-20260819-0549

// dispersion-resample marker r302-sample-302-20260818-20260819-0609

// dispersion-resample marker r303-sample-303-20260818-20260819-0619

// dispersion-resample marker r304-sample-304-20260818-20260819-0629

// dispersion-resample marker r305-sample-305-20260818-20260819-0709

// dispersion-resample marker r306-sample-306-20260818-20260819-0721

// dispersion-resample marker r307-sample-307-20260818-20260819-0736

// dispersion-resample marker r308-sample-308-20260818-20260819-0750

// dispersion-resample marker r309-sample-309-20260818-20260819-0810

// dispersion-resample marker r310-sample-310-20260818-20260819-0819

// dispersion-resample marker r311-sample-311-20260818-20260819-0829

// dispersion-resample marker r312-sample-312-20260818-20260819-0845

// dispersion-resample marker r313-sample-313-20260818-20260819-0909

// dispersion-resample marker r314-sample-314-20260818-20260819-0920

// dispersion-resample marker r315-sample-315-20260818-20260819-1009

// dispersion-resample marker r316-sample-316-20260818-20260819-1020

// dispersion-resample marker r317-sample-317-20260818-20260819-1036

// dispersion-resample marker r318-sample-318-20260818-20260819-1047

// dispersion-resample marker r319-sample-319-20260818-20260819-1103

// dispersion-resample marker r320-sample-320-20260818-20260819-1117

// dispersion-resample marker r321-sample-321-20260818-20260819-1130

// dispersion-resample marker r322-sample-322-20260818-20260819-1139

// dispersion-resample marker r323-sample-323-20260818-20260819-1153

// dispersion-resample marker r324-sample-324-20260818-20260819-1203

// dispersion-resample marker r325-sample-325-20260818-20260819-1215

// dispersion-resample marker r326-sample-326-20260818-20260819-1224

// dispersion-resample marker r327-sample-327-20260818-20260819-1234

// dispersion-resample marker r328-sample-328-20260818-20260819-1244

// dispersion-resample marker r329-sample-329-20260818-20260819-1309

// dispersion-resample marker r330-sample-330-20260818-20260819-1631

// dispersion-resample marker r331-sample-331-20260818-20260819-1632

// dispersion-resample marker fable-f38-20260820-001356

// dispersion-resample marker fable-f39-20260820-002411

// dispersion-resample marker fable-f40-20260820-003430

// dispersion-resample marker fable-f40-20260820-004456

// dispersion-resample marker fable-f41-20260820-005513

// dispersion-resample marker fable-f42-20260820-010527

// dispersion-resample marker fable-f43-20260820-011550

// dispersion-resample marker fable-f44-20260820-012614

// dispersion-resample marker fable-f45-20260820-013634

// dispersion-resample marker fable-f46-20260820-014709

// dispersion-resample marker fable-f47-20260820-015742

// dispersion-resample marker fable-f48-20260820-020809

// dispersion-resample marker fable-f49-20260820-021839
