# Private hook validation

The internal `tract_ffi` Rust crate is not a crates.io release unit. Its crate and
library names remain unchanged to preserve the existing Dart hook asset and C ABI.
`publish = false` prevents accidentally publishing under an unrelated registry name.
The Dart packages remain the consumer-facing distribution units.

Private CI checks the existing hook package with the provisioned Rust 1.94.1
and declared targets before invoking the builder. The hook dependency can otherwise
ask rustup to install missing components; the preflight fails before that path.
CI runs Rust checks, existing Dart/Flutter tests, JavaScript compilation, and the
Linux example's inference integration test without installing shared SDKs.
No GitHub write, registry upload, signing, or deployment step is present.

This lane establishes Linux behavior only. It does not establish Android, Apple,
Windows, offline dependency closure, signed prebuilts, or the proposed Microphone
hook pilot's coexistence/performance requirements. Existing GitHub workflow
platform templates remain historical owner-run templates; their installers are
not copied into private CI.
