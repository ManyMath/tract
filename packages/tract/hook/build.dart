import 'package:hooks/hooks.dart';
import 'package:native_toolchain_rust/native_toolchain_rust.dart';

/// Builds the tract_ffi Rust crate in `native/` for the target being built,
/// and bundles it as the code asset that `tract_ffi_bindings.dart` binds to.
void main(List<String> args) async {
  await build(args, (input, output) async {
    await const RustBuilder(
      assetName: 'src/backends/native/tract_ffi_bindings.dart',
    ).run(input: input, output: output);
  });
}
