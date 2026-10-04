#
# To learn more about a Podspec see http://guides.cocoapods.org/syntax/podspec.html.
# Run `pod lib lint tract_flutter.podspec` to validate before publishing.
#
Pod::Spec.new do |s|
  s.name             = 'tract_flutter'
  s.version          = '0.0.1'
  s.summary          = 'Neural network inference for Flutter via the Rust tract engine.'
  s.description      = <<-DESC
Builds the tract_ffi Rust crate (the sonos/tract ONNX/NNEF inference engine)
with cargokit and links it into the plugin for use over dart:ffi.
                       DESC
  s.homepage         = 'https://github.com/ManyMath/tract'
  s.license          = { :file => '../LICENSE' }
  s.author           = 'ManyMath LLC'

  s.source           = { :path => '.' }
  s.source_files     = 'Classes/**/*'

  # Build the shared `tract_ffi` Rust crate (static lib) via cargokit and
  # force-link it into the plugin framework.
  s.script_phase = {
    :name => 'Build tract_ffi',
    :script => 'sh "$PODS_TARGET_SRCROOT/../cargokit/build_pod.sh" ../../../native/tract_ffi tract_ffi',
    :execution_position => :before_compile,
    :input_files => ['${BUILT_PRODUCTS_DIR}/cargokit_phony'],
    :output_files => ['${BUILT_PRODUCTS_DIR}/libtract_ffi.a'],
  }

  s.dependency 'Flutter'
  s.platform = :ios, '13.0'

  # Flutter.framework does not contain a i386 slice.
  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386',
    'OTHER_LDFLAGS' => '-force_load ${BUILT_PRODUCTS_DIR}/libtract_ffi.a',
  }
  s.swift_version = '5.0'
end
