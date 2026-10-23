#!/bin/sh
# Build the viewer. Some macOS Command Line Tools ship a default SDK newer than their linker can
# read ("tapi error: malformed file ... unknown architecture"); if so, fall back to the SDK that
# matches the running macOS.
cd "$(dirname "$0")"
if ! cargo build --release "$@" 2>/tmp/stadium-viewer-build.log; then
  if grep -q "tapi error" /tmp/stadium-viewer-build.log; then
    major=$(sw_vers -productVersion | cut -d. -f1)
    sdk=$(ls -d /Library/Developer/CommandLineTools/SDKs/MacOSX${major}*.sdk 2>/dev/null | head -1)
    echo "default SDK not linkable; using $sdk" >&2
    SDKROOT="$sdk" cargo build --release "$@"
  else
    cat /tmp/stadium-viewer-build.log >&2
    exit 1
  fi
fi
