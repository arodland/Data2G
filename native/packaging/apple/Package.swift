// swift-tools-version:5.9
// The embedding library (native/embed, docs/EMBEDDING.md) as a Swift package: module CData2G,
// the C ABI of embed/include/CData2G/data2g.h. build-xcframework.sh builds the binary first.
import PackageDescription

let package = Package(
    name: "Data2G",
    platforms: [
        // the first with the floating-point std::to_chars the core uses
        .iOS("16.3"),
        .macOS("13.3"),
    ],
    products: [
        .library(name: "CData2G", targets: ["CData2G"]),
    ],
    targets: [
        .binaryTarget(
            name: "CData2G",
            path: "build/Data2G.xcframework"
        ),
        .testTarget(
            name: "CData2GTests",
            dependencies: ["CData2G"],
            path: "Tests/CData2GTests"
        ),
    ]
)
