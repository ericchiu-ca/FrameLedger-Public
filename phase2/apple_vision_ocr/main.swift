import CoreGraphics
import CoreImage
import Darwin
import Foundation
import ImageIO
import Vision

private let outputSchemaVersion = 1
private let helperProtocol = "frameledger-ocr-helper-v1"

private enum HelperError: Error {
    case usage(String)
    case input(String)
    case unsupportedLanguages(requested: [String], supported: [String])
    case vision(Error)
    case encoding(Error)

    var exitCode: Int32 {
        switch self {
        case .usage:
            return 64 // EX_USAGE
        case .input, .unsupportedLanguages:
            return 65 // EX_DATAERR
        case .vision:
            return 69 // EX_UNAVAILABLE
        case .encoding:
            return 70 // EX_SOFTWARE
        }
    }

    var payload: ErrorPayload {
        switch self {
        case let .usage(message):
            return ErrorPayload(code: "usage", message: message)
        case let .input(message):
            return ErrorPayload(code: "invalid_input", message: message)
        case let .unsupportedLanguages(requested, supported):
            return ErrorPayload(
                code: "unsupported_languages",
                message: "One or more requested recognition languages are unsupported",
                requestedLanguages: requested,
                supportedLanguages: supported
            )
        case let .vision(error):
            let native = error as NSError
            return ErrorPayload(
                code: "vision_request_failed",
                message: native.localizedDescription,
                nativeDomain: native.domain,
                nativeCode: native.code
            )
        case let .encoding(error):
            let native = error as NSError
            return ErrorPayload(
                code: "json_encoding_failed",
                message: native.localizedDescription,
                nativeDomain: native.domain,
                nativeCode: native.code
            )
        }
    }
}

private struct Arguments {
    let imagePath: String
    let languages: [String]

    static func parse(_ raw: [String]) throws -> Arguments {
        var imagePath: String?
        var languages: [String] = []
        var index = 0

        while index < raw.count {
            let option = raw[index]
            switch option {
            case "--image":
                guard index + 1 < raw.count else {
                    throw HelperError.usage("--image requires one PNG path")
                }
                guard imagePath == nil else {
                    throw HelperError.usage("--image may be supplied only once")
                }
                imagePath = raw[index + 1]
                index += 2
            case "--languages":
                guard index + 1 < raw.count else {
                    throw HelperError.usage(
                        "--languages requires a comma-separated BCP-47 language list"
                    )
                }
                languages.append(
                    contentsOf: raw[index + 1]
                        .split(separator: ",", omittingEmptySubsequences: true)
                        .map { String($0).trimmingCharacters(in: .whitespacesAndNewlines) }
                        .filter { !$0.isEmpty }
                )
                index += 2
            case "--language":
                guard index + 1 < raw.count else {
                    throw HelperError.usage("--language requires one BCP-47 language tag")
                }
                let language = raw[index + 1].trimmingCharacters(in: .whitespacesAndNewlines)
                guard !language.isEmpty else {
                    throw HelperError.usage("--language must not be empty")
                }
                languages.append(language)
                index += 2
            default:
                throw HelperError.usage("Unknown argument: \(option)")
            }
        }

        guard let imagePath, !imagePath.isEmpty else {
            throw HelperError.usage(
                "Usage: frameledger-vision-ocr --image IMAGE.png --languages zh-Hans,en-US"
            )
        }
        guard !languages.isEmpty else {
            throw HelperError.usage("At least one recognition language is required")
        }

        var seen: Set<String> = []
        let uniqueLanguages = languages.filter { seen.insert($0).inserted }
        return Arguments(imagePath: imagePath, languages: uniqueLanguages)
    }
}

private struct ProtocolRequest: Decodable {
    let protocolName: String
    let imagePath: String
    let languages: [String]
    let routeKind: String
    let roiNormalized: [Double]
    let bboxOrigin: String

    enum CodingKeys: String, CodingKey {
        case protocolName = "protocol"
        case imagePath = "image_path"
        case languages
        case routeKind = "route_kind"
        case roiNormalized = "roi_normalized"
        case bboxOrigin = "bbox_origin"
    }
}

private struct Invocation {
    let imagePath: String
    let languages: [String]
    let routeKind: String
    let roi: NormalizedROI
    let protocolMode: Bool
}

private struct NormalizedROI {
    let x: Double
    let y: Double
    let width: Double
    let height: Double

    var values: [Double] { [x, y, width, height] }

    static let fullImage = NormalizedROI(x: 0, y: 0, width: 1, height: 1)

    static func validated(_ values: [Double]) throws -> NormalizedROI {
        guard values.count == 4, values.allSatisfy({ $0.isFinite }) else {
            throw HelperError.input("roi_normalized must contain four finite numbers")
        }
        let roi = NormalizedROI(
            x: values[0],
            y: values[1],
            width: values[2],
            height: values[3]
        )
        guard roi.x >= 0, roi.y >= 0, roi.width > 0, roi.height > 0,
              roi.x <= 1, roi.y <= 1,
              roi.x + roi.width <= 1.000001,
              roi.y + roi.height <= 1.000001
        else {
            throw HelperError.input(
                "roi_normalized must be a positive top-left normalized rectangle inside the image"
            )
        }
        return roi
    }
}

private struct BoundingBox: Codable {
    let x: Double
    let y: Double
    let width: Double
    let height: Double
}

private struct Observation: Codable {
    let order: Int
    let visionIndex: Int
    let text: String
    let confidence: Double
    let bbox: [Double]

    enum CodingKeys: String, CodingKey {
        case order
        case visionIndex = "vision_index"
        case text
        case confidence
        case bbox
    }
}

private struct ImageMetadata: Codable {
    let path: String
    let widthPixels: Int
    let heightPixels: Int

    enum CodingKeys: String, CodingKey {
        case path
        case widthPixels = "width_pixels"
        case heightPixels = "height_pixels"
    }
}

private struct OperatingSystemMetadata: Codable {
    let versionString: String
    let major: Int
    let minor: Int
    let patch: Int

    enum CodingKeys: String, CodingKey {
        case versionString = "version_string"
        case major
        case minor
        case patch
    }
}

private struct EngineMetadata: Codable {
    let name: String
    let requestRevision: Int
    let defaultRevision: Int
    let currentRevision: Int
    let supportedRevisions: [Int]
    let recognitionLevel: String
    let usesLanguageCorrection: Bool
    let recognitionLanguages: [String]
    let bboxOrigin: String
    let readingOrder: String
    let orderBase: Int
    let routeKind: String
    let requestedROINormalized: [Double]
    let effectiveROINormalized: [Double]
    let visionFrameworkVersion: String?
    let visionFrameworkBuild: String?
    let operatingSystem: OperatingSystemMetadata

    enum CodingKeys: String, CodingKey {
        case name
        case requestRevision = "request_revision"
        case defaultRevision = "default_revision"
        case currentRevision = "current_revision"
        case supportedRevisions = "supported_revisions"
        case recognitionLevel = "recognition_level"
        case usesLanguageCorrection = "uses_language_correction"
        case recognitionLanguages = "recognition_languages"
        case bboxOrigin = "bbox_origin"
        case readingOrder = "reading_order"
        case orderBase = "order_base"
        case routeKind = "route_kind"
        case requestedROINormalized = "requested_roi_normalized"
        case effectiveROINormalized = "effective_roi_normalized"
        case visionFrameworkVersion = "vision_framework_version"
        case visionFrameworkBuild = "vision_framework_build"
        case operatingSystem = "operating_system"
    }
}

private struct ProtocolSuccessPayload: Codable {
    let protocolName: String
    let engine: EngineMetadata
    let observations: [Observation]

    enum CodingKeys: String, CodingKey {
        case protocolName = "protocol"
        case engine
        case observations
    }
}

private struct SuccessPayload: Codable {
    let schemaVersion: Int
    let engine: EngineMetadata
    let image: ImageMetadata
    let bboxCoordinateSystem: String
    let readingOrder: String
    let orderBase: Int
    let observations: [Observation]

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case engine
        case image
        case bboxCoordinateSystem = "bbox_coordinate_system"
        case readingOrder = "reading_order"
        case orderBase = "order_base"
        case observations
    }
}

private struct ErrorEnvelope: Codable {
    let schemaVersion: Int
    let error: ErrorPayload

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case error
    }
}

private struct ErrorPayload: Codable {
    let code: String
    let message: String
    var nativeDomain: String? = nil
    var nativeCode: Int? = nil
    var requestedLanguages: [String]? = nil
    var supportedLanguages: [String]? = nil

    enum CodingKeys: String, CodingKey {
        case code
        case message
        case nativeDomain = "native_domain"
        case nativeCode = "native_code"
        case requestedLanguages = "requested_languages"
        case supportedLanguages = "supported_languages"
    }
}

private struct UnorderedObservation {
    let visionIndex: Int
    let text: String
    let confidence: Double
    let bbox: BoundingBox
}

private struct LoadedPNG {
    let image: CGImage
    let displayWidth: Int
    let displayHeight: Int
    let canonicalPath: String
}

private struct CroppedImage {
    let image: CGImage
    let effectiveROI: NormalizedROI
}

private func jsonEncoder() -> JSONEncoder {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
    return encoder
}

private func writeJSON<T: Encodable>(_ value: T, to handle: FileHandle) throws {
    var data = try jsonEncoder().encode(value)
    data.append(0x0A)
    handle.write(data)
}

private func clamped(_ value: CGFloat) -> CGFloat {
    min(1.0, max(0.0, value))
}

private func rounded(_ value: CGFloat) -> Double {
    let precision = 1_000_000.0
    return (Double(value) * precision).rounded() / precision
}

private func topLeftBoundingBox(_ visionBox: CGRect) -> BoundingBox {
    let x = clamped(visionBox.origin.x)
    let y = clamped(1.0 - visionBox.origin.y - visionBox.height)
    let width = min(clamped(visionBox.width), 1.0 - x)
    let height = min(clamped(visionBox.height), 1.0 - y)
    return BoundingBox(
        x: rounded(x),
        y: rounded(y),
        width: rounded(width),
        height: rounded(height)
    )
}

private func fullImageBoundingBox(
    _ visionBox: CGRect,
    within roi: NormalizedROI
) -> BoundingBox {
    let local = topLeftBoundingBox(visionBox)
    let x = roi.x + local.x * roi.width
    let y = roi.y + local.y * roi.height
    let width = local.width * roi.width
    let height = local.height * roi.height
    return BoundingBox(
        x: rounded(CGFloat(min(1, max(0, x)))),
        y: rounded(CGFloat(min(1, max(0, y)))),
        width: rounded(CGFloat(min(max(0, 1 - x), max(0, width)))),
        height: rounded(CGFloat(min(max(0, 1 - y), max(0, height))))
    )
}

private func loadPNG(at rawPath: String) throws -> LoadedPNG {
    let url = URL(fileURLWithPath: rawPath).standardizedFileURL
    var isDirectory: ObjCBool = false
    guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory),
          !isDirectory.boolValue
    else {
        throw HelperError.input("Input image does not exist or is not a regular file")
    }

    let data: Data
    do {
        data = try Data(contentsOf: url, options: [.mappedIfSafe])
    } catch {
        throw HelperError.input("Input image could not be read: \(error.localizedDescription)")
    }
    let pngSignature: [UInt8] = [137, 80, 78, 71, 13, 10, 26, 10]
    guard data.count >= pngSignature.count,
          Array(data.prefix(pngSignature.count)) == pngSignature
    else {
        throw HelperError.input("Input must be a PNG file with a valid PNG signature")
    }
    guard let source = CGImageSourceCreateWithData(data as CFData, nil),
          CGImageSourceGetCount(source) == 1,
          let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
    else {
        throw HelperError.input("PNG could not be decoded as one image")
    }

    let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any]
    let orientationRaw = (properties?[kCGImagePropertyOrientation] as? NSNumber)?.uint32Value ?? 1
    let orientation = CGImagePropertyOrientation(rawValue: orientationRaw) ?? .up
    if orientation == .up {
        return LoadedPNG(
            image: image,
            displayWidth: image.width,
            displayHeight: image.height,
            canonicalPath: url.path
        )
    }

    let oriented = CIImage(cgImage: image).oriented(forExifOrientation: Int32(orientation.rawValue))
    let orientedExtent = oriented.extent.integral
    let zeroOrigin = oriented.transformed(
        by: CGAffineTransform(
            translationX: -orientedExtent.origin.x,
            y: -orientedExtent.origin.y
        )
    )
    let displayExtent = CGRect(
        x: 0,
        y: 0,
        width: orientedExtent.width,
        height: orientedExtent.height
    )
    let context = CIContext(options: [
        .cacheIntermediates: false,
        .useSoftwareRenderer: true,
    ])
    guard let displayImage = context.createCGImage(zeroOrigin, from: displayExtent) else {
        throw HelperError.input("PNG orientation could not be normalized")
    }

    return LoadedPNG(
        image: displayImage,
        displayWidth: displayImage.width,
        displayHeight: displayImage.height,
        canonicalPath: url.path
    )
}

private func crop(_ loaded: LoadedPNG, to requestedROI: NormalizedROI) throws -> CroppedImage {
    let imageWidth = Double(loaded.displayWidth)
    let imageHeight = Double(loaded.displayHeight)
    // CGImage cropping treats the first pixel/first data row as (0, 0), so
    // these pixel coordinates already use the protocol's top-left origin.
    let left = floor(requestedROI.x * imageWidth)
    let right = ceil((requestedROI.x + requestedROI.width) * imageWidth)
    let top = floor(requestedROI.y * imageHeight)
    let bottom = ceil((requestedROI.y + requestedROI.height) * imageHeight)
    let pixelRect = CGRect(
        x: max(0, left),
        y: max(0, top),
        width: min(imageWidth, right) - max(0, left),
        height: min(imageHeight, bottom) - max(0, top)
    ).integral
    guard pixelRect.width >= 1, pixelRect.height >= 1 else {
        throw HelperError.input("roi_normalized resolves to an empty pixel rectangle")
    }

    guard let image = loaded.image.cropping(to: pixelRect) else {
        throw HelperError.input("roi_normalized could not be cropped from the PNG")
    }

    let effectiveROI = NormalizedROI(
        x: pixelRect.minX / imageWidth,
        y: pixelRect.minY / imageHeight,
        width: pixelRect.width / imageWidth,
        height: pixelRect.height / imageHeight
    )
    return CroppedImage(image: image, effectiveROI: effectiveROI)
}

private func runOCR(invocation: Invocation) throws -> SuccessPayload {
    let loaded = try loadPNG(at: invocation.imagePath)
    let cropped = try crop(loaded, to: invocation.roi)
    let request = VNRecognizeTextRequest()
    request.revision = VNRecognizeTextRequest.currentRevision
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = true
    request.recognitionLanguages = invocation.languages

    let supportedLanguages: [String]
    do {
        supportedLanguages = try request.supportedRecognitionLanguages()
    } catch {
        throw HelperError.vision(error)
    }
    let supportedSet = Set(supportedLanguages)
    let unsupported = invocation.languages.filter { !supportedSet.contains($0) }
    guard unsupported.isEmpty else {
        throw HelperError.unsupportedLanguages(
            requested: invocation.languages,
            supported: supportedLanguages.sorted()
        )
    }

    let handler = VNImageRequestHandler(
        cgImage: cropped.image,
        orientation: .up,
        options: [:]
    )
    do {
        try handler.perform([request])
    } catch {
        throw HelperError.vision(error)
    }

    let unordered: [UnorderedObservation] = (request.results ?? []).enumerated().compactMap {
        visionIndex, observation in
        guard let candidate = observation.topCandidates(1).first else {
            return nil
        }
        return UnorderedObservation(
            visionIndex: visionIndex,
            text: candidate.string,
            confidence: rounded(CGFloat(candidate.confidence)),
            bbox: fullImageBoundingBox(
                observation.boundingBox,
                within: cropped.effectiveROI
            )
        )
    }
    let sorted = unordered.sorted { left, right in
        if left.bbox.y != right.bbox.y {
            return left.bbox.y < right.bbox.y
        }
        if left.bbox.x != right.bbox.x {
            return left.bbox.x < right.bbox.x
        }
        return left.visionIndex < right.visionIndex
    }
    let observations = sorted.enumerated().map { order, item in
        Observation(
            order: order,
            visionIndex: item.visionIndex,
            text: item.text,
            confidence: item.confidence,
            bbox: [item.bbox.x, item.bbox.y, item.bbox.width, item.bbox.height]
        )
    }

    let os = ProcessInfo.processInfo.operatingSystemVersion
    let frameworkBundle = Bundle(for: VNRequest.self)
    let frameworkVersion = frameworkBundle.object(
        forInfoDictionaryKey: "CFBundleShortVersionString"
    ) as? String
    let frameworkBuild = frameworkBundle.object(
        forInfoDictionaryKey: "CFBundleVersion"
    ) as? String
    let supportedRevisions = VNRecognizeTextRequest.supportedRevisions.map { Int($0) }

    return SuccessPayload(
        schemaVersion: outputSchemaVersion,
        engine: EngineMetadata(
            name: "apple_vision_vnrecognize_text_request",
            requestRevision: Int(request.revision),
            defaultRevision: Int(VNRecognizeTextRequest.defaultRevision),
            currentRevision: Int(VNRecognizeTextRequest.currentRevision),
            supportedRevisions: supportedRevisions,
            recognitionLevel: "accurate",
            usesLanguageCorrection: request.usesLanguageCorrection,
            recognitionLanguages: invocation.languages,
            bboxOrigin: "top_left_normalized",
            readingOrder: "top_to_bottom_then_left_to_right",
            orderBase: 0,
            routeKind: invocation.routeKind,
            requestedROINormalized: invocation.roi.values,
            effectiveROINormalized: cropped.effectiveROI.values,
            visionFrameworkVersion: frameworkVersion,
            visionFrameworkBuild: frameworkBuild,
            operatingSystem: OperatingSystemMetadata(
                versionString: ProcessInfo.processInfo.operatingSystemVersionString,
                major: os.majorVersion,
                minor: os.minorVersion,
                patch: os.patchVersion
            )
        ),
        image: ImageMetadata(
            path: loaded.canonicalPath,
            widthPixels: loaded.displayWidth,
            heightPixels: loaded.displayHeight
        ),
        bboxCoordinateSystem: "top_left_normalized",
        readingOrder: "top_to_bottom_then_left_to_right",
        orderBase: 0,
        observations: observations
    )
}

private func invocationFromStandardInput() throws -> Invocation {
    let input = FileHandle.standardInput.readDataToEndOfFile()
    guard !input.isEmpty else {
        throw HelperError.usage(
            "Expected one frameledger-ocr-helper-v1 JSON request on stdin"
        )
    }
    let request: ProtocolRequest
    do {
        request = try JSONDecoder().decode(ProtocolRequest.self, from: input)
    } catch {
        throw HelperError.input("stdin is not a valid OCR helper request: \(error.localizedDescription)")
    }
    guard request.protocolName == helperProtocol else {
        throw HelperError.input(
            "Unsupported helper protocol: \(request.protocolName)"
        )
    }
    guard request.bboxOrigin == "top_left_normalized" else {
        throw HelperError.input("bbox_origin must be top_left_normalized")
    }
    let knownKinds = Set(["presentation", "table", "chart", "unknown"])
    guard knownKinds.contains(request.routeKind) else {
        throw HelperError.input("route_kind is not recognized: \(request.routeKind)")
    }
    let languages = request.languages.map {
        $0.trimmingCharacters(in: .whitespacesAndNewlines)
    }.filter { !$0.isEmpty }
    guard !languages.isEmpty else {
        throw HelperError.input("languages must contain at least one BCP-47 language tag")
    }
    var seenLanguages: Set<String> = []
    let uniqueLanguages = languages.filter { seenLanguages.insert($0).inserted }
    let roi = try NormalizedROI.validated(request.roiNormalized)
    return Invocation(
        imagePath: request.imagePath,
        languages: uniqueLanguages,
        routeKind: request.routeKind,
        roi: roi,
        protocolMode: true
    )
}

private func invocationFromArguments(_ raw: [String]) throws -> Invocation {
    let arguments = try Arguments.parse(raw)
    return Invocation(
        imagePath: arguments.imagePath,
        languages: arguments.languages,
        routeKind: "unknown",
        roi: .fullImage,
        protocolMode: false
    )
}

do {
    let rawArguments = Array(CommandLine.arguments.dropFirst())
    let invocation = try rawArguments.isEmpty
        ? invocationFromStandardInput()
        : invocationFromArguments(rawArguments)
    let payload = try runOCR(invocation: invocation)
    do {
        if invocation.protocolMode {
            try writeJSON(
                ProtocolSuccessPayload(
                    protocolName: helperProtocol,
                    engine: payload.engine,
                    observations: payload.observations
                ),
                to: .standardOutput
            )
        } else {
            try writeJSON(payload, to: .standardOutput)
        }
    } catch {
        throw HelperError.encoding(error)
    }
} catch let error as HelperError {
    let envelope = ErrorEnvelope(schemaVersion: outputSchemaVersion, error: error.payload)
    if let data = try? jsonEncoder().encode(envelope) {
        FileHandle.standardError.write(data)
        FileHandle.standardError.write(Data([0x0A]))
    } else {
        FileHandle.standardError.write(Data("fatal: could not encode error\n".utf8))
    }
    exit(error.exitCode)
} catch {
    let wrapped = HelperError.vision(error)
    let envelope = ErrorEnvelope(schemaVersion: outputSchemaVersion, error: wrapped.payload)
    if let data = try? jsonEncoder().encode(envelope) {
        FileHandle.standardError.write(data)
        FileHandle.standardError.write(Data([0x0A]))
    }
    exit(wrapped.exitCode)
}
