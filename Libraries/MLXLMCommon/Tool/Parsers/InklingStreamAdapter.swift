// Copyright © 2026 Apple Inc.

// Inkling 2's output protocol, after the chat template and Exo's
// `models/inkling/parser.py` (exo-explore/exo, Apache-2.0).
//
// A turn is a sequence of messages. Each opens with `<|message_model|>` and closes with
// `<|end_message|>`, and the token after the opener says what it holds:
// `<|content_thinking|>` for reasoning, `<|content_text|>` for the answer, or a tool name
// followed by `<|content_invoke_tool_json|>` and `{"name": ..., "args": {...}}` for a call.
// `<|content_model_end_sampling|>` ends the turn.

import Foundation

private enum InklingPayload: Equatable {
    case reasoning
    case response
    case tool(String)
}

/// Token-stream decoder for Inkling 2's message framing and JSON tool calls.
///
/// Markers are matched by token id, so marker-shaped text in a payload is only text.
struct InklingStreamAdapter: TokenStreamDecoder {
    private struct Markers {
        let messageModel: Int
        let contentText: Int
        let contentThinking: Int
        let endMessage: Int
        let invokeToolJSON: Int
        let endSampling: Int
    }

    private enum State {
        /// After `<|message_model|>`: the speaker or tool name, until a content marker.
        case header([Int])
        case payload(InklingPayload, [Int])
        case betweenMessages(reported: Bool)
    }

    /// A header longer than this, or with characters no tool name has, is the
    /// continuation of an answer the prompt began (an assistant prefill).
    private static let maximumNameLength = 64

    private let tokenizer: any Tokenizer
    private let markers: Markers
    private let tools: [[String: any Sendable]]?
    private let validationPolicy: ToolCallValidationPolicy
    // The template has no generation prompt: the model writes `<|message_model|>` itself.
    private var state = State.header([])
    private var detokenizer: any StreamingDetokenizer
    private var stopStringFilter: StopStringFilter
    private var endsInVisibleText = false
    private(set) var rejectedToolCallCount = 0

    let additionalStopTokenIDs: Set<Int>
    let receivesStopTokens = true

    var isInsideReasoning: Bool {
        if case .payload(.reasoning, _) = state { return true }
        return false
    }

    var canEndForSteering: Bool { endsInVisibleText && stopStringFilter.buffer.isEmpty }

    init?(
        tokenizer: any Tokenizer, tools: [[String: any Sendable]]?, stopStrings: Set<String>,
        toolCallPolicy: ToolCallPolicy = .init()
    ) {
        guard let messageModel = tokenizer.convertTokenToId("<|message_model|>"),
            let contentText = tokenizer.convertTokenToId("<|content_text|>"),
            let contentThinking = tokenizer.convertTokenToId("<|content_thinking|>"),
            let endMessage = tokenizer.convertTokenToId("<|end_message|>"),
            let invokeToolJSON = tokenizer.convertTokenToId("<|content_invoke_tool_json|>"),
            let endSampling = tokenizer.convertTokenToId("<|content_model_end_sampling|>"),
            Set([messageModel, contentText, contentThinking, endMessage, invokeToolJSON, endSampling])
                .count == 6
        else { return nil }
        self.tokenizer = tokenizer
        self.markers = Markers(
            messageModel: messageModel, contentText: contentText,
            contentThinking: contentThinking, endMessage: endMessage,
            invokeToolJSON: invokeToolJSON, endSampling: endSampling)
        self.tools = tools
        self.validationPolicy = toolCallPolicy.validation
        self.detokenizer = tokenizer.makeStreamingDetokenizer()
        self.stopStringFilter = StopStringFilter(stopStrings: stopStrings)
        self.additionalStopTokenIDs = [endSampling]
    }

    mutating func push(_ token: Int, emit: (TokenStreamEvent) -> Bool) -> Bool {
        endsInVisibleText = false
        switch token {
        case markers.endSampling:
            let closed = close(emit: emit)
            state = .betweenMessages(reported: false)
            return closed
        case markers.messageModel:
            if case .header(let tokens) = state, tokens.isEmpty { return true }
            let closed = close(emit: emit)
            state = .header([])
            return closed
        case markers.endMessage:
            let closed = close(emit: emit)
            state = .betweenMessages(reported: false)
            return closed
        case markers.contentThinking, markers.contentText:
            guard case .header = state else {
                return emitEvent(.protocolError("Inkling content marker outside a message header"), emit)
            }
            // A header before a content marker names the speaker, which is not shown.
            start(token == markers.contentThinking ? .reasoning : .response)
            return true
        case markers.invokeToolJSON:
            guard case .header(let tokens) = state else {
                return emitEvent(.protocolError("Inkling tool marker outside a message header"), emit)
            }
            let name = tokenizer.decode(tokenIds: tokens, skipSpecialTokens: false)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            guard !name.isEmpty else {
                state = .betweenMessages(reported: false)
                return emitEvent(
                    .rejectedToolCall(
                        RejectedToolCall(
                            reason: .missingToolName, format: .inkling, toolName: nil,
                            rawText: "", detail: "Inkling tool call without a tool name")), emit)
            }
            state = .payload(.tool(name), [])
            return true
        default:
            break
        }

        switch state {
        case .header(var tokens):
            tokens.append(token)
            let text = tokenizer.decode(tokenIds: tokens, skipSpecialTokens: false)
            if Self.couldBeName(text) {
                state = .header(tokens)
                return true
            }
            // Not a name: an answer the prompt began continues here.
            start(.response)
            return tokens.allSatisfy { appendText($0, emit: emit) }
        case .payload(.tool(let name), var tokens):
            tokens.append(token)
            state = .payload(.tool(name), tokens)
            return true
        case .payload:
            return appendText(token, emit: emit)
        case .betweenMessages(let reported):
            guard !reported else { return true }
            state = .betweenMessages(reported: true)
            let text = tokenizer.decode(tokenIds: [token], skipSpecialTokens: false)
            return emitEvent(.protocolError("Unexpected token between Inkling messages: \(text)"), emit)
        }
    }

    mutating func finish(emit: (TokenStreamEvent) -> Bool) -> Bool {
        if case .payload(.tool(let name), _) = state {
            state = .betweenMessages(reported: true)
            return emitEvent(
                .rejectedToolCall(
                    RejectedToolCall(
                        reason: .incompleteOutput, format: .inkling, toolName: name,
                        rawText: "", detail: "Inkling tool call cut off at generation end")),
                emit)
        }
        guard close(emit: emit) else { return false }
        state = .betweenMessages(reported: true)
        if let text = stopStringFilter.finish() { return emit(.response(text)) }
        return true
    }

    private static func couldBeName(_ text: String) -> Bool {
        text.count <= maximumNameLength
            && text.unicodeScalars.allSatisfy {
                CharacterSet.alphanumerics.contains($0) || "_-.".unicodeScalars.contains($0)
            }
    }

    private mutating func start(_ payload: InklingPayload) {
        detokenizer = tokenizer.makeStreamingDetokenizer()
        state = .payload(payload, [])
    }

    /// Streams one token of reasoning or answer text.
    private mutating func appendText(_ token: Int, emit: (TokenStreamEvent) -> Bool) -> Bool {
        guard case .payload(let payload, _) = state, payload == .reasoning || payload == .response
        else { return true }
        detokenizer.append(token: token)
        guard let text = detokenizer.next(), !text.isEmpty else { return true }
        endsInVisibleText = text.last?.isWhitespace == false
        return emitEvent(payload == .reasoning ? .reasoning(text) : .response(text), emit)
    }

    /// Ends the open message: flushes its text, or parses its tool call.
    private mutating func close(emit: (TokenStreamEvent) -> Bool) -> Bool {
        if case .header(let tokens) = state, !tokens.isEmpty {
            // A message that never named its content is the answer the prompt began.
            start(.response)
            guard tokens.allSatisfy({ appendText($0, emit: emit) }) else { return false }
        }
        guard case .payload(let payload, let tokens) = state else { return true }
        state = .betweenMessages(reported: false)
        switch payload {
        case .reasoning:
            return detokenizer.finish().map { emitEvent(.reasoning($0), emit) } ?? true
        case .response:
            return detokenizer.finish().map { emitEvent(.response($0), emit) } ?? true
        case .tool(let name):
            return emitEvent(toolCall(name: name, payload: tokens), emit)
        }
    }

    private mutating func toolCall(name: String, payload: [Int]) -> TokenStreamEvent {
        let text = tokenizer.decode(tokenIds: payload, skipSpecialTokens: false)
        func rejected(_ reason: RejectedToolCall.Reason, _ detail: String) -> TokenStreamEvent {
            .rejectedToolCall(
                RejectedToolCall(
                    reason: reason, format: .inkling, toolName: name, rawText: text,
                    detail: detail))
        }
        guard let data = text.data(using: .utf8),
            let object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
        else {
            return rejected(.malformedSyntax, "Inkling tool call payload is not a JSON object")
        }
        guard (object["name"] as? String).map({ $0 == name }) ?? true else {
            return rejected(.malformedSyntax, "Inkling tool call names two different tools")
        }
        guard let arguments = (object["args"] ?? object["arguments"] ?? [:]) as? [String: Any]
        else {
            return rejected(.invalidArguments, "Inkling tool call arguments are not an object")
        }
        var call = ToolCall(
            function: .init(name: name, arguments: arguments.mapValues { JSONValue.from($0) }),
            id: ToolCallFormat.inkling.generateToolCallID())
        call = ToolArgumentNormalization.normalize(call, tools: tools)
        if validationPolicy == .strict,
            case .invalid(let violations) = ToolSchemaValidator.validate(
                arguments: call.function.arguments, forToolNamed: name, in: tools)
        {
            return rejected(.invalidArguments, ToolSchemaValidator.describe(violations))
        }
        return .toolCall(call)
    }

    private mutating func emitEvent(
        _ event: TokenStreamEvent, _ emit: (TokenStreamEvent) -> Bool
    ) -> Bool {
        switch event {
        case .response(let text):
            let result = stopStringFilter.process(text)
            if let response = result.text, !emit(.response(response)) { return false }
            if result.stopped {
                _ = emit(.stop)
                return false
            }
            return true
        case .toolCall, .rejectedToolCall:
            if case .rejectedToolCall = event { rejectedToolCallCount += 1 }
            if let response = stopStringFilter.finish(), !emit(.response(response)) {
                return false
            }
            return emit(event)
        case .reasoning, .protocolError, .stop:
            return emit(event)
        }
    }
}
