import MLXLMCommon
import SwiftUI

struct ChatMessage: Identifiable {
    let id = UUID()
    let role: String  // "user" | "assistant"
    var text: String

    /// Streamed text with reasoning stripped: hide everything up to a
    /// `</think>` tag, and show a placeholder while a `<think>` is open.
    var visibleText: String {
        if let range = text.range(of: "</think>", options: .backwards) {
            return String(text[range.upperBound...])
                .trimmingCharacters(in: .whitespacesAndNewlines)
        }
        if text.contains("<think>") { return "" }
        return text
    }

    var isThinking: Bool { text.contains("<think>") && !text.contains("</think>") }
}

struct ContentView: View {
    @State private var store = ModelStore()
    @State private var session: ChatSession?
    @State private var messages: [ChatMessage] = []
    @State private var input = ""
    @State private var generating = false
    @AppStorage("hubRepo") private var hubRepo = ""

    var body: some View {
        NavigationStack {
            Group {
                switch store.state {
                case .ready:
                    chat
                default:
                    loader
                }
            }
            .navigationTitle("TroyChat")
            .navigationBarTitleDisplayMode(.inline)
        }
    }

    private var loader: some View {
        VStack(spacing: 16) {
            Image(systemName: "shield.lefthalf.filled")
                .font(.system(size: 48))
            Text("Chat with a model you fine-tuned with Troy — entirely on this device.")
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)

            switch store.state {
            case .loading(let what):
                ProgressView(what)
            case .downloading(let fraction):
                ProgressView("Downloading…", value: fraction)
                    .padding(.horizontal)
            case .failed(let message):
                Text(message)
                    .font(.callout)
                    .foregroundStyle(.red)
                    .multilineTextAlignment(.center)
            default:
                EmptyView()
            }

            if ModelStore.bundledModel == nil {
                TextField("you/your-troy-model", text: $hubRepo)
                    .textFieldStyle(.roundedBorder)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
            }
            Button("Load model") {
                Task { await load() }
            }
            .buttonStyle(.borderedProminent)
            .disabled({ if case .loading = store.state { true } else if case .downloading = store.state { true } else { false } }())

            Spacer()
            Link("Built by Aviraj", destination: URL(string: "https://aviraj.dev")!)
                .font(.footnote)
                .foregroundStyle(.secondary)
        }
        .padding()
    }

    private var chat: some View {
        VStack(spacing: 0) {
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 12) {
                        ForEach(messages) { message in
                            bubble(message)
                        }
                    }
                    .padding()
                }
                .onChange(of: messages.last?.text) {
                    if let last = messages.last {
                        proxy.scrollTo(last.id, anchor: .bottom)
                    }
                }
            }
            HStack {
                TextField("Message", text: $input, axis: .vertical)
                    .textFieldStyle(.roundedBorder)
                    .onSubmit { send() }
                Button {
                    send()
                } label: {
                    Image(systemName: "arrow.up.circle.fill").font(.title2)
                }
                .disabled(generating || input.trimmingCharacters(in: .whitespaces).isEmpty)
            }
            .padding()
        }
    }

    private func bubble(_ message: ChatMessage) -> some View {
        HStack {
            if message.role == "user" { Spacer(minLength: 40) }
            Text(markdown(message.isThinking ? "thinking…" : message.visibleText))
                .padding(10)
                .background(
                    message.role == "user"
                        ? AnyShapeStyle(.tint.opacity(0.2)) : AnyShapeStyle(.fill.tertiary),
                    in: RoundedRectangle(cornerRadius: 12))
            if message.role != "user" { Spacer(minLength: 40) }
        }
        .id(message.id)
    }

    /// Render inline markdown (bold/italic/code); headers become bold lines.
    private func markdown(_ text: String) -> AttributedString {
        guard !text.isEmpty else { return AttributedString("…") }
        let cleaned = text
            .split(separator: "\n", omittingEmptySubsequences: false)
            .map { line -> String in
                let l = line.drop(while: { $0 == "#" })
                return line.first == "#" ? "**\(l.trimmingCharacters(in: .whitespaces))**" : String(line)
            }
            .joined(separator: "\n")
        return (try? AttributedString(
            markdown: cleaned,
            options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace)))
            ?? AttributedString(text)
    }

    private func load() async {
        await store.load(hubRepo: hubRepo)
        if case .ready(let container, _) = store.state {
            // Cap generation and disable Qwen3 thinking (when the model
            // supports the flag) so replies stay fast on-device.
            session = ChatSession(
                container,
                generateParameters: .init(maxTokens: 700),
                additionalContext: ["enable_thinking": false])
        }
    }

    private func send() {
        let prompt = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty, !generating, let session else { return }
        input = ""
        messages.append(ChatMessage(role: "user", text: prompt))
        messages.append(ChatMessage(role: "assistant", text: ""))
        generating = true
        Task {
            do {
                for try await chunk in session.streamResponse(to: prompt) {
                    messages[messages.count - 1].text += chunk
                }
            } catch {
                messages[messages.count - 1].text += "\n[error: \(error.localizedDescription)]"
            }
            generating = false
        }
    }
}
