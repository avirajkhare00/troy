import MLXLMCommon
import SwiftUI

struct ChatMessage: Identifiable {
    let id = UUID()
    let role: String  // "user" | "assistant" | "tool"
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
    @State private var itinerary = Itinerary()
    @State private var session: ChatSession?
    @State private var tools: TravelTools?
    @State private var messages: [ChatMessage] = []
    @State private var input = ""
    @State private var generating = false
    @State private var showItinerary = false
    @AppStorage("hubRepo") private var hubRepo = ""

    private static let instructions = """
        You are a travel planning assistant. Use the available tools to look up \
        flights, hotels and weather, and build the user's trip with \
        add_to_itinerary. Keep replies short. All data comes from the tools. \
        Never guess the departure city — if the user has not said where they \
        are flying from, ask before calling search_flights. You cannot book or \
        purchase anything, and you have no train or bus tools.
        """

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
            .navigationTitle("TroyTravel")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                if case .ready = store.state {
                    Button {
                        showItinerary = true
                    } label: {
                        Label("Itinerary", systemImage: "list.clipboard")
                    }
                    .badge(itinerary.items.count)
                }
            }
            .sheet(isPresented: $showItinerary) { itineraryView }
        }
    }

    private var loader: some View {
        VStack(spacing: 16) {
            Image(systemName: "airplane.circle.fill")
                .font(.system(size: 48))
            Text("Plan a trip with a Troy fine-tuned model — tools and all, entirely on this device.")
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)
            Text("Demo app: flights, hotels and weather are mock data.")
                .font(.footnote)
                .foregroundStyle(.tertiary)

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
                TextField("Plan me a weekend in Jaipur…", text: $input, axis: .vertical)
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

    private var itineraryView: some View {
        NavigationStack {
            List {
                if itinerary.items.isEmpty {
                    Text("Nothing planned yet — ask the assistant to build your trip.")
                        .foregroundStyle(.secondary)
                }
                ForEach(itinerary.items) { item in
                    HStack(alignment: .firstTextBaseline) {
                        Text("Day \(item.day)")
                            .font(.caption.bold())
                            .foregroundStyle(.tint)
                        VStack(alignment: .leading) {
                            Text(item.time).font(.caption).foregroundStyle(.secondary)
                            Text(item.activity)
                        }
                    }
                }
            }
            .navigationTitle("Itinerary")
            .navigationBarTitleDisplayMode(.inline)
        }
        .presentationDetents([.medium, .large])
    }

    private func bubble(_ message: ChatMessage) -> some View {
        HStack {
            if message.role == "user" { Spacer(minLength: 40) }
            if message.role == "tool" {
                Label(message.text, systemImage: "wrench.and.screwdriver")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                Text(markdown(
                    message.isThinking
                        ? "thinking… (\(max(1, message.text.count / 4)) tokens)"
                        : message.visibleText))
                    .padding(10)
                    .background(
                        message.role == "user"
                            ? AnyShapeStyle(.tint.opacity(0.2)) : AnyShapeStyle(.fill.tertiary),
                        in: RoundedRectangle(cornerRadius: 12))
            }
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
            let tools = TravelTools(itinerary: itinerary)
            self.tools = tools
            // temperature 0 — greedy decoding keeps a small fine-tune's
            // tool-calling deterministic; sampling makes it flaky run-to-run.
            // Thinking is unlimited by choice; the live token counter in the
            // thinking bubble is what separates "working" from "frozen".
            session = ChatSession(
                container,
                instructions: Self.instructions,
                generateParameters: .init(temperature: 0),
                tools: tools.specs,
                toolDispatch: { call in
                    let result = try await tools.dispatch(call)
                    await MainActor.run {
                        // A bubble holding only <think> content is dead weight
                        // once a tool call follows it — drop it.
                        if let last = self.messages.last, last.role == "assistant",
                            last.visibleText.isEmpty
                        {
                            self.messages.removeLast()
                        }
                        let preview = result.count > 120
                            ? result.prefix(120) + "…" : Substring(result)
                        self.messages.append(
                            ChatMessage(
                                role: "tool",
                                text: "\(call.function.name) → \(preview)"))
                    }
                    return result
                }
            )
        }
    }

    private func send() {
        let prompt = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty, !generating, let session else { return }
        input = ""
        tools?.resetTurn()
        messages.append(ChatMessage(role: "user", text: prompt))
        generating = true
        Task {
            do {
                for try await chunk in session.streamResponse(to: prompt) {
                    // Only open an assistant bubble when text actually arrives,
                    // so tool-call rounds don't leave empty bubbles behind.
                    if messages.last?.role != "assistant" {
                        if chunk.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                            continue
                        }
                        messages.append(ChatMessage(role: "assistant", text: ""))
                    }
                    messages[messages.count - 1].text += chunk
                }
                // Drop a bubble whose content was entirely <think> reasoning.
                if let last = messages.last, last.role == "assistant",
                    last.visibleText.isEmpty
                {
                    messages.removeLast()
                }
            } catch is ToolLimitReached {
                messages.append(
                    ChatMessage(
                        role: "assistant",
                        text: "I hit the tool limit for this request — check the itinerary "
                            + "(top right) for what I planned so far."))
            } catch {
                messages.append(
                    ChatMessage(role: "assistant", text: "[error: \(error.localizedDescription)]"))
            }
            generating = false
        }
    }
}
