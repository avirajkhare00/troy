import SwiftUI

struct ContentView: View {
    @State private var store = ModelStore()
    @State private var engine = WorkerEngine()
    @AppStorage("meshURL") private var meshURL = ""
    @AppStorage("meshToken") private var meshToken = ""
    @AppStorage("hubRepo") private var hubRepo = "mlx-community/Qwen3-4B-4bit"
    @Environment(\.scenePhase) private var scenePhase

    var body: some View {
        NavigationStack {
            Form {
                Section("Coordinator") {
                    TextField("http://192.168.1.5:8765", text: $meshURL)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .keyboardType(.URL)
                    SecureField("Token from `troy mesh serve`", text: $meshToken)
                }

                Section("Teacher model") {
                    if ModelStore.bundledModel == nil {
                        TextField("mlx-community/Qwen3-4B-4bit", text: $hubRepo)
                            .textInputAutocapitalization(.never)
                            .autocorrectionDisabled()
                    }
                    modelRow
                }

                Section("Worker") {
                    statusRow
                    if engine.isRunning || engine.completed > 0 {
                        LabeledContent("Completed items", value: "\(engine.completed)")
                        LabeledContent("Records accepted", value: "\(engine.accepted)")
                        if engine.meshTarget > 0 {
                            LabeledContent(
                                "Mesh progress",
                                value: "\(engine.meshRecords)/\(engine.meshTarget)")
                        }
                        if let started = engine.startedAt {
                            LabeledContent(
                                "Uptime",
                                value: started.formatted(.relative(presentation: .numeric)))
                        }
                    }
                    startStopButton
                }

                Section {
                    Text("Keep the app in the foreground — the screen stays awake "
                        + "while working. Dropped work is requeued by the Mac.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                    Link("Built by Aviraj", destination: URL(string: "https://aviraj.dev")!)
                        .font(.footnote)
                }
            }
            .navigationTitle("TroyWorker")
            .navigationBarTitleDisplayMode(.inline)
        }
        .onChange(of: scenePhase) {
            if scenePhase != .active, engine.isRunning {
                engine.stop()
            }
        }
    }

    @ViewBuilder
    private var modelRow: some View {
        switch store.state {
        case .idle:
            Button("Load model") { Task { await store.load(hubRepo: hubRepo) } }
        case .loading(let what):
            ProgressView(what)
        case .downloading(let fraction):
            ProgressView("Downloading…", value: fraction)
        case .ready(_, let name):
            Label(name, systemImage: "checkmark.circle.fill")
                .foregroundStyle(.green)
        case .failed(let message):
            Text(message).foregroundStyle(.red).font(.callout)
            Button("Retry") { Task { await store.load(hubRepo: hubRepo) } }
        }
    }

    @ViewBuilder
    private var statusRow: some View {
        switch engine.status {
        case .idle:
            Text("Idle").foregroundStyle(.secondary)
        case .working(let id):
            HStack {
                ProgressView()
                Text("Generating \(id)…").padding(.leading, 6)
            }
        case .waiting:
            Text("Queue empty — waiting…").foregroundStyle(.secondary)
        case .finished:
            Label("Mesh target reached", systemImage: "checkmark.seal.fill")
                .foregroundStyle(.green)
        case .failed(let message):
            Text(message).foregroundStyle(.red).font(.callout)
        }
    }

    @ViewBuilder
    private var startStopButton: some View {
        if engine.isRunning {
            Button("Stop", role: .destructive) { engine.stop() }
        } else {
            Button("Start working") {
                if case .ready(let container, _) = store.state {
                    engine.start(container: container, urlString: meshURL, token: meshToken)
                }
            }
            .disabled(!modelReady || meshURL.isEmpty || meshToken.isEmpty)
        }
    }

    private var modelReady: Bool {
        if case .ready = store.state { return true }
        return false
    }
}
