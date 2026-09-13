import Foundation
import HuggingFace
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers

/// Loads a Troy-exported model: a `TroyModel` folder bundled with the app if
/// present, otherwise a Hugging Face repo (e.g. one uploaded with `troy push`).
@Observable
@MainActor
final class ModelStore {
    enum State {
        case idle
        case loading(String)
        case downloading(Double)
        case ready(ModelContainer, name: String)
        case failed(String)
    }

    private(set) var state: State = .idle

    /// A `TroyModel` folder reference inside the app bundle (drag your
    /// `troy export -f ios` output into Xcode as a folder reference).
    static var bundledModel: URL? {
        guard let url = Bundle.main.url(forResource: "TroyModel", withExtension: nil),
            FileManager.default.fileExists(atPath: url.appendingPathComponent("config.json").path)
        else { return nil }
        return url
    }

    func load(hubRepo: String) async {
        if case .loading = state { return }

        do {
            if let local = Self.bundledModel {
                state = .loading("Loading bundled model…")
                let container = try await loadModelContainer(
                    from: local, using: #huggingFaceTokenizerLoader())
                state = .ready(container, name: "TroyModel (bundled)")
                return
            }

            let repo = hubRepo.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !repo.isEmpty else {
                state = .failed("No bundled TroyModel folder found — enter a Hugging Face repo.")
                return
            }
            state = .downloading(0)
            let container = try await loadModelContainer(
                from: #hubDownloader(),
                using: #huggingFaceTokenizerLoader(),
                configuration: ModelConfiguration(id: repo)
            ) { progress in
                Task { @MainActor [weak self] in
                    self?.state = .downloading(progress.fractionCompleted)
                }
            }
            state = .ready(container, name: repo)
        } catch {
            state = .failed(error.localizedDescription)
        }
    }
}
