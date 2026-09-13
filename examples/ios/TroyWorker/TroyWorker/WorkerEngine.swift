import Foundation
import MLXLMCommon
import UIKit

/// Pulls work from a `troy mesh serve` coordinator, generates with the local
/// teacher model, and posts raw completions back. Foreground-only: the
/// coordinator's lease timeout covers anything we drop when backgrounded.
@Observable
@MainActor
final class WorkerEngine {
    enum Status: Equatable {
        case idle
        case working(String)  // current work item id
        case waiting          // queue momentarily empty
        case finished         // coordinator reached its target
        case failed(String)
    }

    private(set) var status: Status = .idle
    private(set) var completed = 0
    private(set) var accepted = 0
    private(set) var meshRecords = 0
    private(set) var meshTarget = 0
    private(set) var startedAt: Date?

    private var task: Task<Void, Never>?
    private let workerName = UIDevice.current.name

    var isRunning: Bool { task != nil }

    func start(container: ModelContainer, urlString: String, token: String) {
        guard task == nil else { return }
        guard let url = URL(string: urlString.trimmingCharacters(in: .whitespacesAndNewlines)),
            url.scheme?.hasPrefix("http") == true
        else {
            status = .failed("Enter the coordinator URL, e.g. http://192.168.1.5:8765")
            return
        }
        let client = MeshClient(baseURL: url, token: token)
        completed = 0
        accepted = 0
        startedAt = Date()
        UIApplication.shared.isIdleTimerDisabled = true
        task = Task { await run(container: container, client: client) }
    }

    func stop(status newStatus: Status = .idle) {
        task?.cancel()
        task = nil
        UIApplication.shared.isIdleTimerDisabled = false
        status = newStatus
    }

    private func run(container: ModelContainer, client: MeshClient) async {
        var backoff: Double = 1
        while !Task.isCancelled {
            let work: MeshClient.WorkResponse
            do {
                work = try await client.fetchWork(worker: workerName, max: 1)
                backoff = 1
            } catch is CancellationError {
                return
            } catch {
                status = .failed("\(error.localizedDescription) — retrying")
                if !(await sleep(backoff)) { return }
                backoff = min(backoff * 2, 30)
                continue
            }
            if work.done {
                stop(status: .finished)
                return
            }
            guard let item = work.items.first else {
                status = .waiting
                if !(await sleep(work.retrySeconds ?? 5)) { return }
                continue
            }
            status = .working(item.id)

            let raw: String
            do {
                // Fresh session per item: ChatSession is conversational and
                // would leak earlier prompts into later ones.
                let session = ChatSession(
                    container,
                    generateParameters: .init(
                        maxTokens: item.maxTokens,
                        temperature: Float(item.temperature)),
                    additionalContext: ["enable_thinking": false])
                var text = ""
                for try await chunk in session.streamResponse(to: item.prompt) {
                    if Task.isCancelled { return }
                    text += chunk
                }
                raw = text
            } catch {
                status = .failed("Generation failed: \(error.localizedDescription)")
                continue  // coordinator requeues the item after the lease expires
            }

            // The lease requeue also covers a lost POST, so retry a few times
            // and then move on rather than blocking the loop.
            for attempt in 1...3 {
                do {
                    let reply = try await client.postResult(
                        worker: workerName, id: item.id, raw: raw)
                    completed += 1
                    accepted += reply.accepted
                    meshRecords = reply.records
                    meshTarget = reply.target
                    if reply.done {
                        stop(status: .finished)
                        return
                    }
                    break
                } catch is CancellationError {
                    return
                } catch {
                    if attempt == 3 { break }
                    if !(await sleep(Double(attempt * 2))) { return }
                }
            }
        }
    }

    private func sleep(_ seconds: Double) async -> Bool {
        (try? await Task.sleep(for: .seconds(seconds))) != nil
    }
}
